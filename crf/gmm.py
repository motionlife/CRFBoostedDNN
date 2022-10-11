import math
import tensorflow as tf
from tensorflow.keras.layers import Layer
from numpy.polynomial.hermite import hermgauss

pi = math.pi

_2pi = 2 * pi

sg0 = 1 / math.sqrt(_2pi * math.e)

sqrtpi = math.sqrt(pi)

sqrt2 = math.sqrt(2)

sqrt2pi = math.sqrt(_2pi)

# normal pdf constant
ncn = -math.log(_2pi) / 2

# normal distribution entropy constant
ndc = (1 + math.log(_2pi)) / 2


def gausshermite(k=21, dims=None, dtype=tf.float64):
    if dims is None:
        dims = [1, 1, 1, 1, 1, -1]  # Tensor shape = (C, M, N, cliques/unit, L, batch_size, nb)
    x, w = hermgauss(k)
    x, w = tf.cast(x, dtype), tf.cast(w, dtype)
    xn = tf.reshape(x, dims)
    wn = tf.reshape(w / sqrtpi, dims)
    xi, xj = tf.meshgrid(x, x)
    wi, wj = tf.meshgrid(w, w)
    xi = tf.reshape(xi, dims)
    xj = tf.reshape(xj, dims)
    we = tf.reshape(wi * wj / pi, dims)
    return xn, xi, xj, wn, we


class ConditionalGaussianMixture(Layer):
    """conditional marginal mixture of gaussian distribution of y given x"""

    def __init__(self, n_model, height, width, mix_comp, k=21, lr=0.001, dtype=tf.float64):
        super(ConditionalGaussianMixture, self).__init__(name='conditional_gaussian_mixture')
        self.xn, self.xi, self.xj, self.wn, self.we = gausshermite(k, dtype=dtype)
        self.x22 = 2 * self.xn ** 2 - 1
        self.xij = self.xi * self.xj
        self.xi2axj2 = self.xi ** 2 + self.xj ** 2 - 1
        self.xi2mxj2 = self.xi ** 2 - self.xj ** 2
        self.optimizer = tf.keras.optimizers.Adamax(learning_rate=lr)
        self.mu_initializer = tf.random_uniform_initializer(0., 1.)
        self.mu = self.add_weight(name='mu',
                                  shape=(n_model, height, width, 1, mix_comp, 1),
                                  initializer=self.mu_initializer,
                                  constraint=lambda u: tf.clip_by_value(u, 0., 1.),
                                  trainable=True)
        self.sigma_initializer = tf.random_uniform_initializer(0.1, 5.)
        self.sigma = self.add_weight(name='sigma',
                                     shape=(n_model, height, width, 1, mix_comp, 1),
                                     initializer=self.sigma_initializer,
                                     constraint=lambda o: tf.clip_by_value(o, 0.01, 10),
                                     trainable=True)
        self.rou_initializer = tf.random_uniform_initializer(-0.5, 0.5)
        self.rou = self.add_weight(name='rou',
                                   shape=(n_model, height, width, 2, mix_comp, 1),
                                   initializer=self.rou_initializer,
                                   constraint=lambda p: tf.clip_by_value(p, -0.99, 0.99),
                                   trainable=True)
        self.w_initializer = tf.random_uniform_initializer(1, 1)
        self.w = self.add_weight(name='mix_comp_weights',
                                 shape=(n_model, 1, 1, 1, mix_comp, 1),
                                 initializer=self.w_initializer,
                                 constraint=lambda x: tf.clip_by_value(x, -300, 300),
                                 trainable=False)
        self.built = True

    def init_marginal(self):
        for k, initializer in self.__dict__.items():
            if "initializer" in k:
                var = self.__getattribute__(k.replace("_initializer", ""))
                var.assign(initializer(var.shape, var.dtype))
        # reset optimizer, for adam 1st momentum(m) and 2nd momentum(v)
        if self.optimizer.get_weights():
            self.optimizer.set_weights([tf.constant(0).numpy()] +
                                       [tf.zeros_like(v).numpy() for v in self.trainable_variables] * 2)

    def call(self, inputs, full_out=True):
        xn = self.xn * self.sigma * sqrt2 + self.mu
        s = tf.sqrt(1 + self.rou) / 2
        t = tf.sqrt(1 - self.rou) / 2
        a = s + t
        b = s - t
        z1 = a * self.xi + b * self.xj
        z2 = b * self.xi + a * self.xj
        # todo: circle structure may harm the inference by making even more loops (trivial? try pruning)
        u2 = tf.concat([tf.roll(self.mu, -1, 1), tf.roll(self.mu, -1, 2)], 3)
        o2 = tf.concat([tf.roll(self.sigma, -1, 1), tf.roll(self.sigma, -1, 2)], 3)
        x1 = z1 * self.sigma * sqrt2 + self.mu
        x2 = z2 * o2 * sqrt2 + u2
        xe = tf.stack([x1, x2], -1)
        # ------------------------------log pdf -> log of sum of each mixture component-------------------------------
        alpha = tf.nn.softmax(self.w, axis=4)
        alf = tf.expand_dims(alpha, 4)
        u1_ = tf.expand_dims(self.mu, 4)
        o1_ = tf.expand_dims(self.sigma, 4)
        lpn = -((tf.expand_dims(xn, 5) - u1_) / o1_) ** 2 / 2
        lpn = tf.math.log(tf.reduce_sum(tf.exp(lpn) / o1_ * alf, 5) + 1e-307) + ncn
        p = tf.expand_dims(self.rou, 4)
        u2_ = tf.expand_dims(u2, 4)
        o2_ = tf.expand_dims(o2, 4)
        x1_ = (tf.expand_dims(x1, 5) - u1_) / o1_
        x2_ = (tf.expand_dims(x2, 5) - u2_) / o2_
        q = 1 - p ** 2
        z = -(x1_ ** 2 - 2 * p * x1_ * x2_ + x2_ ** 2) / (2 * q)
        lpe = tf.math.log(tf.reduce_sum(tf.exp(z) / (o1_ * o2_ * tf.sqrt(q)) * alf, 5) + 1e-307) + 2 * ncn
        return (xn, xe, lpn, lpe, alpha, z1, z2, o2) if full_out else (xn, xe, lpn, lpe, alpha)

    @tf.function
    def entropy_m1(self):
        """This is the node and edge's entropy contribution to the overall energy term.
        This function used only when the marginal is single gaussian distribution."""
        hn = ndc + tf.math.log(self.sigma + 1e-307)
        o2 = tf.concat([tf.roll(self.sigma, -1, 1), tf.roll(self.sigma, -1, 2)], 3)
        he = 2 * ndc + tf.math.log(self.sigma * o2 + 1e-307) + tf.math.log(1 - self.rou ** 2 + 1e-307) / 2
        return hn, he

    @tf.function
    def bethe_free_energy(self, potential, back_prop=True):
        xn, xe, lpn, lpe, alpha = self(None, full_out=False)
        fn, fe = potential((xn, xe), back_prop=back_prop)
        bfe = -(tf.reduce_sum((fn + 3 * lpn) * self.wn * alpha, [1, 2, 3, 4, 5]) +
                tf.reduce_sum((fe - lpe) * self.we * alpha, [1, 2, 3, 4, 5]))
        return bfe

    @tf.function
    def infer(self, potential, iterations):
        energy = tf.zeros(shape=[self.mu.shape[0]], dtype=self.mu.dtype)
        for i in tf.range(iterations):
            grd, energy = self.gradient(potential)
            self.optimizer.apply_gradients(zip(grd, self.trainable_variables))
            if tf.equal(tf.math.mod(i, 20), 0) or tf.equal(i + 1, iterations):
                tf.print(tf.strings.format('iter: {} dmu = {}, dsigma = {}, drou = {}, Energy = {}', (i,
                         tf.reduce_mean(tf.abs(grd[0])), tf.reduce_mean(tf.abs(grd[1])),
                         tf.reduce_mean(tf.abs(grd[2])), tf.reduce_mean(energy))))
        return energy

    @tf.function
    def gradient(self, potential):
        q = 1 - self.rou ** 2
        sqrtq = tf.sqrt(q)
        xn, xe, lpn, lpe, alpha, z1, z2, o2 = self(None)
        fn_, fe_ = potential((xn, xe))
        fn_ = (fn_ + 3 * lpn) * self.wn
        fe_ = (fe_ - lpe) * self.we
        fn = fn_ * alpha
        fe = fe_ * alpha
        dmu = tf.reduce_sum(fn * self.xn, 5, keepdims=True) / self.sigma * sqrt2
        dsigma = tf.reduce_sum(fn * self.x22, 5, keepdims=True) / self.sigma
        dmu1 = tf.reduce_sum(fe * (z1 - self.rou * z2), 5, keepdims=True) / (self.sigma * q) * sqrt2
        dmu2 = tf.reduce_sum(fe * (z2 - self.rou * z1), 5, keepdims=True) / (o2 * q) * sqrt2
        dsigma1 = tf.reduce_sum(fe * (self.xi2axj2 + self.xi2mxj2 / sqrtq), 5, keepdims=True) / self.sigma
        dsigma2 = tf.reduce_sum(fe * (self.xi2axj2 - self.xi2mxj2 / sqrtq), 5, keepdims=True) / o2
        drou = tf.reduce_sum(fe * (2 * self.xij - self.rou * self.xi2axj2), 5, keepdims=True) / q
        dmu += (tf.reduce_sum(dmu1, 3, True) + tf.roll(dmu2[:, :, :, 0:1, :, :], 1, 1) + tf.roll(dmu2[:, :, :, 1:2, :, :], 1, 2))
        dsigma += (tf.reduce_sum(dsigma1, 3, True) + tf.roll(dsigma2[:, :, :, 0:1, :, :], 1, 1) + tf.roll(dsigma2[:, :, :, 1:2, :, :], 1, 2))
        # dalpha = (tf.reduce_sum(fn_, [1, 2, 3, 5], keepdims=True) + tf.reduce_sum(fe_, [1, 2, 3, 5], keepdims=True))
        # dw = alpha * (dalpha - tf.reduce_sum(dalpha * alpha, 4, keepdims=True))
        energy = tf.reduce_sum(fn, [1, 2, 3, 4, 5]) + tf.reduce_sum(fe, [1, 2, 3, 4, 5])

        return (-dmu, -dsigma, -drou), -energy
