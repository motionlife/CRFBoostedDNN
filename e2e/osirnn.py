import tensorflow as tf
from tensorflow.keras.layers import Layer

from crf.gmm import gausshermite, sqrt2, ndc


class OSI(Layer):
    """One-shot inference - a variational MAP inference reformulated as a RNN layer"""

    def __init__(self, iterations=100, infer_rate=0.01, umin=0., umax=63., k=3, momentum=0.7, **kwargs):
        super(OSI, self).__init__(**kwargs)
        self.iterations = iterations
        self.lr = infer_rate
        self.umin = umin
        self.umax = umax
        self.xn, self.xi, self.xj, self.wn, self.we = gausshermite(k, dims=[1, 1, 1, 1, -1], dtype=self.dtype)
        self.x22 = 2 * self.xn ** 2 - 1
        self.xij = self.xi * self.xj
        self.xi2axj2 = self.xi ** 2 + self.xj ** 2 - 1
        self.xi2mxj2 = self.xi ** 2 - self.xj ** 2

        # hyper-parameter for Adam optimizer (AMSGrad=False)
        # self.beta1 = 0.9
        # self.beta2 = 0.999
        # self.epsilon = 1e-07

        # hyper-parameter for SGD with momentum
        self.momentum = momentum

    def build(self, input_shape):
        # b, h, w = input_shape
        # self.mu = self.add_weight(name='mean',
        #                           shape=(b, h, w, 1, 1),
        #                           constraint=lambda u: tf.clip_by_value(u, 0., 63.),
        #                           trainable=False)
        # self.sigma = self.add_weight(name='standard_deviation',
        #                              shape=(b, h, w, 1, 1),
        #                              initializer=self.sigma_initializer,
        #                              constraint=lambda o: tf.clip_by_value(o, 0.05, 10),
        #                              trainable=False)
        # self.rou = self.add_weight(name='variance_coefficient',
        #                            shape=(b, h, w, 2, 1),
        #                            initializer=self.rou_initializer,
        #                            constraint=lambda p: tf.clip_by_value(p, -0.9, 0.9),
        #                            trainable=False)

        super(OSI, self).build(input_shape)

    @tf.function
    def call(self, inputs, **kwargs):
        unary = tf.expand_dims(tf.expand_dims(inputs[0], -1), -1)
        weight = tf.expand_dims(inputs[1], -1)
        mu = unary
        sigma = tf.ones_like(mu)  # init. sigma as 1.0, may be bigger?
        # initialize rou based on weight, w=inf -> rou=1, w=0 -> rou=0
        rou = weight / (tf.reduce_max(weight) * 1.01 + 1.0)

        v = tf.repeat(tf.zeros_like(rou), repeats=2, axis=3)
        # m = tf.zeros_like(v)

        for _ in tf.range(self.iterations):
            fn = -((self.xn * sigma * sqrt2 + mu) - unary) ** 2
            fn = (fn - 3 * (ndc + tf.math.log(sigma))) * self.wn  # add entropy term

            q = 1 - rou ** 2
            rtq = tf.sqrt(q)
            s = tf.sqrt(1 + rou) / 2
            r = tf.sqrt(1 - rou) / 2
            z1 = (s + r) * self.xi + (s - r) * self.xj
            z2 = (s - r) * self.xi + (s + r) * self.xj

            o2 = tf.concat([tf.roll(sigma, -1, 1), tf.roll(sigma, -1, 2)], 3)
            fe = tf.concat([tf.roll(mu, -1, 1), tf.roll(mu, -1, 2)], 3)
            fe = -weight * ((z1 * sigma * sqrt2 + mu) - (z2 * o2 * sqrt2 + fe)) ** 2
            fe = (fe + (2 * ndc + tf.math.log(sigma * o2) + tf.math.log(q) / 2)) * self.we  # + entropy

            # calculate dmu, dsigma, drou
            dmu = tf.reduce_sum(fn * self.xn, -1, keepdims=True) / sigma * sqrt2
            dmu += tf.reduce_sum(tf.reduce_sum(fe * (z1 - rou * z2), -1, True) / (sigma * q) * sqrt2, 3, True)
            dmu2 = tf.reduce_sum(fe * (z2 - rou * z1), -1, True) / (o2 * q) * sqrt2
            dmu += tf.roll(dmu2[..., 0:1, :], 1, 1) + tf.roll(dmu2[..., 1:2, :], 1, 2)

            dsigma = tf.reduce_sum(fn * self.x22, -1, keepdims=True) / sigma
            dsigma += tf.reduce_sum(tf.reduce_sum(fe * (self.xi2axj2 + self.xi2mxj2 / rtq), -1, True) / sigma, 3, True)
            dsigma2 = tf.reduce_sum(fe * (self.xi2axj2 - self.xi2mxj2 / rtq), -1, True) / o2
            dsigma += tf.roll(dsigma2[..., 0:1, :], 1, 1) + tf.roll(dsigma2[..., 1:2, :], 1, 2)

            drou = tf.reduce_sum(fe * (2 * self.xij - rou * self.xi2axj2), -1, True) / q

            # Adam optimizer
            # grad = -tf.concat([dmu, dsigma, drou], 3)
            # t = tf.cast(t, self.dtype)
            # m = self.beta1 * m + (1 - self.beta1) * grad
            # v = self.beta2 * v + (1 - self.beta2) * grad ** 2
            # grad = self.lr * m / ((1 - self.beta1 ** t) * tf.sqrt(v / (1 - self.beta2 ** t)) + self.epsilon)

            # SGD with momentum optimizer
            v = self.momentum * v + self.lr * tf.concat([dmu, dsigma, drou], 3)

            # update mu, sigma, rou based on corresponding optimizer
            mu = tf.clip_by_value(mu + v[..., 0:1, :], clip_value_min=self.umin, clip_value_max=self.umax)
            sigma = tf.clip_by_value(sigma + v[..., 1:2, :], clip_value_min=0.001, clip_value_max=50.)
            rou = tf.clip_by_value(rou + v[..., 2:4, :], clip_value_min=-0.99, clip_value_max=0.99)

        return tf.squeeze(mu, [-1, -2])

    # def get_config(self):
    #     config = {'iterations': self.iterations,
    #               'infer_rate': self.lr,
    #               'max_disparity': self.max_disparity}
    #     base_config = super(OSI, self).get_config()
    #     return base_config.update(config)


class OSI2(Layer):
    """One-shot inference - a variational MAP inference reformulated as a RNN layer"""

    def __init__(self, iterations=100, infer_rate=0.01, umin=0., umax=63., k=3, momentum=0.7, **kwargs):
        super(OSI2, self).__init__(**kwargs)
        self.iterations = iterations
        self.lr = infer_rate
        self.umin = umin
        self.umax = umax
        self.xn, self.xi, self.xj, self.wn, self.we = gausshermite(k, dims=[1, 1, 1, 1, -1], dtype=self.dtype)
        self.x22 = 2 * self.xn ** 2 - 1
        self.xij = self.xi * self.xj
        self.xi2axj2 = self.xi ** 2 + self.xj ** 2 - 1
        self.xi2mxj2 = self.xi ** 2 - self.xj ** 2

        # hyper-parameter for Adam optimizer (AMSGrad=False)
        # self.beta1 = 0.9
        # self.beta2 = 0.999
        # self.epsilon = 1e-07

        # hyper-parameter for SGD with momentum
        self.momentum = momentum

    @tf.function
    def call(self, inputs, **kwargs):
        yi = tf.expand_dims(tf.expand_dims(inputs[0], -1), -1)
        edge_weight = tf.expand_dims(inputs[1], -1)
        unary_weight1, unary_weight2 = inputs[2][..., 0:1], inputs[2][..., 1:2]
        unary_weight1 = tf.expand_dims(unary_weight1, -1)
        unary_weight2 = tf.expand_dims(unary_weight2, -1)
        mu = yi
        sigma = tf.ones_like(mu)  # init. sigma as 1.0, may be bigger?
        # initialize rou based on weight, w=inf -> rou=1, w=0 -> rou=0
        rou = edge_weight / (tf.reduce_max(edge_weight) * 1.01 + 1.0)

        v = tf.repeat(tf.zeros_like(rou), repeats=2, axis=3)

        for _ in tf.range(self.iterations):
            # fn = -((self.xn * sigma * sqrt2 + mu) - yi) ** 2
            xi = self.xn * sigma * sqrt2 + mu
            fn = xi * (unary_weight1 * xi + unary_weight2 * yi)
            fn = (fn - 3 * (ndc + tf.math.log(sigma))) * self.wn  # add entropy term

            q = 1 - rou ** 2
            rtq = tf.sqrt(q)
            s = tf.sqrt(1 + rou) / 2
            r = tf.sqrt(1 - rou) / 2
            z1 = (s + r) * self.xi + (s - r) * self.xj
            z2 = (s - r) * self.xi + (s + r) * self.xj

            o2 = tf.concat([tf.roll(sigma, -1, 1), tf.roll(sigma, -1, 2)], 3)
            mu2 = tf.concat([tf.roll(mu, -1, 1), tf.roll(mu, -1, 2)], 3)
            # fe = -edge_weight * ((z1 * sigma * sqrt2 + mu) - (z2 * o2 * sqrt2 + mu2)) ** 2
            fe = edge_weight * (z1 * sigma * sqrt2 + mu) * (z2 * o2 * sqrt2 + mu2)
            fe = (fe + (2 * ndc + tf.math.log(sigma * o2) + tf.math.log(q) / 2)) * self.we  # + entropy

            # calculate dmu, dsigma, drou
            dmu = tf.reduce_sum(fn * self.xn, -1, keepdims=True) / sigma * sqrt2
            dmu += tf.reduce_sum(tf.reduce_sum(fe * (z1 - rou * z2), -1, True) / (sigma * q) * sqrt2, 3, True)
            dmu2 = tf.reduce_sum(fe * (z2 - rou * z1), -1, True) / (o2 * q) * sqrt2
            dmu += tf.roll(dmu2[..., 0:1, :], 1, 1) + tf.roll(dmu2[..., 1:2, :], 1, 2)

            dsigma = tf.reduce_sum(fn * self.x22, -1, keepdims=True) / sigma
            dsigma += tf.reduce_sum(tf.reduce_sum(fe * (self.xi2axj2 + self.xi2mxj2 / rtq), -1, True) / sigma, 3, True)
            dsigma2 = tf.reduce_sum(fe * (self.xi2axj2 - self.xi2mxj2 / rtq), -1, True) / o2
            dsigma += tf.roll(dsigma2[..., 0:1, :], 1, 1) + tf.roll(dsigma2[..., 1:2, :], 1, 2)

            drou = tf.reduce_sum(fe * (2 * self.xij - rou * self.xi2axj2), -1, True) / q

            # SGD with momentum optimizer
            v = self.momentum * v + self.lr * tf.concat([dmu, dsigma, drou], 3)

            # update mu, sigma, rou based on corresponding optimizer
            mu = tf.clip_by_value(mu + v[..., 0:1, :], clip_value_min=self.umin, clip_value_max=self.umax)
            sigma = tf.clip_by_value(sigma + v[..., 1:2, :], clip_value_min=0.001, clip_value_max=50.)
            rou = tf.clip_by_value(rou + v[..., 2:4, :], clip_value_min=-0.99, clip_value_max=0.99)

        return tf.squeeze(mu, [-1, -2])

    # def get_config(self):
    #     config = {'iterations': self.iterations,
    #               'infer_rate': self.lr,
    #               'max_disparity': self.max_disparity}
    #     base_config = super(OSI, self).get_config()
    #     return base_config.update(config)
