import tensorflow as tf
from .gmm import ConditionalGaussianMixture
from potential.icnn import ICNNPotential
import matplotlib

matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


class Variational(tf.keras.Model):
    """training the potentials based a variational model"""

    def __init__(self, height, width, mix_comp, u_units, z_units, qp, tr, ir, bs, dtype, problem="stereo"):
        super(Variational, self).__init__(name='variational_model')
        self.gmm = ConditionalGaussianMixture(bs, height, width, mix_comp=mix_comp, k=qp, lr=ir, dtype=dtype)
        self.optimizer = tf.keras.optimizers.Adamax(learning_rate=tr)
        self.potential = ICNNPotential(u_units, z_units)
        self.logZ = tf.zeros((bs,), dtype=dtype)
        self.problem = problem

    def get_conditional_x(self, lbc, rbc, kernel, padding='VALID'):
        imgl = tf.nn.conv2d(input=lbc, filters=kernel, strides=1, padding=padding)
        if self.problem == "denoising":
            return tf.reshape(imgl, imgl.shape.as_list()[:-1] + [1, 1, 1, -1])
        imgr = tf.nn.conv2d(input=rbc, filters=kernel, strides=1, padding=padding)
        return tf.reshape(tf.concat([imgl, imgr], -1), imgl.shape.as_list()[:-1] + [1, 1, 1, -1])

    def train(self, dataset, log_dir, infer_iterations, epochs, kernel, ckp, manager):
        if manager.latest_checkpoint:
            ckp.restore(manager.latest_checkpoint)
            print(f"Restored from {manager.latest_checkpoint}")
        else:
            print("Initializing from scratch.")
        summary_writer = tf.summary.create_file_writer(log_dir)
        for e in tf.range(epochs):
            for i, batch in enumerate(dataset):
                self.potential.xpath(self.get_conditional_x(batch[0], batch[1], kernel))
                self.gmm.init_marginal()
                self.logZ = -self.gmm.infer(self.potential, tf.constant(infer_iterations))

                log_likelihood = -self.train_one_batch(batch[-1]).numpy()
                print(f'epoch {e} iter {i}, Log_likelihood: {log_likelihood}')
                with summary_writer.as_default():
                    tf.summary.scalar('log-likelihood', log_likelihood, step=int(ckp.step))

                if int(ckp.step) % 5 == 0:
                    for j in range(self.gmm.mu.shape[0]):
                        plt.imsave(log_dir + '/mu' + f'-{int(ckp.step)}-{j}' + '.png', self.gmm.mu[j, :, :, 0, 0, 0],
                                   cmap='gray')
                        plt.imsave(log_dir + '/sigma' + f'-{int(ckp.step)}-{j}' + '.png',
                                   self.gmm.sigma[j, :, :, 0, 0, 0], cmap='gray')

                    print(f'Saved checkpoint for step {int(ckp.step)}: {manager.save()}')

                ckp.step.assign_add(1)

    @tf.function
    def train_one_batch(self, y):
        y = tf.reshape(y, y.shape.as_list() + [1, 1])
        ye = tf.stack([tf.tile(y, [1, 1, 1, 2, 1, 1]), tf.concat([tf.roll(y, -1, 1), tf.roll(y, -1, 2)], 3)], -1)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.potential.trainable_variables)
            fn, fe = self.potential(inputs=[y, ye], back_prop=True)
            neg_log_likelihood = -tf.reduce_mean(tf.reduce_sum(fn, [1, 2, 3, 4, 5]) + tf.reduce_sum(fe, [1, 2, 3, 4, 5])
                                                 + self.gmm.bethe_free_energy(self.potential, back_prop=True))
        grd = tape.gradient(neg_log_likelihood, self.potential.trainable_variables)
        self.optimizer.apply_gradients(zip(grd, self.potential.trainable_variables))
        return neg_log_likelihood

    def estimate_disparity(self, lbc, rbc, lr, max_iter, kernel):
        """given left batch and right batch images, return batch disparity map"""
        opt = tf.keras.optimizers.Adamax(learning_rate=lr)
        y = tf.Variable(
            initial_value=tf.random.uniform(minval=3., maxval=25., shape=lbc.shape.as_list() + [1, 1], dtype=lbc.dtype),
            constraint=tf.keras.constraints.non_neg(), trainable=True)
        self.potential.xpath(self.get_conditional_x(lbc, rbc, kernel, padding='SAME'))
        for i in tf.range(max_iter):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch([y])
                ye = tf.stack([tf.tile(y, [1, 1, 1, 2, 1, 1]), tf.concat([tf.roll(y, -1, 1), tf.roll(y, -1, 2)], 3)],
                              -1)
                fn, fe = self.potential(inputs=[y, ye])
                energy = -(tf.reduce_sum(fn, [1, 2, 3, 4, 5]) + tf.reduce_sum(fe, [1, 2, 3, 4, 5]))
            grd = tape.gradient(energy, [y])
            opt.apply_gradients(zip(grd, [y]))
            tf.print(tf.strings.format('infer:{}, energy={}', (i, tf.reduce_mean(energy))))
        return tf.squeeze(y.read_value())
