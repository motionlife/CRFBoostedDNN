import tensorflow as tf
from crf.gmm import ConditionalGaussianMixture
from potential.dnn import NeuralNetPotential
from potential.polynominal import PolynomialPotential
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.keras.backend.set_floatx('float32')
DTYPE = tf.float32

tf.debugging.enable_check_numerics()


class GmmUnitTest(tf.test.TestCase):

    def testGradient(self):
        """test if close form gradient consistent with tensorflow's auto-gradient result"""
        tf.random.set_seed(4)
        c = 10
        m = 28
        n = 28
        l = 1
        k = 3
        gmm = ConditionalGaussianMixture(n_model=c, width=m, height=n, mix_comp=l, k=k, dtype=DTYPE)

        # u = tf.random.uniform([c, m, n, 1, 1, 1], -10, 10)
        # o = tf.random.uniform([c, m, n, 1, 1, 1], 0.01, 10)
        # p = tf.random.uniform((c, m, n, 2, 1, 1), -0.99, 0.99)
        # gmm.mu.assign(tf.tile(u, [1, 1, 1, 1, l, 1]))
        # gmm.sigma.assign(tf.tile(o, [1, 1, 1, 1, l, 1]))
        # gmm.rou.assign(tf.tile(p, [1, 1, 1, 1, l, 1]))

        gmm.mu.assign(tf.random.uniform(gmm.mu.get_shape(), 0, 63, dtype=DTYPE))
        gmm.sigma.assign(tf.random.uniform(gmm.sigma.get_shape(), 0.001, 30, dtype=DTYPE))
        gmm.rou.assign(tf.random.uniform(gmm.rou.get_shape(), -0.99, 0.99, dtype=DTYPE))
        gmm.w.assign(tf.random.uniform(gmm.w.get_shape(), -3, 3, dtype=DTYPE))

        pot = PolynomialPotential(order=2)
        # pot = NeuralNetPotential(node_units=(3, 3, 3, 3), edge_units=(5, 5, 5, 5))
        # --------------------close form gradient calculating---------------------
        grd, bfe = gmm.gradient(pot)

        # --------------------gradient tape gradient calculating---------------------
        gmm2 = ConditionalGaussianMixture(n_model=c, width=m, height=n, mix_comp=l, k=k, dtype=DTYPE)
        gmm2.mu.assign(gmm.mu.read_value())
        gmm2.sigma.assign(gmm.sigma.read_value())
        gmm2.rou.assign(gmm.rou.read_value())
        gmm2.w.assign(gmm.w.read_value())
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(gmm2.trainable_variables)
            bfe2 = gmm2.bethe_free_energy(pot)
        grd2 = tape.gradient(bfe2, gmm2.trainable_variables)

        tolerance = 5e-3
        self.assertAllClose(bfe, bfe2, tolerance, tolerance)
        self.assertAllClose(grd, grd2, rtol=tolerance, atol=tolerance)

        print(f'max grd{[tf.reduce_max(i).numpy() for i in grd]}')
        print(f'max grd2{[tf.reduce_max(i).numpy() for i in grd2]}')
        print(f'min grd{[tf.reduce_min(i).numpy() for i in grd]}')
        print(f'min grd2{[tf.reduce_min(i).numpy() for i in grd2]}')
        for i, v in enumerate(pot.trainable_variables):
            print(f'layer {i} weights min: {tf.reduce_min(v)}, weights max: {tf.reduce_max(v)}')


if __name__ == '__main__':
    tf.test.main()  # run all unit tests
