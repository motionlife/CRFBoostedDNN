import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from potential.icnn import ICNNPotential

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.keras.backend.set_floatx('float64')


class ICNNUnitTest(tf.test.TestCase):
    def testConvexity(self):
        tf.random.set_seed(7)
        y = tf.cast(tf.linspace(-10., 10., 2000), tf.float64)
        icnn = ICNNPotential(u_units=[51, 31, 17], z_units=[7, 7, 7, 1])
        for i in range(20):
            icnn.xpath(tf.random.uniform(shape=[1, 98], minval=0, maxval=1, dtype=tf.float64))
            fig, ax = plt.subplots()
            ax.plot(np.squeeze(y), np.squeeze(icnn((y, tf.ones([1, 2], dtype=tf.float64)))[0]))
            ax.set(xlabel='y', ylabel='f(y|x)', title='PICNN')
            ax.grid()
            plt.show()
            # plt.savefig(f'F:/Documents/ai/NeuralizedCRF/logs/icnn-init/{i}1.png')
            # tf.debugging.assert_non_negative(icnn.fz1.W_z)
            # tf.debugging.assert_non_negative(icnn.fz2.W_z)
            # tf.debugging.assert_non_negative(icnn.fz3.W_z)

        self.assertAllClose(0, 0)


if __name__ == '__main__':
    tf.test.main()  # run all unit tests
