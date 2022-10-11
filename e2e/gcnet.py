"""
Tensorflow implementation of GC-Net from the paper https://arxiv.org/pdf/1703.04309.pdf.
Using keras functional API rather than subclassing.
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv3D, Conv3DTranspose, Activation, BatchNormalization


def conv2d(filters, kernel_size=(3, 3), strides=1, bn=True, layer=1):
    def layer_wrapper(x):
        x = Conv2D(filters, kernel_size, strides, padding='same', name=f'layer_{layer}_c2d')(x)
        if bn:
            x = BatchNormalization(name=f'layer_{layer}_bn')(x)
            x = Activation('relu', name=f'layer_{layer}_relu')(x)
        return x

    return layer_wrapper


def residual_block(filters, kernel_size=(3, 3), bn=True, front_layer=2):
    def layer_wrapper(x):
        o = conv2d(filters, kernel_size, strides=1, bn=bn, layer=front_layer)(x)
        o = conv2d(filters, kernel_size, strides=1, bn=bn, layer=front_layer + 1)(o)
        return o + x

    return layer_wrapper


def conv3d(filters, kernel_size=(3, 3, 3), strides=1, bn=True, layer=0):
    def layer_wrapper(x):
        x = Conv3D(filters, kernel_size, strides, padding='same', name=f'layer_{layer}_c3d')(x)
        if bn:
            x = BatchNormalization(name=f'layer_{layer}_bn')(x)
            x = Activation('relu', name=f'layer_{layer}_relu')(x)
        return x

    return layer_wrapper


def deconv3d(filters, kernel_size=(3, 3, 3), strides=1, bn=True, layer=0):
    def layer_wrapper(x):
        x = Conv3DTranspose(filters, kernel_size, strides, padding='same', name=f'layer_{layer}_c3dtr')(x)
        if bn:
            x = BatchNormalization(name=f'layer_{layer}_bn')(x)
            x = Activation('relu', name=f'layer_{layer}_relu')(x)
        return x

    return layer_wrapper


def GCNet(d=192, name='gc-net'):
    stereo = Input(shape=(None, None, 3))
    # Unary features
    x = conv2d(32, (5, 5), strides=2, bn=True, layer=1)(stereo)  # subsample the original input to its half
    x = residual_block(32, front_layer=2)(x)
    x = residual_block(32, front_layer=4)(x)
    x = residual_block(32, front_layer=6)(x)
    x = residual_block(32, front_layer=8)(x)
    x = residual_block(32, front_layer=10)(x)
    x = residual_block(32, front_layer=12)(x)
    x = residual_block(32, front_layer=14)(x)
    x = residual_block(32, front_layer=16)(x)
    x = conv2d(32, bn=False, layer=18)(x)  # no ReLu or BN

    # Cost volume
    xl, xr = tf.split(x, 2)
    x = tf.map_fn(lambda i: tf.concat([xl, tf.roll(xr, i, axis=2)], -1) * tf.reshape(
        tf.pad(tf.ones(tf.shape(xr)[2] - i), [[i, 0]]), [1, 1, -1, 1]), elems=tf.range(d // 2), dtype=xr.dtype)
    x = tf.transpose(x, [1, 0, 2, 3, 4])  # [B, D/2, H/2, W/2, 2F]

    # Learning regularization using 3-D convolution
    x20 = conv3d(32, layer=20)(conv3d(32, layer=19)(x))
    x = conv3d(64, strides=2, layer=21)(x)
    x23 = conv3d(64, layer=23)(conv3d(64, layer=22)(x))
    x = conv3d(64, strides=2, layer=24)(x)
    x26 = conv3d(64, layer=26)(conv3d(64, layer=25)(x))
    x = conv3d(64, strides=2, layer=27)(x)
    x29 = conv3d(64, layer=29)(conv3d(64, layer=28)(x))
    x = conv3d(128, strides=2, layer=30)(x)
    x = conv3d(128, layer=31)(x)
    x = conv3d(128, layer=32)(x)
    x = deconv3d(64, strides=2, layer=33)(x) + x29
    x = deconv3d(64, strides=2, layer=34)(x) + x26
    x = deconv3d(64, strides=2, layer=35)(x) + x23
    x = deconv3d(32, strides=2, layer=36)(x) + x20
    x = deconv3d(1, strides=2, bn=False, layer=37)(x)  # [B, D, H, W, 1]
    x = tf.nn.softmax(tf.squeeze(x, -1), axis=1)
    x = tf.reduce_sum(x * tf.constant(range(d), shape=[1, d, 1, 1], dtype=x.dtype), axis=1)

    return tf.keras.Model(inputs=stereo, outputs=x, name=name)
