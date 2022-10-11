"""
Tensorflow implementation of PSM-Net from the paper https://arxiv.org/abs/1803.08669.
Using keras functional API rather than subclassing.
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv2D, Conv3D, Conv3DTranspose, Activation, BatchNormalization, \
    AveragePooling2D, UpSampling3D


def conv3dbn(filters, kernel_size=(3, 3, 3), strides=1, transpose=False, activation=True, name='conv3d'):
    def layer_wrapper(x):
        x = (Conv3DTranspose(filters, kernel_size, strides, padding='same', name=name)
             if transpose else Conv3D(filters, kernel_size, strides, padding='same', name=name))(x)
        x = BatchNormalization(name=f'{name}_bn')(x)
        if activation:
            x = Activation('relu', name=f'{name}_relu')(x)
        return x

    return layer_wrapper


def hourglass(filters, name='hourglass'):
    def layer_wrapper(x, presqu, postsqu):
        out = conv3dbn(filters, 3, 2, name=f'{name}_conv3d_1')(x)  # 1/8
        pre = conv3dbn(filters, 3, 1, activation=postsqu is None, name=f'{name}_conv3d_2')(out)
        if postsqu is not None:
            pre = Activation('relu', name=f'{name}_conv3d_2_relu')(pre + postsqu)
        out = conv3dbn(filters, 3, 2, name=f'{name}_conv3d_3')(pre)  # 1/16 filters x 2 ?
        out = conv3dbn(filters, 3, 1, name=f'{name}_conv3d_4')(out)
        out = conv3dbn(filters, 3, 2, transpose=True, activation=False, name=f'{name}_conv3dt_5')(out)  # 1/8
        post = Activation('relu', name=f'{name}_conv3dt_5_relu')(out + (presqu if presqu is not None else pre))
        out = conv3dbn(filters // 2, 3, 2, transpose=True, activation=False, name=f'{name}_conv3dt_6')(post)  # 1/16
        return out, pre, post

    return layer_wrapper


def conv2dbn(filters, kernel_size=(3, 3), strides=1, dilation=1, activation=True, name='conv2d'):
    def layer_wrapper(x):
        x = Conv2D(filters, kernel_size, strides, dilation_rate=dilation, padding='same', name=name)(x)
        x = BatchNormalization(name=f'{name}_bn')(x)
        if activation:
            x = Activation('relu', name=f'{name}_relu')(x)
        return x

    return layer_wrapper


def residual_blocks(filters, strides=1, dilation=1, blocks=3, compat=True, name='res_block'):
    def layer_wrapper(x):
        for i in range(blocks):
            r = x
            x = conv2dbn(filters, 3, strides if i == 0 else 1, dilation, name=f'{name}_blk{i}_l0')(x)
            x = conv2dbn(filters, 3, 1, dilation, activation=False, name=f'{name}_blk{i}_l1')(x)
            x += r if i > 0 or compat else conv2dbn(filters, 1, strides, activation=False, name=f'{name}_compat')(r)
        return x

    return layer_wrapper


def pooling(pool_size, strides, name='pooling'):
    def layer_wrapper(x):
        shape = tf.shape(x)
        x = AveragePooling2D(pool_size, strides, name=name)(x)
        x = conv2dbn(32, 1, 1, name=f'{name}_conv1x1')(x)
        x = tf.image.resize(x, size=shape[1:3], method='bilinear', antialias=True, name=f'{name}_ups2d')
        return x

    return layer_wrapper


class Regression(Layer):
    def __init__(self, max_disp, **kwargs):
        super(Regression, self).__init__(**kwargs)
        self.disp = tf.constant(range(max_disp), shape=[1, max_disp, 1, 1], dtype=self.dtype)
        self.filters = tf.ones([4, 4, 4, 1, 1], dtype=self.dtype)
        self.strides = [1, 1, 1, 1, 1]

    def call(self, inputs, training=None, **kwargs):
        c1, c2, c3 = inputs
        c3 = tf.nn.conv3d(UpSampling3D(4)(c3), self.filters, self.strides, padding='SAME')
        c3 = tf.reduce_sum(tf.nn.softmax(tf.squeeze(c3, -1), axis=1) * self.disp, axis=1)
        if training:
            c1 = tf.nn.conv3d(UpSampling3D(4)(c1), self.filters, self.strides, padding='SAME')
            c1 = tf.reduce_sum(tf.nn.softmax(tf.squeeze(c1, -1), axis=1) * self.disp, axis=1)
            c2 = tf.nn.conv3d(UpSampling3D(4)(c2), self.filters, self.strides, padding='SAME')
            c2 = tf.reduce_sum(tf.nn.softmax(tf.squeeze(c2, -1), axis=1) * self.disp, axis=1)
            return tf.concat([c1, c2, c3], axis=0)
        return c3


def PSMNet(d=192, name='psm-net'):
    stereo = Input(shape=(None, None, 3))
    # CNN Module
    x = conv2dbn(32, 3, 2, name='conv0_0')(stereo)
    x = conv2dbn(32, 3, 1, name='conv0_1')(x)
    x = conv2dbn(32, 3, 1, name='conv0_2')(x)
    x = residual_blocks(32, strides=1, dilation=1, blocks=3, compat=True, name='conv1')(x)
    x_ = residual_blocks(64, strides=2, dilation=1, blocks=16, compat=False, name='conv2')(x)
    x = residual_blocks(128, strides=1, dilation=1, blocks=3, compat=False, name='conv3')(x_)  # dilation=1
    x = residual_blocks(128, strides=1, dilation=2, blocks=3, compat=True, name='conv4')(x)  # dilation=2

    # Spatial Pyramid Pooling Module
    x = tf.concat(
        [x_, x, pooling(8, 8, name='branch4')(x),
         pooling(16, 16, name='branch3')(x),
         pooling(32, 32, name='branch2')(x),
         pooling(64, 64, name='branch1')(x)], axis=-1)
    x = conv2dbn(128, 3, 1, name='fusion0')(x)
    x = Conv2D(32, 1, 1, name='fusion1')(x)  # last conv  no bn no relu, [2B, H/4, W/4, F=32]

    # Cost Volume
    xl, xr = tf.split(x, 2)
    x = tf.map_fn(lambda i: tf.concat([xl, tf.roll(xr, i, axis=2)], -1) * tf.reshape(
        tf.pad(tf.ones(tf.shape(xr)[2] - i), [[i, 0]]), [1, 1, -1, 1]), elems=tf.range(d // 4), dtype=xr.dtype)
    x = tf.transpose(x, [1, 0, 2, 3, 4])  # [B, D/4, H/4, W/4, 2F]

    # 3D CNN - hourglass
    x = conv3dbn(32, 3, 1, name='3Dconv0_1')(conv3dbn(32, 3, 1, name='3Dconv0_0')(x))
    x = conv3dbn(32, 3, 1, activation=False, name='3Dconv1_1')(conv3dbn(32, 3, 1, name='3Dconv1_0')(x)) + x

    out1, pre1, post1 = hourglass(64, name='hourglass1')(x, None, None)
    out1 += x
    out2, pre2, post2 = hourglass(64, name='hourglass2')(out1, pre1, post1)
    out2 += x
    out3, pre3, post3 = hourglass(64, name='hourglass3')(out2, pre1, post2)
    out3 += x
    cost1 = Conv3D(1, 3, 1, padding='same')(conv3dbn(32, 3, 1, name='conv3d_c1')(out1))  # [B, D/4, H/4, W/4, 1]
    cost2 = Conv3D(1, 3, 1, padding='same')(conv3dbn(32, 3, 1, name='conv3d_c2')(out2)) + cost1
    cost3 = Conv3D(1, 3, 1, padding='same')(conv3dbn(32, 3, 1, name='conv3d_c3')(out3)) + cost2

    # Regression and prediction
    predictions = Regression(d)([cost1, cost2, cost3])

    return tf.keras.Model(inputs=stereo, outputs=predictions, name=name)
