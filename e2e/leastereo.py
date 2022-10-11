"""
Tensorflow implementation of best searched result of LEAStereo
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Input, Conv2D, Conv3D, Activation, BatchNormalization, UpSampling3D

fea_num_layers, mat_num_layers = 6, 12
fea_filter_multiplier, fea_block_multiplier, fea_step = 8, 4, 3
mat_filter_multiplier, mat_block_multiplier, mat_step = 8, 4, 3
initial_fm = fea_filter_multiplier * fea_block_multiplier
half_initial_fm = initial_fm // 2
filter_params_arr = tf.constant([1, 2, 4, 8])

network_arch_fea = np.array([[[0., 0., 0.],
                              [1., 0., 0.],
                              [0., 0., 0.],
                              [0., 0., 0.]],
                             [[0., 0., 1.],
                              [0., 0., 0.],
                              [0., 0., 0.],
                              [0., 0., 0.]],
                             [[0., 0., 0.],
                              [1., 0., 0.],
                              [0., 0., 0.],
                              [0., 0., 0.]],
                             [[0., 0., 1.],
                              [0., 0., 0.],
                              [0., 0., 0.],
                              [0., 0., 0.]],
                             [[0., 1., 0.],
                              [0., 0., 0.],
                              [0., 0., 0.],
                              [0., 0., 0.]],
                             [[0., 1., 0.],
                              [0., 0., 0.],
                              [0., 0., 0.],
                              [0., 0., 0.]]])

cell_arch_fea = np.array([[0, 1],
                          [1, 0],
                          [3, 1],
                          [4, 1],
                          [8, 1],
                          [5, 1]])

network_arch_mat = np.array([[[0., 0., 0.],
                              [1., 0., 0.],
                              [0., 0., 0.],
                              [0., 0., 0.]],
                             [[0., 0., 0.],
                              [0., 1., 0.],
                              [0., 0., 0.],
                              [0., 0., 0.]],
                             [[0., 0., 0.],
                              [0., 0., 0.],
                              [1., 0., 0.],
                              [0., 0., 0.]],
                             [[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 0.],
                              [0., 0., 0.]],
                             [[0., 0., 0.],
                              [0., 0., 1.],
                              [0., 0., 0.],
                              [0., 0., 0.]],
                             [[0., 0., 0.],
                              [0., 0., 0.],
                              [1., 0., 0.],
                              [0., 0., 0.]],
                             [[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 0.],
                              [0., 0., 0.]],
                             [[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 0.],
                              [0., 0., 0.]],
                             [[0., 0., 0.],
                              [0., 0., 1.],
                              [0., 0., 0.],
                              [0., 0., 0.]],
                             [[0., 0., 0.],
                              [0., 1., 0.],
                              [0., 0., 0.],
                              [0., 0., 0.]],
                             [[0., 0., 1.],
                              [0., 0., 0.],
                              [0., 0., 0.],
                              [0., 0., 0.]],
                             [[0., 0., 0.],
                              [1., 0., 0.],
                              [0., 0., 0.],
                              [0., 0., 0.]]])

cell_arch_mat = np.array([[1, 1],
                          [0, 1],
                          [3, 1],
                          [4, 1],
                          [8, 1],
                          [6, 1]])

PRIMITIVES = ['skip_connect', 'conv_3x3']
OPS = {
    'skip_connect': lambda C, stride: Layer() if stride == 1 else factorized_reduce(C),
    'conv_3x3': lambda C, stride: ConvBN(C, 3, stride)
}

PRIMITIVES_3d = ['skip_connect', '3d_conv_3x3']
OPS_3d = {
    'skip_connect': lambda C, stride: Layer() if stride == 1 else factorized_reduce(C, is_3d=True),
    '3d_conv_3x3': lambda C, stride: ConvBN(C, 3, stride, is_3d=True)
}


def triLinear(factor):
    def layer_wrapper(x):
        if factor == 1:
            return x
        elif factor > 1:
            fac = int(factor)
            return tf.nn.conv3d(UpSampling3D(fac)(x),
                                tf.ones([fac, fac, fac, x.shape[-1], x.shape[-1]]) / fac ** 3,
                                [1, 1, 1, 1, 1], padding='SAME')
        else:
            fac = int(1 / factor)
            return tf.nn.conv3d(x,
                                tf.ones([fac, fac, fac, x.shape[-1], x.shape[-1]]) / fac ** 3,
                                [1, fac, fac, fac, 1], padding='SAME')

    return layer_wrapper


def ConvBN(filters, kernel_size=3, strides=1, bn=True, relu=True, is_3d=False):
    def layer_wrapper(x):
        x = Conv3D(filters, kernel_size, strides, padding='same', use_bias=False)(x) \
            if is_3d else Conv2D(filters, kernel_size, strides, padding='same', use_bias=False)(x)
        if bn:
            x = BatchNormalization()(x)
        if relu:
            x = Activation('relu')(x)
        return x

    return layer_wrapper


def factorized_reduce(f, is_3d=False):
    conv_1 = Conv3D(f // 2, 1, 2, padding='same', use_bias=False) if is_3d else \
        Conv2D(f // 2, 1, 2, padding='same', use_bias=False)
    conv_2 = Conv3D(f // 2, 1, 2, padding='same', use_bias=False) if is_3d else \
        Conv2D(f // 2, 1, 2, padding='same', use_bias=False)

    def layer_wrapper(x):
        x = tf.concat([conv_1(x), conv_2(x[:, 1:, 1:, 1:, :])], axis=-1)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    return layer_wrapper


def cell(steps, block_multiplier, cell_arch, filter_multiplier, downup_sample, is_3d=False):
    pre_preprocess = ConvBN(filter_multiplier, 1, 1, is_3d=is_3d)
    preprocess = ConvBN(filter_multiplier, 1, 1, is_3d=is_3d)
    if downup_sample == -1:
        scale = 0.5
    elif downup_sample == 1:
        scale = 2.
    primitive = PRIMITIVES_3d if is_3d else PRIMITIVES
    operation = OPS_3d if is_3d else OPS
    ops = [operation[primitive[x[1]]](filter_multiplier, stride=1) for x in cell_arch]

    def layer_wrapper(prev_prev_input, prev_input):
        s0 = prev_prev_input
        s1 = prev_input
        if downup_sample != 0:
            if is_3d:
                s1 = triLinear(scale)(s1)
            else:
                feature_size_h = tf.cast(tf.cast(tf.shape(s1)[1], tf.float32) * scale, tf.int32)
                feature_size_w = tf.cast(tf.cast(tf.shape(s1)[2], tf.float32) * scale, tf.int32)
                s1 = tf.image.resize(s1, size=[feature_size_h, feature_size_w])
        if is_3d:
            s0 = triLinear(s1.shape[1] / s0.shape[1])(s0)
        else:
            s0 = tf.image.resize(s0, size=[tf.shape(s1)[1], tf.shape(s1)[2]])

        s0 = pre_preprocess(s0) if (s0.shape[-1] != filter_multiplier) else s0
        s1 = preprocess(s1)

        states = [s0, s1]
        offset = 0
        ops_index = 0
        for _ in range(steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if branch_index in cell_arch[:, 0]:
                    if prev_prev_input is not None or j > 0:
                        new_state = ops[ops_index](h)
                        new_states.append(new_state)
                    ops_index += 1

            s = sum(new_states)
            offset += len(states)
            states.append(s)

        concat_feature = tf.concat(states[-block_multiplier:], axis=-1)
        return prev_input, concat_feature

    return layer_wrapper


last3 = ConvBN(initial_fm, 1, 1, bn=False, relu=False)
last6 = ConvBN(initial_fm, 1, 1)
last12 = ConvBN(initial_fm * 2, 1, 1)
last24 = ConvBN(initial_fm * 4, 1, 1)

last3_3d = ConvBN(1, 3, 1, bn=False, relu=False, is_3d=True)
last6_3d = ConvBN(initial_fm, 1, 1, is_3d=True)
last12_3d = ConvBN(initial_fm * 2, 1, 1, is_3d=True)
last24_3d = ConvBN(initial_fm * 4, 1, 1, is_3d=True)


def LEAStereo(d=192, name='leastereo'):
    stereo = Input(shape=(None, None, 3))
    # Feature net-----------------------------------------------------------------------------
    stem0 = ConvBN(half_initial_fm, kernel_size=3, strides=1)(stereo)  # [H, W, 16]
    stem1 = ConvBN(initial_fm, kernel_size=3, strides=3)(stem0)  # [H/3, W/3, 32]
    stem2 = ConvBN(initial_fm, kernel_size=3, strides=1)(stem1)  # [H/3, W/3, 32]
    shp = stem2.shape

    for i in range(fea_num_layers):
        level_option = tf.reduce_sum(network_arch_fea[i], axis=1)
        level = tf.argmax(level_option)
        if i == 0:
            downup_sample = - tf.argmax(tf.reduce_sum(network_arch_fea[0], axis=1))
        else:
            three_branch_options = tf.reduce_sum(network_arch_fea[i], axis=0)
            downup_sample = tf.argmax(three_branch_options) - 1
        stem1, stem2 = cell(fea_step, fea_block_multiplier, cell_arch_fea,
                            fea_filter_multiplier * filter_params_arr[level], downup_sample)(stem1, stem2)

    x = stem2
    if x.shape[1] == shp[1]:
        x = last3(x)
    elif x.shape[1] == shp[1] // 2:
        x = last3(tf.image.resize(last6(x), shp[1:3]))
    elif x.shape[1] == shp[1] // 4:
        x = last3(tf.image.resize(last6(tf.image.resize(last12(x), [shp[1] // 2, shp[2] // 2])), shp[1:3]))
    elif x.shape[1] == shp[1] // 8:
        x = last3(tf.image.resize(last6(tf.image.resize(last12(tf.image.resize(last24(x), [shp[1] // 4, shp[2] // 4])),
                                                        [shp[1] // 2, shp[2] // 2])), shp[1:3]))

    # Cost volume
    xl, xr = tf.split(x, 2)
    x = tf.map_fn(lambda j: tf.concat([xl, tf.roll(xr, j, axis=2)], -1) * tf.reshape(
        tf.pad(tf.ones(tf.shape(xr)[2] - j), [[j, 0]]), [1, 1, -1, 1]), elems=tf.range(d // 3), dtype=xr.dtype)
    x = tf.transpose(x, [1, 0, 2, 3, 4])  # [B, D/3, H*, W*, 2F]

    # Matching net------------------------------------------------------------------------------
    shp = x.shape
    stem0 = ConvBN(initial_fm, 3, 1, is_3d=True)(x)
    stem1 = ConvBN(initial_fm, 3, 1, is_3d=True)(stem0)

    for i in range(mat_num_layers):
        level_option = tf.reduce_sum(network_arch_mat[i], axis=1)
        level = tf.argmax(level_option)
        if i == 0:
            downup_sample = - tf.argmax(tf.reduce_sum(network_arch_mat[0], axis=1))
        else:
            three_branch_options = tf.reduce_sum(network_arch_mat[i], axis=0)
            downup_sample = tf.argmax(three_branch_options) - 1
        stem0, stem1 = cell(mat_step, mat_block_multiplier, cell_arch_mat,
                            mat_filter_multiplier * filter_params_arr[level], downup_sample, is_3d=True)(stem0, stem1)
        if i == 1:
            out1 = stem1
        elif i == 4:
            out4 = stem1
            stem1 = ConvBN(initial_fm * 2, 3, 1, is_3d=True)(tf.concat([out1, stem1], axis=-1))
        elif i == 8:
            stem1 = ConvBN(initial_fm * 2, 3, 1, is_3d=True)(tf.concat([out4, stem1], axis=-1))

    x = stem1
    if x.shape[1] == shp[1]:
        x = last3_3d(x)
    elif x.shape[1] == shp[1] // 2:
        x = last3_3d(triLinear(2)(last6_3d(x)))
    elif x.shape[1] == shp[1] // 4:
        x = last3_3d(triLinear(2)(last6_3d(triLinear(2)(last12_3d(x)))))
    elif x.shape[1] == shp[1] // 8:
        x = last3_3d(triLinear(2)(last6_3d(triLinear(2)(last12_3d(triLinear(2)(last24_3d(x)))))))

    # Disp Net-------------------------------------------------------------------------------------
    x = triLinear(3)(x)  # [B, D, H, W, 1]
    x = tf.nn.softmax(tf.squeeze(x, -1), axis=1)
    x = tf.reduce_sum(x * tf.constant(range(d), shape=[1, d, 1, 1], dtype=x.dtype), axis=1)

    return tf.keras.Model(inputs=stereo, outputs=x, name=name)
