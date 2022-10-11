"""VGG16 model for Keras w/ Batch Normalization

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.keras.engine import training
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.utils import layer_utils


def convolution_block(units, activation='relu', batch_normalization=True, block=1, layer=1):
    def layer_wrapper(x):
        x = Conv2D(units, (3, 3), padding='same', name='block{}_conv{}'.format(block, layer))(x)
        if batch_normalization:
            x = BatchNormalization(name='block{}_bn{}'.format(block, layer))(x)
        x = Activation(activation, name='block{}_act{}'.format(block, layer))(x)
        return x

    return layer_wrapper


def VGG16BN(
        bn=True,
        out_layer='block2_conv2',
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        activation='relu',
        model_name='vgg16_bn'):
    """Instantiates the VGG16 architecture with Batch Normalization"""
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = convolution_block(64, activation=activation, batch_normalization=bn, block=1, layer=1)(img_input)
    x = convolution_block(64, activation=activation, batch_normalization=bn, block=1, layer=2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = convolution_block(128, activation=activation, batch_normalization=bn, block=2, layer=1)(x)
    x = convolution_block(128, activation=activation, batch_normalization=bn and model_name == 'vgg16_bn_edge', block=2,
                          layer=2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = convolution_block(256, activation=activation, batch_normalization=bn, block=3, layer=1)(x)
    x = convolution_block(256, activation=activation, batch_normalization=bn, block=3, layer=2)(x)
    x = convolution_block(256, activation=activation, batch_normalization=bn, block=3, layer=3)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = convolution_block(512, activation=activation, batch_normalization=bn, block=4, layer=1)(x)
    x = convolution_block(512, activation=activation, batch_normalization=bn, block=4, layer=2)(x)
    x = convolution_block(512, activation=activation, batch_normalization=bn, block=4, layer=3)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = convolution_block(512, activation=activation, batch_normalization=bn, block=5, layer=1)(x)
    x = convolution_block(512, activation=activation, batch_normalization=bn, block=5, layer=2)(x)
    x = convolution_block(512, activation=activation, batch_normalization=bn, block=5, layer=3)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = training.Model(inputs, x, name=model_name)

    if weights == 'imagenet':
        # Build original VGG16 and load weights then copy its weights to this batch normalized model
        # just ignore batch normalized layer for now until we find pretrained vgg16-bn weights later online
        vgg = VGG16(include_top=include_top, weights='imagenet')
        layer_names = ['block1_conv1', 'block1_conv2',
                       'block2_conv1', 'block2_conv2',
                       'block3_conv1', 'block3_conv2', 'block3_conv3',
                       'block4_conv1', 'block4_conv2', 'block4_conv3',
                       'block5_conv1', 'block5_conv2', 'block5_conv3']
        for ln in layer_names:
            model.get_layer(ln).set_weights(vgg.get_layer(ln).get_weights())
    elif weights is not None:
        model.load_weights(weights)

    return training.Model(inputs=model.input, outputs=model.get_layer(out_layer).output)
