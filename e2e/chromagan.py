import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, concatenate, BatchNormalization, Dense, Flatten, \
    RepeatVector, Reshape, UpSampling2D
from tensorflow.keras.applications.vgg16 import VGG16


def Discriminator(name="discriminator-net"):
    """
    Judge how well the generated data looks like its from the real data
    :param name:
    :return:
    """
    input_l = Input(shape=(None, None, 1), name='l_input')
    input_ab = Input(shape=(None, None, 2), name='ab_input')
    x = concatenate([input_l, input_ab])  # feed lab image to discriminator
    x = Conv2D(64, (4, 4), padding='same', strides=(2, 2))(x)  # 112, 112, 64
    x = LeakyReLU()(x)
    x = Conv2D(128, (4, 4), padding='same', strides=(2, 2))(x)  # 56, 56, 128
    x = LeakyReLU()(x)
    x = Conv2D(256, (4, 4), padding='same', strides=(2, 2))(x)  # 28, 28, 256
    x = LeakyReLU()(x)
    x = Conv2D(512, (4, 4), padding='same', strides=(1, 1))(x)  # 28, 28, 512
    x = LeakyReLU()(x)
    x = Conv2D(1, (4, 4), padding='same', strides=(1, 1))(x)  # 28, 28,1
    return tf.keras.Model(inputs=[input_l, input_ab], outputs=x, name=name)


def Generator(name='generator-net'):
    """
    G_theta consists of two subnetworks: G1 outputs the chrominance information (a,b) = G1(L)
    G2 outputs the class distribution vector y = G2(L)
    :param name:
    :return:
    """

    # VGG16 without top layers (the yellow blocks in the paper figure 2)
    input_img = Input(shape=(None, None, 3))
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    vgg_features = tf.keras.Model(vgg_model.input, vgg_model.layers[-6].output)(input_img)

    # Global Features (the red blocks)
    x = Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu')(vgg_features)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    x1 = Dense(1024)(x)  # todo: why there is no activation? https://github.com/pvitoria/ChromaGAN/issues/8
    x1 = Dense(512)(x1)
    x1 = Dense(256)(x1)
    x1 = RepeatVector(28 * 28)(x1)
    x1 = Reshape((28, 28, 256))(x1)

    # --------------output-class-distribution-the-gray-blocks-ONLY-NEEDED-FOR-TRAINING--------
    y = Dense(4096)(x)
    y = Dense(4096)(y)  # the original vgg has relu as activation for two 4096 Dense
    y = Dense(1000, activation='softmax')(y)
    # -----------------------------------------------------------------------------------------

    # Midlevel Features (the purple blocks)
    x2 = Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(vgg_features)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu')(x2)
    x2 = BatchNormalization()(x2)

    # fusion of (VGG16 + Midlevel) + (VGG16 + Global)
    ab = concatenate([x2, x1])

    # Fusion + Colorization (the blue layers)
    ab = Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu')(ab)
    ab = Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu')(ab)

    ab = UpSampling2D(size=(2, 2), interpolation='bilinear')(ab)
    ab = Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(ab)
    ab = Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(ab)

    ab = UpSampling2D(size=(2, 2), interpolation='bilinear')(ab)
    ab = Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu')(ab)
    ab = Conv2D(2, (3, 3), padding='same', strides=(1, 1), activation='tanh')(ab)
    ab = UpSampling2D(size=(2, 2), interpolation='bilinear')(ab)
    ab = ab * 127  # todo: generator output a,b should be in (-1, 1), otherwise the mse loss will dominate

    return tf.keras.Model(inputs=input_img, outputs=[ab, y], name=name)
