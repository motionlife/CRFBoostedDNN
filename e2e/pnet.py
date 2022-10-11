import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv2D, Conv2DTranspose, MaxPooling2D, Activation, BatchNormalization


def conv2d(filters, kernel, strides, transpose=False, bn=True, relu=True, name='conv2d'):
    def layer_wrapper(x):
        x = (Conv2DTranspose(filters, kernel, strides, padding='same', name=name)
             if transpose else Conv2D(filters, kernel, strides, padding='same', name=name))(x)
        if bn:
            x = BatchNormalization(name=f'{name}_bn')(x)
        if relu:
            x = Activation('relu', name=f'{name}_relu')(x)
        return x

    return layer_wrapper


def conv2dx2(filters, kernel, strides=2, transpose=False, bn=True, relu=True, name='conv2dx2', cat=True):
    def layer_wrapper(x, rem=None):
        x = conv2d(filters, kernel, strides, transpose, bn, relu, name + "_1")(x)
        if rem is not None:
            x = tf.concat([x, rem], -1) if cat else tf.add(x, rem)
        x = conv2d(filters, 3, 1, False, bn, relu, name + "_2")(x)
        return x

    return layer_wrapper


def PWNet(name='pairwise-net'):
    """
    Updated edge-net, hourglass structure:
    1) Increased the depth to enlarge receptive field;
    2) Added batch normalization layer after convolution layers, consistent with unary disparity net.
    """
    ref = Input(shape=(None, None, 3))
    x = conv2d(32, 3, 1, name='basic')(ref)
    x = rem0 = conv2dx2(32, 4, name='conv2d_1')(x)  # 1/2
    x = rem1 = conv2dx2(48, 3, name='conv2d_2')(x)  # 1/4
    x = rem2 = conv2dx2(64, 3, name='conv2d_3')(x)  # 1/8

    x = rem3 = conv2dx2(48, 3, transpose=True, name='conv2dt_4')(x, rem1)  # 1/4
    x = rem4 = conv2dx2(32, 3, transpose=True, name='conv2dt_5')(x, rem0)  # 1/2

    x = rem5 = conv2dx2(48, 3, name='conv2d_6')(x, rem3)  # 1/4
    x = conv2dx2(64, 3, name='conv2d_7')(x, rem2)  # 1/8

    x = conv2dx2(48, 3, transpose=True, name='conv2dt_8')(x, rem5)  # 1/4
    x = conv2dx2(32, 3, transpose=True, name='conv2dt_9')(x, rem4)  # 1/2
    x = conv2dx2(32, 4, transpose=True, name='conv2dt_10')(x)  # original size

    x = tf.abs(conv2d(2, 3, 1, bn=False, relu=False, name='edge-weights')(x))  # [B, H, W, 2]
    x1 = tf.concat([x[:, :-1, :, 0], tf.zeros(shape=[tf.shape(x)[0], 1, tf.shape(x)[2]])], axis=1)
    x2 = tf.concat([x[:, :, :-1, 1], tf.zeros(shape=[tf.shape(x)[0], tf.shape(x)[1], 1])], axis=2)
    out = tf.stack([x1, x2], axis=-1)
    return tf.keras.Model(inputs=ref, outputs=out, name=name)


def UWNet(name='unary-net'):
    ref = Input(shape=(None, None, 3))
    x = conv2d(32, 3, 1, name='basic')(ref)
    x = rem0 = conv2dx2(32, 4, name='conv2d_1')(x)  # 1/2
    x = rem1 = conv2dx2(48, 3, name='conv2d_2')(x)  # 1/4
    x = rem2 = conv2dx2(64, 3, name='conv2d_3')(x)  # 1/8

    x = rem3 = conv2dx2(48, 3, transpose=True, name='conv2dt_4')(x, rem1)  # 1/4
    x = rem4 = conv2dx2(32, 3, transpose=True, name='conv2dt_5')(x, rem0)  # 1/2

    x = rem5 = conv2dx2(48, 3, name='conv2d_6')(x, rem3)  # 1/4
    x = conv2dx2(64, 3, name='conv2d_7')(x, rem2)  # 1/8

    x = conv2dx2(48, 3, transpose=True, name='conv2dt_8')(x, rem5)  # 1/4
    x = conv2dx2(32, 3, transpose=True, name='conv2dt_9')(x, rem4)  # 1/2
    x = conv2dx2(32, 4, transpose=True, name='conv2dt_10')(x)  # original size

    out = conv2d(2, 3, 1, bn=False, relu=False, name='ab-coefficient')(x)  # [B, H, W, 2]
    return tf.keras.Model(inputs=ref, outputs=out, name=name)


class EdgeWeightNet(Layer):
    """
    output pairwise potential weight for each edge, with residual connection.
    """

    def __init__(self, **kwargs):
        super(EdgeWeightNet, self).__init__(**kwargs)
        self.conv1 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same', name='conv1')
        self.conv2 = Conv2D(48, (3, 3), strides=2, activation='relu', padding='same', name='conv2')
        self.conv3 = Conv2D(48, (3, 3), strides=1, activation='relu', padding='same', name='conv3')
        self.conv4 = Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', name='conv4')
        self.conv5 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', name='conv5')

        self.conv6 = Conv2DTranspose(48, (3, 3), strides=2, activation='relu', padding='same', name='conv6')
        self.conv7 = Conv2D(48, (3, 3), strides=1, activation='relu', padding='same', name='conv7')
        self.conv8 = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same', name='conv8')
        self.conv9 = Conv2D(2, (3, 3), strides=1, padding='same', name='conv9')

    def build(self, input_shape):
        super(EdgeWeightNet, self).build(input_shape)

    @tf.function
    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.conv3(self.conv2(x))
        out = x
        x = self.conv5(self.conv4(x))
        x = self.conv7(self.conv6(x) + out)
        x = self.conv9(self.conv8(x))
        x = tf.abs(x)
        x1 = tf.concat([x[:, :-1, :, 0], tf.zeros(shape=[tf.shape(x)[0], 1, tf.shape(x)[2]])], axis=1)
        x2 = tf.concat([x[:, :, :-1, 1], tf.zeros(shape=[tf.shape(x)[0], tf.shape(x)[1], 1])], axis=2)
        return tf.stack([x1, x2], axis=-1)


class EdgeNet(Layer):
    def __init__(self, pretrained, **kwargs):
        super(EdgeNet, self).__init__(**kwargs)
        self.feature_extract = pretrained  # out_shape = [135//2, 120]

        self.conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='edge_conv1')
        # self.bn1 = BatchNormalization(name='edge_bn1')
        # self.act1 = Activation(activation="relu", name='edge_act1')

        self.upscale1 = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same', name='upscale1')
        # self.bn2 = BatchNormalization(name='edge_bn2')
        # self.act2 = Activation(activation="relu", name='edge_act2')

        self.conv2 = Conv2D(64, (3, 3), padding='same', activation='relu', name='edge_conv2')
        # self.bn3 = BatchNormalization(name='edge_bn3')
        # self.act3 = Activation(activation="relu", name='edge_act3')

        self.upscale2 = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same', name='upscale2')
        # self.bn4 = BatchNormalization(name='edge_bn4')
        # self.act4 = Activation(activation="relu", name='edge_act4')

        self.conv3 = Conv2D(2, (3, 3), padding='same', name='edge_weights')  # No AC No BN for last layer

    def build(self, input_shape):
        super(EdgeNet, self).build(input_shape)

    @tf.function
    def call(self, inputs, **kwargs):
        x = self.feature_extract(inputs)
        x = self.conv1(x)
        x = self.upscale1(x)
        x = self.conv2(x)
        x = self.upscale2(x)
        x = tf.abs(self.conv3(x))
        x1 = tf.concat([x[:, :-1, :, 0], tf.zeros(shape=[tf.shape(x)[0], 1, tf.shape(x)[2]])], axis=1)
        x2 = tf.concat([x[:, :, :-1, 1], tf.zeros(shape=[tf.shape(x)[0], tf.shape(x)[1], 1])], axis=2)
        return tf.stack([x1, x2], axis=-1)  # tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]])


class PairwiseNet(Layer):
    """A deep neural network extracting suitable features from images to be used in CRF as pairwise potential.
    Since pairwise potential should be image data-dependent.  e.g. Without training, hand-made filters like
    Gaussian spatial kernel and Gaussian bilateral kernel, etc.
    """

    def __init__(self, **kwargs):
        super(PairwiseNet, self).__init__(**kwargs)
        # Block 1
        self.block1_conv1 = Conv2D(64, (3, 3), activation='selu', padding='same')
        self.block1_conv2 = Conv2D(64, (3, 3), activation='selu', padding='same')
        self.block1_pool = MaxPooling2D((2, 2), strides=(2, 2))  # output_shape=[112, 112, 64],[160, 208, 64],[152,200]

        # Block 2
        self.block2_conv1 = Conv2D(128, (3, 3), activation='selu', padding='same')
        self.block2_conv2 = Conv2D(128, (3, 3), activation='selu', padding='same')
        self.block2_pool = MaxPooling2D((2, 2), strides=(2, 2))  # output_shape=[56, 56, 128],[80, 104, 128],[76,100]

        # Block 3
        self.block3_conv1 = Conv2D(256, (3, 3), activation='selu', padding='same')
        self.block3_conv2 = Conv2D(256, (3, 3), activation='selu', padding='same')
        self.block3_pool = MaxPooling2D((2, 2), strides=(2, 2))  # output_shape=[28, 28, 256],[40, 52, 128],[38, 50]

        # De-Convolution to upscale images
        self.scale1 = Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='selu')
        self.scale10 = Conv2D(32, (3, 3), padding='same', activation='selu')
        self.pool2s = Conv2D(32, (3, 3), padding='same', activation='selu', name='connect-pool2')

        self.scale2 = Conv2DTranspose(16, (3, 3), strides=2, padding='same', activation='selu')
        self.scale20 = Conv2D(16, (3, 3), padding='same', activation='selu')
        self.pool1s = Conv2D(16, (3, 3), padding='same', activation='selu', name='connect-pool1')

        # Final scale out to original input image size
        self.score = Conv2DTranspose(2, (3, 3), strides=2, padding='same')

    def build(self, input_shape):
        super(PairwiseNet, self).build(input_shape)

    @tf.function
    def call(self, inputs, **kwargs):
        # [300, 400] -> [304, 400]
        x = tf.pad(inputs, [[0, 0], [2, 2], [0, 0], [0, 0]])

        x = self.block1_conv1(x)
        x = self.block1_conv2(x)
        x = self.block1_pool(x)
        p1 = x

        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_pool(x)
        p2 = x

        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = self.block3_pool(x)

        x = self.scale10(self.scale1(x)) + self.pool2s(p2)
        x = self.scale20(self.scale2(x)) + self.pool1s(p1)

        x = tf.abs(self.score(x))[:, 2:-2, ...]
        x1 = tf.concat([x[:, :-1, :, 0], tf.zeros(shape=[tf.shape(x)[0], 1, tf.shape(x)[2]])], axis=1)
        x2 = tf.concat([x[:, :, :-1, 1], tf.zeros(shape=[tf.shape(x)[0], tf.shape(x)[1], 1])], axis=2)
        return tf.stack([x1, x2], axis=-1)


class PairwiseNetSmall(Layer):
    """A deep neural network extracting suitable features from images to be used in CRF as pairwise potential.
    Since pairwise potential should be image data-dependent.  e.g. Without training, hand-made filters like
    Gaussian spatial kernel and Gaussian bilateral kernel, etc.
    """

    def __init__(self, **kwargs):
        super(PairwiseNetSmall, self).__init__(**kwargs)
        # Block 1
        self.block1_conv1 = Conv2D(64, (3, 3), activation='selu', padding='same')
        self.block1_conv2 = Conv2D(64, (3, 3), activation='selu', padding='same')
        self.block1_pool = MaxPooling2D((2, 2), strides=(2, 2))

        # Block 2
        self.block2_conv1 = Conv2D(128, (3, 3), activation='selu', padding='same')
        self.block2_conv2 = Conv2D(128, (3, 3), activation='selu', padding='same')
        self.block2_pool = MaxPooling2D((2, 2), strides=(2, 2))

        # Block 3
        self.block3_conv1 = Conv2D(256, (3, 3), activation='selu', padding='same')
        self.block3_conv2 = Conv2D(256, (3, 3), activation='selu', padding='same')

        # De-Convolution
        self.deconv1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='selu')
        self.deconv2 = Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', activation='selu')

        self.deconv3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='selu')
        self.deconv4 = Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', activation='selu')

        self.deconv5 = Conv2DTranspose(2, (3, 3), strides=(1, 1), padding='same')

    def build(self, input_shape):
        super(PairwiseNetSmall, self).build(input_shape)

    @tf.function
    def call(self, inputs, **kwargs):
        x = self.block1_conv1(inputs[0])
        x = self.block1_conv2(x)
        x = self.block1_pool(x)
        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_pool(x)
        x = self.block3_conv1(x)
        x = self.block3_conv2(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = tf.abs(x)
        x1 = tf.concat([x[:, :-1, :, 0], tf.zeros(shape=[tf.shape(x)[0], 1, tf.shape(x)[2]])], axis=1)
        x2 = tf.concat([x[:, :, :-1, 1], tf.zeros(shape=[tf.shape(x)[0], tf.shape(x)[1], 1])], axis=2)
        return tf.stack([x1, x2], axis=-1)
