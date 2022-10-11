import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv2D, Conv2DTranspose, UpSampling2D, Conv3DTranspose
from .pnet import conv2d, conv2dx2


class DisparityNet(Layer):
    """A deep neural network estimating coarse disparity used in CRF unary potential"""

    def __init__(self, num_disparity, pretrained, shift_mode='before', **kwargs):
        super(DisparityNet, self).__init__(**kwargs)
        self.numDisparity = num_disparity
        self.disparities = tf.constant(range(self.numDisparity))
        self.D = tf.constant(range(self.numDisparity), shape=[self.numDisparity, 1, 1, 1], dtype=tf.float64)
        self.C = tf.constant(range(self.numDisparity), shape=[self.numDisparity, 1, 1, 1], dtype=self.dtype) * 1e-21
        self.feature_extract = pretrained
        self.shift_mode = shift_mode
        # TODO: TRY Transposed Conv2D layer to do up-scaling?
        self.upscale = UpSampling2D(size=(2, 2), interpolation='bilinear',  # try bicubic and gaussian?
                                    data_format='channels_first' if shift_mode == 'before' else 'channels_last')

    def build(self, input_shape):
        super(DisparityNet, self).build(input_shape)

    @tf.function
    def call(self, inputs, **kwargs):
        if self.shift_mode == 'after':
            x = tf.concat(inputs, axis=0)
            x = self.upscale(self.feature_extract(x))  # (2b, h/2, w/2, fs) = (2b, 150, 200, 128)
            xl, xr = tf.split(x, 2)
            x = tf.map_fn(lambda i: tf.reduce_sum((xl - tf.roll(xr, i, axis=2)) ** 2, axis=-1), elems=self.disparities,
                          dtype=xl.dtype)
        else:
            xl, xr = inputs
            xr = tf.map_fn(lambda i: tf.roll(xr, i, axis=2), elems=self.disparities, dtype=xr.dtype)
            x = tf.concat([xl, tf.reshape(xr, tf.concat([tf.constant([-1]), tf.shape(xr)[2:]], 0))], axis=0)
            x = self.feature_extract(x)  # [(1+numDisparity)*b, h/2, w/2, c*]
            x = tf.reshape(x, tf.concat([tf.constant([1 + self.numDisparity, -1]), tf.shape(x)[1:]], 0))
            # Cost volume aggregation, TODO:  try Weighted Euclidean Distance on feature dimension.
            # TODO: if tensor is too large, use tf.map_fn to unroll along disparity's dimension
            x = tf.reduce_sum((x[0:1, ...] - x[1:, ...]) ** 2, axis=-1) + self.C  # (D, B, H/2, W/2)
            x = self.upscale(x)

        # differentiable soft argmin, todo: how to prevent diminished gradient? mixed precision?
        x = tf.cast(-177. * x / tf.reduce_max(x, 0, True), tf.float64)
        x = tf.reduce_sum(self.D * tf.nn.softmax(x, axis=0), axis=0)
        return tf.cast(x, self.dtype)

    # def get_config(self):
    #     config = {'num_disparity': self.numDisparity,
    #               'pretrained': self.feature_extract,
    #               'shift_mode': self.shift_mode}
    #     base_config = super(DisparityNet, self).get_config()
    #     return base_config.update(config)


class DisparityNet2(Layer):
    """A deep neural network estimating coarse disparity used in CRF unary potential version 2,
        the network design mainly follows GA-Net-15 feature extraction pattern
    """

    def __init__(self, num_disparity, **kwargs):
        super(DisparityNet2, self).__init__(**kwargs)
        self.numDisparity = num_disparity
        self.disparities = tf.constant(range(self.numDisparity))
        self.D = tf.constant(range(self.numDisparity), shape=[self.numDisparity, 1, 1, 1], dtype=tf.float64)

        self.cv1 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same', name='conv1')
        # encoder0 --------------------------------------------------------------------------------
        self.cv2 = Conv2D(48, (3, 3), strides=3, activation='relu', padding='same', name='conv2')
        self.cv3 = Conv2D(48, (3, 3), strides=1, activation='relu', padding='same', name='conv3')
        self.cv4 = Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', name='conv4')
        self.cv5 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', name='conv5')
        self.cv6 = Conv2D(96, (3, 3), strides=2, activation='relu', padding='same', name='conv6')
        self.cv7 = Conv2D(96, (3, 3), strides=1, activation='relu', padding='same', name='conv7')
        self.cv8 = Conv2D(128, (3, 3), strides=2, activation='relu', padding='same', name='conv8')
        self.cv9 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same', name='conv9')
        # decoder0 ---------------------------------------------------------------------------------
        self.cv10 = Conv2DTranspose(96, (3, 3), strides=2, activation='relu', padding='same', name='conv10')
        self.cv11 = Conv2D(96, (3, 3), strides=1, activation='relu', padding='same', name='conv11')
        self.cv12 = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same', name='conv12')
        self.cv13 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', name='conv13')
        self.cv14 = Conv2DTranspose(48, (3, 3), strides=2, activation='relu', padding='same', name='conv14')
        self.cv15 = Conv2D(48, (3, 3), strides=1, activation='relu', padding='same', name='conv15')
        # encoder1 --------------------------------------------------------------------------------
        self.cv16 = Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', name='conv16')
        self.cv17 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', name='conv17')
        self.cv18 = Conv2D(96, (3, 3), strides=2, activation='relu', padding='same', name='conv18')
        self.cv19 = Conv2D(96, (3, 3), strides=1, activation='relu', padding='same', name='conv19')
        self.cv20 = Conv2D(128, (3, 3), strides=2, activation='relu', padding='same', name='conv20')
        self.cv21 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same', name='conv21')
        # decoder1 ---------------------------------------------------------------------------------
        self.cv22 = Conv2DTranspose(96, (3, 3), strides=2, activation='relu', padding='same', name='conv22')
        self.cv23 = Conv2D(96, (3, 3), strides=1, activation='relu', padding='same', name='conv23')
        self.cv24 = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same', name='conv24')
        self.cv25 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', name='conv25')
        self.cv26 = Conv2DTranspose(48, (3, 3), strides=2, activation='relu', padding='same', name='conv26')
        self.cv27 = Conv2D(48, (3, 3), strides=1, activation='relu', padding='same', name='conv27')
        # -------------------------------------------------------------------------------------------
        self.cv28 = Conv2DTranspose(32, (3, 3), strides=3, activation='relu', padding='same', name='conv28')

    def build(self, input_shape):
        super(DisparityNet2, self).build(input_shape)

    @tf.function
    def call(self, inputs, **kwargs):
        xl, xr = inputs
        shp = tf.shape(xr)
        xr = tf.map_fn(lambda i: tf.roll(xr, i, axis=2), elems=self.disparities, dtype=xr.dtype)
        x = tf.concat([xl, tf.reshape(xr, [-1, shp[1], shp[2], shp[3]])], axis=0)

        # ------------begin-feature-extraction-----------------------------------------------------
        x = self.cv1(x)  # 1 x 1 x 32
        x = self.cv3(self.cv2(x))
        o1 = x  # 1/3 x 1/3 x 48
        x = self.cv5(self.cv4(x))
        o2 = x  # 1/6 x 1/6 x 64
        x = self.cv7(self.cv6(x))
        o3 = x  # 1/12 x 1/12 x 96
        x = self.cv9(self.cv8(x))
        o4 = x  # 1/24 x 1/24 x 128
        x = self.cv11(self.cv10(x) + o3)
        o5 = x
        x = self.cv13(self.cv12(x) + o2)
        o6 = x
        x = self.cv15(self.cv14(x) + o1)
        o7 = x
        x = self.cv17(self.cv16(x) + o6)
        o8 = x
        x = self.cv19(self.cv18(x) + o5)
        o9 = x
        x = self.cv21(self.cv20(x) + o4)
        x = self.cv23(self.cv22(x) + o9)
        x = self.cv25(self.cv24(x) + o8)
        x = self.cv27(self.cv26(x) + o7)
        x = self.cv28(x)  # out_shape = [(1+numDisparity)*B, H, W, 32]
        # -------------end-feature-extraction-----------------------------------------------------

        x = tf.reshape(x, [1 + self.numDisparity, -1, shp[1], shp[2], 32])
        # x = tf.reduce_sum((x[0:1, ...] - x[1:, ...]) ** 2, axis=-1)  # (D, B, H, W)
        x = tf.reduce_sum(tf.nn.l2_normalize(x[0:1, ...], -1) * tf.nn.l2_normalize(x[1:, ...], -1), axis=-1)
        x = tf.reduce_sum(self.D * tf.nn.softmax(tf.cast(173 * x, tf.float64), axis=0), axis=0)
        return tf.cast(x, self.dtype)


def Hourglass(d=192, name='jhg-net'):
    """A simple disparity net contains only feature extraction convolution layers,
    which is stacked hourglass and using cosine similarity as cost aggregation method.
    """
    stereo = Input(shape=(None, None, 3))
    # Unary features
    x = conv2d(32, 3, 1, name='basic')(stereo)
    x = rem0 = conv2dx2(32, 5, strides=3, name='conv2d_0')(x)  # 1/3
    x = rem1 = conv2dx2(48, 3, name='conv2d_1')(x)  # 1/6
    x = rem2 = conv2dx2(64, 3, name='conv2d_2')(x)  # 1/12
    x = rem3 = conv2dx2(96, 3, name='conv2d_3')(x)  # 1/24
    x = rem4 = conv2dx2(128, 3, name='conv2d_4')(x)  # 1/48

    x = rem5 = conv2dx2(96, 4, transpose=True, name='conv2dt_5')(x, rem3)  # 1/24
    x = rem6 = conv2dx2(64, 4, transpose=True, name='conv2dt_6')(x, rem2)  # 1/12
    x = rem7 = conv2dx2(48, 4, transpose=True, name='conv2dt_7')(x, rem1)  # 1/6
    x = rem8 = conv2dx2(32, 4, transpose=True, name='conv2dt_8')(x, rem0)  # 1/3

    x = rem9 = conv2dx2(48, 3, name='conv2d_9')(x, rem7)  # 1/6
    x = rem10 = conv2dx2(64, 3, name='conv2d_10')(x, rem6)  # 1/12
    x = rem11 = conv2dx2(96, 3, name='conv2d_11')(x, rem5)  # 1/24
    x = conv2dx2(128, 3, name='conv2d_12')(x, rem4)  # 1/48

    x = conv2dx2(96, 4, transpose=True, name='conv2dt_13')(x, rem11)  # 1/24
    x = conv2dx2(64, 4, transpose=True, name='conv2dt_14')(x, rem10)  # 1/12
    x = conv2dx2(48, 4, transpose=True, name='conv2dt_15')(x, rem9)  # 1/6
    x = conv2dx2(32, 4, transpose=True, name='conv2dt_16')(x, rem8)  # 1/3

    x = conv2d(32, 3, 1, bn=False, relu=False, name='last_ft_no-bn')(x)  # [2B, H/3, W/3, F]

    # Cost volume
    xl, xr = tf.split(x, 2)
    xl = tf.expand_dims(xl, 0)
    xr = tf.map_fn(
        lambda i: tf.roll(xr, i, axis=2) * tf.reshape(tf.pad(tf.ones(tf.shape(xr)[2] - i), [[i, 0]]), [1, 1, -1, 1]),
        elems=tf.range(d // 3), dtype=xr.dtype)  # [D/3, B, H/3, W/3, F]

    # Cost aggregation: cosine similarity
    x = tf.reduce_sum(tf.nn.l2_normalize(xl, -1) * tf.nn.l2_normalize(xr, -1), axis=-1, keepdims=True)
    x = tf.transpose(x, [1, 0, 2, 3, 4])  # [B, D/3, H/3, W/3, 1]

    # Up-sampling to original size and soft argmax regression
    x = Conv3DTranspose(1, 4, strides=3, padding='same', name='up-sampler')(x)  # [B, D, H, W, 1]
    x = tf.nn.softmax(tf.squeeze(x, -1) * 173., axis=1)
    x = tf.reduce_sum(x * tf.constant(range(d), shape=[1, d, 1, 1], dtype=x.dtype), axis=1)

    return tf.keras.Model(inputs=stereo, outputs=x, name=name)
