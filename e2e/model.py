import tensorflow as tf
from .dnet import DisparityNet, Hourglass
from .pnet import EdgeNet, EdgeWeightNet, PWNet, UWNet
from .osirnn import OSI, OSI2
from .gcnet import GCNet
from .psmnet import PSMNet
from .leastereo import LEAStereo
from .chromagan import Generator, Discriminator


class CrfChromaGan(tf.keras.Model):
    """The complete keras model consists of original chroma net (unary potential) and OSI-RNN layer"""

    def __init__(self, infer_iter=21, infer_rate=0.01, q_points=3, momentum=0.7, use_crf=True, gpw=10.,
                 supermodular=False, **kwargs):
        super(CrfChromaGan, self).__init__(**kwargs)
        self.use_crf = use_crf
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.gradient_penalty_weight = gpw
        if use_crf:
            self.pairwise_net = PWNet()
            self.unary_net = UWNet() if supermodular else (lambda x: x)
            # reasonable range of a value (-86.18126, 98.23517), b value (-107.78852, 94.47578)
            one_shot_infer = OSI2 if supermodular else OSI
            self.osi = one_shot_infer(iterations=infer_iter, infer_rate=infer_rate, umin=-107.78852, umax=98.23517,
                                      k=q_points, momentum=momentum)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        if not training:
            l = inputs
        else:
            # inputs are lab color images when training
            l, rab = inputs[..., 0:1], inputs[..., 1:3]

        l3 = tf.tile(l, [1, 1, 1, 3])
        pab, y = self.generator(l3)  # B, H, W, 2
        if self.use_crf:
            pab = tf.reshape(tf.transpose(pab, [3, 0, 1, 2]), [-1, tf.shape(pab)[1], tf.shape(pab)[2]])
            edge_w = tf.tile(self.pairwise_net(l3), [2, 1, 1, 1])
            node_w = tf.tile(self.unary_net(l3), [2, 1, 1, 1])
            pab = self.osi([pab, edge_w, node_w])  # 2B, H, W
            pab = tf.transpose(tf.reshape(pab, [2, -1, tf.shape(pab)[1], tf.shape(pab)[2]]), [1, 2, 3, 0])

        if not training:
            return pab

        disc_pab_loss = tf.reduce_mean(self.discriminator([l, tf.stop_gradient(pab)]))

        # gradient penalty loss
        w = tf.random.uniform([tf.shape(pab)[0], 1, 1, 1])
        avg_samples = w * rab + (1 - w) * tf.stop_gradient(pab)
        disc_avg = self.discriminator([l, avg_samples])
        gradients = tf.gradients(disc_avg, avg_samples)[0]
        gradients_sqr_sum = tf.reduce_sum(tf.square(gradients), axis=tf.range(1, tf.rank(gradients)))
        gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
        gradient_penalty = self.gradient_penalty_weight * tf.square(1 - gradient_l2_norm)

        return (tf.reduce_mean(tf.math.squared_difference(pab, rab), axis=-1),  # predictedAB - mse loss
                y,  # classVector, to compare with y_true
                tf.reduce_mean(self.discriminator([l, pab])),
                -disc_pab_loss,  # to cancel gradient of discriminator weights produced by above term (smart move)
                # ↑↑↑↑-Combined(Generator) Model----------↓↓↓-Discriminator Model---------
                tf.reduce_mean(self.discriminator([l, rab])),  # discRealAB - loss
                disc_pab_loss,  # discPredAB - loss,
                tf.reduce_mean(gradient_penalty))


class CrfRnn(tf.keras.Model):
    """The complete keras model consists of disparity net (unary potential) and OSI-RNN layer"""

    def __init__(self, max_disparity, infer_iter=21, infer_rate=0.01, q_points=3, net='gc', crf=True,
                 supermodular=False, **kwargs):
        super(CrfRnn, self).__init__(**kwargs)
        # self.train_offset = max_disparity // 2
        self.crf = crf
        self.net = net
        if net == 'gc':
            self.unary_potential = GCNet(d=max_disparity)
        elif net == 'psm':
            self.unary_potential = PSMNet(d=max_disparity)
        elif net == 'leastereo':
            self.unary_potential = LEAStereo(d=max_disparity)
        else:
            self.unary_potential = Hourglass(d=max_disparity)

        if crf:
            self.unary_net = UWNet() if supermodular else (lambda x: x)
            self.pairwise_net = PWNet() if net in ['gc', 'psm', 'leastereo'] else EdgeWeightNet()
            one_shot_infer = OSI2 if supermodular else OSI
            self.inference = one_shot_infer(iterations=infer_iter, infer_rate=infer_rate, umax=max_disparity,
                                            k=q_points)

    def call(self, inputs, training=None, mask=None):
        disparity = self.unary_potential(tf.concat(inputs, axis=0))
        if self.crf:
            edge = self.pairwise_net(inputs[0])
            node = self.unary_net(inputs[0])
            if self.net == 'psm' and training:
                edge = tf.tile(edge, [tf.shape(disparity)[0] // tf.shape(edge)[0], 1, 1, 1])
                node = tf.tile(node, [tf.shape(disparity)[0] // tf.shape(node)[0], 1, 1, 1])
            disparity = self.inference([disparity, edge, node])
        return disparity


class CRF(tf.keras.Model):
    """The complete keras model consists of disparity net (unary potential) and OSI-RNN layer, with pre-train option"""

    def __init__(self, max_disparity, unary_pretrained, edge_pretrained, shift_mode='before',
                 infer_iter=100, infer_rate=0.01, q_points=3, **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.train_offset = max_disparity // 2
        self.unary_potential = DisparityNet(max_disparity, unary_pretrained, shift_mode)
        self.pairwise_potential = EdgeNet(pretrained=edge_pretrained)
        self.inference = OSI(iterations=infer_iter, infer_rate=infer_rate, umax=max_disparity, k=q_points)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        depth = self.inference([self.unary_potential(inputs), self.pairwise_potential(inputs[0])])
        return depth[:, :, self.train_offset:] if training else depth

    # def get_config(self):
    #     config = {'max_disparity': self.numDisparities,
    #               'pretrained': self.unary_potential.feature_extract,
    #               'shift_mode': self.unary_potential.shift_mode,
    #               'infer_iter': self.inference.iterations,
    #               'infer_rate': self.inference.lr}
    #     base_config = super(CRF, self).get_config()
    #     return base_config.update(config)
