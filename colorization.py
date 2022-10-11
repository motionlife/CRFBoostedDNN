# end to end architecture of DNN and CRF
import os
import json
import argparse
import datetime
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[:1], 'GPU')
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as KB
from e2e.model import CrfChromaGan
from utils.rgb_lab import rgb_to_lab, lab_to_rgb
import matplotlib.pyplot as plt

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
tf.config.optimizer.set_jit(True)  # enable XLA on GPU
# tf.debugging.enable_check_numerics()
K.set_floatx('float32')
DTYPE = tf.float32
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 36
INFER_ITER = 21  # try 7, 12, 14, 21 when inference
INFER_RATE = 0.01
QUADRATURE_PTS = 3
LEARN_RT = 2e-5
DS_NAME = 'IMAGE-NET'
DATA_DIR = "/data/hao/vision/image-net/ILSVRC2012/ILSVRC2012_img_train"
CLASS_PATH = "/data/hao/vision/image-net/imagenet_class_index.json"
IMG_SIZE = 224
NET = 'chromaGAN'
USE_CRF = True
global CTB


def file2class(path):
    with open(path) as f:
        CLASS_INDEX = json.load(f)
    serial_no = []
    for i in range(1000):
        serial_no.append(CLASS_INDEX[str(i)][0])
    return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(tf.constant(serial_no), tf.range(1000)), -1)


def getXY(file_name):
    img = tf.io.read_file(file_name)
    img = tf.image.decode_image(img, channels=3, dtype=DTYPE, expand_animations=False)  # rgb(0.0->1.0)
    img = tf.image.resize(img, size=[IMG_SIZE, IMG_SIZE])
    x = rgb_to_lab(img)  # convert from rgb space to lab space
    y = tf.one_hot(CTB.lookup(tf.strings.split(file_name, "/")[-2]), depth=1000)
    return x, (0, y, 0, 0, 0, 0, 0)


def getLChannel(file_name):
    img = tf.io.read_file(file_name)
    img = tf.image.decode_image(img, channels=3, dtype=DTYPE, expand_animations=False)  # rgb(0.0->1.0)
    img_gs = tf.image.rgb_to_grayscale(img)
    img_l = rgb_to_lab(tf.tile(img_gs, [1, 1, 3]))[..., 0:1]
    # h, w = tf.shape(img)[0], tf.shape(img)[1]
    return tf.image.resize(img_l, [IMG_SIZE, IMG_SIZE]), tf.image.resize(img, [IMG_SIZE, IMG_SIZE])


def get_data(folder, pattern="/*/*", bs=BATCH_SIZE, shuffle=True, training=True):
    ds = tf.data.Dataset.list_files(folder + pattern, shuffle=False)
    if shuffle:
        ds = ds.shuffle(buffer_size=1_000_000, reshuffle_each_iteration=True)
    ds = ds.map((getXY if training else getLChannel), AUTOTUNE)
    # todo: replicate every one batch, one for generator training another for discriminator
    return ds.batch(bs, drop_remainder=training).prefetch(AUTOTUNE)


def colorize(folder):
    log_dir = 'logs/gan/20211006-223210-IMAGE-NET_IR-0.01_ITER-21_TR-2e-05_BS-36_chromaGAN-supermodular'
    os.makedirs("data/results-e46-val/", exist_ok=True)
    strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0'])
    with strategy.scope():
        model = CrfChromaGan(infer_iter=INFER_ITER, infer_rate=INFER_RATE, q_points=QUADRATURE_PTS,
                             momentum=0.7, gpw=10., use_crf=USE_CRF)
        model.load_weights(log_dir + '/weights-e046-loss727491.9375000.ckpt')
    bs = 1
    dataset = get_data(folder, pattern="/*", bs=bs, shuffle=False, training=False)
    for i, (lt, lo) in enumerate(dataset):
        pab = model.predict_on_batch(lt)
        for j in range(tf.shape(pab)[0]):
            lab = tf.concat([lo[j], tf.image.resize(pab[j], size=lo[j].shape[:2])], -1)
            plt.imsave(f"data/results-e46-val/e46-cg-{i * bs + j:05d}.png", lab_to_rgb(lab).numpy())


def psnr(folder, weight_ckpt_path):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = CrfChromaGan(infer_iter=INFER_ITER, infer_rate=INFER_RATE, q_points=QUADRATURE_PTS,
                             momentum=0.7, gpw=10., use_crf=USE_CRF, supermodular=True)
        model.load_weights(weight_ckpt_path)
    bs = 1
    dataset = get_data(folder, pattern="/*", bs=bs, shuffle=False, training=False)
    PSNR_TOTAL = []
    total = 0.
    for i, (lt, img) in enumerate(dataset):
        pab = model.predict_on_batch(lt)
        lab = tf.concat([lt, pab], -1)
        rgb = lab_to_rgb(lab)
        psnr = tf.image.psnr(rgb, img, max_val=1.0).numpy()
        PSNR_TOTAL.append(psnr)
        total += psnr
        if i % 50 == 0:
            print(f'image {i} psnr = {psnr}, avg_psnr = {total / (i + 1)}')

    return PSNR_TOTAL


@tf.function
def loss(y_true, y_pred):
    return tf.nn.compute_average_loss(per_example_loss=tf.expand_dims(y_pred, axis=-1),
                                      global_batch_size=BATCH_SIZE)


class DynamicLossWeights(KB.Callback):
    def __init__(self, lw0, lw1, lw2, lw3, lw4, lw5, lw6):
        super(DynamicLossWeights, self).__init__()
        self.lw0 = lw0
        self.lw1 = lw1
        self.lw2 = lw2
        self.lw3 = lw3
        self.lw4 = lw4
        self.lw5 = lw5
        self.lw6 = lw6

    def on_train_batch_begin(self, batch, logs=None):
        if batch % 2 == 0:  # first train G then train D
            K.set_value(self.lw0, 1.0)
            K.set_value(self.lw1, 0.003)
            K.set_value(self.lw2, -0.1)
            K.set_value(self.lw3, -0.1)
            K.set_value(self.lw4, 0.)
            K.set_value(self.lw5, 0.)
            K.set_value(self.lw6, 0.)
        else:
            K.set_value(self.lw0, 0.)
            K.set_value(self.lw1, 0.)
            K.set_value(self.lw2, 0.)
            K.set_value(self.lw3, 0.)
            K.set_value(self.lw4, -1.)
            K.set_value(self.lw5, 1.)
            K.set_value(self.lw6, 1.)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--colorize', dest='colorize', default=False, action='store_true')
    parser.add_argument('--psnr', dest='psnr', default=False, action='store_true')
    parser.add_argument('--img_dir', dest='img_dir', type=str, default=DATA_DIR)
    args = parser.parse_args()
    if args.colorize:
        colorize(folder=args.img_dir)
    elif args.psnr:
        result = psnr(folder=args.img_dir, weight_ckpt_path='logs/gan/20211006-223210-IMAGE-NET_IR-0.01_ITER-21_TR-2e-05_BS-36_chromaGAN-supermodular/weights-e001-loss739263.7500000.ckpt')
    else:
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'logs/gan/' + time + f'-{DS_NAME}_IR-{INFER_RATE}_ITER-{INFER_ITER}_TR-{LEARN_RT}_BS-{BATCH_SIZE}_{NET}'
        weights_dir = 'logs/gan/20201103-131231-IMAGE-NET_IR-0.01_ITER-21_TR-2e-05_BS-36_chromaGAN'
        # scheduler_cbk = KB.LearningRateScheduler((lambda e: 0.98 ** e * LEARN_RT), verbose=1)
        CTB = file2class(path=CLASS_PATH)

        mirrored_strategy = tf.distribute.MirroredStrategy()
        # Move the creation, compiling  and weights loading of Keras model inside strategy.scope
        with mirrored_strategy.scope():
            model = CrfChromaGan(infer_iter=INFER_ITER, infer_rate=INFER_RATE, q_points=QUADRATURE_PTS,
                                 momentum=0.7, gpw=10., use_crf=USE_CRF, supermodular=True)
            # dynamic loss weights to mimic training 2 parts separately
            # solution: adding loss-weights mask on_train_end, see https://github.com/keras-team/keras/issues/2595
            l0, l1, l2, l3 = K.variable(0.), K.variable(0.), K.variable(0.), K.variable(0.)
            l4, l5, l6 = K.variable(0.), K.variable(0.), K.variable(0.)

            callbacks = [KB.TensorBoard(log_dir=log_dir, write_graph=False, profile_batch=0, update_freq=5000),
                         # scheduler_cbk,
                         DynamicLossWeights(l0, l1, l2, l3, l4, l5, l6),
                         KB.ModelCheckpoint(filepath=log_dir + "/weights-e{epoch:03d}-loss{loss:.7f}.ckpt", verbose=0,
                                            save_weights_only=True, save_freq='epoch')]

            model.compile(loss=[loss, "kld", loss, loss, loss, loss, loss],
                          # loss_weights=[1.0, 0.003, -0.1, -0.1, -1.0, 1.0, 1.0],
                          loss_weights=[l0, l1, l2, l3, l4, l5, l6],
                          optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RT, beta_1=0.5))

            model.load_weights(weights_dir + '/weights-e046-loss727491.9375000.ckpt')
            K.set_value(model.optimizer.learning_rate, LEARN_RT)
            K.set_value(model.optimizer.iterations, 0)

        dataset = get_data(folder=DATA_DIR)
        history = model.fit(dataset, callbacks=callbacks, epochs=100, initial_epoch=0)
