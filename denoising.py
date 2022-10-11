import datetime
import numpy as np
import tensorflow as tf
from setup import select_device
from crf.variational import Variational

# tf.config.optimizer.set_jit(True)  # enable XLA on GPU
# $ TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
select_device(0)
tf.keras.backend.set_floatx('float64')
DTYPE = tf.float64
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 4
DS_NAME = 'realworld'
IMG_WIDTH = 200  # 400
IMG_HEIGHT = 150  # 300
FILTER_WIDTH = 9
FILTER_HEIGHT = 9
INFER_RATE = 0.02
INFER_ITERATIONS = 2000
TRAIN_RATE = 0.002
EPOCHS = 100
MIXTURE_COMP = 1
QUADRATURE_PTS = 11
start_w = FILTER_WIDTH // 2
start_h = FILTER_HEIGHT // 2
w_end = IMG_WIDTH - start_w
h_end = IMG_HEIGHT - start_h
FILTER = tf.expand_dims(tf.transpose(tf.reshape(
    tf.constant(([1] + [0] * FILTER_HEIGHT * FILTER_WIDTH) * (FILTER_HEIGHT * FILTER_WIDTH - 1) + [1], DTYPE),
    [-1, FILTER_HEIGHT, FILTER_WIDTH]), [1, 2, 0]), -2)


def decode_img(img_file):
    img = tf.io.read_file(img_file)
    img = tf.image.decode_png(img, channels=1)[75:225, 100:300, :]
    img = tf.image.convert_image_dtype(img, DTYPE)
    return img


def pre_process(noise, gt):
    return decode_img(noise), decode_img(gt)[start_h: h_end, start_w: w_end, ...]


def get_dataset(shuffle=False):
    # dataset pipeline
    ds_noise = tf.data.Dataset.list_files('data/small/' + DS_NAME + '/noise/*.png', shuffle=False)
    ds_gt = tf.data.Dataset.list_files('data/small/' + DS_NAME + '/gt/*.png', shuffle=False)
    dst = tf.data.Dataset.zip((ds_noise, ds_gt))
    if shuffle:
        dst = dst.shuffle(buffer_size=100)
    dst = dst.map(pre_process, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE, drop_remainder=True)
    return dst


if __name__ == '__main__':
    ds = get_dataset(shuffle=True)
    crf = Variational(height=IMG_HEIGHT - FILTER_HEIGHT + 1, width=IMG_WIDTH - FILTER_WIDTH + 1, bs=BATCH_SIZE,
                      mix_comp=MIXTURE_COMP, qp=QUADRATURE_PTS, tr=TRAIN_RATE, ir=INFER_RATE,
                      u_units=[32, 16, 8], z_units=[7, 7, 7, 1], dtype=DTYPE, problem="denoising")

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/denoising/' + time + f'-{DS_NAME}-FLT_{FILTER_HEIGHT}x{FILTER_WIDTH}-IR_{INFER_RATE}-ITR_{INFER_ITERATIONS}-TR_{TRAIN_RATE}-BS_{BATCH_SIZE}-QP_{QUADRATURE_PTS}-MIX_{MIXTURE_COMP}-rou-picnn'
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=crf.optimizer, model=crf)
    manager = tf.train.CheckpointManager(checkpoint, log_dir + '/checkpoints', max_to_keep=2)
    crf.train(dataset=ds, log_dir=log_dir, infer_iterations=INFER_ITERATIONS, epochs=EPOCHS, kernel=FILTER,
              ckp=checkpoint, manager=manager)
