# end to end architecture of DNN and CRF
import os
import time
import tensorflow as tf
from e2e.model import CrfRnn
from setup import select_device
import matplotlib.pyplot as plt
from PIL import Image

# from utils.vis import disp_to_color

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
select_device(0)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
tf.config.optimizer.set_jit(True)  # enable XLA on GPU
tf.keras.backend.set_floatx('float32')
DTYPE = tf.float32
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 4
INFER_ITER = 12
INFER_RATE = 0.005
QUADRATURE_PTS = 3
DISPARITIES = 192
NET = 'psm'
START_NUM = 560
MUL = 16 if NET == 'psm' else (24 if NET == 'leastereo' else 32)
CRF = True
EVAL = False  # evaluation (w/ ground truth) or just prediction and save disparity map ( w/o ground truth)
DS_NAME = 'hangzhou'
# DATA_DIR = "/data/hao/vision/vkitti/Scene06"
# DATA_DIR = "/data/hao/vision/kitti/val"
DATA_DIR = '/data/hao/vision/' + ('DrivingStereo/test' if DS_NAME == 'hangzhou' else 'kitti/stereo/testing')
# DATA_DIR = "/data/hao/vision/sceneflow/test_finalpass"
# DATA_DIR = "/data/hao/vision/sceneflow/fly3dfull/benchmark"
SCENE = '' # 'all_scenes'  # 'TEST/*/*'  # '' if DS_NAME == 'hangzhou' else ('validation-2015' if EVAL else DS_NAME[-4:])
W_DIR = "logs/e2e/compare/20200830-105134-weathers-256x768_d192_IR-0.01_ITER-21_TR-0.0002_BS-2_psm/040.ckpt"
if not EVAL:
    ODR = W_DIR[:-8] + f"viz/epk{int(W_DIR[-8:-5])}/{SCENE}"
    os.makedirs(ODR, exist_ok=True)


def get_image(img_file, channels=3, dtype=tf.uint8):
    img = tf.io.read_file(img_file)
    img = tf.image.decode_png(img, channels=channels, dtype=dtype)
    img = tf.cast(img, DTYPE)
    return img


def get_disparity(file_path):
    # decode kitti or driving stereo disparity, both are sparse real-world data
    if tf.strings.regex_full_match(file_path, '.*_10.png$|.*2018-.*png$'):
        disp = get_image(file_path, channels=0, dtype=tf.uint16) / 256
        return tf.where(disp > 0, disp, -1) if EVAL else disp
    # decode virtual kitti 2 depth map and convert it to disparity
    if tf.strings.regex_full_match(file_path, '.*depth_.*png$'):
        depth = get_image(file_path, 0, tf.uint16)
        return 53.2725 * 725.0087 / tf.where(depth < 65535, depth, DTYPE.max)  # depth to disparity conversion
    # decode MPI-sintel disparity
    if tf.strings.regex_full_match(file_path, '.*_frame_.*png$'):
        disp = get_image(file_path)
        return disp[..., 0:1] * 4 + disp[..., 1:2] / 64 + disp[..., 2:3] / 16384
    # decode disparity map of scene flow dataset using pure tensorflow ops, making it possible to run on cloud tpu
    disparity = tf.io.read_file(file_path)
    N = tf.strings.length(disparity)
    disparity = tf.strings.substr(disparity, N - 540 * 960 * 4, N, unit='BYTE')
    disparity = tf.reshape(tf.io.decode_raw(disparity, out_type=DTYPE), [540, 960, 1])
    disparity = tf.abs(tf.reverse(disparity, [0]))  # flip upside down, keep + for both 18-byte and 21-byte header
    return disparity


def path2data(lp, rp, tl=None):
    images = tf.stack([get_image(lp), get_image(rp)])
    dh, dw = (MUL - tf.shape(images)[1] % MUL) % MUL, (MUL - tf.shape(images)[2] % MUL) % MUL
    images = tf.pad(images, [[0, 0], [dh, 0], [dw, 0], [0, 0]])
    mean, variance = tf.nn.moments(images, axes=[0, 1, 2], keepdims=True)
    images = tf.nn.batch_normalization(images, mean, variance, offset=None, scale=None, variance_epsilon=1e-17)
    return tuple(tf.unstack(images)), get_disparity(tl) if EVAL else (dh, dw)


def get_dataset():
    if 'Scene06' in DATA_DIR:
        ls_L = tf.data.Dataset.list_files(DATA_DIR + f"/*/frames/rgb/Camera_0/*.jpg", False)
        ls_R = tf.data.Dataset.list_files(DATA_DIR + f"/*/frames/rgb/Camera_1/*.jpg", False)
        ls_TL = tf.data.Dataset.list_files(DATA_DIR + f"/*/frames/depth/Camera_0/*.png", False) if EVAL else None
    else:
        ls_L = tf.data.Dataset.list_files(DATA_DIR + f'/rgb/{SCENE}/left/*', shuffle=False)
        ls_R = tf.data.Dataset.list_files(DATA_DIR + f'/rgb/{SCENE}/right/*', shuffle=False)
        ls_TL = tf.data.Dataset.list_files(DATA_DIR + f"/disparity/{SCENE}/left/*", shuffle=False) if EVAL else None
    dst = tf.data.Dataset.zip((ls_L, ls_R, ls_TL) if EVAL else (ls_L, ls_R)).cache().map(path2data)
    return dst.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)


def save_disparity(md, ds, cmap='jet'):
    for i, (x, (dh, dw)) in enumerate(ds):
        # predict is designed for performance in large scale inputs and tf.distribute. small batch use __call__
        if i * BATCH_SIZE < START_NUM:
            continue
        disparity = md.predict_on_batch(x)
        for j in range(BATCH_SIZE):
            disp = disparity[j, dh[j]:, dw[j]:]
            # Image.fromarray(tf.cast(disp * 256., tf.uint16).numpy()).save(f"{ODR}/{i * BATCH_SIZE + j:06d}_10.png")
            plt.imsave(f"{ODR}/{i * BATCH_SIZE + j:06d}.png", disp, cmap=cmap)


@tf.function
def metric(y_true, y_pred, thresh):
    # only consider valid disparities that also less than or equal to the maximum setting
    mask = tf.logical_and(y_true >= 0, y_true <= DISPARITIES)
    total = tf.reduce_sum(tf.cast(mask, DTYPE))
    if total == 0:
        return 0.
    error = tf.abs(y_true - tf.where(mask, y_pred[:, -tf.shape(y_true)[1]:, -tf.shape(y_true)[2]:], y_true))
    if thresh == 0:
        return tf.reduce_sum(error) / total
    return tf.reduce_sum(tf.cast(error > thresh, DTYPE)) / total


def mae(y_true, y_pred):
    return metric(y_true, y_pred, 0)


def d1p(y_true, y_pred):
    return metric(y_true, y_pred, 1.)


def d3p(y_true, y_pred):
    return metric(y_true, y_pred, 3.)


def d5p(y_true, y_pred):
    return metric(y_true, y_pred, 5.)


def bench(m, ds, start, end):
    for e in range(start, end):
        with mirrored_strategy.scope():
            m.load_weights(W_DIR + f'/{e:03d}.ckpt')
        r = m.evaluate(ds, verbose=1)
        with open(W_DIR + "/eval.txt", 'a') as f:
            f.write(f'Epoch {e}, mae: {r[1]:.5f} - d1p: {r[2]:.5f} - d3p: {r[3]:.5f} - d5p: {r[4]:.5f}\n')


if __name__ == "__main__":
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = CrfRnn(max_disparity=DISPARITIES, infer_iter=INFER_ITER, infer_rate=INFER_RATE, q_points=QUADRATURE_PTS,
                       net=NET, crf=CRF)
        model.compile(metrics=[mae, d1p, d3p, d5p])
        model.load_weights(W_DIR)
        # model = tf.keras.models.load_model(W_DIR)
    #     if not EVAL:
    #         model.load_weights(W_DIR)

    dataset = get_dataset()
    if EVAL:
        s = 488
        while True:
            if os.path.exists(W_DIR + f'/{s:03d}.ckpt.index'):
                bench(model, dataset, s, s + 1)
                s += 1
            else:
                time.sleep(7)
    else:
        save_disparity(model, dataset, cmap='jet')  # cmap='viridis'

    # path = 'logs/e2e/20210411-111024-sceneflow-408x720_d192_IR-0.01_ITER-21_TR-0.0002_BS-6_leastereo/057.ckpt'
    # with mirrored_strategy.scope():
    #     model.load_weights(path)
    # result = model.evaluate(dataset, verbose=1)
