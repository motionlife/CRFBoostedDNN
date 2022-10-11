# end to end architecture of DNN and CRF
import datetime
import tensorflow as tf
# import tensorflow_addons as tfa
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[1:], 'GPU')

import os
from e2e.model import CrfRnn

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
tf.config.optimizer.set_jit(True)  # enable XLA on GPU
# tf.debugging.enable_check_numerics()
tf.keras.backend.set_floatx('float32')
DTYPE = tf.float32
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 9
INFER_ITER = 21  # try 7, 12, 14, 21 when inference
INFER_RATE = 0.01
QUADRATURE_PTS = 3
LEARN_RT = 0.00002
DISPARITIES = 192
DS_NAME = 'sceneflow'
DATA_DIR = "/data/hao/vision/vkitti"
NET = 'leastereo'
CRF = True
CROPPED_H, CROPPED_W = (264, 648)  # (408, 720) # (288, 640)  # (304, 640)  # (240, 624) # (320, 1216/1040+16)
MIRROR = False


def get_image(img_file, channels=3, dtype=tf.uint8):
    img = tf.io.read_file(img_file)
    img = tf.image.decode_image(img, channels=channels, dtype=dtype)
    img = tf.cast(img, DTYPE)
    return img


def get_disparity(file_path):
    # decode kitti or driving stereo disparity, both are sparse real-world data
    if tf.strings.regex_full_match(file_path, '.*_10.png$|.*2018-.*png$'):
        disp = get_image(file_path, channels=0, dtype=tf.uint16) / 256
        return tf.where(disp > 0, disp, -1)
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


def path2data(lp, rp, tl, tr=None):
    imgl, imgr = get_image(lp), get_image(rp)
    disparities = get_disparity(tl)
    if MIRROR:
        images = tf.stack([imgl, tf.image.flip_left_right(imgr), imgr, tf.image.flip_left_right(imgl)])
        disparities = tf.stack([disparities, tf.image.flip_left_right(get_disparity(tr))])
    else:
        images = tf.stack([imgl, imgr])
    # random crop for training data
    limit = [1, tf.shape(images)[1] - CROPPED_H + 1, tf.shape(images)[2] - CROPPED_W + 1, 1]
    offset = tf.random.uniform(shape=[4], dtype=tf.int32, maxval=tf.int32.max) % limit
    images = tf.slice(images, offset, size=[tf.shape(images)[0], CROPPED_H, CROPPED_W, 3])
    if MIRROR:
        disparities = tf.slice(disparities, offset, size=[tf.shape(disparities)[0], CROPPED_H, CROPPED_W, 1])
    else:
        disparities = tf.slice(disparities, offset[1:], size=[CROPPED_H, CROPPED_W, 1])
    # normalize the input image pair
    mean, variance = tf.nn.moments(images, axes=[0, 1, 2], keepdims=True)
    images = tf.nn.batch_normalization(images, mean, variance, offset=None, scale=None, variance_epsilon=1e-17)
    return tuple(tf.split(images, 2) if MIRROR else tf.unstack(images, 2)), disparities


def get_dataset(shuffle=True, name='*', scene='*', list_path=""):
    if list_path:
        flist = open(list_path, 'r').read().splitlines()
        lpath, ltp, rpath, rtp = [], [], [], []
        for l in flist:
            lpath.append(DATA_DIR + "frames_finalpass/" + l)
            ltp.append(DATA_DIR + "disparity/" + l[:-3] + "pfm")
            r = l.replace("/left/", "/right/")
            rpath.append(DATA_DIR + "frames_finalpass/" + r)
            rtp.append(DATA_DIR + "disparity/" + r[:-3] + "pfm")
        lpath = tf.data.Dataset.from_tensor_slices(lpath)
        rpath = tf.data.Dataset.from_tensor_slices(rpath)
        ltp = tf.data.Dataset.from_tensor_slices(ltp)
        rtp = tf.data.Dataset.from_tensor_slices(rtp)
    else:
        lpath = tf.data.Dataset.list_files(DATA_DIR + f"/{name}/{scene}/frames/rgb/Camera_0/*", False)
        rpath = tf.data.Dataset.list_files(DATA_DIR + f"/{name}/{scene}/frames/rgb/Camera_1/*", False)
        ltp = tf.data.Dataset.list_files(DATA_DIR + f"/{name}/{scene}/frames/depth/Camera_0/*", False)
        rtp = tf.data.Dataset.list_files(DATA_DIR + f"/{name}/{scene}/frames/depth/Camera_1/*",
                                         False) if MIRROR else None
    dst = tf.data.Dataset.zip((lpath, rpath, ltp, rtp) if MIRROR else (lpath, rpath, ltp))
    dst = dst.cache()
    if shuffle:
        dst = dst.shuffle(buffer_size=46536, reshuffle_each_iteration=True)
    dst = dst.map(path2data, AUTOTUNE, False)
    if MIRROR:
        dst = dst.unbatch()
    return dst.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)


# @tf.function
# def loss(y_true, y_pred):
#     mask = tf.logical_and(y_true >= 0, y_true <= DISPARITIES)
#     per_example_loss = huber(y_true, tf.where(mask, y_pred, y_true))
#     # sample_weight = tf.where(tf.reduce_any(y_true < 0, axis=[1, 2]), 1.2, 1.)  # slightly increase weight of real data
#     return tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE)

class Loss(tf.keras.losses.Loss):
    def __int__(self, name="loss_wrapper"):
        super(Loss, self).__int__(name=name)

    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true, axis=-1)
        mask = tf.logical_and(y_true >= 0, y_true <= DISPARITIES)
        return tf.keras.losses.Huber(delta=1.0)(y_true, tf.where(mask, y_pred, y_true))


huber = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.NONE)


@tf.function
def loss(y_true, y_pred):
    mask = tf.logical_and(y_true >= 0, y_true <= DISPARITIES)
    per_example_loss = huber(y_true, tf.where(mask, y_pred, y_true))
    # sample_weight = tf.where(tf.reduce_any(y_true < 0, axis=[1, 2]), 1.2, 1.)  # slightly increase weight of real data
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE)


@tf.function
def loss3(y_true, y_pred):
    b = tf.shape(y_true)[0]
    # if tf.equal(b, tf.shape(y_pred)[0]):
    #     return loss(y_true, y_pred[-b:])
    mask = tf.logical_and(y_true >= 0, y_true <= DISPARITIES)
    h1 = huber(y_true, tf.where(mask, y_pred[:b], y_true))
    h2 = huber(y_true, tf.where(mask, y_pred[b:2 * b], y_true))
    h3 = huber(y_true, tf.where(mask, y_pred[-b:], y_true))
    per_example_loss = 0.5 * h1 + 0.7 * h2 + 1.0 * h3
    # sample_weight = tf.where(tf.reduce_any(y_true < 0, axis=[1, 2]), 1.2, 1.)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE)


@tf.function
def mae(y_true, y_pred):
    mask = tf.logical_and(y_true >= 0, y_true <= DISPARITIES)
    total = tf.reduce_sum(tf.cast(mask, DTYPE))
    y_pred = tf.where(mask, y_pred[-tf.shape(y_true)[0]:], y_true)
    return tf.reduce_sum(tf.abs(y_true - y_pred)) / total if total != 0 else 0.


if __name__ == "__main__":
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/e2e/' + time + f'-{DS_NAME}-{CROPPED_H}x{CROPPED_W}_d{DISPARITIES}_IR-{INFER_RATE}_ITER-{INFER_ITER}_TR-{LEARN_RT}_BS-{BATCH_SIZE}_{NET}-supermodular'
    weights_dir = 'logs/e2e/20210909-100832-sceneflow-408x720_d192_IR-0.01_ITER-21_TR-1e-05_BS-6_leastereo-supermodular/old'
    # scheduler_cbk = tf.keras.callbacks.LearningRateScheduler((lambda e: 0.98 ** e * LEARN_RT), verbose=1)
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False, profile_batch=0),
                 # scheduler_cbk,
                 tf.keras.callbacks.ModelCheckpoint(
                     filepath=log_dir + "/{epoch:03d}.ckpt",
                     verbose=0,
                     save_weights_only=True,
                     save_freq='epoch')]

    mirrored_strategy = tf.distribute.MirroredStrategy()
    # Move the creation, compiling  and weights loading of Keras model inside strategy.scope
    # tmp_model = CrfRnn(max_disparity=DISPARITIES, infer_iter=INFER_ITER, infer_rate=INFER_RATE, q_points=QUADRATURE_PTS,
    #                    net='psm', crf=CRF, supermodular=True)
    # tmp_model.load_weights('logs/e2e/20210922-112711-sceneflow-256x640_d192_IR-0.01_ITER-21_TR-4e-05_BS-6_psm-supermodular/053.ckpt')
    with mirrored_strategy.scope():
        model = CrfRnn(max_disparity=DISPARITIES, infer_iter=INFER_ITER, infer_rate=INFER_RATE, q_points=QUADRATURE_PTS,
                       net=NET, crf=CRF, supermodular=True)
        # optimizer = tfa.optimizers.RectifiedAdam(
        #     lr=LEARN_RT * 5,
        #     total_steps=7777,
        #     warmup_proportion=0.3,
        #     min_lr=LEARN_RT)
        optimizer = tf.keras.optimizers.Adam(lr=LEARN_RT)
        # optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=lambda: 1e-3 * schedule(step))
        model.compile(loss=loss3 if NET == 'psm' else loss, metrics=[mae], optimizer=optimizer)
        model.load_weights(weights_dir + '/089.ckpt')
        # model.unary_net.set_weights(tmp_model.unary_net.get_weights())
        # model.pairwise_net.set_weights(tmp_model.pairwise_net.get_weights())
        tf.keras.backend.set_value(model.optimizer.learning_rate, LEARN_RT)
        tf.keras.backend.set_value(model.optimizer.iterations, 0)

    dataset = get_dataset(shuffle=True, name=DS_NAME, scene='fly3d', list_path="")
    # val_data = get_dataset(shuffle=False, name='real_kitti', scene='validation-201*')
    model.fit(dataset, callbacks=callbacks, epochs=128, initial_epoch=0)
