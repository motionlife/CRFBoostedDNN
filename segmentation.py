import os
import datetime
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[:1], 'GPU')
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as KB
from e2e.deeplabv3plus import DeepLabV3plus

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_enable_xla_devices'
tf.config.optimizer.set_jit(True)  # enable XLA on GPU
# tf.debugging.enable_check_numerics()

AUTOTUNE = tf.data.experimental.AUTOTUNE
# NUM_CLASS = 134  # coco panoptic 133 + 1 (void)
NUM_CLASS = 55  # coco panoptic 53 (stuff) + 1 (all things) + 1 (void)
IMG_SIZE = 256  # 512
LEARN_RT = 1e-5
BATCH_SIZE = 10  # multiple of 3 Tesla V100
RANDOM_FLIP = False
RANDOM_ROTATE = False
CRF = True
INFER_ITER = 17
INFER_RATE = 0.01


def get_xy(record):
    img, mask, seg_ids, seg_lbs = record['image'], record['panoptic_image'], record['panoptic_objects']['id'], \
                                  record['panoptic_objects']['label']
    img = tf.image.resize(img, size=[IMG_SIZE, IMG_SIZE]) / 255.  # normalize to [0, 1] better than [-1, 1]!

    mask, seg_ids, seg_lbs = tf.cast(mask, tf.int32), tf.cast(seg_ids, tf.int32), tf.cast(seg_lbs, tf.int32)
    mask = mask[:, :, 0] + mask[:, :, 1] * 256 + mask[:, :, 2] * 256 ** 2
    # mask = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(seg_ids, seg_lbs) + 1), 0).lookup(mask)
    for i in range(len(seg_ids)):
        mask = tf.where(mask == seg_ids[i], -(seg_lbs[i] + 1), mask)
    mask = tf.abs(mask)
    mask = tf.where(tf.math.logical_and(mask > 0, mask < 81), 1, mask)  # mark all things as 1
    mask = tf.where(mask > 80, mask - 79, mask)  # offset stuff label start from 2

    # resize to 512 with Nearest neighbor interpolation
    mask = tf.image.resize(tf.expand_dims(mask, -1), size=[IMG_SIZE, IMG_SIZE], method='nearest')

    # random flip
    if RANDOM_FLIP and tf.random.uniform([]) < 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    if RANDOM_ROTATE:
        # random rotate
        theta = tf.random.normal([], 0., 0.1)  # to mimic the real world camera horizontal turbulence
        img = tfa.image.rotate(img, angles=theta, interpolation="BILINEAR")
        mask = tfa.image.rotate(mask, angles=theta, interpolation="NEAREST")

    return img, tf.squeeze(mask, -1)


def get_data(name='coco/2017_panoptic', split='train+validation', shuffle=True):
    dataset = tfds.load(name, split=split, shuffle_files=shuffle)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=5000, reshuffle_each_iteration=True)
    dataset = dataset.map(get_xy, num_parallel_calls=AUTOTUNE) \
        .batch(BATCH_SIZE, drop_remainder=True) \
        .prefetch(AUTOTUNE)
    return dataset


if __name__ == "__main__":
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/segmentation/' + time + f'-lr_{LEARN_RT}-bs_{BATCH_SIZE}-NoDecay-flipped_{RANDOM_FLIP}-rotate_{RANDOM_ROTATE}-stuff-sm_crf'
    weight_dir = 'logs/segmentation/20211011-224416-lr_1e-05-bs_18-NoDecay-flipped_True-rotate_True-stuff-sm_crf'
    callbacks = [KB.TensorBoard(log_dir=log_dir, write_graph=False, profile_batch=0, update_freq='epoch'),
                 KB.ModelCheckpoint(filepath=log_dir + "/{epoch:03d}.ckpt", verbose=0,
                                    save_weights_only=True, save_freq='epoch')]

    # step = tf.Variable(0, trainable=False)
    # schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=LEARN_RT,
    #     decay_steps=5870 * 2,
    #     decay_rate=0.94)
    # PiecewiseConstantDecay or PolynomialDecay
    # schedule = tf.optimizers.schedules.PiecewiseConstantDecay([10000, 15000], [1e-0, 1e-1, 1e-2])

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        # Due to huge memory use with OS=8, Xception backbone should be trained with OS=16 and only inferred with OS=8
        model = DeepLabV3plus(weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3), classes=NUM_CLASS, backbone='xception',
                              OS=16, activation=None, use_crf=CRF, supermodular=True, osi_iter=INFER_ITER,
                              infer_rate=INFER_RATE)
        # rectified_adam = tfa.optimizers.RectifiedAdam(
        #     lr=LEARN_RT * 3,
        #     total_steps=10000,
        #     warmup_proportion=0.3,
        #     min_lr=LEARN_RT,
        #     epsilon=1e-08
        # )
        adam = tf.optimizers.Adam(learning_rate=LEARN_RT,  # lambda: schedule(step),
                                  # resnet 1e-4, xception and mobilenet-v3 use 4(7)e-5 for weight decay
                                  # weight_decay=lambda: schedule(step),
                                  epsilon=1e-08
                                  )
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.MeanIoU(NUM_CLASS, name='mIoU'), 'sparse_categorical_accuracy'],
                      optimizer=adam)

        model.load_weights(weight_dir + "/023.ckpt")
        # model = tf.keras.models.load_model(log_dir + "/model081")
        # K.set_value(model.optimizer.learning_rate, LEARN_RT)
        # K.set_value(model.optimizer.min_lr, LEARN_RT)
        # K.set_value(model.optimizer.iterations, 0)

    # train_ds = get_data(name='coco/2017_panoptic', split='train')
    # model.fit(train_ds, callbacks=callbacks, epochs=180, initial_epoch=0)
    validata = get_data(name='coco/2017_panoptic', split='validation', shuffle=False)
    results = model.evaluate(validata, verbose=1)
    # with mirrored_strategy.scope():
    #     model.load_weights(weight_dir + "/092.ckpt")
    # idx = 0
    # from utils.visual import vis_segmentation
    # from matplotlib import pyplot as plt
    #
    # for k, (imgs, mask) in enumerate(validata):
    #     res = model.predict_on_batch(imgs)
    #     for img, rs in zip(imgs, res):
    #         idx += 1
    #         vis_segmentation(tf.cast(img * 255, tf.uint8), tf.argmax(rs, -1).numpy(), ds='coco-stuff',
    #                          save_name=log_dir + f'/visual92-gt/{idx:04d}.png')
