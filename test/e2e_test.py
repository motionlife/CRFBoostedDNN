import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def readPFM(file, outlier=None):
    file = open(file, 'rb')
    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        # scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)

    # replace impossible disparities, clip to upper bound
    if outlier is not None and (isinstance(outlier, int) or isinstance(outlier, float)):
        data[data > outlier] = float(outlier)

    return data


def tfDecodePfm(pfm_file):
    """pure tensorflow way to decode pfm file"""
    disparity = tf.io.read_file(pfm_file)
    N = tf.strings.length(disparity)
    disparity = tf.strings.substr(disparity, N - 540 * 960 * 4, N, unit='BYTE')
    disparity = tf.reshape(tf.io.decode_raw(disparity, out_type=tf.float32), [540, 960])
    disparity = tf.abs(tf.reverse(disparity, [0]))
    return disparity


class EndModelTest(tf.test.TestCase):

    def testTfDecodePfmFile(self):
        # file_path = "../../../data/vision/sceneflow/softlinks/driving/disparity/15mm_fl_fw_fa/left/0078.pfm"
        file_path = "../../../data/vision/sceneflow/fly3dfull/benchmark/disparity/all_scenes/left/0000011.pfm"
        # file_path = "../../../data/vision/sceneflow/FlyingThings3D_fullset/disparity/TRAIN/A/0000/right/0008.pfm"
        tf_pfm = tfDecodePfm(file_path)
        true_pfm = tf.abs(readPFM(file_path))

        self.assertAllEqual(tf_pfm, true_pfm)
        # self.assertAllGreaterEqual(tf_pfm, tf.zeros_like(tf_pfm))
        print(tf.reduce_min(tf_pfm), tf.reduce_max(tf_pfm))
        plt.imsave('0000011.png', tf_pfm, cmap='jet')

        # show the 2 disparity maps for visual check
        # f = plt.figure()
        # f.add_subplot(1, 2, 1)
        # plt.imshow(np.rot90(tf_pfm, 2))
        # f.add_subplot(1, 2, 2)
        # plt.imshow(np.rot90(true_pfm, 2))
        #
        # f.add_subplot(1, 2, 1)
        # nearest = tf.image.resize(tf.expand_dims(tf_pfm, -1), [180, 320], method='nearest')
        # plt.imshow(nearest[..., 0])
        #
        # f.add_subplot(1, 2, 2)
        # gaussian = tf.image.resize(tf.expand_dims(tf_pfm, -1), [180, 320], method='gaussian')
        # plt.imshow(np.rot90(gaussian[..., 0], 2))
        plt.show(block=True)

    def testTfDecodePfmAll(self):
        DATA_DIR = "/data/hao/vision/sceneflow/softlinks/fly3d"
        ds_tl = tf.data.Dataset.list_files(DATA_DIR + "/disparity/*/left/*.pfm", shuffle=False)
        ds_tr = tf.data.Dataset.list_files(DATA_DIR + "/disparity/*/right/*.pfm", shuffle=False)
        dst = tf.data.Dataset.zip((ds_tl, ds_tr))
        for tl, tr in dst:
            tf_pfm_left, tf_pfm_right = tfDecodePfm(tl), tfDecodePfm(tr)
            true_pfm_left, true_pfm_right = tf.abs(readPFM(tl.numpy())), tf.abs(readPFM(tr.numpy()))
            self.assertAllEqual(tf_pfm_left, true_pfm_left)
            self.assertAllEqual(tf_pfm_right, true_pfm_right)

    def testPathMatched(self):
        DATA_DIR = "/data/hao/vision/sceneflow/softlinks"
        ds_left = tf.data.Dataset.list_files(DATA_DIR + "/*/rgb/*/left/*.png", shuffle=False)
        ds_right = tf.data.Dataset.list_files(DATA_DIR + "/*/rgb/*/right/*.png", shuffle=False)
        ds_tl = tf.data.Dataset.list_files(DATA_DIR + "/*/disparity/*/left/*.pfm", shuffle=False)
        ds_tr = tf.data.Dataset.list_files(DATA_DIR + "/*/disparity/*/right/*.pfm", shuffle=False)
        dst = tf.data.Dataset.zip((ds_left, ds_right, ds_tl, ds_tr)).shuffle(buffer_size=35000)
        for lp, rp, tl, tr in dst:
            lps = tf.strings.split(lp, "/")
            rps = tf.strings.split(rp, "/")
            tls = tf.strings.split(tl, "/")
            trs = tf.strings.split(tr, "/")
            # verify image number consistent
            self.assertAllEqual(lps[-1], rps[-1])
            img_num = tf.strings.split(lps[-1], '.')[0]
            self.assertAllEqual(img_num, tf.strings.split(tls[-1], '.')[0])
            self.assertAllEqual(img_num, tf.strings.split(trs[-1], '.')[0])
            # verify scene name consistent
            self.assertAllEqual(lps[-3], rps[-3])
            self.assertAllEqual(lps[-3], tls[-3])
            self.assertAllEqual(lps[-3], trs[-3])
            # verify dataset name consistent
            self.assertAllEqual(lps[-5], rps[-5])
            self.assertAllEqual(lps[-5], tls[-5])
            self.assertAllEqual(lps[-5], trs[-5])


if __name__ == '__main__':
    tf.test.main()
