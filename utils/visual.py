import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt

# PASCAL VOC class names
LABEL_NAMES_CITYSCAPES = np.asarray(['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                                     'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person',
                                     'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'])

LABEL_NAMES_VOC = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

LABEL_NAMES_COCO = np.asarray([
    'unknown', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge',
    'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light',
    'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea',
    'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other',
    'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
    'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged',
    'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged',
    'rug-merged'
])

LABEL_NAMES_COCO_STUFF = np.asarray([
    'unknown', 'things', 'banner', 'blanket', 'bridge',
    'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light',
    'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea',
    'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other',
    'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
    'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged',
    'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged',
    'rug-merged'
])

LABELS = {'coco-stuff': LABEL_NAMES_COCO_STUFF, 'coco': LABEL_NAMES_COCO, 'pascal_voc': LABEL_NAMES_VOC,
          'cityscapes': LABEL_NAMES_CITYSCAPES}


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC or COCO Panoptic segmentation benchmark.

    Returns:
      A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation(image, seg_map, alpha=0.5, colorized=None, ds='coco', save_name=''):
    """Visualizes input image, segmentation map and overlay view."""
    labels = LABELS[ds.lower()] if ds else LABEL_NAMES_COCO
    plt.figure(figsize=(17 if colorized is None else 23, 7))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[7, 7, 7, 1], wspace=0.02) \
        if colorized is None else gridspec.GridSpec(1, 6, width_ratios=[7, 7, 7, 7, 7, 1], wspace=0.02)
    idx = 0
    plt.subplot(grid_spec[idx])
    idx += 1
    plt.imshow(image)
    plt.axis('off')
    plt.title('Input Image' if colorized is None else "Ground Truth Reference")

    if colorized is not None:
        plt.subplot(grid_spec[idx])
        idx += 1
        plt.imshow(np.mean(image, axis=-1), cmap="gray")
        plt.axis('off')
        plt.title('Grayscale Input')

    plt.subplot(grid_spec[idx])
    idx += 1
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('Segmentation Map')

    plt.subplot(grid_spec[idx])
    idx += 1
    plt.imshow(image)
    plt.imshow(seg_image, alpha=alpha)
    plt.axis('off')
    plt.title('Segmentation Overlay')

    if colorized is not None:
        plt.subplot(grid_spec[idx])
        idx += 1
        plt.imshow(colorized)
        plt.axis('off')
        plt.title('Generated Color Image')

    unique_labels, pixel_counts = np.unique(seg_map, return_counts=True)
    unique_labels = unique_labels[pixel_counts > 10]  # ignore labels that havel total number of pixels less than 11
    ax = plt.subplot(grid_spec[idx])
    idx += 1
    full_color_map = label_to_color_image(np.arange(len(labels)).reshape(len(labels), 1))
    plt.imshow(full_color_map[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), labels[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    if save_name:
        plt.savefig(save_name, dpi=150)
    plt.show()
