"""Loads training and test data sets.
"""

import numpy as np
import struct


DIR = 'raw_data'
TRAIN_IMG_FNAME = '%s/train-images-idx3-ubyte' % DIR
TRAIN_LABEL_FNAME = '%s/train-labels-idx1-ubyte' % DIR
TEST_IMG_FNAME = '%s/t10k-images-idx3-ubyte' % DIR
TEST_LABEL_FNAME = '%s/t10k-labels-idx1-ubyte' % DIR


class Dataset(object):
    """Represents training or test data set.
    """

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels


def load(image_fname, label_fname):
    """Loads image and label data from Yann LeCun's raw IDX files.
    """
    with open(label_fname, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.fromstring(f.read(), dtype=np.uint8)

    with open(image_fname, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.fromstring(f.read(), dtype=np.uint8)

    assert num_labels == num_images

    images = np.zeros((num_images, rows * cols))
    for i in range(num_images):
        start = i * rows * cols
        end = (i + 1) * rows * cols
        images[i][:] = data[start:end]

    return images, labels


train_images, train_labels = load(TRAIN_IMG_FNAME, TRAIN_LABEL_FNAME)
assert train_images.shape[0] == 60000
assert train_images.shape[0] == train_labels.size

test_images, test_labels = load(TEST_IMG_FNAME, TEST_LABEL_FNAME)
assert test_images.shape[0] == 10000
assert test_images.shape[0] == test_labels.size

train = Dataset(train_images, train_labels)
test = Dataset(test_images, test_labels)
