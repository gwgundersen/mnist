"""
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import mnist


def render(image, width=None):
    """Renders a square, grayscale image from a numpy array.
    """
    d = width if width else np.sqrt(image.size)
    assert d == int(d)
    d = int(d)
    image = image.reshape(d, d)
    plt.imshow(Image.fromarray(image))
    plt.show()


def render_all_n(dataset, n, max=100):
    """Renders all images for a particular number, to a limit (default 10).
    """
    dim = np.sqrt(max)
    assert dim == int(dim)
    dim = int(dim)
    rows = [None] * dim

    for i, image in enumerate(dataset.images[dataset.labels == n]):
        if i >= max:
            break

        d = np.sqrt(image.size)
        assert d == int(d)
        d = int(d)
        image = image.reshape((d, d))
        row_ix = int(np.floor(i / dim))

        if i % dim == 0:
            rows[row_ix] = image
        else:
            row = rows[row_ix]
            rows[row_ix] = np.concatenate((row, image), axis=1)

    agg = None
    for row in rows:
        if _exists(agg):
            agg = np.concatenate((agg, row), axis=0)
        else:
            agg = row

    render(agg)


def _exists(ndarray):
    exists = True
    try:
        ndarray.shape
    except AttributeError:
        exists = False
    return exists


render_all_n(mnist.train, 2, max=400)
