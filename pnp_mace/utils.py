# -*- coding: utf-8 -*-

"""Utility functions."""

import requests
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image


def load_img(path, convert_to_gray=True, convert_to_float=True):
    """Load an image given a filename or url.

    Args:
        path: data path, may be local, or url as a string beginning with
          `http`
        convert_to_gray:  True to convert color images to grayscale
        convert_to_float: True to convert to floats and divide by 255

    Returns:
        Grayscale image (numpy ndarray)
    """

    if path[0:4] == "http":
        response = requests.get(path)
        local_image = Image.open(BytesIO(response.content))
    else:
        local_image = Image.open(path)  # read image

    local_image = np.asarray(local_image)

    if convert_to_float:
        local_image = local_image.astype(float)
        local_image = local_image / 255.0

    if (convert_to_gray and len(local_image.shape) > 2 and
        local_image.shape[2] == 3):
        local_image = (local_image[:, :, 0] * 299 / 1000.0 +
                       local_image[:, :, 1] * 587 / 1000.0 +
                       local_image[:, :, 2] * 114 / 1000.0)

    return np.asarray(local_image)


def display_image_nrmse(input_image, reference_image, title="", cmap='gray',
                        fig=None, ax=None):
    """Display an image along with the NRMSE relative to the reference
    image in the title.

    Args:
        input_image: image to be displayed
        reference_image: reference image for calculating nrmse of input_image
        title: title for the plot
        cmap: colormap for image display
        fig : draw in specified figure instead of creating one
        ax : plot in specified axes instead of current axes of figure
    """
    title = title + "  [NRMSE: %.3f]" % nrmse(input_image, reference_image)
    display_image(input_image, title=title, cmap=cmap, fig=fig, ax=ax)


def display_image(input_image, title=None, vmin=0, vmax=1, cmap='gray',
                  fig=None, ax=None):
    """Display an image in console using :mod:`matplotlib.pyplot`

    Args:
        input_image: image to be displayed
        title: title for the plot
        cmap: colormap for image display
        fig : draw in specified figure instead of creating one
        ax : plot in specified axes instead of current axes of figure
    """
    figp = fig
    if fig is None:
        fig = plt.figure()
        fig.clf()
        ax = fig.gca()
    elif ax is None:
        ax = fig.gca()

    im = ax.imshow(np.asarray(input_image), vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_yticklabels([])
    ax.set_xticklabels([])

    if title is not None:
        ax.set_title(title)

    shape = np.asarray(input_image).shape
    orient = 'vertical' if shape[0] >= shape[1] else 'horizontal'
    pos = 'right' if orient == 'vertical' else 'bottom'
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(pos, size="5%", pad=0.2)
    plt.colorbar(im, ax=ax, cax=cax, orientation=orient)

    if figp is None:
        fig.show()


def stack_init_image(init_image, num_images):
    """Create a list from a single image.

    Args:
        init_image: a single image to be copied and stacked
        num_images: number of copies to be included

    Returns:
        A list of copies of the original image  (numpy ndarrays)
    """
    init_images = []
    for j in range(num_images):
        init_images.append(np.asarray(init_image.copy()))

    return init_images


def downscale(input_image, scale_factor, resample):
    """Downscale the image via decimation by the given factor in each
    direction.

    Args:
        input_image: input image as a numpy array
        scale_factor: upscale factor
        resample: interpolation type as in PIL.Image
           (NEAREST = NONE = 0, LANCZOS = 1, BILINEAR = 2,
           BICUBIC = 3, BOX = 4, HAMMING = 5)

    Returns:
        Downscaled image (numpy ndarray)
    """
    new_image = upscale(input_image, 1 / scale_factor, resample)
    return np.asarray(new_image)


def upscale(input_image, scale_factor, resample):
    """Upscale the image via replication by the given factor in each
    direction.

    Args:
        input_image: input image
        scale_factor: upscale factor
        resample: interpolation type as in PIL.Image
           (NEAREST = NONE = 0, LANCZOS = 1, BILINEAR = 2,
           BICUBIC = 3, BOX = 4, HAMMING = 5)

    Returns:
        Upscaled image (numpy ndarray)
    """
    new_img = Image.fromarray(input_image)
    size = np.round(scale_factor * np.array(new_img.size)).astype(int)
    return np.asarray(new_img.resize(size, resample))


def nrmse(image, reference):
    """Calculate the NRMSE of an image with respect to a reference.

    Args:
        image: input image to be compared with reference
        reference: reference image

    Returns:
        Root mean square error of the difference of image and reference,
        divided by the root mean square of the reference
    """
    nrmse = (np.sqrt(np.mean((np.asarray(image) -
                              np.asarray(reference)) ** 2)) /
             np.sqrt(np.mean(np.asarray(reference) ** 2)))
    return np.round(nrmse, decimals=3)


def add_noise(clean_image, noise_std, seed=None):
    """Add Gaussian white noise to an image.

    Args:
        clean_image: input image
        noise_std: standard deviation of noise to be added
        seed: seed for random number generator

    Returns:
        image with noise added, clipped to valid range of values
        (numpy ndarray)
    """
    if seed is not None:
        np.random.seed(seed)
    noise = noise_std * np.random.standard_normal(
        np.asarray(clean_image).shape)
    noise = np.squeeze(noise)
    noisy_data = np.asarray(clean_image) + noise
    noisy_image = np.clip(noisy_data, 0, 1)
    return noisy_image
