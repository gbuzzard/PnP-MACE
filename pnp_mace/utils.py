import numpy as np
import skimage as ski
from skimage import io, color, transform
import matplotlib.pyplot as plt
from PIL import Image


def load_img(path):
    """
    Given the image path, return the image.

    Args:
        path: data path.

    Returns:
        Grayscale image
    """

    local_image = io.imread(path)  # read image
    local_image = ski.img_as_float(local_image)
    local_image = color.rgb2gray(local_image)

    return local_image


def display_img_console(local_img, title="", cmap='gray'):
    """
    Display an image in console using matplotlib.pyplot

    Args:
        title: title for the plot
        cmap: colormap for image display
        local_img: image to be displayed
    """
    plt.imshow(local_img, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.show()


def downscale(local_img, scale_factor):
    """
    Downscale the image by the given factor in each
    direction using block averaging.  The image is 0-padded at the end
    if needed.  This could probably be done more efficiently in a way
    closer to upscale.

    Args:
        local_img: input image
        scale_factor: downscale factor

    Returns:
        Downscaled image
    """
    new_img = ski.transform.downscale_local_mean(local_img, (scale_factor, scale_factor))
    return new_img


def upscale(local_img, scale_factor, resample):
    """
    Upscale the image by the given factor in each direction
    using replication

    Args:
        local_img: input image as a numpy array
        scale_factor: upscale factor
        resample: interpolation type as in PIL.Image.py
                    NEAREST = NONE = 0, LANCZOS = 1, BILINEAR = 2, BICUBIC = 3, BOX = 4, HAMMING = 5

    Returns:
        Upscaled image
    """
    im = Image.fromarray(local_img)
    im = im.resize(scale_factor * np.array(im.size), resample)
    new_img = np.array(im)

    return new_img


