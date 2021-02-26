# -*- coding: utf-8 -*-
# Copyright (C) by Greg Buzzard <buzzard@purdue.edu>
# All rights reserved.
# Portions of this file are based on
# https://github.com/gbuzzard/CT-Tutorial/blob/master/CT_Tutorial.ipynb

import imageio
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import radon, resize, iradon
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet)
import pnp_mace as pnpm
from dotmap import DotMap


def ct_demo():
    r"""
    Overview: This demo illustrates a 2D tomography example using a high-dynamic range phantom.
    First the radon transform is applied using relatively spare views.  Then proportional noise is added and filtered
    backprojection applied to produce an initial reconstruction.

    The forward model is the forward radon transform, the backprojection can be chosen as either
    filtered or unfiltered backprojection, and the prior agent is the bm3d denoiser.
    """

    # Set basic parameters
    img_path = "https://www.math.purdue.edu/~buzzard/software/image01.png"
    num_views = 60
    mu0 = 0.6  # Forward agent weight
    num_iters = 30

    # Set the filter for filtered back projection.
    filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None]
    filter_index = 0
    filter_name = filters[filter_index]

    # Set the denoiser for the prior agent
    def denoiser(x, params):
        denoised_x = denoise_tv_chambolle(x, weight=params.noise_std)
        # denoised_x = denoise_bilateral(np.clip(x, a_min=0, a_max=None), sigma_spatial=1.5)
        # denoised_x = denoise_wavelet(x, sigma=0.2)
        return denoised_x

    #
    # Get the image, sinogram and baseline reconstruction
    #
    theta = np.linspace(0., 180., num_views, endpoint=False)
    print("Reading data")
    ground_truth, mask, image_scale = get_image_and_mask(img_path)
    print("Creating sinogram")
    sinogram, sino_scale = get_scaled_sinogram(ground_truth, theta)
    noisy_sinogram = add_noise(sinogram)
    print("Creating FBP reconstruction")
    fbp_recon, fbp_noisy_recon = baseline(sinogram, noisy_sinogram, sino_scale, theta)

    #
    # Display the original and reconstructed images
    #
    image_list = [ground_truth, fbp_noisy_recon]
    titles = ['Ground truth', 'FBP from noisy data']
    display_images(image_list, titles, ground_truth)

    #
    # Set up the forward agent.
    # We'll use a linear prox map, so we need to define A (radon transform) and AT (back projection).
    #
    def A(x):
        return sino_scale * radon(x, theta=theta, circle=True)

    def AT(x):
        return iradon(x, theta=theta, circle=True, filter_name=filter_name) / sino_scale

    step_size = 0.1
    forward_agent = pnpm.LinearProxForwardAgent(noisy_sinogram, A, AT, step_size)

    #
    # Set up the prior agent - a denoiser together with the mask to ensure a proper ROI.
    #
    def prior_agent_method(x, params):
        denoised_x = denoiser(x, params)
        return mask * denoised_x

    prior_params = DotMap()
    prior_params.noise_std = step_size

    prior_agent = pnpm.PriorAgent(prior_agent_method, prior_params)

    #
    # Compute and display one step of forward and prior agents.
    #
    init_recon = fbp_noisy_recon
    one_step_forward = forward_agent.step(np.asarray(init_recon))
    one_step_prior = prior_agent.step(np.asarray(init_recon))

    #
    # Set up the equilibrium problem
    #
    mu = [mu0, 1 - mu0]
    rho = 0.5
    keep_all_images = False

    equil_params = DotMap()
    equil_params.mu = mu
    equil_params.rho = rho
    equil_params.num_iters = num_iters
    equil_params.keep_all_images = keep_all_images
    equil_params.verbose = True

    agents = [forward_agent, prior_agent]
    equil_prob = pnpm.EquilibriumProblem(agents, pnpm.mann_iteration_mace,
                                         equil_params)

    init_images = pnpm.stack_init_image(init_recon, len(agents))

    #
    # Compute MACE solution
    #
    final_images, residuals, vectors, all_images = equil_prob.solve(init_images)
    v_sum = mu[0] * vectors[0] + mu[1] * vectors[1]

    #
    # Display results.
    #
    im = final_images[0]
    display_images([im], ['Final recon'], ground_truth)

    input("Press 'Return' to exit: ")


def get_image_and_mask(image_path, imsize=None):
    r"""

    The images in this demo are designed to mimic single-energy (~100 KeV) CT images with high dynamic range.
    The pixel values are in Hounsfield units, with air as 0 and water as 1000. Hounsfield units are closely
    related to the atomic weight of the associated material. We then scale so that water is 1 and air is 0.
    In these scaled units, steel and other dense metals are about 12 to 15, but may go up to 20.

    The demo image has high dynamic range with some characteristics like those seen in
    CT scans of baggage.

    With parallel beam CT, only the center circular region can be properly reconstructed. Here we mask to that region.

    Note that if the image is resized here, then the pixel pitch (physical distance between pixels as
    measured on the object being imaged) must be adjusted later as part of scaling.

    Args:
        image_path: path to image file
        imsize: Leave imsize = [] to use the natural image size or set to an ordered pair to resize to that shape

    Returns:
        Ground truth image in scaled Hounsfield units
        mask to circular region of interest
        image_scale (1 if the image size is unchanged)
    """

    # The image is in Hounsfield units
    # Convert to Hounsfield units/1000
    orig_img = imageio.imread(image_path).astype(float)
    ground_truth = orig_img / 1000.

    # Set up the image mask to restrict to a circular ROI
    cur_size = min(ground_truth.shape)
    image_scale = 1

    if imsize is not None and imsize != cur_size:
        image_scale = cur_size / imsize
        new_size = ground_truth.shape * image_scale
        ground_truth = resize(ground_truth, new_size.astype(int))

    radius = min(ground_truth.shape) // 2

    c0, c1 = np.ogrid[0:ground_truth.shape[0], 0:ground_truth.shape[1]]
    mask = ((c0 - ground_truth.shape[0] // 2) ** 2
            + (c1 - ground_truth.shape[1] // 2) ** 2)
    mask = (mask <= radius ** 2)
    ground_truth = ground_truth * mask
    return ground_truth, mask, image_scale


def get_scaled_sinogram(ground_truth, theta, image_scale=1):
    r"""
    Scaling:

    All CT scans are relative to a baseline scan with no objects - i.e., a scan of air, which makes air 0.

    The raw projection operator (radon function in python or matlab) sums along pixels, with assumed distance 1 between
    pixels. To get the correct units, we need to scale. The first step is raw projection:

    :math:`\text{Raw projection} = Ax`
    (:math:`x` is the image as scaled above, :math:`Ax` is the output of the radon method)

    This result needs to be scaled according to physical units.

    We scale by the pixel pitch (distance between pixels) times the x-ray density of water.

    We assume pixel pitch of 0.93 mm = 0.093 cm and water density at 100KeV of 0.17 :math:`\text{cm}^{-1}`.
    With an image size of 512x512 pixels, this corresponds to a width of 512*0.93/10 = 47.16 cm or 18.75 in.
    Scaling of the projection is then

    .. math::
        y \text{ (or scaled projection) } = Ax * \text{pixel_pitch} * \text{water_xray_density}

    Note that if the original image was resized, then we have to adjust the pixel pitch accordingly so that
    the image has the same physical dimensions. For example, if the image was resized from 512x512 to 256x256,
    then the pixel pitch is increased by a factor of two.

    The sinogram values should be on the order of 0-5 for most images. Here are approximate X-ray densities
    for common materials you might have in an image

    water ~0.17 :math:`\text{cm}^{-1}`
    steel ~3.0 :math:`\text{cm}^{-1}`
    aluminum ~1.0 :math:`\text{cm}^{-1}`
    plastic ~0.1 :math:`\text{cm}^{-1}`

    Args:
        ground_truth: Ground truth image from get_image_and_mask
        theta: Vector of angles of views to use
        image_scale: image scale from get_image_and_mask

    Returns:
        sinogram of the ground truth, scaled as described above.
        sino_scale so that the sinogram is the radon transform times the sino_scale

    """
    # Set the sinogram scaling factor
    pixel_pitch = 0.093  # in cm
    water_xray_density = 0.17  # in cm^{-1}

    pixel_pitch = pixel_pitch / image_scale
    sino_scale = pixel_pitch * water_xray_density

    # Forward projection to get the sinogram, scaled as above.
    sinogram = radon(ground_truth, theta=theta, circle=True)
    sinogram *= sino_scale

    return sinogram, sino_scale


def add_noise(sinogram):
    r"""
    Noise modeling:

    The noise is modeled as additive noise but with variance that depends on the signal.

    .. math::
        y_{noise} = y + w

    For an entry :math:`y(i)', the noise :math:`w(i)`$` is :math:`N(0, \sigma^2(i))`$` with

    .. math::
        \sigma^2(i) = \frac{1}{\lambda_0} \exp(y(i)),

    where :math:`\lambda_0` is the photon count at the detector for an empty scan.
    For convenience, we define :math:`\alpha = 1/\lambda_0`.

    Args:
        sinogram: sinogram from get_scaled_sinogram

    Returns:
        noisy sinogram

    """

    # Sinogram noise variance scaling
    lambda0 = 16000  # photon count for blank scan
    alpha = 1 / lambda0

    # Add noise to the sinogram
    w = np.sqrt(alpha * np.exp(sinogram)) * np.random.standard_normal(sinogram.shape)
    noisy_sinogram = sinogram + w  # This will be used as the noisy data to fit

    return noisy_sinogram


def baseline(sinogram, noisy_sinogram, sino_scale, theta):
    r"""
    Reconstruction:

    Invert the radon transform using filtered backprojection for both the noise-free and noisy sinograms.

    Since the sinogram was scaled using sino_scale as described above, we need to apply the inverse scaling
    to the reconstructions.

    Args:
        sinogram: sinogram from get_scaled_sinogram
        noisy_sinogram: noisy sinogram from add_noise
        sino_scale: scale from get_scaled_sinogram
        theta: angles of the views in the sinogram

    Returns:
        Filtered backprojection for sinogram and noisy_sinogram
    """

    # Invert the radon transform in the noise-free and noisy cases
    fbp_recon = iradon(sinogram, theta=theta, circle=True, filter_name='ramp')
    fbp_noisy_recon = iradon(noisy_sinogram, theta=theta, circle=True, filter_name='ramp')

    # Scale to recover the appropriate units
    fbp_recon = fbp_recon / sino_scale
    fbp_noisy_recon = fbp_noisy_recon / sino_scale

    return fbp_recon, fbp_noisy_recon


def display_images(image_list, image_titles, ground_truth):
    r"""
    Display images to capture high dynamic range.

    Note that these images have a scale bar to indicate the intensity and that all the images are scaled
    to have the same intensity range. We show several intensity bands to highlight the high dynamic range.
    The reconstructions have values outside the given range, so the intensities are clipped, but the full range
    is shown in the titles.

    Args:
        image_list: list of images to display
        image_titles: title for the images
        ground_truth: ground truth image for calculating NRMSE

    Returns:
        None
    """

    titles = []
    for img, title in zip(image_list, image_titles):
        cur_min = np.round(np.amin(img), 1)
        cur_max = np.round(np.amax(img), 1)
        bounds = '{} to {}'.format(str(cur_min), str(cur_max))
        nrmse = pnpm.nrmse(img, ground_truth)
        titles.append(title + ' [NRMSE: ' + str(nrmse) + ']\n (full range is ' + bounds + ' )')

    vmin = [0, 2, 8]
    vmax = [2, 8, 15]

    num_scales = len(vmin)

    for img, title in zip(image_list, titles):

        fig, ax = plt.subplots(nrows=1, ncols=num_scales, figsize=(4.5*num_scales, 5))

        for k in range(num_scales):
            # display at various scales
            range_title = "Range: " + str(vmin[k]) + " to " + str(vmax[k])
            pnpm.display_image(img, title=range_title, fig=fig, ax=ax[k], vmin=vmin[k], vmax=vmax[k], cmap="viridis")
            #plt.colorbar()

        plt.suptitle(title)
        plt.tight_layout()
        fig.show()


if __name__ == '__main__':
    ct_demo()
