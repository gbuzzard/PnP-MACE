# -*- coding: utf-8 -*-
# Copyright (C) by Greg Buzzard <buzzard@purdue.edu>
# All rights reserved.

r"""
This demo illustrates the solution of a MACE problem
using Mann iteration and the stacked operators F and G.  The forward
model is a subsampling operation, and the prior agent is a
denoiser, with several different options.

First a clean image is subsampled, then white noise is added to
produce noisy data.  This is used to define a forward agent that
updates to better fit the data.

In a classical Bayesian approach, this update has the form :math:`F(x)
= x + c A^T (y - Ax)`, for a constant c.  In some contexts, it's
useful to have a mismatched backprojector, which is equivalent to
replacing :math:`A^T` with an alternative matrix designed to promote
better or faster reconstruction.  As shown in a paper by Emma Reid,
this is equivalent to using the standard back projector but changing
the prior.

This demo provides the ability to explore mismatched backprojectors by
changing the upsampling method used to define :math:`A^T`.  It also
provides the ability to change the relative weight of data-fitting and
denoising by changing mu.

This demo uses parallel construction based on the MACE formulation and Mann
iterations, while superres_pnp.py uses the standard Plug and Play method.
"""

from dotmap import DotMap
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle

import pnp_mace as pnpm


def superres_mace_demo():
    """Illustrate a MACE reconstruction on an image superresolution problem."""

    """
    Load test image.
    """
    print("Reading image and creating noisy, subsampled data.")
    img_path = ("https://raw.githubusercontent.com/bwohlberg/sporco/master/"
               "sporco/data/kodim23.png")
    test_image = pnpm.load_img(img_path, convert_to_gray=True,
                               convert_to_float=True)  # create the image
    test_image = np.asarray(Image.fromarray(test_image).crop((100, 100, 356, 312)))

    """
    Adjust image shape as needed to allow for up/down sampling.
    """
    factor = 4  # Downsampling factor
    new_size = factor * np.floor(np.double(test_image.shape) / np.double(factor))
    new_size = new_size.astype(int)
    resized_image = Image.fromarray(test_image).crop((0, 0, new_size[1], new_size[0]))
    resample = Image.NONE
    ground_truth = np.asarray(resized_image)
    clean_data = pnpm.downscale(ground_truth, factor, resample)

    """
    Create noisy downsampled image.
    """
    noise_std = 0.05  # Noise standard deviation
    seed = 0          # Seed for pseudorandom noise realization
    noisy_data = pnpm.add_noise(clean_data, noise_std, seed)

    """
    Generate initial solution for MACE.
    """
    init_image = pnpm.upscale(noisy_data, factor, Image.BICUBIC)

    """
    Display test images.
    """
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
    pnpm.display_image(ground_truth, title="Original", fig=fig, ax=ax[0, 0])
    pnpm.display_image(clean_data, title="Downsampled", fig=fig, ax=ax[0, 1])
    pnpm.display_image(noisy_data, title="Noisy downsampled", fig=fig,
                       ax=ax[1, 0])
    pnpm.display_image_nrmse(init_image, ground_truth,
                             title="Bicubic reconstruction", fig=fig, ax=ax[1, 1])
    fig.show()

    """
    Set up the forward agent. We'll use a linear prox map, so we need to
    define A and AT.
    """
    print("Setting up the agents and equilibrium problem.")
    downscale_type = Image.BICUBIC
    upscale_type = Image.BICUBIC


    def A(x):
        return pnpm.downscale(x, factor, downscale_type)


    def AT(x):
        return pnpm.upscale(x, factor, upscale_type)


    step_size = 0.1
    forward_agent = pnpm.LinearProxForwardAgent(noisy_data, A, AT, step_size)

    """
    Set up the prior agent.
    """

    # Set the denoiser for the prior agent
    def denoiser(x, params):
        # denoised_x = denoise_tv_chambolle(x, weight=0.01)
        denoised_x = pnpm.bm3d_method(x, params)
        return denoised_x


    prior_agent_method = denoiser

    prior_params = DotMap()
    prior_params.noise_std = noise_std

    prior_agent = pnpm.PriorAgent(prior_agent_method, prior_params)

    """
    Compute and display one step of forward and prior agents.
    """
    print("Applying one step of each of the forward and prior agents for "
          "illustration.")
    one_step_forward = forward_agent.step(np.asarray(init_image))
    one_step_prior = prior_agent.step(np.asarray(init_image))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 5))
    pnpm.display_image_nrmse(one_step_forward, ground_truth,
                             title="One step of forward model", fig=fig, ax=ax[0])
    pnpm.display_image_nrmse(one_step_prior, ground_truth,
                             title="One step of prior model", fig=fig, ax=ax[1])
    fig.show()

    """
    Set up the equilibrium problem
    """
    mu0 = 0.5  # Forward agent weight
    mu = [mu0, 1 - mu0]
    rho = 0.5
    num_iters = 20
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

    init_images = pnpm.stack_init_image(init_image, len(agents))

    """
    Compute MACE iterations.
    """
    print("Computing the solution.")
    final_images, residuals, vectors, all_images = equil_prob.solve(init_images)
    v_sum = mu[0] * vectors[0] + mu[1] * vectors[1]
    i0 = Image.fromarray(final_images[0])

    """
    Display results.
    """
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(13.5, 5))
    pnpm.display_image(ground_truth, title="Original", fig=fig, ax=ax[0])
    pnpm.display_image_nrmse(init_image, ground_truth,
                             title="Bicubic reconstruction", fig=fig, ax=ax[1])
    pnpm.display_image_nrmse(i0, ground_truth, title="MACE reconstruction",
                             fig=fig, ax=ax[2])
    fig.show()


    input("Press 'Return' to exit: ")


if __name__ == '__main__':
    superres_mace_demo()
