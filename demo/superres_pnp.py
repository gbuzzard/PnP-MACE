# -*- coding: utf-8 -*-
# Copyright (C) by Greg Buzzard <buzzard@purdue.edu>
# All rights reserved.

r"""
Overview: A simple demo to demonstrate the solution of a PnP problem
The forward model is a subsampling operation, and the prior agent is
the bm3d denoiser.

First a clean image is subsampled, then white noise is added to
produce noisy data.  This is used to define a forward agent that
updates to better fit the data.
"""

from dotmap import DotMap
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import pnp_mace as pnpm

"""
Load test image.
"""
print("Reading image and creating noisy, subsampled data.")
img_path = ("https://raw.githubusercontent.com/bwohlberg/sporco/master/"
           "sporco/data/kodim23.png")
test_image = pnpm.load_img(img_path, convert_to_gray=True,
                           convert_to_float=True)
test_image = np.asarray(Image.fromarray(test_image).crop((100, 100, 356, 312)))

"""
Adjust ground truth image shape as needed to allow for up/down sampling.
"""
factor = 4  # Downsampling factor
new_size = factor * np.floor(np.double(test_image.shape) / np.double(factor))
new_size = new_size.astype(int)
resized_image = Image.fromarray(test_image).crop((0, 0, new_size[1],
                                                  new_size[0]))
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
print("Setting up the agents.")
downscale_type = Image.BICUBIC
upscale_type = Image.BICUBIC


def A(x):
    return pnpm.downscale(x, factor, downscale_type)


def AT(x):
    return pnpm.upscale(x, factor, upscale_type)


step_size = 0.03
forward_agent = pnpm.LinearProxForwardAgent(noisy_data, A, AT, step_size)

"""
Set up the prior agent.
"""
prior_agent_method = pnpm.bm3d_method

prior_params = DotMap()
prior_params.noise_std = step_size

prior_agent = pnpm.PriorAgent(prior_agent_method, prior_params)

"""
Compute and display one step of forward and prior agents.
"""
one_step_forward = forward_agent.step(np.asarray(init_image))
one_step_prior = prior_agent.step(np.asarray(init_image))

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 5))
pnpm.display_image_nrmse(one_step_forward, ground_truth,
                         title="One step of forward model", fig=fig, ax=ax[0])
pnpm.display_image_nrmse(one_step_prior, ground_truth,
                         title="One step of prior model", fig=fig, ax=ax[1])
fig.show()

"""
Set up the plug and play problem
"""
num_iters = 20

pnp_params = DotMap()
pnp_params.num_iters = num_iters

pnp_prob = pnpm.PlugAndPlayADMM(forward_agent, prior_agent,
                                np.asarray(init_image),
                                pnp_params)


"""
Compute PnP ADMM iterations.
"""
print("Computing the solution.")
verbose_output = True
final_image = pnp_prob.solve(verbose_output=verbose_output)

"""
Display results.
"""
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(13.5, 5))
pnpm.display_image(ground_truth, title="Original", fig=fig, ax=ax[0])
pnpm.display_image_nrmse(init_image, ground_truth,
                         title="Bicubic reconstruction", fig=fig, ax=ax[1])
pnpm.display_image_nrmse(final_image, ground_truth, title="PnP reconstruction",
                         fig=fig, ax=ax[2])
fig.show()


input("Press 'Return' to exit:")
