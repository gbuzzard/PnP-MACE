import pnp_mace as pnpm
from dotmap import DotMap
from PIL import Image
import numpy as np

if __name__ == '__main__':

    img_path = "https://www.math.purdue.edu/~buzzard/software/cameraman_clean.jpg"  # original image is loaded to this path
    test_image = pnpm.load_img(img_path)  # create the image
    image_data = np.asarray(test_image.convert("F")) / 255.0
    ground_truth = Image.fromarray(image_data)
    pnpm.display_image(ground_truth, title="Original")

    #########################
    # Adjust shape as needed to allow for up/down sampling
    factor = 4
    new_size = factor * np.floor(ground_truth.size / np.double(factor))
    new_size = new_size.astype(int)
    ground_truth = ground_truth.crop((0, 0, new_size[0], new_size[1]))
    resample = Image.NONE # NEAREST = NONE = 0, LANCZOS = 1, BILINEAR = 2, BICUBIC = 3, BOX = 4, HAMMING = 5
    clean_data = pnpm.downscale(ground_truth, factor, resample)
    pnpm.display_image(clean_data, title="Downsampled")

    #########################
    # Create noisy data
    # Note: the BM3D demos have examples to show how to create additive spatially correlated noise
    noise_std = 0.05  # Noise standard deviation
    seed = 0  # seed for pseudorandom noise realization
    noisy_image = pnpm.add_noise(clean_data, noise_std, seed)

    # Generate initial reconstruction.  This could be all 0s if there's no good initial estimate.
    init_image = pnpm.upscale(noisy_image, factor, Image.BICUBIC)  # initial image for PnP

    # Display the initial reconstruction
    pnpm.display_image(noisy_image, title="Noisy data")
    pnpm.display_image_nrmse(init_image, ground_truth, title="Initial reconstruction")

    #########################
    # Set up the forward agent
    downscale_type = Image.BICUBIC
    upscale_type = Image.BICUBIC

    # We'll use a linear prox map, so we need to define A and AT
    def A(x): return np.asarray(pnpm.downscale(Image.fromarray(x), factor, downscale_type))
    def AT(x): return np.asarray(pnpm.upscale(Image.fromarray(x), factor, upscale_type))

    step_size = 0.1
    forward_agent = pnpm.LinearProxForwardAgent(noisy_image, A, AT, step_size)
    one_step_forward = forward_agent.step(np.asarray(init_image))
    pnpm.display_image_nrmse(one_step_forward, ground_truth, title="One step of forward model")

    #########################
    # Set up the prior agent
    prior_agent_method = pnpm.bm3d_method

    prior_params = DotMap()
    prior_params.noise_std = step_size

    prior_agent = pnpm.PriorAgent(prior_agent_method, prior_params)
    one_step_prior = prior_agent.step(np.asarray(init_image))
    pnpm.display_image_nrmse(one_step_prior, ground_truth, title="One step of prior model")

    #########################
    # Set up the equilibrium problem
    mu0 = 0.5  # Forward agent weight
    mu = [mu0, 1 - mu0]
    rho = 0.5
    num_iters = 10
    keep_all_images = False

    equil_params = DotMap()
    equil_params.mu = mu
    equil_params.rho = rho
    equil_params.num_iters = num_iters
    equil_params.keep_all_images = keep_all_images

    agents = [forward_agent, prior_agent]
    equil_prob = pnpm.EquilibriumProblem(agents, pnpm.mann_iteration_mace, equil_params)

    init_images = pnpm.stack_init_image(init_image, len(agents))

    # Iterate as directed
    final_images, residuals, vectors, all_images = equil_prob.solve(init_images)
    v_sum = mu[0] * vectors[0] + mu[1] * vectors[1]
    i0 = Image.fromarray(final_images[0])
    pnpm.display_image_nrmse(i0, ground_truth, title="Final reconstruction")
