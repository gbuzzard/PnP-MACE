import numpy as np
from pnp_mace.utils import load_img, display_img_console, downscale, upscale
from experiment_funcs import get_experiment_noise, get_psnr, get_cropped_psnr
import pnp_mace.prioragent as prior
import pnp_mace.forwardagent as forward
from pnp_mace.equilibriumproblem import *
from dotmap import DotMap
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == '__main__':

    img_path = "./images/image_0002.jpg"  # original image is loaded to this path
    ground_truth = load_img(img_path)  # create the image
    # display_img_console(ground_truth, title="Original")

    #########################
    # Adjust shape as needed to allow for up/down sampling
    factor = 8
    new_size = factor * np.floor(ground_truth.shape / np.double(factor))
    new_size = new_size.astype(np.int)
    ground_truth = ground_truth[:new_size[0], :new_size[1]]
    clean_data = downscale(ground_truth, factor)

    #########################
    # Create data using BM3D demo
    # Possible noise types to be generated 'gw', 'g1', 'g2', 'g3', 'g4', 'g1w',
    # 'g2w', 'g3w', 'g4w'.
    noise_type = 'gw'
    noise_var = 0.0001  # Noise variance
    seed = 0  # seed for pseudorandom noise realization

    # Generate noise with given PSD
    noise, psd, kernel = get_experiment_noise(noise_type, noise_var, seed, clean_data.shape)
    # N.B.: For the sake of simulating a more realistic acquisition scenario,
    # the generated noise is *not* circulant. Therefore there is a slight
    # discrepancy between PSD and the actual PSD computed from infinitely many
    # realizations of this noise with different seeds.

    # Generate noisy image corrupted by additive spatially correlated noise
    # with noise power spectrum PSD
    noise = np.squeeze(noise)
    noisy_data = clean_data + noise
    init_image =  upscale(noisy_data, factor, Image.BICUBIC)  # ground_truth  #

    #########################
    # Set up the forward agent
    forward_agent_method = forward.prox_decimation  # forward.prox_fullsize  #

    forward_params = DotMap()
    forward_params.factor = factor
    forward_params.alpha = 0.01
    forward_params.sigmasq = 0.01
    forward_params.resample = Image.BICUBIC

    for_agent = forward.ForwardAgent(noisy_data, forward_agent_method, forward_params)

    from scipy.sparse.linalg import eigs, LinearOperator

    base = for_agent.step(0*init_image)

    def mv(vec):
        img = np.reshape(vec, init_image.shape)
        mv_prod = for_agent.step(img) - base
        return np.reshape(mv_prod, vec.shape)
    vec_len = np.prod(init_image.shape).astype(int)
    A = LinearOperator((vec_len,vec_len), matvec=mv)
    w,v = eigs(A, )
    #########################
    # Set up the prior agent
    prior_agent_method = prior.tv1_2d  # prior.bm3d_agent

    prior_params = DotMap()
    prior_params.noise_std = np.sqrt(noise_var)

    prior_agent = prior.PriorAgent(prior_agent_method, prior_params)

    #########################
    # Set up the equilibrium problem
    mu0 = 0.5  # Forward agent weight
    mu = [mu0, 1 - mu0]
    rho = 0.5
    num_iters = 1000
    keep_all_images = True

    equil_params = DotMap()
    equil_params.mu = mu
    equil_params.rho = rho
    #equil_params.added_noise_std = 0.01
    equil_params.num_iters = num_iters
    equil_params.keep_all_images = keep_all_images

    agents = [for_agent, prior_agent]
    equil_prob = EquilibriumProblem(agents, mann_iteration_mace, equil_params)

    tiled_images = np.tile(init_image[:,:,np.newaxis], (1,1,len(agents)))
    init_images = []
    for j in range(len(agents)):
        init_images.append(tiled_images[:,:,j])

    # Iterate as directed
    final_images, residuals, vectors, all_images = equil_prob.solve(init_images)
    v_sum = mu[0] * vectors[0] + mu[1] * vectors[1]
    i0 = final_images[0]
    view_stack = np.moveaxis(all_images, [0, 1, 2], [1,0,2])
    np.save("view_stack.npy", view_stack)
    residuals = np.array(residuals)
    vectors = np.array(vectors)
    with napari.gui_qt():
        viewer1 = napari.view_image(noisy_data, rgb=False)
        viewer2 = napari.view_image(v_sum, rgb=False)
        viewer4 = napari.view_image(residuals, rgb=False)
        if keep_all_images:
            viewer4 = napari.view_image(view_stack, rgb=False, order=(2,1,0))
        viewer5 = napari.view_image(vectors, rgb=False)

"""
    # denoise image (filter using 2D TV-L1, noise level = 0.03)
    denoiser = ptv.tv1_2d
    img_denoise = denoiser(ground_truth, 0.03)
    display_img_console(img_denoise, title="Denoised")

    # downscale image by local averaging
    factor = 8
    img_downscaled = downscale(ground_truth, factor)
    display_img_console(img_downscaled, title="Downscaled")
    print(img_downscaled.shape)

    b = upscale(img_downscaled, factor)
    display_img_console(b, title="Upscaled")
    # with napari.gui_qt():
    #     viewer = napari.view_image(b, rgb=False)


    # Call BM3D With the default settings.
    y_est = bm3d(noisy_full_image, psd)

    # To include refiltering:
    # y_est = bm3d(z, psd, 'refilter')

    # For other settings, use BM3DProfile.
    # profile = BM3DProfile(); # equivalent to profile = BM3DProfile('np');
    # profile.gamma = 6;  # redefine value of gamma parameter
    # y_est = bm3d(z, psd, profile);

    # Note: For white noise, you may instead of the PSD
    # also pass a standard deviation
    # y_est = bm3d(z, sqrt(noise_var));

    psnr = get_psnr(noisy_full_image, y_est)
    print("PSNR:", psnr)

    # PSNR ignoring 16-pixel wide borders (as used in the paper), due to refiltering potentially leaving artifacts
    # on the pixels near the boundary of the image when noise is not circulant
    psnr_cropped = get_cropped_psnr(noisy_full_image, y_est, [16, 16])
    print("PSNR cropped:", psnr_cropped)

    # Ignore values outside range for display (or plt gives an error for multichannel input)
    y_est = np.minimum(np.maximum(y_est, 0), 1)
    z_rang = np.minimum(np.maximum(noisy_full_image, 0), 1)
    plt.title("y, z, y_est")
    output_image = np.concatenate((noisy_full_image, np.squeeze(z_rang), y_est), axis=1)
    plt.imshow(output_image, cmap='gray')
    plt.show()

    with napari.gui_qt():
        viewer = napari.view_image(output_image, rgb=False)
"""
