import numpy as np
import pnp_mace.utils as utils
import pnp_mace.prioragent as prior
import pnp_mace.forwardagent as forward
from pnp_mace.equilibriumproblem import *
from dotmap import DotMap
from PIL import Image

if __name__ == '__main__':

    img_path = "https://www.math.purdue.edu/~buzzard/software/cameraman_clean.jpg"  # original image is loaded to this path
    ground_truth = utils.load_img(img_path)  # create the image
    utils.display_img_console(ground_truth, title="Original")

    #########################
    # Adjust shape as needed to allow for up/down sampling
    factor = 4
    new_size = factor * np.floor(ground_truth.size / np.double(factor))
    new_size = new_size.astype(int)
    ground_truth = ground_truth.crop((0, 0, new_size[0], new_size[1]))
    resample = Image.NONE # NEAREST = NONE = 0, LANCZOS = 1, BILINEAR = 2, BICUBIC = 3, BOX = 4, HAMMING = 5
    clean_data = utils.downscale(ground_truth, factor, resample)
    utils.display_img_console(clean_data, title="Downsampled")

    #########################
    # Create noisy data
    noise_std = 0.1  # Noise standard deviation
    seed = 0  # seed for pseudorandom noise realization

    # Generate noisy image
    # Note: the BM3D demos have examples to show how to create additive spatially correlated noise
    noisy_image = utils.add_noise(clean_data, noise_std, seed)
    init_image = utils.upscale(noisy_image, factor, Image.BICUBIC)  # initial image for PnP #

    # Display
    utils.display_img_console(noisy_image, title="Noisy data")
    nrmse = utils.nrmse(init_image, ground_truth)
    title = "Initial reconstruction, nrmse = " + str(nrmse)
    utils.display_img_console(init_image, title=title)

    #########################
    # Set up the forward agent
    forward_agent_method = forward.prox_downsample  # forward.prox_fullsize  #

    forward_params = DotMap()
    forward_params.factor = factor
    forward_params.alpha = 0.01
    forward_params.sigmasq = 0.01
    forward_params.resample = Image.BICUBIC

    for_agent = forward.ForwardAgent(noisy_image, forward_agent_method, forward_params)

    #########################
    # Set up the prior agent
    prior_agent_method = prior.tv1_2d  # prior.bm3d_agent

    prior_params = DotMap()
    prior_params.noise_std = np.sqrt(noise_std)

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
    utils.display_img_console(img_denoise, title="Denoised")

    # utils.downscale image by local averaging
    factor = 8
    img_utils.downscaled = utils.downscale(ground_truth, factor)
    utils.display_img_console(img_utils.downscaled, title="utils.downscaled")
    print(img_utils.downscaled.shape)

    b = utils.upscale(img_utils.downscaled, factor)
    utils.display_img_console(b, title="utils.upscaled")
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
