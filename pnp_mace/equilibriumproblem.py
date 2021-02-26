# -*- coding: utf-8 -*-

"""EquilibriumProblem class and solution methods."""


import numpy as np
import copy


class EquilibriumProblem:
    """Class providing a container and interface to various equilibrium
    formulations and solution algorithms.
    """

    def __init__(self, agents, solution_method, params):
        """
        Define the basic elements of the equilibrium problem.

        Args:
            agents: list of forward and prior agents
            solution_method: callable method to find a solution
            params: parameters used by the solution method
        """
        self.agents = agents
        self.solution_method = solution_method
        self.solution_params = params

    def solve(self, init_image):
        """
        Interface to algorithm to solve the problem.

        Args:
            init_image: Initial image for iterative reconstruction

        Returns:
            DotMap with solution and any information about convergence
            and residual
        """
        return self.solution_method(init_image, self.agents,
                                    self.solution_params)


# Particular solution methods below

def mann_iteration_mace(init_images, agents, params):
    r"""Apply Mann iterations.

    Apply Mann iterations of the form :math:`x(k+1) = (1 - \rho) x(k) +
    \rho (2G - I)(2F - I)(x)` :cite:`buzzard-2018-plug`

    Args:
        init_images: A list of images to be used as a stacked vector
           input to the agents
        agents: A list of agents, both forward and prior
        params: `params.mu` gives weighting of agents (should be positive
           adding to 1), `params.rho` gives Mann parameter,
           `params.num_iters` gives number of iterations,
           `params.added_noise_std` gives the noise level added to each
           iteration for generative MACE

    Returns:
        A list of output images after num_iters
    """
    mu = params.mu  # array of positive real numbers adding to 1
    rho = params.rho  # Mann iteration weight - 0 gives the identity map
    num_iters = params.num_iters
    added_noise_std = 0
    if "added_noise_std" in params:
        added_noise_std = params.added_noise_std
    num_agents = len(agents)
    assert (len(params.mu) == num_agents)
    assert (np.min(mu) > 0)
    assert (np.abs(np.sum(mu) - 1) < 0.00001)

    # Get memory
    cur_images = init_images
    next_images = copy.deepcopy(init_images)
    save_all_images = False
    all_images = []
    if ("keep_all_images" in params) and params.keep_all_images:
        save_all_images = True
        all_images = np.zeros(([init_images[0].shape[0],
                                init_images[0].shape[1], num_iters]))

    if ("verbose" in params) and params.verbose:
        print_output = True
        print("Starting Mann iterations")
    else:
        print_output = False

    # Apply Mann iterations
    for j in range(num_iters):
        if added_noise_std > 0:
            cur_images = [cur + added_noise_std*np.random.randn(
                cur.shape[0], cur.shape[1]) for cur in cur_images]
        temp_images = F(agents, cur_images)
        # 2F-I
        temp_images = [2 * temp - cur for temp, cur in zip(
            temp_images, cur_images)]
        next_images = G(mu, temp_images, next_images)
        if save_all_images:
            all_images[:, :, j] = next_images[0]
        # 2G-I
        next_images = [2 * nxt - temp for nxt, temp in zip(
            next_images, temp_images)]
        # (1-rho) I + rho (2G - I) (2F - I)
        next_images = [(1 - rho) * cur + rho * nxt for cur, nxt in
                       zip(cur_images, next_images)]
        cur_images = copy.deepcopy(next_images)
        if print_output:
            print("Finished iteration " + str(j+1) + " of " + str(num_iters))

    # Calculate the residuals and vectors
    next_images = G(mu, cur_images, next_images)
    temp_images = F(agents, cur_images)
    vectors = [temp - cur for temp, cur in zip(temp_images, cur_images)]
    residuals = [nxt - temp for nxt, temp in zip(next_images, temp_images)]
    return cur_images, residuals, vectors, all_images


def F(agents, images_in):
    """The stacked agent map.

    Takes a list of agents and a list of images and applies each agent
    to the corresponding image.

    Args:
        agents: List of agents
        images_in: List of input images

    Returns:
        List of output images after applying the agents
    """
    images_out = [agent(img) for agent, img in zip(agents, images_in)]
    return images_out


def G(mu, images_in, images_out):
    """The stacked averaging map.

    Takes a list of images, applies a weighted average, and redistributes
    them in a list.

    Args:
        mu: Weights for the average
        images_in: List of input images
        images_out: List of output images (used to provide memory for
           the output)

    Returns:
        Averaged input images, copied into the list images_out
    """
    image_average = np.sum([m * img for m, img in zip(mu, images_in)], axis=0)
    for j in range(len(images_in)):
        images_out[j] = np.copy(image_average)
    return images_out
