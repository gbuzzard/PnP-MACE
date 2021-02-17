import pnp_mace.utils as utils
import pnp_mace.agent as agent
import numpy as np


# This file contains the class declaration for ForwardAgent as well as particular
# forward agent methods.


class ForwardAgent(agent.Agent):
    """
    Class to provide a container and interface to various forward models.
    """

    def __init__(self, data_to_fit, forward_agent_method, params):
        """
        Define the basic elements of the forward agent: the data to fit and the
        method used to update the input to reflect the data.

        Args:
            data_to_fit: data in form used by forward_agent_method
            forward_agent_method: method to update input to reflect data
            params: parameters used by the forward agent method
        """
        super().__init__()
        self.data = data_to_fit
        self.method = forward_agent_method
        self.params = params

    def step(self, agent_input):
        """
        Apply the update method one time

        Args:
            agent_input: The current reconstruction

        Returns:
            The reconstruction after one application of the forward method
        """
        return self.method(self.data, agent_input, self.params)


######################
# Prox  agent

class ProxAgent(ForwardAgent):
    """
    Class to provide a container and interface to proximal map forward model.
    """

    def __init__(self, data_to_fit, cost_function, sigma, params):
        r"""
        Define the basic elements of the forward agent: the data to fit and the
        cost function used to promote data fitting.

        This subclass focuses on proximal maps.  The cost function should have the form f(x, data, params), where x is a
        candidate reconstruction and data is the data to fit.

        The implementation approximates a solution to

        .. math::
           F(x) = \mathrm{argmin}_v \; f(v, data, params) + (1 / (2\sigma^2)) \| x - v \|^2

        Args:
            data_to_fit: data in a form used by forward_agent_method
            cost_function: function that accepts (v, data, params), where v is a candidate reconstruction and
                           data is the data to be fit, and returns a cost
            sigma: estimate of desired step size - small sigma leads to small steps
            params: parameters used by the cost function
        """
        def forward_agent_method(data, x, cost_params):
            return prox_approximation(x, data, cost_function, sigma, cost_params)

        super().__init__(data_to_fit, forward_agent_method, params)
        self.previous_output = None


def prox_approximation(x, data, cost_function, sigma, cost_params):
    r"""
        Return an approximate solution to

        .. math::
           F(x) = \mathrm{argmin}_v \; f(v, data, params) + (1 / (2\sigma^2)) \| x - v \|^2

        Args:
            x: Candidate reconstruction
            data: Data to fit
            cost_function: function that accepts (v, data, params), where v is a candidate reconstruction and
                           data is the data to be fit, and returns a cost
            sigma: estimate of desired step size - small sigma leads to small steps
            cost_params: parameters used by the cost function

        Returns:
            An approximation of F(x) as defined above.
    """
    # TODO:  implement using sporco
    pass


def prox_downsample(data_to_fit, agent_input, params):
    r"""
    Proximal map for downsampling forward model ~ ||y - Ax||^2 where A is a downsampling matrix
    The proximal map has the form

    .. math::
       F(x) = \mathrm{argmin}_v \;
        (\alpha /2) \| y - Av \|^2 + (1 / (2\sigma^2)) \| x - v \|^2

    The solution is

    .. math::
       F(x) = x + \alpha \sigma^2 A^T(I + \alpha \sigma^2 A A^T)^{-1} (y - Ax)

    For large images, this is not practical unless A A^T is a multiple of the identity matrix, which is true for
    some important special cases, but not in general.

    Instead of resorting to solving the optimization problem directly, F(x) = v* can be written as an implicit step by

    .. math::
       v^* = x + \alpha \sigma^2 A^T (y - Av^*)

    As in V. Sridhar, X. Wang, G. T. Buzzard and C. A. Bouman, "Distributed Iterative CT Reconstruction Using Multi-Agent
    Consensus Equilibrium," in IEEE Transactions on Computational Imaging, vol. 6, pp. 1153-1166, 2020,
    doi: 10.1109/TCI.2020.3008782, we implement this version using the previous output of F(x), which is saved in an
    instance variable (available through the superclass Agent).

    Args:
        data_to_fit:  downsampled image data, assumed to be noise plus A applied to a clean image
        agent_input: current full-size reconstruction
        params:  params.alpha and params.sigmasq for step size,
                    params.factor for downsampling factor
                    params.downsample for downsampling type as in PIL.Image.py - default is NEAREST (subsampling)
                    NEAREST = NONE = 0, LANCZOS = 1, BILINEAR = 2, BICUBIC = 3, BOX = 4, HAMMING = 5
                    params.upsample for upsampling type (same possible values) - default is the same as params.downsample

    Returns:
        new full-size reconstruction after update
    """
    factor = params.factor
    downsample = 0
    if "downsample" in params:
        downsample = min([params.downsample, 5])
    upsample = 0
    if "upsample" in params:
        upsample = min([params.upsample, 5])

    x = agent_input
    v = self.previous_output
    cAx = utils.downscale(x, factor, downsample)
    y = data_to_fit
    diff = y - cAx
    unscaled_step = utils.upscale(diff, factor, upsample)
    scaled_step = (params.alpha / params.sigmasq) * unscaled_step
    return x + scaled_step


def prox_fullsize(data_to_fit, agent_input, params):
    """
    As shown in a paper by Emma Reid, et al., replacing the linear map applied to y-Ax in this last expression is
    equivalent to changing the prior agent. This can be achieved in this function by choosing the upsampling and
    downsampling parameters separately.

    Proximal map for upsampled data forward model ~ ||A^T y - x||^2 with A^T block replication

    Args:
        data_to_fit: downsampled image data (block mean)
        agent_input: full-size reconstruction
        params: params.alpha and params.sigmasq for step size and
                    params.factor for downsampling factor

    Returns:
        new full-size reconstruction after update
    """
    factor = params.factor
    resample = 0
    x = agent_input
    y = data_to_fit
    ATy = utils.upscale(y, factor, resample)
    diff = x - ATy
    scaled_step = (params.alpha / (1 + params.sigmasq)) * diff
    return x - scaled_step
