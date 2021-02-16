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
# Particular forward agent methods

def prox_downsample(data_to_fit, agent_input, params):
    """
    Proximal map for downsampling forward model ~ ||y - Ax||^2 where A is a downsampling matrix
    The proximal map has the form

    .. math::
       F(x) = \mathrm{argmin}_v \;
        (1/2) \| y - Av \|^2 + (1 / (2\sigma^2)) \| x - v \|^2

    This can be shown to be

    .. math::
       F(x) = x + \sigma^2 A^T(I + \sigma^2 A A^T)^{-1} (y - Ax)

    As shown in a paper by Emma Reid, et al., replacing the linear map applied to y-Ax in this last expression is
    equivalent to changing the prior agent. This can be achieved in this function by choosing the upsampling and
    downsampling parameters separately.

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
    resample = 0
    if "resample" in params:
        resample = min([params.resample, 5])
    else:
        resample = 0
    x = agent_input
    cAx = utils.downscale(x, factor, resample)
    y = data_to_fit
    diff = cAx - y
    unscaled_step = utils.upscale(diff, factor, resample)
    scaled_step = (params.alpha / params.sigmasq) * unscaled_step
    return x - scaled_step


def prox_fullsize(data_to_fit, agent_input, params):
    """
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
