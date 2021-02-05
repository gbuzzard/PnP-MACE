from utils import upscale, downscale
from agent import *
import numpy as np

# This file contains the class declaration for ForwardAgent as well as particular
# forward agent methods.


class ForwardAgent(Agent):
    """
    Class to provide a container and interface to various forward models.
    """

    def __init__(self, data_to_fit, forward_agent_method, params):
        """
        Define the basic elements of the forward agent: the data to fit and the
        method used to update the input to reflect the data.

        :param data_to_fit: data in form used by forward_agent_method
        :param forward_agent_method: method to update input to reflect data
        :param params: parameters used by the forward agent method
        """
        super().__init__()
        self.data = data_to_fit
        self.method = forward_agent_method
        self.params = params

    def step(self, agent_input):
        """
        Apply the update method one time

        :param agent_input: The current reconstruction
        :return: The reconstruction after one application of the forward method
        """
        return self.method(self.data, agent_input, self.params)


######################
# Particular forward agent methods

def prox_decimation(data_to_fit, agent_input, params):
    """
    Proximal map for downsampling forward model ~ ||y - Ax||^2 with A a block averaging matrix

    :param data_to_fit:  downsampled image data (block mean)
    :param agent_input: full-size reconstruction
    :param params:  params.alpha and params.sigmasq for step size,
                    params.factor for downsampling factor
                    params.resample for interpolation type as in PIL.Image.py
                    NEAREST = NONE = 0, LANCZOS = 1, BILINEAR = 2, BICUBIC = 3, BOX = 4, HAMMING = 5
    :return:  new full-size reconstruction after update
    """
    factor = params.factor
    resample = 0
    if "resample" in params:
        resample = min([params.resample, 5])
    x = agent_input
    cAx = downscale(x, factor)  # This is (1/factor^2) A, where A is block sum
    y = data_to_fit
    diff = cAx - y
    unscaled_step = upscale(diff, factor, resample)  # This is A.T if resample=0, otherwise another upscaling matrix
    scaled_step = (params.alpha / params.sigmasq) * unscaled_step
    return x - scaled_step


def prox_fullsize(data_to_fit, agent_input, params):
    """
    Proximal map for upsampled data forward model ~ ||A^T y - x||^2 with A^T block replication

    :param data_to_fit: downsampled image data (block mean)
    :param agent_input: full-size reconstruction
    :param params: params.alpha and params.sigmasq for step size and
                    params.factor for downsampling factor
    :return: new full-size reconstruction after update
    """
    factor = params.factor
    resample = 0
    x = agent_input
    y = data_to_fit
    ATy = upscale(y, factor, resample)
    diff = x - ATy
    scaled_step = (params.alpha / (1 + params.sigmasq)) * diff
    return x - scaled_step
