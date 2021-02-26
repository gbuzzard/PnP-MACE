# -*- coding: utf-8 -*-

"""Prior agents."""


import pnp_mace.agent as agent
from bm3d import bm3d


# This file contains the class declaration for PriorAgent as well as particular
# prior agent methods.


class PriorAgent(agent.Agent):
    """
    Class providing a container and interface to various prior models.
    """

    def __init__(self, prior_agent_method, params):
        """
        Define the basic elements of the prior agent: the data to fit
        and the method used to update the input to reflect the data.

        Args:
            prior_agent_method: method to update input to reflect data
            params: parameters used by the prior agent method
        """
        super().__init__()
        self.method = prior_agent_method
        self.params = params

    def __call__(self, agent_input):
        """
        Apply the update method one time

        Args:
            agent_input: The current reconstruction

        Returns:
            The reconstruction after one application of the prior method
        """
        return self.method(agent_input, self.params)


# Particular prior agent methods
def bm3d_method(agent_input, params):
    """
    BM3D prior

    Args:
        agent_input: full-size reconstruction
        params:  params.noise_std for noise standard deviation

    Returns:
        New full-size reconstruction after update
    """

    return bm3d(agent_input, params.noise_std)
