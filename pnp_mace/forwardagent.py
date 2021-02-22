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
        self.data = np.asarray(data_to_fit)
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


#############################
# Proximal map forward  agent

class ProxForwardAgent(ForwardAgent):
    """
    Class to provide a container and interface to proximal map forward model.
    """

    def __init__(self, data_to_fit, cost_function, sigma, params):
        r"""
        Define the basic elements of the proximal map forward agent: the data to fit and the
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
            return utils.prox_approximation(x, data, cost_function, sigma, cost_params)

        super().__init__(data_to_fit, forward_agent_method, params)
        self.previous_output = None


class LinearProxForwardAgent(ForwardAgent):
    r"""
    Class to provide a container and interface to a proximal map forward model for a cost function of the form

    .. math::
       f(x) = (1/2) \| y - Av \|^2

    The associated proximal map has the form

    .. math::
       F(x) = \mathrm{argmin}_v \;
        (1/2) \| y - Av \|^2 + (1 / (2\sigma^2)) \| x - v \|^2

    The solution is

    .. math::
       F(x) = x + \sigma^2 (I + \sigma^2 A^T A)^{-1} A^T (y - Ax)

    This form is typically not practical unless A A^T is a multiple of the identity matrix, which is true for
    some important special cases, but not in general.

    Instead of solving the optimization problem as in the ProxAgent, F(x) = v* can be written as an implicit step using

    .. math::
       v^* = x + \alpha \sigma^2 A^T (y - Av^*)

    As in V. Sridhar, X. Wang, G. T. Buzzard and C. A. Bouman, "Distributed Iterative CT Reconstruction Using Multi-Agent
    Consensus Equilibrium," in IEEE Transactions on Computational Imaging, vol. 6, pp. 1153-1166, 2020,
    doi: 10.1109/TCI.2020.3008782, we implement this version using the previous output of v* = F(x), which is saved in an
    instance variable.
    """

    def __init__(self, data_to_fit, A, AT, sigma):
        r"""
        Define the basic elements of the forward agent: the data to fit and the data-fitting
        cost function (determined by the forward and back projectors A and AT).

        As shown in a paper by Emma Reid, et al., replacing AT by a linear operator B that approximates AT is
        equivalent to changing the prior agent.  In some contexts, this is known as mismatched back-projection.

        One example is to use one form of downsampling for A and a type of upsampling for B that approximates AT but is
        not exactly AT.  Another example is to use forward projection from the Radon transform as A and filtered
        back-projection as B.  In some cases, this improves the final reconstruction.

        Args:
            data_to_fit: data in a form consistent with the output of A
            A: linear operator - forward projector
            AT: linear operator - back projector (see notes above on mismatched back-projection)
            sigma: estimate of desired step size - small sigma leads to small steps
        """
        def forward_agent_method(data, x, cost_params):
            return self.linear_prox_implicit_step(x, data, A, AT, sigma**2)

        params = None
        super().__init__(data_to_fit, forward_agent_method, params)
        self.previous_output = None

    def restart(self):
        self.previous_output = None

    def linear_prox_implicit_step(self, x, data, A, AT, sigmasq):
        r"""
        Instead of solving the proximal map optimization problem as in the ProxAgent, F(x) = v* can be written as an
        implicit step using

        .. math::
           v^{k+1} = x + \sigma^2 A^T (y - Av^k)

        where y = data. We implement this version using the previous output of v* = F(x), which is saved in an instance variable.

        Args:
            x: Candidate reconstruction
            data: Data to fit
            A: linear operator - forward projector.
               Assumed to be either an operator or a numpy array that multiplies x.
            AT: linear operator - back projector (see notes above on mismatched back-projection).
                Assumed to be either an operator or a numpy array that multiplies Ax.
            sigmasq: estimate of desired step size - small sigma leads to small steps

        Returns:
            An approximation of F(x) = v* as defined above.
        """
        v = self.previous_output
        if v is None:
            # Apply one gradient descent step to initialize v
            self.previous_output = x
            v = self.linear_prox_implicit_step(x, data, A, AT, sigmasq)

        # Check on the type of A and apply it appropriately
        if callable(A):
            Av = A(v)
        else:
            Av = np.matmul(A, v)
        # Get the data difference
        y = data

        # Check on the type of AT and apply it appropriately
        diff = y - Av
        if callable(AT):
            step = AT(diff)
        else:
            step = np.matmul(AT, diff)
        # Take a step
        scaled_step = sigmasq * step
        new_x = x + scaled_step
        self.previous_output = np.copy(new_x)
        return new_x

