# -*- coding: utf-8 -*-

"""Plug and Play ADMM alorithm."""

import numpy as np


class PlugAndPlayADMM:
    """Class implementing Plug and Play ADMM."""

    def __init__(self, F, G, v0, params):
        """Initialize object.

        Args:
            F: Forward model agent
            G: Prior model agent
            v0: Initial solution
            params: algorithm parameters
        """
        self.F = F
        self.G = G
        self.v = v0.copy()
        self.u = np.zeros(v0.shape)
        self.params = params

    def solve(self, verbose_output=True):
        """Compute a solution via ADMM iterations."""
        for niter in range(self.params.num_iters):
            self.x = self.F(self.v - self.u)
            self.v = self.G(self.x + self.u)
            self.u += self.x - self.v
            if verbose_output:
                print("Completed iteration %3d of %3d" %
                      (niter+1, self.params.num_iters))
        return self.v
