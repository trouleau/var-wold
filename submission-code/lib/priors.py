import abc

import torch


class Prior(metaclass=abc.ABCMeta):

    def __init__(self, C):
        """

        Parameters:
        C : torch.Tensor or float
            Parameter of the prior
        """
        if isinstance(C, float):
            self.C = C
            self.n_params = None
        elif isinstance(C, torch.Tensor):
            self.C = C
            self.n_params = len(C)
        else:
            raise ValueError(('Parameter `C` should be a float or a tensor of '
                              'length `n_params`.'))

    @abc.abstractmethod
    def logprior(self, z):
        """Log pdf of Prior"""

    def __call__(self, z):
        """Negative log-prior, to use for penalties"""
        return -1.0 * self.logprior(z)

    @abc.abstractmethod
    def opt_hyper(self, z):
        """Optimal regularization weights for the current value of z"""


class GaussianPrior(Prior):

    def logprior(self, z):
        """Log pdf of Prior"""
        return -torch.sum((z ** 2) / self.C)

    def opt_hyper(self, z):
        """Optimal regularization weights for the current value of z"""
        return 2 * z ** 2


class LaplacianPrior(Prior):

    def logprior(self, z):
        """Log pdf of Prior"""
        return -torch.sum(z / self.C)

    def opt_hyper(self, z):
        """Optimal regularization weights for the current value of z"""
        return z


class GaussianLaplacianPrior(Prior):
    """
    Gaussian & Laplacian prior
    """

    def __init__(self, C, mask_gaus):
        """

        Parameters:
        C : torch.Tensor or float
            Parameter of the prior
        mask_gaus : torch.Tensor
            Boolean mask indicating which parameters should be Gaussian
            distributed, the others are Laplacian distributed
        """
        super().__init__(C)
        if self.n_params:
            if len(mask_gaus) != self.n_params:
                raise ValueError(('Parameter `mask_gaus` should be a boolean '
                                  'tensor of length `n_params`.'))
        else:
            self.n_params = len(mask_gaus)
            self.C = self.C * torch.ones(self.n_params, dtype=torch.float64)
        self.mask_gaus = mask_gaus

    def logprior(self, z):
        """Log pdf of Prior"""
        return (- torch.sum(z[self.mask_gaus] ** 2 / self.C[self.mask_gaus])
                - torch.sum(z[~self.mask_gaus] / self.C[~self.mask_gaus]))

    def opt_hyper(self, z):
        """Optimal regularization weights for the current value of z"""
        opt_C = torch.zeros_like(self.C, dtype=torch.float64)
        opt_C[self.mask_gaus] = 2 * z[self.mask_gaus] ** 2
        opt_C[~self.mask_gaus] = z[~self.mask_gaus]
        return opt_C
