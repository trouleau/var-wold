import numpy as np

import torch


class Prior:

    def __init__(self, C):
        if not isinstance(C, torch.Tensor):
            raise ValueError('Parameter `C` should be a tensor of length `n_params`.')
        self.n_params = len(C)
        self.C = C

    def logprior(self, z):
        """
        Log Prior
        """
        raise NotImplementedError('Must be implemented in child class')

    def opt_hyper(self, z):
        """
        Optimal regularization weights for the current value of z
        """
        raise NotImplementedError('Must be implemented in child class')


class GaussianPrior(Prior):

    def logprior(self, z):
        """
        Log Prior
        """
        return -torch.sum((z ** 2) / self.C)

    def opt_hyper(self, z):
        """
        Optimal regularization weights for the current value of z
        """
        return 2 * z ** 2


class LaplacianPrior(Prior):

    def logprior(self, z):
        """
        Log Prior
        """
        return -torch.sum(z / self.C)

    def opt_hyper(self, z):
        """
        Optimal regularization weights for the current value of z
        """
        return z


class GaussianLaplacianPrior(Prior):
    """
    Gaussian & Laplacian prior
    """

    def __init__(self, C, mask_gaus):
        """

        Parameters:
        C : torch.Tensor
            Parameter of the prior
        mask_gaus : torch.Tensor
            Boolean mask indicating which parameters should be Gaussian 
            distributed, the others are Laplacian distributed
        """
        if not isinstance(C, torch.Tensor):
            raise ValueError('Parameter `C` should be a tensor of length `n_params`.')
        self.n_params = len(C)
        self.C = C
        if ((not isinstance(mask_gaus, torch.Tensor)) or 
            (len(mask_gaus) != self.n_params) or 
            (mask_gaus.dtype != torch.bool)):
            raise ValueError('Parameter `mask_gaus` should be a boolean tensor of length `n_params`.')
        self.mask_gaus = mask_gaus

    def logprior(self, z):
        """
        Prior
        """
        return - torch.sum(z[self.mask_gaus] ** 2 / self.C[self.mask_gaus]) \
               - torch.sum(z[~self.mask_gaus] / self.C[~self.mask_gaus])
    
    def opt_hyper(self, z):
        """
        Optimal regularization weights for the current value of z
        """
        opt_C = torch.zeros_like(self.C)
        opt_C[self.mask_gaus] = 2 * z[self.mask_gaus] ** 2
        opt_C[~self.mask_gaus] = z[~self.mask_gaus]
        return opt_C
