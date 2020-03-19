import numpy as np
import torch

from .utils.decorators import enforce_observed


class FitterSGD:
    """Simple SGD Fitter"""

    def _check_convergence(self):
        """Check convergence of `fit`"""
        try:
            if torch.abs(self.coeffs - self.coeffs_prev).max() < self.tol:
                return True
        except AttributeError:
            # First time: just init `coeffs_prev`
            pass
        self.coeffs_prev = self.coeffs.detach().clone()
        return False

    def _take_gradient_step(self):
        # Gradient update
        self.optimizer.zero_grad()
        self._loss = self.objective_func(self.coeffs)
        self._loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        # Project to positive
        with torch.no_grad():
            self.coeffs[self.coeffs < 1e-10] = 1e-10

    @enforce_observed
    def fit(self, *, objective_func, x0, optimizer, lr, lr_sched, tol, max_iter,
            seed=None, callback=None):
        """
        Fit the model.

        Arguments:
        ----------
        objective_func : callable
            Objective function to minimize
        x0 : torch.tensor
            Initial estimate
        optimizer : torch.optim
            Optimizer object
        lr : float
            Learning rate of optimizer
        lr_sched : float
            Exponential decay of learning rate scheduler
        tol : float
            Tolerence for convergence
        max_iter : int
            Maximum number of iterations
        seed : int
            Random seed (for both `numpy` and `torch`)
        callback : callable
            Callback function that takes as input `self`

        Returns:
        --------
        converged : bool
            Indicator of convergence
        """
        # Set random seed
        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
        # Set callable if None
        if callback is None:
            def callback(arg):
                pass
        # Set alias for objective function
        self.objective_func = objective_func
        # Initialize estimate
        self.coeffs = x0.clone().detach().requires_grad_(True)
        # Reset optimizer & scheduler
        self.optimizer = optimizer([self.coeffs], lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=lr_sched)
        for t in range(max_iter):
            self._n_iter_done = t
            self._take_gradient_step()
            # Check that the optimization did not fail
            if torch.isnan(self.coeffs).any():
                raise ValueError('NaNs in coeffs! Stop optimization...')
            # Convergence check and callback
            if self._check_convergence():
                if callback:  # Callback before the end
                    callback(self, end='\n')
                return True
            elif callback:  # Callback at each iteration
                callback(self)
        return False
