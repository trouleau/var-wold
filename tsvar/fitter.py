import numpy as np
import torch
import abc

from .utils.decorators import enforce_observed


class Fitter(metaclass=abc.ABCMeta):

    def _check_convergence(self, tol):
        """Check convergence of `fit`"""
        # Keep this feature in `numpy`
        coeffs = self.coeffs
        if isinstance(coeffs, torch.Tensor):
            coeffs = coeffs.detach().numpy()
        if hasattr(self, 'coeffs_prev'):
            if np.abs(coeffs - self.coeffs_prev).max() < tol:
                return True
        self.coeffs_prev = coeffs.copy()
        return False


class FitterIterativeNumpy(Fitter):
    """
    Basic fitter for iterative algorithms (no gradient descent) using `numpy`
    """

    @enforce_observed
    def fit(self, *, step_function, tol, max_iter, seed=None, callback=None):
        """
        Fit the model.

        Arguments:
        step_function : callable
            Function to evaluate at each iteration
        tol : float
            Tolerence for convergence
        max_iter : int
            Maximum number of iterations
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
        # Set callable if None
        if callback is None:
            def callback(arg, end=''): pass
        for t in range(max_iter):
            self._n_iter_done = t
            # Run iteration
            step_function()

            # Sanity check that the optimization did not fail
            if np.isnan(self.coeffs).any():
                raise ValueError('NaNs in coeffs! Stop optimization...')

            if (t+1) % 10 == 0:
                # Check convergence in callback (if available)
                if hasattr(callback, 'has_converged'):
                    if callback.has_converged():
                        callback(self, end='\n')  # Callback before the end
                        return True
                # Or, check convergence in fitter, and then callback
                if self._check_convergence(tol):
                    callback(self, end='\n')  # Callback before the end
                    return True

            callback(self)  # Callback at each iteration
        return False


class FitterSGD(Fitter):
    """
    Simple SGD Fitter projected on positive hyperplane.

    Methods
    -------
    fit -- Fit the model
    _take_gradient_step -- take one gradient step
    """

    def _take_gradient_step(self):
        # Gradient update
        self.optimizer.zero_grad()
        self._loss = self._objective_func(self.coeffs)
        if self.penalty:
            self._loss += self.penalty(self.coeffs)
        self._loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        # Project to positive
        with torch.no_grad():
            self.coeffs[self.coeffs < 1e-10] = 1e-10

    @enforce_observed
    def fit(self, *, objective_func, x0, optimizer, lr, lr_sched, tol, max_iter,
            penalty=None, C=1.0, seed=None, callback=None):
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
        penalty : Prior
            Penalty term
        C : float or torch.tensor
            Penalty weight
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
            def callback(arg): pass
        # Set alias for objective function
        self._objective_func = objective_func
        # Set penalty term
        if penalty:
            self.penalty = penalty(C=C)
        else:
            self.penalty = None
        # Initialize estimate
        self.coeffs = x0.clone().detach().to(self.device).requires_grad_(True)
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

            # # Convergence check and callback
            # if self._check_convergence(tol):
            #     callback(self, end='\n')  # Callback before the end
            #     return True

            if (t+1) % 100 == 0:
                # # Check convergence in callback (if available)
                # if hasattr(callback, 'has_converged'):
                #     if callback.has_converged(n=10):
                #         callback(self, end='\n')  # Callback before the end
                #         return True
                # Or, check convergence in fitter, and then callback
                if self._check_convergence(tol):
                    callback(self, end='\n')  # Callback before the end
                    return True

            callback(self)  # Callback at each iteration
        return False


class FitterVariationalEM(Fitter):
    """
    Variational EM Fitter


    """

    def _e_step(self):
        """"
        Perform a signle gradient updates of posterior coefficients `coeffs`
        """
        # Gradient update
        self.optimizer.zero_grad()
        self._loss = self._objective_func(self.coeffs)
        self._loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def _m_step(self):
        """"Update the parameters of the prior `C`"""
        # Update hyper-parameters
        condition = ((self._n_iter_done + 1) % self.mstep_interval == 0
                     and self._n_iter_done > self.mstep_offset)
        if condition:
            self.hyper_parameter_learn(self.coeffs.detach().to(self.device),
                                       momentum=self.mstep_momentum)

    @enforce_observed
    def fit(self, *, objective_func, x0, optimizer, lr, lr_sched, tol, max_iter,
            mstep_interval=100, mstep_offset=0, mstep_momentum=0.5, seed=None,
            callback=None):
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
        mstep_interval : int
            Number of iterations between M-Step updates
        mstep_offset : int
            Number of iterations before first M-Step
        mstep_momentum : float
            Momentum of M-step
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
            def callback(arg): pass
        # Set alias for objective function
        self._objective_func = objective_func
        # Set the attributes for the E and M steps
        self.mstep_interval = mstep_interval
        self.mstep_offset = mstep_offset
        self.mstep_momentum = mstep_momentum
        # Initialize estimate
        self.coeffs = x0.clone().detach().to(self.device).requires_grad_(True)
        # Reset optimizer & scheduler
        self.optimizer = optimizer([self.coeffs], lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=lr_sched)
        for t in range(max_iter):
            self._n_iter_done = t
            # E step
            self._e_step()
            # M step
            self._m_step()
            # Check that the optimization did not fail
            if torch.isnan(self.coeffs).any():
                raise ValueError('NaNs in coeffs! Stop optimization...')

            # # Convergence check and callback
            # if self._check_convergence(tol):
            #     callback(self, end='\n')  # Callback before the end
            #     print('Converged!')
            #     return True

            if (t+1) % 100 == 0:
                # Check convergence in callback (if available)
                if hasattr(callback, 'has_converged'):
                    if callback.has_converged(n=10):
                        callback(self, end='\n')  # Callback before the end
                        return True
                # Or, check convergence in fitter, and then callback
                if self._check_convergence(tol):
                    callback(self, end='\n')  # Callback before the end
                    return True

            callback(self)  # Callback at each iteration
        return False
