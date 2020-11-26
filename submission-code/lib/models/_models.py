import abc
import numpy as np

import torch

from ..utils import softmax
from ..utils.decorators import enforce_observed
from ..posteriors import Posterior
from ..priors import Prior


class Model(metaclass=abc.ABCMeta):
    """Base class for models with a log-likelihood function"""

    def __init__(self, verbose=False, device='cpu'):
        """Initialize the model
        """
        self.verbose = verbose  # Indicate verbosity behavior
        # Device to use for torch ('cpu' or 'cuda')
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        # Data-related attributes set with data in `observe`
        self.n_jumps = None     # Total Number of jumps observed
        self.dim = None         # Number of dimensions
        self.n_params = None    # Number of parameters
        self._fitted = False    # Indicate if data is properly set

    def observe(self, events, end_time=None):
        """
        Set the data for the model as well as various attributes.
        Child class must set attribute `n_params`
        """
        # Events must be tensor to use torch's automatic differentiation
        assert isinstance(events, list), "`events` should be a list of `torch.Tensor` or `numpy.ndarray`."
        # Cast to tensor if necessary
        if not isinstance(events[0], torch.Tensor):
            events = [torch.FloatTensor(num) for num in events]
        # Move the tensors to GPU if available
        self.events = [num.to(self.device) for num in events]
        self.end_time = end_time or max([max(num) for num in events if len(num) > 0])
        self.dim = len(events)
        self.n_jumps = list(map(len, events))
        # Check that all dimensions have at least one event, otherwise the
        # computation of the log-likelihood is not correct
        assert min(self.n_jumps) > 0, "Each dimension should have at least one event."
        # self.n_params  # set in child class

    @abc.abstractmethod
    @enforce_observed
    def log_likelihood(self, coeffs):
        """Evaluate the log likelihood of the model for the given parameters"""


class ModelBlackBoxVariational(Model):
    """
    Base class for models with a log-likelihood function to be learnt via
    Importance Weighted Black Box Variational Inference, based on
        https://arxiv.org/abs/1509.00519
    """

    def __init__(self, posterior, prior, C, n_samples, n_weights=1,
                 weight_temp=1, verbose=False, device='cpu', prior_kwargs={}):
        """
        Initialize the model

        Arguments:
        ----------
        posterior : Posterior
            Posterior object
        prior : Prior
            Prior object
        C : float or torch.tensor
            Prior weight (i.e., scale for `GaussianPrior`)
        n_samples : int
            Number of samples used fort he Monte Carlo estimate of expectations
        n_weights : int (optional, default: 1)
            Number of samples used for the importance weighted posterior
        weight_temp : float (optional, default: 1)
            Tempering weight of the importance weights
        verbose : bool (optional, default: False)
            Verbosity behavior
        device : str (optional, default: 'cpu')
            Device for `torch` tensors
        """
        super().__init__(verbose=verbose, device=device)
        self.posterior = posterior()             # Coeffs posterior
        self.prior = prior(C=C, **prior_kwargs)  # Coeffs prior
        self.n_samples = n_samples               # Number of samples for BBVI
        self.n_weights = n_weights               # Number of weights for Weighted-BBVI
        self.weight_temp = weight_temp           # Weight temperatur for Weighted-BBVI
        # Data-related attributes set with data in `observe`
        self.n_var_params = None  # Number of parameters of the posterior
        self.alpha = None         # loc/shape of the posterior
        self.beta = None          # rate/scale of the posterior

    def observe(self, events, end_time):
        super().observe(events=events, end_time=end_time)
        self.n_var_params = 2 * self.n_params  # must be set in child class

    def _log_importance_weight(self, eps, alpha, beta):
        """
        Compute the value of a single importance weight `log(w_i)`
        """
        # Reparametrize the variational parameters
        z = self.posterior.g(eps, alpha, beta)
        # Evaluate the log-likelihood
        loglik = self.log_likelihood(z)
        # Compute the log-prior
        logprior = self.prior.logprior(z)
        # Compute log-posterior
        logpost = self.posterior.logpdf(eps, alpha, beta)
        # print(f"{loglik:.2f} + {logprior:.2f} - {logpost:.2f}")
        return loglik + logprior - logpost

    def _objective_l(self, eps_l, alpha, beta):
        log_w_arr = torch.zeros(self.n_weights, dtype=torch.float64)
        for i in range(self.n_weights):
            eps = eps_l[i]
            # Compute the importance weights (and their gradients)
            log_w_arr[i] = self._log_importance_weight(eps, alpha, beta)
        # Temper the weights
        log_w_arr /= self.weight_temp
        w_tilde = softmax(log_w_arr).detach()  # Detach `w_tilde` from backward computations
        # Compute the weighted average over all `n_weights` samples
        value_i = w_tilde * log_w_arr
        return value_i.sum()

    @enforce_observed
    def bbvi_objective(self, x, seed=None):
        """
        Importance weighted variational objective function

        Arguments:
        ----------
        x : torch.Tensor
             The parameters to optimize
        seed : int (optional)
            Random seed for samples
        """
        if seed:
            np.random.seed(seed)  # Sampling is done in posterior with numpy
        # Split the parameters into `alpha` and `beta`
        alpha = x[:self.n_params]
        beta = x[self.n_params:]
        # Sample noise
        sample_size = (self.n_samples, self.n_weights, self.n_params)
        eps_arr = self.posterior.sample_epsilon(size=sample_size)
        # Initialize the output variables
        value = 0.0
        # Compute a Monte Carlo estimate of the expectation
        for l in range(self.n_samples):
            value += self._objective_l(eps_arr[l],  alpha, beta)
        value /= self.n_samples
        value *= -1.0  # Negative ELBO (we want to minimize it)
        return value

    @enforce_observed
    def hyper_parameter_learn(self, x, momentum=0.5):
        """
        Learn the hyper parameters of the model
        """
        opt_C_now = torch.zeros((self.n_weights, self.n_params), dtype=torch.float64)
        log_w_arr = torch.zeros(self.n_weights, dtype=torch.float64)
        # Split the parameters into `alpha` and `beta`
        alpha = x[:self.n_params]
        beta = x[self.n_params:]
        # Sample noise
        sample_size = (self.n_weights, self.n_params)
        eps_arr = self.posterior.sample_epsilon(size=sample_size)
        for i in range(self.n_weights):
            eps = eps_arr[i]
            # Compute the importance weights (and their gradients)
            log_w_arr[i] = self._log_importance_weight(eps, alpha, beta)
            z_i = self.posterior.g(eps, alpha, beta)
            opt_C_now[i] = self.prior.opt_hyper(z_i)
        # Temper the weights
        log_w_arr /= self.weight_temp
        w_tilde = softmax(log_w_arr).detach()  # Detach `w_tilde` from backward computations
        # Compute the weighted average over all `n_weights` samples
        opt_C = torch.matmul(w_tilde.unsqueeze(0), opt_C_now).squeeze().to(self.device)
        self.prior.C = (1-momentum) * opt_C + momentum * self.prior.C

    def _sample_from_expected_importance_weighted_distribution(self, eps_arr_l, alpha, beta):
        # Reparametrize the variational parameters
        log_w_arr = torch.zeros(self.n_weights, dtype=torch.float64)
        for i in range(self.n_weights):
            eps = eps_arr_l[i]
            # Compute the importance weights (and their gradients)
            log_w_arr[i] = self._log_importance_weight(eps, alpha, beta)
        # Sample one of the `z` in `z_arr` w.p. proportional to `softmax(log_w_arr)`
        j = np.random.multinomial(n=1, pvals=softmax(log_w_arr).detach().numpy()).argmax()
        return self.posterior.g(eps_arr_l[j], alpha, beta).detach().numpy()

    @enforce_observed
    def expected_importance_weighted_estimate(self, alpha, beta, n_samples=None, seed=None):
        """
        Return the mean of the expected importance weighted distribution at the
        parameters `x`

        Arguments:
        ----------
        alpha : np.ndarray (shape: `n_params`)
            The value of the variational parameters
        beta : np.ndarray (shape: `n_params`)
            The value of the variational parameters
        n_sample : int (optional)
            The number of Montre Carlo samples to use (if None, then the
            default `n_samples` attribute will be used)
        seed : int (optional)
            Random seed generator for `numpy.random`
        """
        n_samples = n_samples or self.n_samples
        if seed:
            np.random.seed(seed)
        # Sample noise
        eps_arr_t = self.posterior.sample_epsilon(
            size=(n_samples, self.n_weights, self.n_params))
        # Compute a Monte Carlo estimate of the expectation
        value = np.zeros(self.n_params)
        for l in range(n_samples):
            value += self._sample_from_expected_importance_weighted_distribution(
                eps_arr_t[l], alpha, beta)
        value /= n_samples
        return value

    def _check_convergence(self, tol):
        """Check convergence of `fit`"""
        if hasattr(self, 'coeffs_prev'):
            if torch.abs(self.coeffs - self.coeffs_prev).max() < tol:
                return True
        self.coeffs_prev = self.coeffs.detach().clone()
        return False

    def _fit(self, x0, optimizer, lr, lr_sched, tol, max_iter,
            mstep_interval=100, mstep_offset=0, mstep_momentum=0.5,
            seed=None, callback=None):
        if seed:
            np.random.seed(seed)
        # Initialize estimate
        self.coeffs = x0.clone().detach().requires_grad_(True)
        self.coeffs_prev = self.coeffs.detach().clone()
        # Reset optimizer
        self.optimizer = optimizer([self.coeffs], lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=lr_sched)
        for t in range(max_iter):
            self._n_iter_done = t

            # Gradient update
            self.optimizer.zero_grad()
            self._loss = self.bbvi_objective(self.coeffs)
            self._loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Check that the optimization did not fail
            if torch.isnan(self.coeffs).any():
                raise ValueError('NaNs in coeffs! Stop optimization...')
            # Convergence check
            if self._check_convergence(tol):
                if callback:  # Callback before the end
                    callback(self, end='\n')
                    print('Converged!')
                break
            elif callback:  # Callback at each iteration
                callback(self)
            # Update hyper-parameters
            if (t+1) % mstep_interval == 0 and t > mstep_offset:
                self.hyper_parameter_learn(self.coeffs.detach(),
                                           momentum=mstep_momentum)
        return self.coeffs
