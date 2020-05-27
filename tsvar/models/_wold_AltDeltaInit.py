import torch
import numba
import warnings
import numpy as np

from . import Model
from ..fitter import FitterSGD
from ..utils.decorators import enforce_observed


warnings.filterwarnings(action='ignore', module='numba',
                        category=numba.NumbaPendingDeprecationWarning)


@numba.jit(nopython=True, fastmath=True)
def _wold_model_init_cache(events):
    dim = len(events)
    n_jumps = [len(events[i]) for i in range(dim)]
    delta_ikj = [np.zeros((n_jumps[i], dim)) for i in range(dim)]
    # For each reiceiving dimension
    for i in range(dim):
        last_idx_tlj = [-1 for j in range(dim)]
        last_tki = 0.0
        # For each observed event, compute the inter-arrival time with
        # each dimension
        for k, tki in enumerate(events[i]):
            if k == 0:
                # Delta should be ignored for the first event.
                # Mark has invalid
                delta_ikj[i][k, :] = events[i][0]
                continue
            last_tki = events[i][k-1]
            # For each incoming dimension
            for j in range(dim):
                if (last_idx_tlj[j] < 0) and (events[j][0] >= last_tki):
                    # If the 1st event in dim `j` comes after `last_tki`, it should be ignored.
                    # Mark as invalid
                    delta_ikj[i][k, :] = events[i][0]
                    continue
                # Update last index for dim `j`
                l = max(last_idx_tlj[j], 0)
                while (events[j][l] < float(last_tki)):
                    l += 1
                    if l == n_jumps[j]:
                        break
                l -= 1
                last_idx_tlj[int(j)] = int(l)
                # Set delta_ikj
                delta_ikj[i][k, j] = last_tki - events[j][l]
        last_tki = tki
    return delta_ikj


class WoldModelOther(Model):
    """Class for the Multivariate Wold Point Process Model with intensity
    function

        $\lamda_i(t) = \alpha_0^i + \sum_{j=1}^D \frac{\alpha_j^i}{\beta_j^i + \delta_j^i(t)}$

    where $D$ is the number of dimensions and $\delta_j^i(t)$ is the time
    between the last event $t_k^i$ in dim. $i$ and the latest event $t_l^j$ in
    dim. $j$ prior to $t_k^i$.

    Note: When setting the data with `observe`, an artificial event at the end
    of the observation window, i.e. `end_time`, is added as last event in each
    dimension. This is a trick to make the computation of the log-likelihood
    easier. Indeed, we need to evaluate the intensity function at each events,
    as well as at the end of the observation window in order to compute the
    integral term of the log-likelihood. Therefore, the last event in each
    dimension of the `events` attribute is a fictious event occuring at time
    `end_time`.

    Note: to make the computation of the log-likelihood faster, we pre-compute
    once the inter-arrival times in attribute `delta_ikj`, where
    `delta_ikj[i][k, j]` holds $t_{k-1}^i - t_l^j$, where $l$ is the index of
    the latest event in dimension $j$ such that $t_l^j < t_{k-1}^i$, which is
    used to computed the intensity function for dimension $i$ at time $t_k^i$.
    """

    def observe(self, events, end_time=None):
        """Set the data for the model as well as various attributes, and cache
        computations of inter-arrival time for future log-likelihood calls.
        """
        super().observe(events, end_time)

        #
        # TODO: Observed events, add a virtual event at `end_time` for easier
        # log-likelihood computation. Remove the virtual event, it's nasty and
        # will eventually introduce bugs.
        for i in range(self.dim):
            self.events[i] = torch.cat((self.events[i], torch.tensor([self.end_time], dtype=torch.float, device=self.device)))
        self.n_jumps = list(map(len, self.events))
        #
        # Number of parameters of the model
        self.n_params = (self.dim                # shape baseline (mu)
                         + self.dim * self.dim   # shape beta
                         + self.dim * self.dim)  # shape adjacency (alpha)
        # Init cache if necessary
        self._init_cache()

    def _init_cache(self):
        events_ = [ev.numpy() for ev in self.events]
        # Compute cache with numba
        self.delta_ikj = _wold_model_init_cache(events_)
        # Cast numpy output to torch
        self.delta_ikj = [torch.tensor(
            self.delta_ikj[i], dtype=torch.float, device=self.device)
            for i in range(self.dim)]
        self._observed = True

    @enforce_observed
    def log_likelihood(self, coeffs):
        """Log likelihood of Hawkes Process for the given parameters.
        The parameters `coeffs` of the model are parameterized as a
        one-dimensional tensor with $[\mu, \beta, \alpha]$. To get each set of
        parameters back, do:

        ```
        mu = coeffs[:dim]
        beta = coeffs[dim:2*dim]
        W = coeffs[2*dim:].reshape(dim, dim)
        ```

        Arguments:
        ----------
        coeffs : torch.Tensor
            Parameters of the model (shape: (dim^2 + 2 dim) x 1)
        """
        # Extract each set of parameters
        mu = coeffs[:self.dim]
        beta = coeffs[self.dim:self.dim + self.dim**2].reshape(self.dim, self.dim)
        alpha = coeffs[self.dim + self.dim**2:].reshape(self.dim, self.dim)
        # Compute the log-likelihood
        log_like = 0
        for i in range(self.dim):
            # Compute the intensity at each event
            lam_ik_arr = mu[i] + torch.sum(
                self.valid_mask_ikj[i] * alpha[:, i] / (
                    beta[:, i] + 1 + self.delta_ikj[i]), axis=1)
            # Add the log-intensity term (ignore the last 'fake' event)
            log_like += lam_ik_arr[:-1].log().sum()
            # Subtract the integral term
            log_like -= lam_ik_arr[0] * self.events[i][0]
            log_like -= torch.sum(lam_ik_arr[1:]
                                  * (self.events[i][1:] - self.events[i][:-1]))
        return log_like


class WoldModelMLE(WoldModelOther, FitterSGD):
    """Wold Model with Maximum Likelihoof Estimation fitter"""

    @enforce_observed
    def mle_objective(self, coeffs):
        """Objectvie function for MLE: Averaged negative log-likelihood"""
        return -1.0 * self.log_likelihood(coeffs) / sum(self.n_jumps)

    def fit(self, *args, **kwargs):
        return super().fit(objective_func=self.mle_objective, *args, **kwargs)
