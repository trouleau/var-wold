import abc
import torch
import numba
import warnings
import numpy as np
from scipy.special import digamma

from .utils.decorators import enforce_fitted


warnings.filterwarnings(action='ignore', module='numba',
                        category=numba.NumbaPendingDeprecationWarning)


class Model(metaclass=abc.ABCMeta):
    """Base class for models with a log-likelihood function"""

    def __init__(self, verbose=False, device='cpu'):
        """Initialize the model
        """
        self.n_jumps = None  # Total Number of jumps observed
        self.dim = None  # Number of dimensions
        self.n_params = None  # Number of parameters
        self._fitted = False  # Indicate if data is properly set
        self.verbose = verbose  # Indicate verbosity behavior
        # Device to use for torch ('cpu' or 'cuda')
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'

    @abc.abstractmethod
    def set_data(self, events, end_time=None):
        """Set the data for the model as well as various attributes, and cache
        some computations for future log-likelihood calls"""

    @abc.abstractmethod
    def log_likelihood(self, coeffs):
        """Evaluate the log likelihood of the model for the given parameters"""


@numba.jit(nopython=True, fastmath=True)
def _wold_model_init_cache(events):
    dim = len(events)
    n_jumps = [len(events[i]) for i in range(dim)]
    delta_ikj = [np.zeros((n_jumps[i], dim)) for i in range(dim)]
    valid_mask_ikj = [np.ones((n_jumps[i], dim), dtype=np.bool_) for i in range(dim)]
    # For each reiceiving dimension
    for i in range(dim):
        last_idx_tlj = [-1 for j in range(dim)]
        last_tki = events[i][0]
        # For each observed event, compute the inter-arrival time with
        # each dimension
        for k, tki in enumerate(events[i]):
            if k == 0:
                # Delta should be ignored for the first event.
                # Mark has invalid
                valid_mask_ikj[i][k,:] = 0
                continue
            last_tki = events[i][k-1]
            # For each incoming dimension
            for j in range(dim):
                if (last_idx_tlj[j] < 0) and (events[j][0] >= last_tki):
                    # If the 1st event in dim `j` comes after `last_tki`, it should be ignored.
                    # Mark as invalid
                    valid_mask_ikj[i][k,j] = 0
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
                delta_ikj[i][k,j] = last_tki - events[j][l]
        last_tki = tki
    return delta_ikj, valid_mask_ikj


class WoldModel(Model):
    """Class for the Multivariate Wold Point Process Model

    Note: When setting the data with `set_data`, an artificial event at the end
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

    def set_data(self, events, end_time=None):
        """Set the data for the model as well as various attributes, and cache
        some computations for future log-likelihood calls
        """
        # Events must be tensor to use torch's automatic differentiation
        assert isinstance(events[0], torch.Tensor), "`events` should be a list of `torch.Tensor`."
        # Number of dimensions
        self.dim = len(events)
        # End of the observation window
        self.end_time = end_time or max([max(num) for num in events if len(num) > 0])
        # Observed events, add a virtual event at `end_time` for easier log-likelihood computation
        # TODO: Remove the virtual event, it's nasty and will eventually introduce bugs
        self.events = []
        for i in range(self.dim):
            self.events.append(torch.cat((
                events[i], torch.tensor([self.end_time], dtype=torch.float))))
        # Number of events per dimension
        self.n_jumps = list(map(len, self.events))
        # Check that all dimensions have at least one event, otherwise the c
        # omputation of the log-likelihood is not correct
        assert min(self.n_jumps) > 0, "Each dimension should have at least one event."
        # Number of parameters of the model
        self.n_params = self.dim * self.dim + self.dim + self.dim
        # Init cache if necessary
        self._init_cache()

    def _init_cache(self):
        events_ = [ev.numpy() for ev in self.events]
        self.delta_ikj, self.valid_mask_ikj = _wold_model_init_cache(events_)
        self.delta_ikj = [torch.tensor(
            self.delta_ikj[i], dtype=torch.float) for i in range(self.dim)]
        self.valid_mask_ikj = [torch.tensor(
            self.valid_mask_ikj[i], dtype=torch.float) for i in range(self.dim)]
        self._fitted = True

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
        beta = coeffs[self.dim:2 * self.dim]
        alpha = coeffs[2 * self.dim:].reshape(self.dim, self.dim)
        # Compute the log-likelihood
        log_like = 0
        for i in range(self.dim):
            # Compute the intensity at each event
            lam_ik_arr = mu[i] + torch.sum(
                self.valid_mask_ikj[i] * alpha[:, i] / (
                    beta.unsqueeze(0) + 1 + self.delta_ikj[i]), axis=1)
            # Add the log-intensity term
            log_like += lam_ik_arr[:-1].log().sum()
            # Subtract the integral term
            log_like -= lam_ik_arr[0] * self.events[i][0]
            log_like -= torch.sum(
                lam_ik_arr[1:] * (self.events[i][1:] - self.events[i][:-1]))
        return log_like


# @numba.jit(nopython=True)
def _update_alpha(as_pr, ar_pr, zp_po, bs_po, br_po, last_t):
    dim = as_pr.shape[1]
    as_po = np.zeros_like(as_pr)  # Alpha posterior shape, to return
    ar_po = np.zeros_like(as_pr)  # Alpha posterior rate, to return
    for i in range(dim):
        as_po[:, i] = as_pr[:, i] + zp_po[i].sum(axis=0)
        ar_po[0, i] = ar_pr[0, i] + last_t[i]
        ar_po[1:, i] = ar_pr[1:, i] + (bs_po[:, i] / br_po[:, i]) * last_t[i]
    return as_po, ar_po


# @numba.jit(nopython=True)
def _update_beta(bs_pr, br_pr, zp_po, as_po, ar_po, delta_ikj, last_t):
    bs_po = np.zeros_like(bs_pr)  # Alpha posterior shape, to return
    br_po = np.zeros_like(br_pr)  # Alpha posterior rate, to return
    dim = as_po.shape[1]
    for i in range(dim):
        bs_po[:, i] = bs_pr[:, i] + np.sum(zp_po[i][:, 1:], axis=0)
        br_po[:, i] = (br_pr[:, i]
                       + np.sum(zp_po[i][:, 1:] * delta_ikj[i][:, 1:], axis=0)
                       + (as_po[1:, i] / ar_po[1:, i]) * last_t[i])
    return bs_po, br_po


def _update_z(as_po, ar_po, bs_po, br_po, delta_ikj):
    dim = as_po.shape[1]
    zp = list()
    for i in range(dim):
        # Expected value
        epi = np.zeros_like(delta_ikj[i])
        epi += (digamma(as_po[np.newaxis, :, i])
                - np.log(ar_po[np.newaxis, :, i]))
        epi[:, 1:] -= (np.log(br_po[:, i]) - digamma(bs_po[:, i])
                       + (delta_ikj[i][:, 1:] + 1) * bs_po[:, i] / br_po[:, i])
        # Softmax
        epi = epi - epi.max(axis=1)[:, np.newaxis]
        epi = np.exp(epi)
        epi /= epi.sum(axis=1)[:, np.newaxis]
        zp.append(epi)
    return zp


class VariationalWoldModel(WoldModel):

    def set_data(self, events, end_time=None):
        super().set_data(events, end_time)
        # TODO: fix this once virtual events in fixed in parent class
        for i in range(self.dim):
            self.events[i] = self.events[i][:-1].numpy()
            self.valid_mask_ikj[i] = np.hstack((
                np.ones((self.n_jumps[i]-1, 1)),
                self.valid_mask_ikj[i][:-1, :].numpy()))
            self.delta_ikj[i] = np.hstack((
                np.zeros((self.n_jumps[i]-1, 1)),
                self.delta_ikj[i][:-1, :].numpy()))
        self.last_t = [self.events[i][-1] for i in range(self.dim)]

        # Number of events per dimension
        self.n_jumps = np.array(list(map(len, self.events)))

    @enforce_fitted
    def fit(self, *, as_pr, ar_pr, bs_pr, br_pr, zc_pr, max_iter=100, tol=1e-5):
        self._as_pr = as_pr  # Alpha prior, shape of Gamma distribution
        self._ar_pr = ar_pr  # Alpha prior, rate of Gamma distribution
        self._bs_pr = bs_pr  # Beta prior, shape of Gamma distribution
        self._br_pr = br_pr  # Beta prior, rate of Gamma distribution
        self._zc_pr = zc_pr  # Z prior, concentration

        # shape: (dim+1: j, dim: i)
        self._as_po = self._as_pr.copy()  # Alpha posterior, shape of Gamma distribution
        self._ar_po = self._ar_pr.copy()  # Alpha posterior, rate of Gamma distribution

        # shape: (dim: j, dim: i)
        self._bs_po = self._bs_pr.copy()  # Beta posterior, shape of InvGamma distribution
        self._br_po = self._br_pr.copy()  # Beta posterior, rate of InvGamma distribution

        # shape: (dim: i, #events_i: k, dim: j)
        self._zp_po = list()  # Z posterior, probabilities of Categorical distribution
        for i in range(self.dim):
            self._zp_po.append(self._zc_pr[i]
                               / self._zc_pr[i].sum(axis=1)[:, np.newaxis])

        print('-'*50, 0)
        print('Alpha posterior mean:')
        print(np.round(self._as_po / self._ar_po, 2))
        print('Z[0] posterior probabilities')
        print(self._zp_po[0])

        for i in range(max_iter):
            print('-'*50, i+1)

            self._as_po, self._ar_po = _update_alpha(as_pr=self._as_pr,
                                                     ar_pr=self._ar_pr,
                                                     zp_po=self._zp_po,
                                                     bs_po=self._bs_po,
                                                     br_po=self._br_po,
                                                     last_t=self.last_t)

            self._bs_po, self._br_po = _update_beta(bs_pr=self._bs_pr,
                                                    br_pr=self._br_pr,
                                                    zp_po=self._zp_po,
                                                    as_po=self._as_po,
                                                    ar_po=self._ar_po,
                                                    delta_ikj=self.delta_ikj,
                                                    last_t=self.last_t)

            print('Alpha posterior mean:')
            print(np.round(self._as_po / self._ar_po, 2))

            if self._as_po.min() < 0:
                raise RuntimeError("Negative alpha shape!")
            if self._as_po.min() < 0:
                raise RuntimeError("Negative alpha rate!")

            self._zp_po = _update_z(as_po=self._as_po,
                                    ar_po=self._ar_po,
                                    bs_po=self._bs_po,
                                    br_po=self._br_po,
                                    delta_ikj=self.delta_ikj)

            print('Z posterior probabilities')
            print(self._zp_po[0])
