import abc
import torch
import numba
import warnings
import numpy as np


warnings.filterwarnings(action='ignore',
    category=numba.NumbaPendingDeprecationWarning, module='numba')


class Model(metaclass=abc.ABCMeta):
    """Base class for models with a log-likelihood function"""

    @abc.abstractmethod
    def set_data(self, events, end_time=None):
        """Set the data for the model as well as various attributes, and cache 
        some computations for future log-likelihood calls"""
    
    @abc.abstractmethod
    def log_likelihood(self, params):
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
                    # Set delta_ijk
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
    once the inter-arrival times in attribute `delta_ijk`, where 
    `delta_ijk[i][j,k]` holds $t_{k-1}^i - t_l^j$, where $l$ is the index of 
    the latest event in dimension $j$ such that $t_l^j < t_{k-1}^i$, which is 
    used to computed the intensity function for dimension $i$ at time $t_k^i$.
    """

    def __init__(self, verbose=False, device='cpu'):
        """Initialize the model
        """
        self.n_jumps = None
        self.dim = None
        self.n_params = None
        self._fitted = False
        self.verbose = verbose
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'

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
        self.events = []
        for i in range(self.dim):
            self.events.append(torch.cat((events[i], torch.tensor([self.end_time]))))
        # Number of events per dimension
        self.n_jumps_per_dim = list(map(len, self.events))
        # Check that all dimensions have at least one event, otherwise the computation of the 
        # log-likelihood is not correct
        assert min(self.n_jumps_per_dim) > 0, "Each dimension should have at least one event."
        # Total number of events
        self.n_jumps = sum(self.n_jumps_per_dim)
        # Number of parameters of the model
        self.n_params = self.dim * (self.dim + 2)
        # Init cache if necessary
        self._init_cache()

    def _init_cache(self):
        events_ = [ev.numpy() for ev in self.events]
        self.delta_ikj, self.valid_mask_ikj = _wold_model_init_cache(events_)
        self.delta_ikj = [torch.tensor(self.delta_ikj[i], dtype=torch.float) for i in range(self.dim)]
        self.valid_mask_ikj = [torch.tensor(self.valid_mask_ikj[i], dtype=torch.float) for i in range(self.dim)]
        

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
        beta = coeffs[self.dim:2*self.dim]
        alpha = coeffs[2*self.dim:].reshape(self.dim, self.dim)
        # Compute the log-likelihood
        log_like = 0
        for i in range(self.dim):
            # Compute the intensity at each event
            lam_ik_arr = mu[i] + torch.sum(self.valid_mask_ikj[i] * alpha[:,i] /(beta.unsqueeze(0) + self.delta_ikj[i]), axis=1)
            # Add the log-intensity term
            log_like += lam_ik_arr[:-1].log().sum()
            # Subtract the integral term
            log_like -= lam_ik_arr[0] * self.events[i][0]
            log_like -= torch.sum(lam_ik_arr[1:] * (self.events[i][1:] - self.events[i][:-1]))
        return log_like