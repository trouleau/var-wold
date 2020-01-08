import abc
import torch
import numpy as np


class Model(metaclass=abc.ABCMeta):
    """Base class for models with a log-likelihood function"""

    @abc.abstractmethod
    def set_data(self, events, end_time=None):
        """Set the data for the model as well as various attributes, and cache 
        some computations for future log-likelihood calls"""
    
    @abc.abstractmethod
    def log_likelihood(self, params):
        """Evaluate the log likelihood of the model for the given parameters"""



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
        
        self.delta_ijk = {
            i: torch.zeros((self.dim, self.n_jumps_per_dim[i])) 
            for i in range(self.dim)}
        self.valid_mask_ijk = {
            i: torch.ones((self.dim, self.n_jumps_per_dim[i]), dtype=torch.bool) 
            for i in range(self.dim)}
        
        # For each reiceiving dimension
        for i in range(self.dim):
            last_idx_tlj = {j: None for j in range(self.dim)}
            last_tki = None
            
            # For each observed event, compute the inter-arrival time with
            # each dimension
            for k in range(self.n_jumps_per_dim[i]):
                if k == 0:
                    # Delta should be ignored for the first event.
                    # Mark has invalid
                    self.valid_mask_ijk[i][:,k] = 0
                    continue
                last_tki = self.events[i][k-1]
                # For each incoming dimension
                for j in range(self.dim):
                    if (last_idx_tlj[j] is None) and (self.events[j][0] >= last_tki):
                        # If the 1st event in dim `j` comes after `last_tki`, it should be ignored.
                        # Mark as invalid
                        self.valid_mask_ijk[i][j,k] = 0
                        continue
                    # Update last index for dim `j`
                    l = last_idx_tlj[j] or 0
                    while (self.events[j][l] < last_tki):
                        l += 1
                        if l == self.n_jumps_per_dim[j]:
                            break
                    l -= 1
                    last_idx_tlj[j] = l
                    # Set delta_ijk
                    self.delta_ijk[i][j,k] = last_tki - self.events[j][l]

    def log_likelihood(self, mu, alpha, beta):
        """Log likelihood of Hawkes Process for the given parameters mu and W

        Arguments:
        ----------
        mu : torch.Tensor
            Exogenous intensities (shape: dim x 1)
        alpha : torch.Tensor
            Endogenous weights for each pair of nodes (shape: dim x dim)
            This is the scaling parameters in the numerator of the excitation function
        beta : torch.Tensor
            Excitation weights for each node (shape: dim x 1)
            This is the denominator of the excitation function for each node
        """
        log_like = 0
        for i in range(self.dim):
            lam_ik_arr = torch.zeros(self.n_jumps_per_dim[i])
            for k in range(self.n_jumps_per_dim[i]):
                lam_ik_arr[k] = mu[i] + torch.sum(self.valid_mask_ijk[i][:,k] * alpha[:,i] / (beta + self.delta_ijk[i][:,k]))
            log_like += lam_ik_arr[:-1].log().sum()
            log_like -= lam_ik_arr[0] * self.events[i][0]
            log_like -= torch.sum(lam_ik_arr[1:] * (self.events[i][1:] - self.events[i][:-1]))
        return log_like