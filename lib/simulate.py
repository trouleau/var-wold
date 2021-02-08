"""
The simulation code in this module is adapted from
    https://github.com/flaviovdf/granger-busca/blob/master/gb/simulate.py
"""
import random as rd
import numpy as np

import numba


#@numba.njit
def _total_intensity(mu, adj, beta, delta, t):
    return mu + np.sum(adj / (beta + 1 + delta), axis=0)


#@numba.njit
def _simulate(mu, adj, beta, last, delta, start_t, start_n, max_t, max_n, seed=None):
    dim = len(mu)
    # FIXME: Add fake 0.0 events to avoid numba complaining of unknown type
    events = [[0.0] for i in range(dim)]
    if seed:
        rd.seed(seed)
    # Init time
    t = float(start_t)
    max_time = t + max_t
    # Init number of jumps
    n_jumps = int(start_n)
    max_jumps = n_jumps + max_n
    while (t < max_time) and (n_jumps < max_jumps):
        # Compute intensity at each node
        lambdas_t = _total_intensity(mu, adj, beta, delta, t)
        # Compute total intensity
        sum_lambdas_t = lambdas_t.cumsum()
        # Sample next event time
        dt = rd.expovariate(sum_lambdas_t[-1])
        # Increase current time
        t = float(t + dt)
        n_jumps += 1
        if t > max_time:
            break
        # Sample next event dimension
        u = rd.random() * sum_lambdas_t[-1]
        i = np.searchsorted(sum_lambdas_t, u)
        # Add event to the history
        events[i].append(t)
        # Update cache for intensity computation
        delta[:, i] = t - last
        last[i] = t
    # FIXME: Reove fake 0.0 events
    events = [ev[1:] for ev in events]
    return events, last, delta, t, n_jumps


class MultivariateWoldSimulator(object):

    def __init__(self, baseline, adjacency, beta):
        self.baseline = np.asanyarray(baseline)
        assert len(self.baseline.shape) == 1
        self.dim = len(self.baseline)
        self.adjacency = np.asanyarray(adjacency)
        assert self.adjacency.shape == (self.dim, self.dim)
        self.beta = np.asanyarray(beta)
        assert self.beta.shape == (self.dim, self.dim)

        self.last = 0.0 * np.ones(self.dim)  # Time of previous event in each dim
        self.delta = 0.0 * np.ones((self.dim, self.dim))  # Last inter-event dim
        self.events = [[] for i in range(self.dim)]  # List of events
        self.t = 0.0
        self.n_jumps = 0

    def simulate(self, *, max_time=np.inf, max_jumps=np.inf, seed=None):
        if seed is None:
            seed = rd.randint(0, 2 ** 32 - 1)
        if not ((max_time < np.inf) ^ (max_jumps < np.inf)):
            raise ValueError('Either `max_time` or `max_jumps` must be set, but not both.')
        new_events, last, delta, new_time, new_jumps = _simulate(
            mu=self.baseline, adj=self.adjacency, beta=self.beta,
            last=self.last, delta=self.delta, start_t=self.t,
            start_n=self.n_jumps, max_t=max_time, max_n=max_jumps, seed=seed)
        self.last = last.copy()
        self.delta = delta.copy()
        for i in range(self.dim):
            self.events[i].extend(new_events[i])
        self.t = new_time
        self.n_jumps = new_jumps
        return list(map(np.array, self.events))

    @property
    def end_time(self):
        return self.t

    def spectral_radius(self):
        eigs = np.linalg.eigvals(self.adjacency / (self.beta + 1))
        return eigs.max() - eigs.min()
