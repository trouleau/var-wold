# -*- coding: utf8
import random as rd
import numpy as np

import numba


@numba.njit
def _total_intensity(mu, adj, beta, delta, t):
    return mu + np.sum(adj / (beta + delta), axis=0)


@numba.njit
def _simulate(mu, adj, beta, last, delta, start_t, max_t, seed=None):
    dim = len(mu)
    # FIXME: Add fake 0.0 events to avoid numba complaining of unknown type
    events = [[0.0] for i in range(dim)]
    if seed:
        rd.seed(seed)
    t = float(start_t)
    max_time = start_t + max_t
    while t < max_time:
        # Compute intensity at each node
        lambdas_t = _total_intensity(mu, adj, beta, delta, t)
        # Compute total intensity
        sum_lambdas_t = lambdas_t.cumsum()
        # Sample next event time
        dt = rd.expovariate(sum_lambdas_t[-1])
        # Increase current time
        t = float(t + dt)
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
    return events, last, delta


class GrangerBuscaSimulator(object):

    def __init__(self, mu_rates, Alpha_ba, Beta_b):
        self.mu_rates = np.asanyarray(mu_rates)
        assert len(self.mu_rates.shape) == 1
        self.dim = len(self.mu_rates)
        self.Alpha_ba = np.asanyarray(Alpha_ba)
        assert self.Alpha_ba.shape == (self.dim, self.dim)
        self.Beta_b = np.asanyarray(Beta_b)[:, np.newaxis]
        assert self.Beta_b.shape == (self.dim, 1)

        self.last = -np.inf * np.ones(self.dim)
        self.delta = np.inf * np.ones((self.dim, self.dim))
        self.events = [[] for i in range(self.dim)]
        self.t = 0.0

    def simulate(self, max_time, seed=None):
        if seed is None:
            seed = rd.randint(0, 2 ** 32 - 1)
        new_events, last, delta = _simulate(
            mu=self.mu_rates, adj=self.Alpha_ba, beta=self.Beta_b,
            last=self.last, delta=self.delta, start_t=self.t,
            max_t=max_time, seed=seed)
        self.last = last.copy()
        self.delta = delta.copy()
        for i in range(self.dim):
            self.events[i].extend(new_events[i])
        return list(map(np.array, self.events))
