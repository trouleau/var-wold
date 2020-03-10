# -*- coding: utf8
"""
Code modified from library GrangerBusca:
https://github.com/flaviovdf/granger-busca/blob/master/gb/simulate.py
"""
from bisect import bisect
import random as rd
import numpy as np


class GrangerBuscaSimulator(object):

    def __init__(self, mu_rates, Alpha_ba, Beta_b):
        self.mu_rates = np.asanyarray(mu_rates)
        assert len(self.mu_rates.shape) == 1
        self.dim = len(self.mu_rates)
        
        self.Alpha_ba = np.asanyarray(Alpha_ba)
        assert self.Alpha_ba.shape == (self.dim, self.dim)
        
        self.Beta_b = np.asanyarray(Beta_b)[:,np.newaxis] 
        assert self.Beta_b.shape == (self.dim,1)
       
        self.events = [[] for i in range(self.dim)]
        self.last = np.nan * np.zeros(self.dim)
        self.delta = np.nan * np.zeros((self.dim, self.dim))

        self.t = 0.0

    # def total_intensity___(self, t):
    #     lambdas_t = np.zeros(self.mu_rates.shape[0], dtype='d')
    #     for proc_a in range(self.Alpha_ba.shape[0]):
    #         lambdas_t[proc_a] = self.mu_rates[proc_a]
    #         if len(self.events[proc_a]) == 0:
    #             continue

    #         tp = self.events[proc_a][-1]
    #         assert tp <= t
    #         for proc_b in range(self.Alpha_ba.shape[0]):
    #             # If no past event, then no excitation
    #             if len(self.events[proc_b]) == 0:
    #                 continue

    #             # Find delta time different
    #             tpp_idx = bisect(self.events[proc_b], tp)
    #             if tpp_idx == len(self.events[proc_b]):
    #                 tpp_idx -= 1
    #             tpp = self.events[proc_b][tpp_idx]
    #             while tpp >= tp and tpp_idx > 0:
    #                 tpp_idx -= 1
    #                 tpp = self.events[proc_b][tpp_idx]
    #             # If no past event prior to tp, then no excitation
    #             if tpp >= tp:
    #                 continue
                
    #             # If `tpp` successfully found, add to rate
    #             busca_rate = self.Alpha_ba[proc_b, proc_a]
    #             busca_rate /= (self.Beta_b[proc_b] + tp - tpp)
                
    #             lambdas_t[proc_a] += busca_rate
    #     return lambdas_t

    def _total_intensity(self, t):
        # lambdas_t = np.zeros(self.dim)
        # for i in range(self.dim):
        #     lambdas_t[i] = self.mu_rates[i] + np.nansum(self.Alpha_ba[:,i] / (self.Beta_b + self.delta[i]))
        lambdas_t = self.mu_rates + np.nansum(self.Alpha_ba / (self.Beta_b+ self.delta), axis=0)
        return lambdas_t

    def simulate(self, forward, seed=None):
        if seed:
            rd.seed(seed)

        t = self.t
        max_time = t + forward
        while t < max_time:

            # Compute intensity at each node
            lambdas_t = self._total_intensity(t)

            # Compute total intensity
            sum_lambdas_t = lambdas_t.cumsum()

            # Sample next event time
            dt = rd.expovariate(sum_lambdas_t[-1])

            # # Increase current time
            t = t + dt
            if t > max_time:
                break

            # Sample next event dimension
            u = rd.random() * sum_lambdas_t[-1]
            i = bisect(sum_lambdas_t, u)

            # Add event to the history
            self.events[i].append(t)

            # Update cache for intensity computation
            self.delta[:,i] = t - self.last
            self.last[i] = t

            self.t = t
        
        return list(map(np.array, self.events))
