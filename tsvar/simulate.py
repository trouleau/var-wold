# -*- coding: utf8

from bisect import bisect

import numpy as np


class GrangeBuscaSimulator(object):

    def __init__(self, mu_rates, Alpha_ba, Beta_b):
        self.mu_rates = np.asanyarray(mu_rates)
        self.Alpha_ba = np.asanyarray(Alpha_ba)
        self.Beta_b = np.asanyarray(Beta_b)
        self.past = [[] for i in range(self.Alpha_ba.shape[0])]
        self.upper_bound = 0.0
        for proc_a in range(self.Alpha_ba.shape[0]):
            self.upper_bound += self.mu_rates[proc_a]
            for proc_b in range(self.Alpha_ba.shape[0]):
                self.upper_bound += self.Alpha_ba[proc_b, proc_a] / \
                    self.Beta_b[proc_b]
        self.t = 0

    def total_intensity(self, t):
        lambdas_t = np.zeros(self.mu_rates.shape[0], dtype='d')
        for proc_a in range(self.Alpha_ba.shape[0]):
            lambdas_t[proc_a] = self.mu_rates[proc_a]
            if len(self.past[proc_a]) == 0:
                continue

            tp = self.past[proc_a][-1]
            assert tp <= t
            for proc_b in range(self.Alpha_ba.shape[0]):
                # If no past event, then no excitation
                if len(self.past[proc_b]) == 0:
                    continue

                # Find delta time different
                tpp_idx = bisect(self.past[proc_b], tp)
                if tpp_idx == len(self.past[proc_b]):
                    tpp_idx -= 1
                tpp = self.past[proc_b][tpp_idx]
                while tpp >= tp and tpp_idx > 0:
                    tpp_idx -= 1
                    tpp = self.past[proc_b][tpp_idx]
                # If no past event prior to tp, then no excitation
                if tpp >= tp:
                    continue
                
                # If `tpp` successfully found, add to rate
                busca_rate = self.Alpha_ba[proc_b, proc_a]
                busca_rate /= (self.Beta_b[proc_b] + tp - tpp)
                
                lambdas_t[proc_a] += busca_rate
        return lambdas_t

    def simulate(self, forward):
        t = self.t
        max_time = t + forward
        while t < max_time:
            # Compute intensity at each node
            lambdas_t = self.total_intensity(t)
            # Compute total intensity
            sum_lambdas_t = lambdas_t.cumsum()
            # Sample next event time
            dt = np.random.exponential(1.0 / sum_lambdas_t[-1])

            # Increase current time
            t = t + dt
            if t > max_time:
                break

            # Sample next event dimension
            u = np.random.rand() * sum_lambdas_t[-1]
            i = 0
            while i < self.Alpha_ba.shape[0]:
                if sum_lambdas_t[i] >= u:
                    break
                i += 1

            # Add event to the history
            self.past[i].append(t)
        self.t = t
        return self.past
