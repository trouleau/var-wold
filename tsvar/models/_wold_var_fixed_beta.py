import numba
import numpy as np
from scipy.special import digamma

from . import WoldModel
from ..utils.decorators import enforce_observed
from ..fitter import FitterIterativeNumpy


@numba.jit(nopython=True)
def _update_alpha(as_pr, ar_pr, zp_po, D_ikj):
    dim = as_pr.shape[1]
    as_po = np.zeros_like(as_pr)  # Alpha posterior shape, to return
    ar_po = np.zeros_like(as_pr)  # Alpha posterior rate, to return
    for i in range(dim):
        as_po[:, i] = (as_pr[:, i] + zp_po[i].sum(axis=0))
        ar_po[:, i] = (ar_pr[:, i] + D_ikj[i].sum(axis=0))
    return as_po, ar_po


def _update_z(as_po, ar_po, D_ikj):
    dim = len(D_ikj)
    zp = list()
    for i in range(dim):
        epi = (digamma(as_po[np.newaxis, :, i])
               - np.log(ar_po[np.newaxis, :, i])
               + np.log(D_ikj[i] + 1e-10))
        # epi -= np.max(epi, axis=1)[:, np.newaxis]
        epi = np.exp(epi)
        epi /= epi.sum(axis=1)[:, np.newaxis]
        zp.append(epi)
    return zp


class WoldModelVariationalFixedBeta(WoldModel, FitterIterativeNumpy):

    def observe(self, events, beta, end_time=None):
        super().observe(events, end_time)
        # TODO: fix this once virtual events in fixed in parent class
        self.beta = beta
        self.D_ikj = [np.zeros_like(arr) for arr in self.delta_ikj]
        for i in range(self.dim):
            self.events[i] = self.events[i][:-1].numpy()
            valid_mask_ikj_i = np.hstack((
                np.ones((self.n_jumps[i]-1, 1)),
                self.valid_mask_ikj[i][:-1, :].numpy()))
            delta_ikj_i = np.hstack((
                np.zeros((self.n_jumps[i]-1, 1)),
                self.delta_ikj[i][:-1, :].numpy()))
            dts = np.hstack((self.events[i][0], np.diff(self.events[i])))
            self.D_ikj[i] = (valid_mask_ikj_i
                             * dts[:, np.newaxis]
                             / (beta[np.newaxis, :, i] + delta_ikj_i + 1e-20))
            self.D_ikj[i][~valid_mask_ikj_i.astype(bool)] = 1e-20
        # Remove `delta_ikj` and `valid_mask_ikj`, we only need `D_ikj` to fit
        del self.delta_ikj
        del self.valid_mask_ikj
        # Number of events per dimension
        self.n_jumps = np.array(list(map(len, self.events)))

    @enforce_observed
    def _init_fit(self, as_pr, ar_pr, zc_pr):
        """Set attributes of priors and init posteriors"""
        self._as_pr = as_pr  # Alpha prior, shape of Gamma distribution
        self._ar_pr = ar_pr  # Alpha prior, rate of Gamma distribution
        self._zc_pr = zc_pr  # Z prior, concentration
        self._as_po = self._as_pr.copy()  # Alpha posterior, shape of Gamma distribution
        self._ar_po = self._ar_pr.copy()  # Alpha posterior, rate of Gamma distribution
        self._zp_po = list()
        for i in range(self.dim):
            # Z posterior, probabilities of Categorical distribution
            self._zp_po.append((self._zc_pr[i]
                                / self._zc_pr[i].sum(axis=1)[:, np.newaxis]))

    def _iteration(self):
        # Update Alpha
        self._as_po, self._ar_po = _update_alpha(as_pr=self._as_pr,
                                                 ar_pr=self._ar_pr,
                                                 zp_po=self._zp_po,
                                                 D_ikj=self.D_ikj)
        # Raise error if invalid parameters
        if (self._as_po.min() < 0) or (self._ar_po.min() < 0):
            raise RuntimeError("Negative posterior parameter!")
        # Update Z
        self._zp_po = _update_z(as_po=self._as_po, ar_po=self._ar_po,
                                D_ikj=self.D_ikj)
        # Set coeffs attribute for Fitter to assess convergence
        self.coeffs = (self._as_po / self._ar_po).flatten()

    @enforce_observed
    def fit(self, as_pr, ar_pr, zc_pr, *args, **kwargs):
        self._init_fit(as_pr, ar_pr, zc_pr)
        super().fit(step_function=self._iteration, *args, **kwargs)

    def alpha_posterior_mean(self):
        return self._as_po / self._ar_po

    def alpha_posterior_mode(self):
        return (self._as_po >= 1) * (self._as_po - 1) / self._ar_po
