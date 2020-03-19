import numpy as np
import scipy.special as sc
import numba

from . import WoldModel
from ..utils.decorators import enforce_observed


# @numba.jit(nopython=True)
def _update_alpha(as_pr, ar_pr, zp_po, bs_po, br_po, dt_ikj, delta_ikj, valid_mask_ikj):
    dim = as_pr.shape[1]
    as_po = np.zeros_like(as_pr)  # Alpha posterior shape, to return
    ar_po = np.zeros_like(as_pr)  # Alpha posterior rate, to return
    for i in range(dim):
        # update shape
        as_po[:, i] = as_pr[:, i] + zp_po[i].sum(axis=0)
        # update rate
        x = br_po[:, i] / (delta_ikj[i][:, 1:] + 1e-20)
        D_i_kj = (valid_mask_ikj[i][:, 1:] * dt_ikj[i][:, np.newaxis] *
                  ((bs_po[:, i] / br_po[:, i])
                  * sc.gammainc(bs_po[:, i] + 1, x) / (sc.gammainc(bs_po[:, i], x) + 1e-20)))
        ar_po[0, i] = ar_pr[0, i] + dt_ikj[i].sum(axis=0)
        ar_po[1:, i] = ar_pr[1:, i] + D_i_kj.sum(axis=0)
    return as_po, ar_po


# @numba.jit(nopython=True)
def _update_beta(bs_pr, br_pr, zp_po, as_po, ar_po, last_t, delta_ikj, valid_mask_ikj, dt_ikj):
    bs_po = np.zeros_like(bs_pr)  # Alpha posterior shape, to return
    br_po = np.zeros_like(br_pr)  # Alpha posterior rate, to return
    dim = as_po.shape[1]
    for i in range(dim):
        bs_po[:, i] = bs_pr[:, i] + np.sum(zp_po[i][:, 1:], axis=0)

        br_po[:, i] = (br_pr[:, i] + (as_po[1:, i] / ar_po[1:, i]
                                      * np.sum(dt_ikj[i][:, np.newaxis] * valid_mask_ikj[i][:, 1:], axis=0)))
    return bs_po, br_po


def _update_z(as_po, ar_po, bs_po, br_po, delta_ikj, valid_mask_ikj, dt_ikj):
    dim = as_po.shape[1]
    zp = list()
    for i in range(dim):
        # Expected value
        epi = np.zeros_like(delta_ikj[i])
        epi += (sc.digamma(as_po[np.newaxis, :, i])
                - np.log(ar_po[np.newaxis, :, i]))

        a = bs_po[:, i]
        x = br_po[:, i] / (delta_ikj[i][:, 1:] + 1e-20)
        epi[:, 1:] -= (np.log(br_po[:, i]) - sc.digamma(bs_po[:, i])
                       - (valid_mask_ikj[i][:, 1:]
                          * (sc.gammainc(a + 1e-5, x) / (sc.gammainc(a, x) + 1e-20) - 1) * 1e5 ))
        # Softmax
        epi = epi - epi.max(axis=1)[:, np.newaxis]
        epi = np.exp(epi)
        epi /= epi.sum(axis=1)[:, np.newaxis]
        zp.append(epi)
    return zp


class WoldModelVariational(WoldModel):

    def observe(self, events, end_time=None):
        super().observe(events, end_time)
        self.dt_ikj = list()
        for i in range(self.dim):
            # TODO: fix this once virtual events in fixed in parent class
            self.events[i] = self.events[i][:-1].numpy()
            self.n_jumps[i] -= 1
            # TODO: fix this once virtual events in fixed in parent class
            self.valid_mask_ikj[i] = np.hstack((
                np.ones((self.n_jumps[i], 1)),  # set to 1 for j=0
                self.valid_mask_ikj[i][:-1, :].numpy()))
            # TODO: fix this once virtual events in fixed in parent class
            self.delta_ikj[i] = np.hstack((
                np.zeros((self.n_jumps[i], 1)),  # set \delta_0^ik = 0
                self.delta_ikj[i][:-1, :].numpy()))
            # Cache Inter-arrival time
            dt_i = np.hstack((self.events[i][0], np.diff(self.events[i])))
            self.dt_ikj.append(dt_i)
        # Cache last arrival time
        self.last_t = [self.events[i][-1] for i in range(self.dim)]
        # Sanity check
        assert np.allclose(self.n_jumps, np.array(list(map(len, self.events))))

    @enforce_observed
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
                                                     dt_ikj=self.dt_ikj,
                                                     delta_ikj=self.delta_ikj,
                                                     valid_mask_ikj=self.valid_mask_ikj)

            self._bs_po, self._br_po = _update_beta(bs_pr=self._bs_pr,
                                                    br_pr=self._br_pr,
                                                    zp_po=self._zp_po,
                                                    as_po=self._as_po,
                                                    ar_po=self._ar_po,
                                                    last_t=self.last_t,
                                                    delta_ikj=self.delta_ikj,
                                                    valid_mask_ikj=self.valid_mask_ikj,
                                                    dt_ikj=self.dt_ikj)

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
                                    delta_ikj=self.delta_ikj,
                                    valid_mask_ikj=self.valid_mask_ikj,
                                    dt_ikj=self.dt_ikj)

            print('Z posterior probabilities')
            print(self._zp_po[0])
