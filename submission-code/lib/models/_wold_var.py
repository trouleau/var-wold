import warnings
import numpy as np
# import scipy.special as sc
import math
import numba

from . import WoldModel
from ..utils.decorators import enforce_observed
from ..fitter import FitterIterativeNumpy


MOMENT_ORDER = 1.7  # Moment of equation to solve for beta update
EPS = 1e-8  # Finite-difference gradient epsilon

warnings.filterwarnings("ignore")  # To handle NumbaPendingDeprecationWarning

@numba.jit(nopython=True, fastmath=True)
def digamma(arr, eps=EPS):
    """Digamma function (arr is assumed to be 1 or 2 dimensional)"""
    lgamma_prime = np.zeros_like(arr)
    if arr.ndim == 1:
        for i in numba.prange(arr.shape[0]):
            lgamma_prime[i] = (math.lgamma(arr[i] + eps) - math.lgamma(arr[i])) / eps
    elif arr.ndim == 2:
        for j in numba.prange(arr.shape[0]):
            for i in numba.prange(arr.shape[1]):
                lgamma_prime[j, i] = (math.lgamma(arr[j, i] + eps) - math.lgamma(arr[j, i])) / eps
    return lgamma_prime


@numba.jit(nopython=True, fastmath=True)
def expect_alpha(as_po, ar_po):
    """Compute the expectation of alpha"""
    return as_po / ar_po


@numba.jit(nopython=True, fastmath=True)
def expect_log_alpha(as_po, ar_po):
    """Compute the expectation of log(alpha)"""
    return digamma(as_po) - np.log(ar_po)


@numba.jit(nopython=True, fastmath=True)
def expect_z(zp_po):
    """Compute the expectation of z"""
    return zp_po


@numba.jit(nopython=True, fastmath=True)
def expect_inv_beta_p_delta(bs_po, br_po, delta):
    """Compute the expectation of 1/(beta + delta)"""
    b_mean = br_po / (bs_po - 1)
    return 1 / (b_mean + delta)


@numba.jit(nopython=True, fastmath=True)
def expect_log_beta_p_delta(bs_po, br_po, delta):
    """Compute the expectation of log(beta + delta)"""
    b_mean = br_po / (bs_po - 1)
    return np.log(b_mean + delta)


@numba.jit(nopython=True, fastmath=True)
def _beta_funcs(x, j, i, n, bs_pr, br_pr, as_po, ar_po, zp_po, dts, delta, return_fprime2=True):

    a_mean = expect_alpha(as_po[j+1, i], ar_po[j+1, i])
    x_p_delta = x + delta[i][:, j+1] + 1e-20

    term1 = (bs_pr[j, i] + 1 - n) / x
    term2 = br_pr[j, i] / (x ** 2)

    term31 = zp_po[i][:, j+1] / x_p_delta
    term32 = a_mean * dts[i] / (x_p_delta ** 2)

    func = term1 - term2 + np.sum(term31 - term32)

    term1 *= -1
    term1 /= x
    term2 *= -2
    term2 /= x
    term31 *= -1
    term31 /= x_p_delta
    term32 *= -2
    term32 /= x_p_delta

    fprime = term1 - term2 + np.sum(term31 - term32)

    if not return_fprime2:
        return func, fprime, 0.0

    term1 *= -2
    term1 /= x
    term2 *= -3
    term2 /= x
    term31 *= -2
    term31 /= x_p_delta
    term32 *= -3
    term32 /= x_p_delta

    fprime2 = term1 - term2 + np.sum(term31 - term32)

    return func, fprime, fprime2


@numba.jit(nopython=True, fastmath=True)
def solve_binary_search(x_min, x_max, max_iter, tol, j, i, n, bs_pr, br_pr, as_po, ar_po, zp_po, dts, delta):
    f = tol + 1
    f_max, _, _ = _beta_funcs(x_max, j, i, n, bs_pr, br_pr, as_po, ar_po, zp_po,
                              dts, delta, return_fprime2=False)
    for it in range(max_iter):
        mid = (x_min + x_max) / 2
        f, _, _ = _beta_funcs(mid, j, i, n, bs_pr, br_pr, as_po, ar_po, zp_po,
                              dts, delta, return_fprime2=False)
        if f * f_max > 0:
            x_max = mid
        else:
            x_min = mid
        if abs(f) < tol:
            break
    return mid


@numba.jit(nopython=True, fastmath=True)
def solve_newton(xstart, max_iter, tol, j, i, n, bs_pr, br_pr, as_po, ar_po, zp_po, dts, delta):
    x = float(xstart)
    for it in range(max_iter):
        f, fp, _ = _beta_funcs(x, j, i, n, bs_pr, br_pr, as_po, ar_po, zp_po,
                               dts, delta, return_fprime2=False)
        x_new = x - f / fp
        if abs(f) < tol:
            # print('it', it+1)
            return x
        x = x_new
    # print('it', it+1)
    return x


@numba.jit(nopython=True, fastmath=True)
def solve_halley(xstart, max_iter, tol, j, i, n, bs_pr, br_pr, as_po, ar_po, zp_po, dts, delta):
    """
    Solve the equation to to find the parameters of the posterior of beta.

    Parameters:
    -----------
    xstart : starting point
    max_iter : maximum number of iterations
    tol : tolerance for convergence
    j, i : indices of beta to solve for, as in beta_{j, i}
    n : the order to solve for
    bs_pr : beta shape prior
    br_pr : beta rate prior
    zp_po : z probability posterior
    as_po : alpha shape posterior
    ar_po : alpha rate posterior
    dts[k] : within-dimension inter-arrival time (i.e. poisson interval length,
        for numerator) t^i_k - t^i_{k-1}
    delta : inter-arrival time (Wold influence, for denominator)

    Returns:
    --------
    x : float
        Solution of the equation
    conv : bool
        Indicator of convergence
    """
    x = float(xstart)
    for it in range(max_iter):
        f, fp, fpp = _beta_funcs(x, j, i, n, bs_pr, br_pr, as_po, ar_po, zp_po,
                                 dts, delta)
        x_new = x - (2 * f * fp) / (2 * fp**2 - f * fpp)
        if abs(x - x_new) < tol:
            return x
        x = x_new
    return x


@numba.jit(nopython=True, fastmath=True)
def _update_alpha(as_pr, ar_pr, zp_po, bs_po, br_po, delta_ikj, dt_ik, last_t):
    dim = as_pr.shape[1]
    as_po = np.zeros_like(as_pr)  # Alpha posterior shape, to return
    ar_po = np.zeros_like(as_pr)  # Alpha posterior rate, to return
    for i in numba.prange(dim):
        # update shape
        as_po[:, i] = as_pr[:, i] + zp_po[i].sum(axis=0)
        # update rate
        ar_po[0, i] = ar_pr[0, i] + last_t[i]
        D_i_kj = (np.expand_dims(dt_ik[i], 1) *
                  expect_inv_beta_p_delta(bs_po[:, i], br_po[:, i],
                                          delta_ikj[i][:, 1:] + 1e-20))
        ar_po[1:, i] = ar_pr[1:, i] + D_i_kj.sum(axis=0)
    return as_po, ar_po


@numba.jit(nopython=True, fastmath=True)
def _update_beta(*, x0, xn, as_po, ar_po, zp_po, bs_pr, br_pr, dt_ik, delta_ikj):
    dim = as_po.shape[1]
    max_iter = 10
    tol = 1e-5
    for j in numba.prange(dim):
        for i in numba.prange(dim):
            x0[j, i] = solve_newton(
                xstart=x0[j, i],  # xstart=0.1,
                max_iter=max_iter,
                tol=tol, j=j, i=i, n=0,
                bs_pr=bs_pr, br_pr=br_pr,
                as_po=as_po, ar_po=ar_po,
                zp_po=zp_po,
                dts=dt_ik,
                delta=delta_ikj)
            if x0[j, i] < 0:
                print('Beta optim failed for x0, swtich to bin search')
                x0[j, i] = solve_binary_search(
                    x_min=0.01, x_max=200.0,
                    max_iter=20, tol=1e-2,
                    j=j, i=i, n=0,
                    bs_pr=bs_pr, br_pr=br_pr,
                    as_po=as_po, ar_po=ar_po,
                    zp_po=zp_po,
                    dts=dt_ik,
                    delta=delta_ikj)
            xn[j, i] = solve_newton(
                xstart=xn[j, i],  # xstart=0.1,
                max_iter=max_iter,
                tol=tol, j=j, i=i, n=MOMENT_ORDER,
                bs_pr=bs_pr, br_pr=br_pr,
                as_po=as_po, ar_po=ar_po,
                zp_po=zp_po,
                dts=dt_ik,
                delta=delta_ikj)
            if xn[j, i] < 0:
                print('Beta optim failed for xn, switch to bin search')
                xn[j, i] = solve_binary_search(
                    x_min=0.01, x_max=200.0,
                    max_iter=20, tol=1e-2,
                    j=j, i=i, n=MOMENT_ORDER,
                    bs_pr=bs_pr, br_pr=br_pr,
                    as_po=as_po, ar_po=ar_po,
                    zp_po=zp_po,
                    dts=dt_ik,
                    delta=delta_ikj)
    bs_po = MOMENT_ORDER * xn / (xn - x0 + 1e-10) - 1
    br_po = MOMENT_ORDER * xn * x0 / (xn - x0 + 1e-10)
    return bs_po, br_po, x0, xn


@numba.jit(nopython=True, fastmath=True)
def _compute_epi(i, as_po, ar_po, bs_po, br_po, dt_ik, delta_ikj):
    # log(inter-arrival time), only for valid events
    epi = np.zeros_like(delta_ikj[i])
    epi += np.log(np.expand_dims(dt_ik[i], 1) + 1e-20)
    # Expected value log(alpha)
    epi += np.expand_dims(expect_log_alpha(as_po[:, i], ar_po[:, i]), 0)
    # Expected value log(beta + delta), only for j>=1, i.e. ignore baseline
    epi[:, 1:] -= expect_log_beta_p_delta(bs_po[:, i], br_po[:, i],
                                          delta_ikj[i][:, 1:] + 1e-20)
    return epi


@numba.jit(nopython=True, fastmath=True)
def _update_z(as_po, ar_po, bs_po, br_po, dt_ik, delta_ikj):
    dim = as_po.shape[1]
    zs = list()
    for i in range(dim):
        epi = _compute_epi(i, as_po, ar_po, bs_po, br_po, dt_ik, delta_ikj)
        # Softmax
        epi -= epi.max()
        epi = np.exp(epi)
        epi /= np.expand_dims(epi.sum(axis=1), 1)
        zs.append(epi)
    return zs


class WoldModelVariational(WoldModel, FitterIterativeNumpy):

    def observe(self, events, end_time=None):
        super().observe(events, end_time)
        self.dt_ik = list()
        for i in range(self.dim):
            # TODO: fix this once virtual events is removed in parent class
            self.events[i] = self.events[i][:-1].numpy()
            self.n_jumps[i] -= 1
            # TODO: fix this once virtual events is removed in parent class
            self.delta_ikj[i] = np.hstack((
                np.zeros((self.n_jumps[i], 1)),  # set \delta_0^ik = 0
                self.delta_ikj[i][:-1, :].numpy()))
            # Cache Inter-arrival time
            dt_i = np.hstack((self.events[i][0], np.diff(self.events[i])))
            self.dt_ik.append(dt_i)
        # Cache last arrival time
        self.last_t = [self.events[i][-1] for i in range(self.dim)]
        # Sanity check
        assert np.allclose(self.n_jumps, np.array(list(map(len, self.events))))

    def _init_fit(self, as_pr, ar_pr, bs_pr, br_pr, zc_pr):
        self._as_pr = as_pr.copy()  # Alpha prior, shape of Gamma distribution
        self._ar_pr = ar_pr.copy()  # Alpha prior, rate of Gamma distribution
        self._bs_pr = bs_pr.copy()  # Beta prior, shape of Gamma distribution
        self._br_pr = br_pr.copy()  # Beta prior, rate of Gamma distribution
        self._zc_pr = zc_pr.copy()  # Z prior, concentration
        # shape: (dim+1: j, dim: i)
        self._as_po = self._as_pr.copy()  # Alpha posterior, shape of Gamma distribution
        self._ar_po = self._ar_pr.copy()  # Alpha posterior, rate of Gamma distribution
        # shape: (dim: j, dim: i)
        self._bs_po = self._bs_pr.copy()  # Beta posterior, shape of InvGamma distribution
        self._br_po = self._br_pr.copy()  # Beta posterior, rate of InvGamma distribution
        self._b_x0 = self._br_po / (self._bs_po - 1)
        self._b_xn = 0.1 * self._br_po / (self._bs_po - 1)
        # shape: (dim: i, #events_i: k, dim: j)
        self._zp_po = list()  # Z posterior, probabilities of Categorical distribution
        for i in range(self.dim):
            self._zp_po.append(self._zc_pr[i]
                               / self._zc_pr[i].sum(axis=1)[:, None])

    def _iteration(self):

        # Update alpha
        self._as_po, self._ar_po = _update_alpha(as_pr=self._as_pr,
                                                 ar_pr=self._ar_pr,
                                                 zp_po=self._zp_po,
                                                 bs_po=self._bs_po,
                                                 br_po=self._br_po,
                                                 dt_ik=self.dt_ik,
                                                 last_t=self.last_t,
                                                 delta_ikj=self.delta_ikj)

        #print('---- Alpha')
        #print(f'    as: min:{self._as_po.min():+.2e}, max:{self._as_po.max():+.2e}')
        #print(f'    ar: min:{self._ar_po.min():+.2e}, max:{self._ar_po.max():+.2e}')
        #a_mean = self._as_po / self._ar_po
        #print(f'a_mean: min:{a_mean.min():+.2e}, max:{a_mean.max():+.2e}')
        # (debug) Sanity check
        if np.isnan(self._as_po).any() or np.isnan(self._ar_po).any():
            raise RuntimeError("NaNs in Alpha parameters")
        if (np.min(self._as_po) < 0) or (np.min(self._ar_po) < 0):
            raise RuntimeError("Negative Alpha parameters")

        # Update beta
        self._bs_po, self._br_po, self._b_x0, self._b_xn = _update_beta(
            x0=self._b_x0, xn=self._b_xn,
            as_po=self._as_po, ar_po=self._ar_po, zp_po=self._zp_po,
            bs_pr=self._bs_pr, br_pr=self._br_pr, dt_ik=self.dt_ik,
            delta_ikj=self.delta_ikj)

        #print('---- Beta')
        #print(f'    x0: min:{self._b_x0.min():+.2e}, max:{self._b_x0.max():+.2e}')
        #print(f'    xn: min:{self._b_xn.min():+.2e}, max:{self._b_xn.max():+.2e}')
        #print(f'    bs: min:{self._bs_po.min():+.2e}, max:{self._bs_po.max():+.2e}')
        #print(f'    br: min:{self._br_po.min():+.2e}, max:{self._br_po.max():+.2e}')
        #b_mean = self._br_po / (self._bs_po - 1) * (self._bs_po > 1)
        # Deal with numerical instability
        if (self._bs_po.min() <= 0) and (self._bs_po.min() + 1e-3 > 0):
            self._bs_po[self._bs_po < 0] = 1e-5
        if (self._br_po.min() <= 0) and (self._br_po.min() + 1e-3 > 0):
            self._br_po[self._br_po < 0] = 1e-5
        #print(f'b_mean: min:{b_mean.min():+.2e}, max:{b_mean.max():+.2e}')
        # (debug) Sanity check
        if np.isnan(self._bs_po).any() or np.isnan(self._br_po).any():
            raise RuntimeError("NaNs in Beta parameters")
        if (self._bs_po.min() <= 0) or (self._bs_po.min() <= 0):
            raise RuntimeError("Negative Beta parameter!")
        if np.any(np.isnan(self._b_x0)) or np.any(np.isnan(self._b_xn)):
            raise RuntimeError('NaNs in optimization results of beta update!')
        if np.any(np.abs(self._b_x0) > 1e10) or np.any(self._b_xn > 1e10):
            raise RuntimeError('Optimization results of beta update is diverging!')
        if (self._b_x0.min() < 0) or (self._b_xn.min() < 0):
            raise RuntimeError('Optimization results of beta update is negative!')

        # Update Z
        self._zp_po = _update_z(as_po=self._as_po,
                                ar_po=self._ar_po,
                                bs_po=self._bs_po,
                                br_po=self._br_po,
                                dt_ik=self.dt_ik,
                                delta_ikj=self.delta_ikj)

        # Set coeffs attribute for Fitter to assess convergence
        self.coeffs = self.alpha_posterior_mode()[1:, :].flatten()

    @enforce_observed
    def fit(self, as_pr, ar_pr, bs_pr, br_pr, zc_pr, *args, **kwargs):
        self._init_fit(as_pr, ar_pr, bs_pr, br_pr, zc_pr)
        return super().fit(step_function=self._iteration, *args, **kwargs)

    def alpha_posterior_mean(self, as_po=None, ar_po=None):
        if (as_po is None) and (ar_po is None):
            as_po = self._as_po
            ar_po = self._ar_po
        return as_po / ar_po

    def alpha_posterior_mode(self, as_po=None, ar_po=None):
        if (as_po is None) and (ar_po is None):
            as_po = self._as_po
            ar_po = self._ar_po
        return (as_po > 1) * (as_po - 1) / ar_po
