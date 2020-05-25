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


CACHE = False
PARALLEL = True

warnings.filterwarnings("ignore")  # To handle NumbaPendingDeprecationWarning


def exact_beta_density(beta_range, bs_pr, br_pr, as_po, ar_po, zp_po, dts, delta, valid_mask):
    """
    Evaluate the exact un-normalized posterior of beta at `beta_range`. The
    range is assumed to cover the whole density and is used to normalize it.
    """
    a_mean = as_po / ar_po
    term1 = beta_range[None, :] ** (-bs_pr - 1)
    term1[term1 > 1e10] = 1e10
    term2 = np.exp(- br_pr / beta_range[None, :])
    term2[term2 > 1e10] = 1e10
    term3_1 = (beta_range[None, :] * delta[:, None]) ** (- zp_po[:, None])
    term3_1[term3_1 > 1e10] = 1e10
    term3_2 = np.exp(
        - (a_mean * dts[:, None]) / (beta_range[None, :] * delta[:, None]))
    term3_2[term3_2 > 1e10] = 1e10
    term3 = term3_1 * term3_2
    term3[valid_mask == 0.0] = 1.0
    term3 = np.prod(term3, axis=0)
    post = term1 * term2 * term3
    post = np.squeeze(post)
    post /= post.sum()  # FIXME: does not normalize correctly, it should be multiplied by bin size
    return post


def approx_beta_density(beta_range, j, i, x0, xn, n, as_po, ar_po, zp_po, bs_pr, br_pr, dt_ik, delta_ikj, valid_mask_ikj):
    """
    Evaluate the approximate Inverse-Gamma posterior of beta at `beta_range`.
    """
    x0_out, _ = solve_halley(func, fprime, fprime2, x0, 100, 1e-5, j, i, 0,
                             bs_pr, br_pr, as_po, ar_po, zp_po, dt_ik, delta_ikj, valid_mask_ikj)
    xn_out, _ = solve_halley(func, fprime, fprime2, xn, 100, 1e-5, j, i, n,
                             bs_pr, br_pr, as_po, ar_po, zp_po, dt_ik, delta_ikj, valid_mask_ikj)
    bs_po = n * xn_out / (xn_out - x0_out) - 1
    br_po = n * xn_out * x0_out / (xn_out - x0_out)
    print(bs_po, br_po)
    import scipy.stats
    post = scipy.stats.invgamma(a=bs_po, scale=br_po).pdf(beta_range)
    return post


@numba.jit(nopython=True, fastmath=True, parallel=PARALLEL, cache=CACHE)
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


@numba.jit(nopython=True, fastmath=True, parallel=PARALLEL, cache=CACHE)
def expect_alpha(as_po, ar_po):
    """Compute the expectation of alpha"""
    return as_po / ar_po


@numba.jit(nopython=True, fastmath=True, parallel=PARALLEL, cache=CACHE)
def expect_log_alpha(as_po, ar_po):
    """Compute the expectation of log(alpha)"""
    return digamma(as_po) - np.log(ar_po)


@numba.jit(nopython=True, fastmath=True, parallel=PARALLEL, cache=CACHE)
def expect_z(zp_po):
    """Compute the expectation of z"""
    return zp_po


@numba.jit(nopython=True, fastmath=True, parallel=PARALLEL, cache=CACHE)
def expect_inv_beta_p_delta(bs_po, br_po, delta):
    """Compute the expectation of 1/(beta + delta)"""
    b_mean = br_po / (bs_po - 1)
    return 1 / (b_mean + delta)


@numba.jit(nopython=True, fastmath=True, parallel=PARALLEL, cache=CACHE)
def expect_log_beta_p_delta(bs_po, br_po, delta):
    """Compute the expectation of log(beta + delta)"""
    b_mean = br_po / (bs_po - 1)
    return np.log(b_mean + delta)


@numba.jit(nopython=True, fastmath=True, parallel=PARALLEL, cache=CACHE)
def _beta_funcs(x, j, i, n, bs_pr, br_pr, as_po, ar_po, zp_po, dts, delta, valid_mask):

    a_mean = expect_alpha(as_po[j+1, i], ar_po[j+1, i])
    x_p_delta = x + delta[i][:, j+1] + 1e-20

    term1 = (bs_pr[j, i] + 1 - n) / x
    term2 = br_pr[j, i] / (x ** 2)

    mask = valid_mask[i][:, j+1]
    term31 = zp_po[i][:, j+1] / x_p_delta
    term32 = a_mean * dts[i] / (x_p_delta ** 2)

    func = term1 - term2 + np.sum(mask * (term31 - term32))

    term1 *= -1
    term1 /= x
    term2 *= -2
    term2 /= x
    term31 *= -1
    term31 /= x_p_delta
    term32 *= -2
    term32 /= x_p_delta

    fprime = term1 - term2 + np.sum(mask * (term31 - term32))

    term1 *= -2
    term1 /= x
    term2 *= -3
    term2 /= x
    term31 *= -2
    term31 /= x_p_delta
    term32 *= -3
    term32 /= x_p_delta

    fprime2 = term1 - term2 + np.sum(mask * (term31 - term32))

    return func, fprime, fprime2


@numba.jit(nopython=True, fastmath=True, parallel=PARALLEL, cache=CACHE)
def solve_halley(xstart, max_iter, tol, j, i, n, bs_pr, br_pr, as_po, ar_po, zp_po, dts, delta, valid_mask):
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
    # print('-'*10)
    # print(f'j={j}, i={i}  (n={n})')
    for it in numba.prange(max_iter):
        f, fp, fpp = _beta_funcs(x, j, i, n, bs_pr, br_pr, as_po, ar_po, zp_po,
                                 dts, delta, valid_mask)

        x_new = x - (2 * f * fp) / (2 * fp**2 - f * fpp)

        # if (j == 8) and (i == 66):
        #     print(f'it: {it}, f={f:.2e}, fp={fp:.2e}, fpp={fpp:.2e}, x={x:.2e}, xnew={x_new:.2e}')

        # print(f'it: {it}, f={f:.2e}, fp={fp:.2e}, fpp={fpp:.2e}, x={x:.2e}, xnew={x_new:.2e}')

        if abs(x - x_new) < tol:
            return x
        x = x_new
    return x


@numba.jit(nopython=True, fastmath=True, cache=CACHE)
def _update_alpha(as_pr, ar_pr, zp_po, bs_po, br_po, dt_ik, delta_ikj, valid_mask_ikj):
    dim = as_pr.shape[1]
    as_po = np.zeros_like(as_pr)  # Alpha posterior shape, to return
    ar_po = np.zeros_like(as_pr)  # Alpha posterior rate, to return
    for i in numba.prange(dim):
        # update shape
        as_po[:, i] = as_pr[:, i] + zp_po[i].sum(axis=0)
        # update rate
        ar_po[0, i] = ar_pr[0, i] + (valid_mask_ikj[i][:, 0] * dt_ik[i]).sum(axis=0)

        D_i_kj = (valid_mask_ikj[i][:, 1:] * np.expand_dims(dt_ik[i], 1) *
                  expect_inv_beta_p_delta(bs_po[:, i], br_po[:, i],
                                          delta_ikj[i][:, 1:] + 1e-20))

        ar_po[1:, i] = ar_pr[1:, i] + D_i_kj.sum(axis=0)
    return as_po, ar_po


@numba.jit(nopython=True, fastmath=True, parallel=PARALLEL, cache=CACHE)
def _update_beta(*, x0, xn, n, as_po, ar_po, zp_po, bs_pr, br_pr,
                 dt_ik, delta_ikj, valid_mask_ikj):
    dim = as_po.shape[1]
    max_iter = 10
    tol = 1e-3
    bs_po = np.ones_like(bs_pr)
    br_po = np.ones_like(bs_pr)
    for j in numba.prange(dim):
        for i in numba.prange(dim):
            x0[j, i] = solve_halley(xstart=0.01,  # float(x0[j, i]),
                                    max_iter=max_iter,
                                    tol=tol, j=j, i=i, n=0,
                                    bs_pr=bs_pr, br_pr=br_pr,
                                    as_po=as_po, ar_po=ar_po,
                                    zp_po=zp_po,
                                    dts=dt_ik,
                                    delta=delta_ikj,
                                    valid_mask=valid_mask_ikj)
            xn[j, i] = solve_halley(xstart=0.01,  # float(xn[j, i]),
                                    max_iter=max_iter,
                                    tol=tol, j=j, i=i, n=MOMENT_ORDER,
                                    bs_pr=bs_pr, br_pr=br_pr,
                                    as_po=as_po, ar_po=ar_po,
                                    zp_po=zp_po,
                                    dts=dt_ik,
                                    delta=delta_ikj,
                                    valid_mask=valid_mask_ikj)
    bs_po = n * xn / (xn - x0) - 1
    br_po = n * xn * x0 / (xn - x0)
    return bs_po, br_po, x0, xn


@numba.jit(nopython=True, fastmath=True, cache=CACHE)
def _compute_epi(i, as_po, ar_po, bs_po, br_po, dt_ik, delta_ikj, valid_mask_ikj):
    # log(inter-arrival time), only for valid events
    epi = np.log(valid_mask_ikj[i] * np.expand_dims(dt_ik[i], 1) + 1e-20)
    # Expected value log(alpha)
    a_mean = expect_log_alpha(as_po[:, i], ar_po[:, i])
    epi += np.expand_dims(a_mean, 0)
    # Expected value log(beta + delta), only for j>=1, i.e. ignore baseline
    epi[:, 1:] -= (valid_mask_ikj[i][:, 1:] *
                   expect_log_beta_p_delta(bs_po[:, i], br_po[:, i],
                                           delta_ikj[i][:, 1:] + 1e-20))
    return epi


@numba.jit(nopython=True, fastmath=True, cache=CACHE)
def _update_z(as_po, ar_po, bs_po, br_po, dt_ik, delta_ikj, valid_mask_ikj):
    dim = as_po.shape[1]
    zs = list()
    for i in range(dim):
        epi = _compute_epi(i, as_po, ar_po, bs_po, br_po, dt_ik, delta_ikj,
                           valid_mask_ikj)
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
            # TODO: fix this once virtual events in fixed in parent class
            self.events[i] = self.events[i][:-1].numpy()
            self.n_jumps[i] -= 1
            # TODO: fix this once virtual events in fixed in parent class
            self.valid_mask_ikj[i] = np.hstack((
                np.ones((self.n_jumps[i], 1)),  # set to 1 for j=0
                self.valid_mask_ikj[i][:-1, :].numpy()))
            # TODO: fix this once virtual events in fixed in parent class
            self.delta_ikj[i] = np.hstack((
                np.zeros((self.n_jumps[i], 1)),  # set \delta_0^ik = 0 (not used)
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
        self._b_x0 = 0.1 * np.ones_like(self._br_po)
        self._b_xn = 0.1 * np.ones_like(self._br_po)
        # shape: (dim: i, #events_i: k, dim: j)
        self._zp_po = list()  # Z posterior, probabilities of Categorical distribution
        for i in range(self.dim):
            self._zp_po.append(self._zc_pr[i]
                               / self._zc_pr[i].sum(axis=1)[:, None])

    def _iteration(self):

        # print('\n', '#'*50, 'iter:', self._n_iter_done)

        # Update alpha
        self._as_po, self._ar_po = _update_alpha(as_pr=self._as_pr,
                                                 ar_pr=self._ar_pr,
                                                 zp_po=self._zp_po,
                                                 bs_po=self._bs_po,
                                                 br_po=self._br_po,
                                                 dt_ik=self.dt_ik,
                                                 delta_ikj=self.delta_ikj,
                                                 valid_mask_ikj=self.valid_mask_ikj)

        print('---- Alpha')
        print(f'    as: {self._as_po.min():+.2e}, {self._as_po.max():.2e}')
        print(f'    ar: {self._ar_po.min():+.2e}, {self._ar_po.max():.2e}')
        a_mean = self._as_po / self._ar_po
        print(f'a_mean: {a_mean.min():.2e}, {a_mean.max():.2e}')

        # (debug) Sanity check
        if np.isnan(self._as_po).any() or np.isnan(self._ar_po).any():
            raise RuntimeError("NaNs in Alpha parameters")
        if (np.min(self._as_po) < 0) or (np.min(self._ar_po) < 0):
            raise RuntimeError("Negative Alpha parameters")

        # Update beta
        self._bs_po, self._br_po, self._b_x0, self._b_xn = _update_beta(
            x0=self._b_x0, xn=self._b_xn, n=MOMENT_ORDER,  # Init equations with previous solutions
            as_po=self._as_po, ar_po=self._ar_po, zp_po=self._zp_po,
            bs_pr=self._bs_pr, br_pr=self._br_pr, dt_ik=self.dt_ik,
            delta_ikj=self.delta_ikj, valid_mask_ikj=self.valid_mask_ikj)

        print('---- Beta')
        print(f'    x0: {self._b_x0.min():+.2e}, {self._b_x0.max():.2e}')
        print(f'    xn: {self._b_xn.min():+.2e}, {self._b_xn.max():.2e}')
        print(f'    bs: {self._bs_po.min():+.2e}, {self._bs_po.max():.2e}')
        print(f'    br: {self._br_po.min():+.2e}, {self._br_po.max():.2e}')
        b_mean = self._br_po / (self._bs_po - 1) * (self._bs_po > 1)
        print(f'b_mean: {b_mean.min():+.2e}, {b_mean.max():.2e}')

        # (debug) Sanity check
        if np.isnan(self._bs_po).any() or np.isnan(self._br_po).any():
            raise RuntimeError("NaNs in Beta parameters")
        if (self._as_po.min() < 0) or (self._as_po.min() < 0):
            raise RuntimeError("Negative posterior parameter!")
        if np.any(np.isnan(self._b_x0)) or np.any(np.isnan(self._b_xn)):
            raise RuntimeError('NaNs in optimization results of beta update!')
        if np.any(np.abs(self._b_x0) > 1e10) or np.any(self._b_xn > 1e10):
            raise RuntimeError('Optimization results of beta update is diverging!')

        # Update Z
        self._zp_po = _update_z(as_po=self._as_po,
                                ar_po=self._ar_po,
                                bs_po=self._bs_po,
                                br_po=self._br_po,
                                dt_ik=self.dt_ik,
                                delta_ikj=self.delta_ikj,
                                valid_mask_ikj=self.valid_mask_ikj)

        # Set coeffs attribute for Fitter to assess convergence
        self.coeffs = expect_alpha(as_po=self._as_po, ar_po=self._ar_po)[1:, :].flatten()

        # print('zp:')
        # print(self._zp_po[0])


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
        return (as_po >= 1) * (as_po - 1) / ar_po
