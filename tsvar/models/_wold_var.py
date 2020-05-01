import warnings
import numpy as np
import scipy.special as sc
import numba

from . import WoldModel
from ..utils.decorators import enforce_observed
from ..fitter import FitterIterativeNumpy


MOMENT_ORDER = 5


warnings.filterwarnings("ignore")  # To handle NumbaPendingDeprecationWarning


def exact_beta_density(beta_range, bs_pr, br_pr, as_po, ar_po, zp_po, dts, delta, valid_mask):
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
    # print(term1)
    # print(term2)
    # print(term3)
    # print(term3_1)
    # print(term3_2)
    post = term1 * term2 * term3
    post = np.squeeze(post)
    post /= post.sum()
    # return post, term1, term2, term3
    return post


def approx_beta_density(beta_range, j, i, x0, xn, n, as_po, ar_po, zp_po, bs_pr, br_pr, dt_ik, delta_ikj, valid_mask_ikj):
    x0_out, _ = solve_halley(func, fprime, fprime2, x0, 100, 1e-5, j, i, 0,
                             bs_pr, br_pr, as_po, ar_po, zp_po, dt_ik, delta_ikj, valid_mask_ikj)
    xn_out, _ = solve_halley(func, fprime, fprime2, xn, 100, 1e-5, j, i, n,
                             bs_pr, br_pr, as_po, ar_po, zp_po, dt_ik, delta_ikj, valid_mask_ikj)
    bs_po = n * xn_out / (xn_out - x0_out) - 1
    br_po = n * xn_out * x0_out / (xn_out - x0_out)
    print(bs_po, br_po)
    import scipy.stats
    post = scipy.stats.invgamma(a=bs_po, scale=br_po).pdf(beta_range)
    post /= post.sum()
    return post


# @numba.jit(nopython=True)
def expect_alpha(as_po, ar_po):
    return as_po / ar_po


# @numba.jit(nopython=True)  # NOTE: cannot jit due to call to `sc.digamma`
def expect_log_alpha(as_po, ar_po):
    return sc.digamma(as_po) - np.log(ar_po)


@numba.jit(nopython=True)
def expect_z(zp_po):
    return zp_po


# @numba.jit(nopython=True)
def expect_inv_beta_p_delta(bs_po, br_po, delta):
    b_mean = br_po / (bs_po - 1)
    return 1 / (b_mean + delta)


@numba.jit(nopython=True)
def expect_log_beta_p_delta(bs_po, br_po, delta):
    b_mean = br_po / (bs_po - 1)
    return np.log(b_mean + delta)


# FIXME: between func, fprime and fprime2, we recompute many times the same thing
# NOTE: the `j` indices differ by 1 betwee parameters of beta and the other ones... This indexing makes nasty bugs easily introduced.
# @numba.jit(nopython=True)
def func(x, j, i, n, bs_pr, br_pr, as_po, ar_po, zp_po, dts, delta, valid_mask):
    x_p_delta = x + delta[i][:, j+1] + 1e-20
    a_mean = expect_alpha(as_po[j+1, i], ar_po[j+1, i])
    return ((bs_pr[j, i] + 1 - n) / x
            - br_pr[j, i] / (x ** 2)
            + np.sum(valid_mask[i][:, j+1] * (
                zp_po[i][:, j+1] / x_p_delta
                - a_mean * dts[i] / (x_p_delta ** 2))))


# @numba.jit(nopython=True)
def fprime(x, j, i, n, bs_pr, br_pr, as_po, ar_po, zp_po, dts, delta, valid_mask):
    x_p_delta = x + delta[i][:, j+1] + 1e-20
    a_mean = expect_alpha(as_po[j+1, i], ar_po[j+1, i])
    return (-(bs_pr[j, i] + 1 - n) / x ** 2
            + 2 * br_pr[j, i] / x ** 3
            + np.sum(valid_mask[i][:, j+1] * (
                -zp_po[i][:, j+1] / x_p_delta ** 2
                + 2 * a_mean * dts[i] / x_p_delta ** 3)))


# @numba.jit(nopython=True)
def fprime2(x, j, i, n, bs_pr, br_pr, as_po, ar_po, zp_po, dts, delta, valid_mask):
    x_p_delta = x + delta[i][:, j+1] + 1e-20
    a_mean = expect_alpha(as_po[j+1, i], ar_po[j+1, i])
    return (2 * (bs_pr[j, i] + 1 - n) / x ** 3
            - 6 * br_pr[j, i] / x ** 4
            + np.sum(valid_mask[i][:, j+1] * (
                2 * zp_po[i][:, j+1] / x_p_delta ** 3
                - 6 * a_mean * dts[i] / x_p_delta ** 4)))


# @numba.jit(nopython=True)
def solve_halley(func, fprime, fprime2, x0, max_iter, tol, j, i, n, bs_pr, br_pr, as_po, ar_po, zp_po, dts, delta, valid_mask):
    """

    required kwargs:
    ---------------

    j, i : indices of beta to solve for, as in beta_{j, i}
    n : the order to solve for
    bs_pr : beta shape prior
    br_pr : beta rate prior
    zp_po : z probability posterior
    as_po : alpha shape posterior
    ar_po : alpha rate posterior
    dts[k] : within-dimension inter-arrival time (poisson interval length, for numerator) t^i_k - t^i_{k-1}
    delta : inter-arrival time (Wold influence, for denominator)
    """
    x = float(x0)
    for it in range(max_iter):
        f = func(x, j, i, n, bs_pr, br_pr, as_po, ar_po, zp_po, dts, delta, valid_mask)
        fp = fprime(x, j, i, n, bs_pr, br_pr, as_po, ar_po, zp_po, dts, delta, valid_mask)
        fpp = fprime2(x, j, i, n, bs_pr, br_pr, as_po, ar_po, zp_po, dts, delta, valid_mask)
        x_new = x - (2 * f * fp) / (2 * fp**2 - f * fpp)
        # print(it, x_new)
        if abs(x - x_new) < tol:
            # print(it+1)
            return x, True
        x = x_new
    # print(it+1)
    return x, False


# @numba.jit(nopython=True)
def _update_alpha(as_pr, ar_pr, zp_po, bs_po, br_po, dt_ik, delta_ikj, valid_mask_ikj):
    dim = as_pr.shape[1]
    as_po = np.zeros_like(as_pr)  # Alpha posterior shape, to return
    ar_po = np.zeros_like(as_pr)  # Alpha posterior rate, to return
    for i in range(dim):
        # update shape
        as_po[:, i] = as_pr[:, i] + zp_po[i].sum(axis=0)
        # update rate
        D_i_kj = (valid_mask_ikj[i][:, 1:] * dt_ik[i][:, np.newaxis] *
                  expect_inv_beta_p_delta(bs_po[:, i], br_po[:, i],
                                          delta_ikj[i][:, 1:] + 1e-20))
        ar_po[0, i] = ar_pr[0, i] + dt_ik[i].sum(axis=0)
        ar_po[1:, i] = ar_pr[1:, i] + D_i_kj.sum(axis=0)
    return as_po, ar_po


# @numba.jit(nopython=True)
def _update_beta(x0, xn, n, as_po, ar_po, zp_po, bs_pr, br_pr,
                 dt_ik, delta_ikj, valid_mask_ikj):
    dim = as_po.shape[1]
    max_iter = 10
    tol = 1e-10
    bs_po = np.zeros_like(bs_pr)
    br_po = np.zeros_like(bs_pr)
    for j in range(dim):
        for i in range(dim):
            x0[j, i], _ = solve_halley(func, fprime, fprime2, x0[j, i], max_iter, tol, j, i, 0,
                                       bs_pr, br_pr, as_po, ar_po, zp_po, dt_ik, delta_ikj, valid_mask_ikj)
            xn[j, i], _ = solve_halley(func, fprime, fprime2, xn[j, i], max_iter, tol, j, i, n,
                                       bs_pr, br_pr, as_po, ar_po, zp_po, dt_ik, delta_ikj, valid_mask_ikj)
    bs_po = n * xn / (xn - x0) - 1
    br_po = n * xn * x0 / (xn - x0)
    return bs_po, br_po, x0, xn


# @numba.jit(nopython=True)  # NOTE: cannot jit due to call to `sc.digamma`
def _update_z(as_po, ar_po, bs_po, br_po, delta_ikj, valid_mask_ikj, eps):
    dim = as_po.shape[1]
    zp = list()
    for i in range(dim):
        epi = np.zeros_like(delta_ikj[i])

        # Expected value log(alpha)

        # epi += (sc.digamma(as_po[np.newaxis, :, i])
        #         - np.log(ar_po[np.newaxis, :, i]))
        epi += expect_log_alpha(as_po[np.newaxis, :, i],
                                ar_po[np.newaxis, :, i])

        # a, x from p(a, x) in notes
        # a = bs_po[:, i]
        # x = br_po[:, i] / (delta_ikj[i][:, 1:] + 1e-20)
        # epi[:, 1:] -= (np.log(br_po[:, i]) - sc.digamma(bs_po[:, i])
        #                - (valid_mask_ikj[i][:, 1:]
        #                   * (sc.gammainc(a + eps, x) / (sc.gammainc(a, x) + 1e-20) - 1) / eps))
        epi[:, 1:] -= expect_log_beta_p_delta(bs_po[:, i],
                                              br_po[:, i],
                                              delta_ikj[i][:, 1:] + 1e-20)

        # Softmax
        epi = epi - epi.max(axis=1)[:, np.newaxis]
        epi = np.exp(epi)
        epi /= epi.sum(axis=1)[:, np.newaxis]
        zp.append(epi)
    return zp


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
        self._b_x0 = self._br_po / (self._bs_po - 1)
        self._b_xn = 0.1 * self._br_po / (self._bs_po - 1)
        # shape: (dim: i, #events_i: k, dim: j)
        self._zp_po = list()  # Z posterior, probabilities of Categorical distribution
        for i in range(self.dim):
            self._zp_po.append(self._zc_pr[i]
                               / self._zc_pr[i].sum(axis=1)[:, np.newaxis])

    def _iteration(self):

        print('#'*50, 'start iter', self._n_iter_done)
        print()

        print('initial values:')

        print('-'*15, 'ALPHA')
        print('shape')
        print(self._as_po)
        print('rate')
        print(self._ar_po)
        print('mean ***')
        print(self._as_po / self._ar_po)

        print('-'*15, 'BETA')
        print('x0')
        print(self._b_x0)
        print(f'xn (n={MOMENT_ORDER})')
        print(self._b_xn)
        print('shape')
        print(self._bs_po)
        print('rate')
        print(self._br_po)
        print('mean ******')
        print(self._br_po / (self._bs_po - 1))

        # print('-'*15, 'Z')
        # print('dim 0')
        # print(self._zp_po[0])
        # print('dim 1')
        # print(self._zp_po[1])

        print()

        # print('do updates:')
        # print()

        # Update alpha
        self._as_po, self._ar_po = _update_alpha(as_pr=self._as_pr,
                                                 ar_pr=self._ar_pr,
                                                 zp_po=self._zp_po,
                                                 bs_po=self._bs_po,
                                                 br_po=self._br_po,
                                                 dt_ik=self.dt_ik,
                                                 delta_ikj=self.delta_ikj,
                                                 valid_mask_ikj=self.valid_mask_ikj)

        # print('='*15, 'ALPHA')
        # print('shape')
        # print(self._as_po)
        # print('rate')
        # print(self._ar_po)
        # print('mean')
        # print(self._as_po / self._ar_po)
        # print()

        # print('='*15, 'BETA')
        # print('--- before')
        # print('x0')
        # print(self._b_x0)
        # print(f'xn (n={MOMENT_ORDER})')
        # print(self._b_xn)
        # print('shape')
        # print(self._bs_po)
        # print('rate')
        # print(self._br_po)
        # print('mean')
        # print(self._br_po / (self._bs_po - 1))

        # Update beta
        self._bs_po, self._br_po, self._b_x0, self._b_xn = _update_beta(
            x0=self._b_x0, xn=self._b_xn, n=MOMENT_ORDER,  # Init equations with previous solutions
            as_po=self._as_po, ar_po=self._ar_po, zp_po=self._zp_po,
            bs_pr=self._bs_pr, br_pr=self._br_pr, dt_ik=self.dt_ik,
            delta_ikj=self.delta_ikj, valid_mask_ikj=self.valid_mask_ikj)

        # print('--- after')
        # print('x0')
        # print(self._b_x0)
        # print(f'xn (n={MOMENT_ORDER})')
        # print(self._b_xn)
        # print('shape')
        # print(self._bs_po)
        # print('rate')
        # print(self._br_po)
        # print('mean')
        # print(self._br_po / (self._bs_po - 1))
        # print()

        if (np.any(np.isnan(self._b_x0)) or np.any(np.isnan(self._b_xn)) or np.any(self._b_x0 > 1e10) or np.any(self._b_xn > 1e10)):
            raise RuntimeError('Nope nope nope...')

        # # Sanity check
        # if (self._as_po.min() < 0) or (self._as_po.min() < 0):
        #     raise RuntimeError("Negative posterior parameter!")

        # Update Z
        self._zp_po = _update_z(as_po=self._as_po,
                                ar_po=self._ar_po,
                                bs_po=self._bs_po,
                                br_po=self._br_po,
                                delta_ikj=self.delta_ikj,
                                valid_mask_ikj=self.valid_mask_ikj,
                                eps=1e-8)

        # print('='*15, 'Z')
        # print(self._zp_po[0])
        # print(self._zp_po[1])
        # print()

        # Set coeffs attribute for Fitter to assess convergence
        self.coeffs = np.hstack((
            self._as_po.flatten(), self._ar_po.flatten(),
            self._bs_po.flatten(), self._br_po.flatten(),
        ))

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
