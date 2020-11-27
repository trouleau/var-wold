#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: language_level=3

from libc.math cimport lgamma
from scipy.optimize.cython_optimize cimport ridder

import numpy as np  # For internal testing of the cython documentation
cimport numpy as np  # "cimport" is used to import special compile-time stuff

DTYPE_d = np.float
ctypedef np.float_t DTYPE_d_t


cdef expect_inv_beta_p_delta(np.ndarray[DTYPE_d_t, ndim=1] bs_po,
                             np.ndarray[DTYPE_d_t, ndim=1] br_po,
                             np.ndarray[DTYPE_d_t, ndim=2] delta):
    """Compute the expectation of 1/(beta + delta)"""
    b_mean = br_po / (bs_po - 1)
    return 1 / (b_mean + delta)


def _update_alpha(np.ndarray[DTYPE_d_t, ndim=2] as_pr,
                  np.ndarray[DTYPE_d_t, ndim=2] ar_pr,
                  list zp_po,
                  np.ndarray[DTYPE_d_t, ndim=2] bs_po,
                  np.ndarray[DTYPE_d_t, ndim=2] br_po,
                  list delta_ikj,
                  list dt_ik,
                  np.ndarray[DTYPE_d_t, ndim=1] last_t):
    cdef:
        dim = as_pr.shape[1]
        np.ndarray[DTYPE_d_t, ndim=2] as_po = np.zeros((dim+1, dim))  # Alpha posterior shape, to return
        np.ndarray[DTYPE_d_t, ndim=2] ar_po = np.zeros((dim+1, dim))  # Alpha posterior rate, to return
        np.ndarray[DTYPE_d_t, ndim=2] zp_po_i
    for i in range(dim):
        # update shape
        as_po[:, i] = as_pr[:, i] + zp_po[i].sum(axis=0)
        # update rate
        ar_po[0, i] = ar_pr[0, i] + last_t[i]
        D_i_kj = (np.expand_dims(dt_ik[i], 1) *
                  expect_inv_beta_p_delta(bs_po[:, i], br_po[:, i],
                                          delta_ikj[i][:, 1:] + 1e-20))
        ar_po[1:, i] = ar_pr[1:, i] + D_i_kj.sum(axis=0)
    return as_po, ar_po


cdef digamma(np.ndarray[DTYPE_d_t, ndim=1] arr):
    """Digamma function (arr is assumed to be 1 or 2 dimensional)"""
    cdef:
        int dim0 = arr.shape[0]
        int dim1 = arr.shape[1]
        double eps = 1e-8
    lgamma_prime = np.zeros_like(arr)
    if arr.ndim == 1:
        for i in range(dim0):
            lgamma_prime[i] = (lgamma(arr[i] + eps) - lgamma(arr[i])) / eps
    elif arr.ndim == 2:
        for j in range(dim0):
            for i in range(dim1):
                lgamma_prime[j, i] = (lgamma(arr[j, i] + eps) - lgamma(arr[j, i])) / eps
    return lgamma_prime



cdef expect_log_alpha(np.ndarray[DTYPE_d_t, ndim=1] as_po, np.ndarray[DTYPE_d_t, ndim=1] ar_po):
    """Compute the expectation of log(alpha)"""
    return digamma(as_po) - np.log(ar_po)


cdef expect_log_beta_p_delta(np.ndarray[DTYPE_d_t, ndim=1] bs_po,
                            np.ndarray[DTYPE_d_t, ndim=1] br_po,
                            np.ndarray[DTYPE_d_t, ndim=2] delta):
    """Compute the expectation of log(beta + delta)"""
    b_mean = br_po / (bs_po - 1)
    return np.log(b_mean + delta)


cdef _compute_epi(int i,
                 np.ndarray[DTYPE_d_t, ndim=2] as_po,
                 np.ndarray[DTYPE_d_t, ndim=2] ar_po,
                 np.ndarray[DTYPE_d_t, ndim=2] bs_po,
                 np.ndarray[DTYPE_d_t, ndim=2] br_po,
                 list dt_ik,
                 list delta_ikj):
    # log(inter-arrival time), only for valid events
    epi = np.zeros_like(delta_ikj[i])
    epi += np.log(np.expand_dims(dt_ik[i], 1) + 1e-20)
    # Expected value log(alpha)
    epi += np.expand_dims(expect_log_alpha(as_po[:, i], ar_po[:, i]), 0)
    # Expected value log(beta + delta), only for j>=1, i.e. ignore baseline
    epi[:, 1:] -= expect_log_beta_p_delta(bs_po[:, i], br_po[:, i],
                                          delta_ikj[i][:, 1:] + 1e-20)
    return epi



def _update_z(np.ndarray[DTYPE_d_t, ndim=2] as_po,
              np.ndarray[DTYPE_d_t, ndim=2] ar_po,
              np.ndarray[DTYPE_d_t, ndim=2] bs_po,
              np.ndarray[DTYPE_d_t, ndim=2] br_po,
              list dt_ik,
              list delta_ikj):
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
