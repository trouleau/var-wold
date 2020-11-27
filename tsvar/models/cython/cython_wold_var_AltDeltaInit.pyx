#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: language_level=3

from libc.math cimport sin
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
