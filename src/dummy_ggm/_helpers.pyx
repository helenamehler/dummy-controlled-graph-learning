# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
"""Low-level kernels for the nodewise (neighborhood selection) forward path.

Unlike the global formulation, neighborhood selection works on an explicit
augmented design matrix ``[X | D]`` per node, so these routines operate on
dense column blocks directly.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt


def cholesky_rank1_update(np.ndarray[np.float64_t, ndim=2] R,
                          np.ndarray[np.float64_t, ndim=2] X_active,
                          np.ndarray[np.float64_t, ndim=1] x_new,
                          double eps=1e-12):
    """Append column ``x_new`` to the active set via a rank-1 Cholesky update.

    Returns the updated factor, or ``None`` if the new column is numerically
    dependent on the active set (caller treats this as "skip this candidate").
    """
    cdef Py_ssize_t i, j, t = X_active.shape[1]
    cdef np.ndarray[np.float64_t, ndim=1] v = np.dot(X_active.T, x_new)
    cdef double s = np.dot(x_new, x_new)

    # Forward substitution: R^T z = v.
    cdef np.ndarray[np.float64_t, ndim=1] z = np.zeros(t)
    for i in range(t):
        z[i] = v[i]
        for j in range(i):
            z[i] -= R[j, i] * z[j]
        z[i] /= R[i, i]

    cdef double r_squared = s - np.dot(z, z)
    if r_squared < eps:
        return None
    cdef double r = sqrt(r_squared)

    cdef np.ndarray[np.float64_t, ndim=2] R_new = np.zeros((t + 1, t + 1))
    for i in range(t):
        for j in range(i, t):
            R_new[i, j] = R[i, j]
        R_new[i, t] = z[i]
    R_new[t, t] = r
    return R_new


def compute_stepsize_gamma(np.ndarray[np.float64_t, ndim=1] corr,
                           np.ndarray[np.float64_t, ndim=1] a,
                           double A_active,
                           double C,
                           double tiny32,
                           np.ndarray[np.float64_t, ndim=1] gammas,
                           np.ndarray[np.int32_t, ndim=1] actives):
    """LARS step sizes for every inactive predictor (smallest one is taken)."""
    cdef int p = corr.shape[0]
    cdef bint is_active
    cdef int j, k
    cdef double gamma_plus, gamma_minus

    for j in range(p):
        is_active = False
        for k in range(actives.shape[0]):
            if actives[k] == j:
                is_active = True
                break
        if not is_active:
            gamma_minus = (C - corr[j]) / (A_active - a[j] + tiny32)
            gamma_plus = (C + corr[j]) / (A_active + a[j] + tiny32)

            if gamma_minus > tiny32 and gamma_plus > tiny32:
                gammas[j] = min(gamma_minus, gamma_plus)
            elif gamma_minus > tiny32:
                gammas[j] = gamma_minus
            elif gamma_plus > tiny32:
                gammas[j] = gamma_plus
            else:
                gammas[j] = np.inf
    return gammas
