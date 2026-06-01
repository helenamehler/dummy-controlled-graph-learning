# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
"""Low-level kernels for the global (SCPL) edge-level forward selection.

In the symmetry-constrained pseudo-likelihood formulation the design matrix
columns correspond to undirected edges ``(i, j)`` of the augmented graph. Each
column is fully determined by its index pair, so all forward-selection
operations (inner products, Cholesky updates, step sizes) can be computed
directly from the node vectors without ever materialising the huge design
matrix. These routines implement those operations.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from cython.parallel import prange
from numpy.linalg import LinAlgError
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

# Flat lookup table mapping a linear edge index to its (i, j) node pair.
# Filled once per problem size by ``init_index_to_edge``.
cdef vector[int] _idx2edge_cpp

ctypedef np.float64_t DTYPE_t


cpdef init_index_to_edge(int p):
    """Build the linear-index -> (i, j) edge table for ``p`` nodes."""
    cdef int M = p * (p - 1) // 2
    _idx2edge_cpp.clear()
    _idx2edge_cpp.reserve(2 * M)
    for i in range(p - 1):
        for j in range(i + 1, p):
            _idx2edge_cpp.push_back(i)
            _idx2edge_cpp.push_back(j)


cpdef tuple index_to_edge(int idx):
    """Return the node pair ``(i, j)`` for a linear edge index ``idx``."""
    cdef int i = _idx2edge_cpp[2 * idx]
    cdef int j = _idx2edge_cpp[2 * idx + 1]
    return i, j


cpdef double inner_product(
        double[:, :] X,
        np.int64_t[::1] idx1,  # length-2 edge (a1, b1)
        np.int64_t[::1] idx2,  # length-2 edge (a2, b2)
        double s1,
        double s2):
    """Inner product of two signed edge columns, computed from node vectors.

    For edges ``(a1, b1)`` and ``(a2, b2)`` the column inner product reduces to
    a few scalar products of columns of ``X`` (see the paper, Sec. II-B):
    the self-product sums both endpoint norms, edges sharing one node give a
    single cross term, and disjoint edges are orthogonal.
    """
    cdef int a1 = idx1[0]
    cdef int b1 = idx1[1]
    cdef int a2 = idx2[0]
    cdef int b2 = idx2[1]
    cdef double res = 0.0
    cdef int n = X.shape[0]
    cdef int i
    cdef int o1, o2

    # Same edge: <z, z> = <x_a, x_a> + <x_b, x_b>.
    if a1 == a2 and b1 == b2:
        for i in range(n):
            res += X[i, a1] * X[i, a1] + X[i, b1] * X[i, b1]
        return res

    # Edges sharing one vertex: reduce to a single cross term.
    if a1 == a2:
        o1, o2 = b1, b2
    elif a1 == b2:
        o1, o2 = b1, a2
    elif b1 == a2:
        o1, o2 = a1, b2
    elif b1 == b2:
        o1, o2 = a1, a2
    else:
        return 0.0  # disjoint edges are orthogonal

    for i in range(n):
        res += s1 * s2 * X[i, o1] * X[i, o2]
    return res


cpdef np.ndarray[DTYPE_t, ndim=2] cholesky_rank1_update(
        np.ndarray[DTYPE_t, ndim=2] R,             # current factor, shape (t, t)
        np.ndarray[np.int64_t, ndim=2] active_edges,  # active edges, shape (t, 2)
        np.ndarray[np.int64_t, ndim=1] new_edge,      # the new edge, length 2
        np.ndarray[DTYPE_t, ndim=1] signs,         # signs, length t + 1
        np.ndarray[DTYPE_t, ndim=2] X,             # data matrix, shape (n, p)
        double eps=1e-12):
    """Append one edge to the active set via a rank-1 Cholesky update.

    The cross products ``v[i] = <active_edges[i], new_edge>`` and the
    self-product of the new edge are computed on the fly with
    :func:`inner_product`, so the design matrix is never formed explicitly.
    """
    cdef Py_ssize_t t = active_edges.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] v = np.empty(t, dtype=np.float64)
    cdef double new_diag
    cdef Py_ssize_t i, j

    # 1) cross inner products of the new edge with each active edge.
    for i in range(t):
        v[i] = inner_product(X, active_edges[i], new_edge, signs[i], signs[t])

    # 2) self inner product of the new edge.
    new_diag = inner_product(X, new_edge, new_edge, signs[t], signs[t])

    # 3) forward-solve R^T z = v.
    cdef np.ndarray[DTYPE_t, ndim=1] z = np.empty(t, dtype=np.float64)
    for i in range(t):
        z[i] = v[i]
        for j in range(i):
            z[i] -= R[j, i] * z[j]
        z[i] /= R[i, i]

    # 4) new diagonal entry r = sqrt(new_diag - z^T z).
    cdef double r2 = new_diag
    for i in range(t):
        r2 -= z[i] * z[i]
    if r2 < -eps:
        raise LinAlgError("Cholesky update failed: not positive definite")
    if r2 < 0.0:
        r2 = 0.0
    cdef double r = sqrt(r2)

    # 5) assemble the (t + 1, t + 1) updated factor.
    cdef np.ndarray[DTYPE_t, ndim=2] R_new = np.zeros((t + 1, t + 1), dtype=np.float64)
    for i in range(t):
        for j in range(i, t):
            R_new[i, j] = R[i, j]
        R_new[i, t] = z[i]
    R_new[t, t] = r
    return R_new


cpdef inline double[::1] compute_stepsize_gamma(
        double[::1] corr,    # shape (M,)
        double[::1] a,       # shape (M,)
        double A_active,
        double C,
        double tiny32,
        double[::1] gammas,  # output buffer, shape (M,)
        int[::1] actives) nogil:
    """LARS step sizes for every inactive edge (smallest one is taken)."""
    cdef int p = corr.shape[0]
    cdef int j, t = actives.shape[0]
    cdef double gm, gp
    cdef char *is_act

    is_act = <char*>malloc(p * sizeof(char))
    if not is_act:
        return gammas
    for j in range(p):
        is_act[j] = 0
        gammas[j] = 1e300
    for j in range(t):
        is_act[actives[j]] = 1

    for j in prange(p, nogil=True):
        if not is_act[j]:
            gm = (C - corr[j]) / (A_active - a[j] + tiny32)
            gp = (C + corr[j]) / (A_active + a[j] + tiny32)
            if gm > tiny32 and gm < gp:
                gammas[j] = gm
            elif gp > tiny32:
                gammas[j] = gp

    free(is_act)
    return gammas


cpdef np.ndarray[DTYPE_t, ndim=1] compute_u(
        double[:, :] X,             # shape (n, p)
        long[:, ::1] active_edges,  # shape (t, 2)
        double[::1] w):             # length t
    """Equiangular direction ``u = Z_active @ w`` in the n*p ambient space.

    Each edge column places ``x_b`` in block ``a`` and ``x_a`` in block ``b``;
    this accumulates the weighted contributions directly into ``u``.
    """
    cdef int n = X.shape[0]
    cdef int t = active_edges.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] u = np.zeros(n * X.shape[1], dtype=np.float64)
    cdef int i, j, a, b, base_a, base_b
    cdef double wi

    for i in range(t):
        a = active_edges[i, 0]
        b = active_edges[i, 1]
        wi = w[i]
        base_a = a * n
        base_b = b * n
        for j in range(n):
            u[base_a + j] += wi * X[j, b]
            u[base_b + j] += wi * X[j, a]
    return u


cpdef np.ndarray[DTYPE_t, ndim=1] compute_X_T_and_vector(
        double[:, :] X,
        np.ndarray[DTYPE_t, ndim=1] r):
    """Correlations of every edge column with vector ``r`` (i.e. ``Z^T r``).

    Computed block-wise through a single dense ``X^T R`` product, then
    symmetrised and read off the strict upper triangle.
    """
    cdef int p = X.shape[1]
    cdef int n = X.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] Rm = r.reshape((p, n)).T
    cdef np.ndarray[DTYPE_t, ndim=2] C = X.T @ Rm
    C += C.T
    return C[np.triu_indices(p, 1)]
