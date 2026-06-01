"""Global, symmetry-constrained pseudo-likelihood (SCPL) T-LARS.

This is the global formulation of the paper (Sec. II-B). All ``p`` neighborhood
regressions are stacked into one linear model whose design-matrix columns are
the undirected edges of the augmented graph. Selecting an edge column enforces
the symmetry constraint ``beta_(i,j) = beta_(j,i)`` automatically, so no
post-hoc symmetrisation is needed.

The design matrix is never materialised: every column is determined by its node
pair, and all forward-selection quantities are computed from the node vectors by
the kernels in :mod:`dummy_ggm._graph_utils`. Selection stops once ``T`` dummy
edges (data-dummy or dummy-dummy) have entered the active set.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.linalg import solve_triangular
from scipy.linalg.blas import daxpy

from dummy_ggm._graph_utils import (
    cholesky_rank1_update,
    compute_stepsize_gamma,
    compute_u,
    compute_X_T_and_vector,
    index_to_edge,
    init_index_to_edge,
    inner_product,
)

__all__ = ["TLarsGraph", "init_index_to_edge", "index_to_edge"]


class TLarsGraph:
    """Edge-level LARS forward selection for the SCPL formulation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, p + num_dummies)
        Augmented data matrix (original variables followed by dummy nodes). Its
        columns are centred and normalised internally.
    num_dummies : int
        Number of dummy nodes appended to the original variables.
    y : ndarray, optional
        Stacked response ``vec(X)``. Defaults to the flattened normalised ``X``.
    max_steps : int, optional
        Maximum number of LARS steps.
    counter : {"T", "TXD", "TDD"}, optional
        Which dummy-edge count triggers the stopping rule: all dummy edges
        (``"T"``), only data-dummy edges (``"TXD"``), or only dummy-dummy edges
        (``"TDD"``).
    dummy_true_edges : iterable of (int, int), optional
        Edges among the dummy nodes that should *not* be counted as dummy
        selections (used only for structured dummies; empty by default).
    verbose : bool, optional
        If True, emit debug logging.
    """

    def __init__(self, X, num_dummies, y=None, max_steps=None, dummy_true_edges=None, verbose: bool = False):
        X_centered = X - np.mean(X, axis=0)
        self.X = X_centered / np.linalg.norm(X_centered, axis=0)

        self.n, self.p = X.shape
        self.y = self.X.ravel(order="F").copy() if y is None else y
        self.num_dummies = num_dummies
        self.max_steps = max_steps or min(self.n * self.p, self.p * (self.p - 1) // 2)

        if dummy_true_edges is not None:
            offset = self.p - self.num_dummies
            self.dummy_true_edges = {(i + offset, j + offset) for i, j in dummy_true_edges}
        else:
            self.dummy_true_edges = set()

        # Warm-start state.
        self.step = 0
        self.R = None
        self.w = None
        self.a = None
        self.u = None
        self.A_active = None
        self.beta = np.zeros(self.p * (self.p - 1) // 2)
        self.new_edges = []

        # Pre-allocated buffers (avoid per-iteration allocations).
        self._M = self.p * (self.p - 1) // 2
        self._gammas = np.empty(self._M, dtype=float)
        self._d = np.empty(self._M, dtype=float)
        self._actives_buf = np.empty(self._M, dtype=np.int32)

        # Tracking.
        self.beta_path = []
        self.residuals = self.y.copy()
        self.eps = np.finfo(np.float64).eps
        self.actives = []
        self.active_edges = []
        self.signs = None
        self.selected_dummies = []
        self.selected_num_dummies = 0
        self.selected_XD = 0  # data-dummy edges
        self.selected_DD = 0  # dummy-dummy edges

        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

        init_index_to_edge(self.p)

    def _prepare_initial_state(self):
        if self.step == 0:
            self.beta_path = [np.zeros(self._M)]
            self.mu = np.zeros_like(self.y)
            self.corr = compute_X_T_and_vector(self.X, self.residuals)
        else:
            self.beta_path = [self.beta_path[-1].copy()]
            self.mu = self.y - self.residuals

    def _find_and_add_active(self):
        C = np.max(np.abs(self.corr))
        if C < self.eps:
            self.logger.debug("Max correlation below eps; stopping.")
            raise StopIteration

        cand = np.where(np.abs(self.corr) >= C - self.eps)[0]
        new_idxs = [i for i in cand if i not in self.actives]
        self.actives.extend(new_idxs)
        self.new_edges = [index_to_edge(i) for i in new_idxs]
        self.active_edges.extend(self.new_edges)
        self.signs = np.sign(self.corr[self.actives])

    def _update_factor(self):
        if len(self.actives) == 1:
            edge = np.array(self.active_edges[0], dtype=np.int64)
            val = inner_product(self.X, edge, edge, 1.0, 1.0)
            self.R = np.array([[np.sqrt(val)]], dtype=float)
            return

        prev = np.ascontiguousarray(np.array(self.active_edges[:-1], dtype=np.int64))
        new = np.ascontiguousarray(np.array(self.active_edges[-1], dtype=np.int64))
        try:
            self.R = cholesky_rank1_update(self.R, prev, new, self.signs, self.X, self.eps)
        except np.linalg.LinAlgError:
            self.logger.error("Cholesky update failed: not positive definite.")
            raise StopIteration

    def _compute_direction(self):
        ones = np.ones(len(self.active_edges))
        z = solve_triangular(self.R.T, ones, lower=True)
        try:
            w = solve_triangular(self.R, z, lower=False)
        except Exception as err:
            self.logger.error("Triangular solve failed at step %d: %s", self.step, err)
            raise StopIteration
        A = 1.0 / np.sqrt(ones.dot(w))
        self.w = w * A
        w_sign = self.w * self.signs
        edges_np = np.ascontiguousarray(np.array(self.active_edges, dtype=np.int64))
        self.u = compute_u(self.X, edges_np, w_sign)
        self.a = compute_X_T_and_vector(self.X, self.u)
        self.A_active = A

    def _take_step(self):
        self._gammas.fill(np.inf)
        self._d.fill(0.0)
        k = len(self.actives)
        self._actives_buf[:k] = self.actives
        self._d[self.actives] = self.signs * self.w

        gammas = compute_stepsize_gamma(
            self.corr, self.a, self.A_active, np.max(np.abs(self.corr)),
            self.eps, self._gammas, self._actives_buf[:k],
        )
        gamma = np.min(gammas)

        self.mu = daxpy(self.u, self.mu, a=gamma)
        beta_new = self.beta_path[-1] + gamma * self._d
        self.residuals = self.y - self.mu
        self.corr = daxpy(self.a, self.corr, a=-gamma)
        self.beta_path.append(beta_new)
        self.beta = beta_new

    def run(self, T: int = 1) -> np.ndarray:
        """Run (or resume) the path until ``T`` dummy edges have been selected.

        Returns the coefficient path as an array of shape ``(n_steps, M)``,
        where ``M = p * (p - 1) / 2`` is the number of undirected edges.
        """
        self._prepare_initial_state()

        for step in range(self.step, self.max_steps):
            self._find_and_add_active()
            self._update_factor()
            self._compute_direction()
            self._take_step()

            # Count newly selected dummy edges (an edge whose larger endpoint is
            # a dummy node) and split them into data-dummy / dummy-dummy.
            threshold = self.p - self.num_dummies
            for edge in self.new_edges:
                if edge[1] >= threshold and edge not in self.dummy_true_edges:
                    self.selected_dummies.append(edge)
                    self.selected_num_dummies += 1
                    if edge[0] >= threshold:
                        self.selected_DD += 1
                    else:
                        self.selected_XD += 1
                    if self.selected_num_dummies >= T:
                        self.step = step + 1
                        break

            if self.selected_num_dummies >= T:
                break

        return np.array(self.beta_path)
