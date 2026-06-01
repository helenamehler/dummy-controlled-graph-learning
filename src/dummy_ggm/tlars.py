"""Dummy-augmented nodewise T-LARS for neighborhood selection (NS).

This is the local formulation of the paper (Sec. II-A): for a single response
``y`` and a design matrix ``[X | D]`` augmented with dummy predictors ``D``, a
LARS forward-selection path is run and terminated once ``T`` dummy predictors
have entered the active set. Repeating this over all nodes and random
experiments yields the dummy-controlled neighborhood selection estimate.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from dummy_ggm._helpers import cholesky_rank1_update, compute_stepsize_gamma


class TLarsNodewise:
    """LARS forward selection on a dummy-augmented design matrix.

    The active set grows one (signed) predictor at a time, maintaining a
    Cholesky factor of the active Gram matrix via rank-1 updates. Selection
    stops once ``T`` dummy predictors (columns with index ``>= p - num_dummies``)
    have been picked up. The path supports warm starts: calling :meth:`run`
    again with a larger ``T`` resumes from where the previous call stopped.
    """

    def __init__(self, X, y, num_dummies, max_steps=None, lasso=False,
                 normalize=False, verbose: bool = False):
        """
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Augmented predictor matrix; the last ``num_dummies`` columns are
            the dummy predictors.
        y : ndarray of shape (n_samples,)
            Response vector.
        num_dummies : int
            Number of dummy predictors at the end of ``X``.
        max_steps : int, optional
            Maximum number of LARS steps. Defaults to ``min(n, p)``.
        lasso : bool, optional
            If True, run the LASSO variant (with variable removals).
        normalize : bool, optional
            If True, center ``y`` and center/normalize the columns of ``X``.
        verbose : bool, optional
            If True, print diagnostic messages.
        """
        self.X = X
        self.y = y
        self.num_dummies = num_dummies
        self.max_steps = max_steps
        self.lasso = lasso
        self.normalize = normalize
        self.verbose = verbose

        self.n, self.p = X.shape
        self.beta_path = []
        self.actives = []
        self.dropid = []

        self.tiny32 = np.finfo(np.float32).tiny
        self.eps = 1e2 * np.finfo(np.float64).eps
        self.normx = np.ones(self.p)

        self.selected_dummies = []
        self.selected_num_dummies = 0
        self.step = 0  # next step index (enables warm starts)
        self.R = None

        if self.max_steps is None:
            self.max_steps = min(self.n, self.p)

        if self.normalize:
            Xc = self.X - self.X.mean(axis=0)
            normx = np.maximum(np.linalg.norm(Xc, axis=0), np.finfo(np.float64).tiny)
            self.X = Xc / normx
            self.y = self.y - self.y.mean()
            self.normx = normx

        self.residuals = self.y.copy()
        self.corr = self.X.T @ self.residuals

    def run(self, T=1):
        """Run (or resume) the path until ``T`` dummy predictors are selected.

        Returns the coefficient path as an array of shape ``(n_steps, p)``,
        rescaled back to the original predictor scale.
        """
        if self.step == 0:
            beta = np.zeros(self.p)
            self.beta_path = [beta.copy()]
            mu = np.zeros(self.n)
        else:
            beta = self.beta_path[-1].copy()
            mu = self.y - self.residuals

        range_val = 8 * self.max_steps if self.lasso else self.max_steps

        for step in range(self.step, range_val):
            self.corr = self.X.T @ self.residuals
            C = np.max(np.abs(self.corr))
            if C < 100 * self.eps:
                if self.verbose:
                    print("Max |corr| ~ 0; exiting.")
                break

            # Candidates are the predictors currently tied for max correlation.
            candidates = np.where(np.abs(self.corr) >= C - self.eps)[0]

            # Add candidates one at a time (a candidate that is numerically
            # dependent on the active set is dropped rather than added).
            for cand in candidates:
                if cand in self.actives or cand in self.dropid:
                    continue
                if len(self.actives) >= self.max_steps:
                    if self.verbose:
                        print(f"Reached max_steps = {self.max_steps}; stop adding.")
                    break

                if len(self.actives) > 0:
                    s_existing = np.where(self.corr[self.actives] >= 0.0, 1.0, -1.0)
                    X_active_prev = self.X[:, self.actives] * s_existing[np.newaxis, :]
                else:
                    X_active_prev = np.zeros((self.n, 0))

                s_cand = 1.0 if self.corr[cand] >= 0.0 else -1.0
                new_col = self.X[:, cand] * s_cand

                try:
                    if len(self.actives) == 0:
                        r0 = np.sqrt(np.dot(new_col, new_col))
                        if r0 <= self.tiny32:
                            raise ValueError("Zero norm for first active predictor")
                        R_new = np.array([[r0]])
                    else:
                        R_new = cholesky_rank1_update(self.R, X_active_prev, new_col)
                        if R_new is None:
                            raise ValueError("cholesky_rank1_update returned None")
                except Exception as err:
                    if self.verbose:
                        print(f"Ignoring candidate {cand} (Cholesky failure): {err}")
                    self.dropid.append(cand)
                    continue

                self.R = R_new
                self.actives.append(cand)

            if len(self.actives) == 0:
                if self.verbose:
                    print("No active predictors after candidate processing; breaking.")
                break

            s = np.where(self.corr[self.actives] >= 0.0, 1.0, -1.0)
            X_active = self.X[:, self.actives]
            X_active_prime = X_active * s[np.newaxis, :]

            if self.R is None:
                if self.verbose:
                    print("R is None after candidate processing; breaking.")
                break

            # Equiangular direction and its correlation profile.
            ones_vec = np.ones(self.R.shape[0])
            z = solve_triangular(self.R.T, ones_vec, lower=True)
            w = solve_triangular(self.R, z, lower=False)
            A_active = 1.0 / np.sqrt(np.dot(ones_vec, w))
            w = w * A_active
            u = X_active_prime @ w
            a = self.X.T @ u

            d = np.zeros(self.p)
            for i_idx, active in enumerate(self.actives):
                d[active] = s[i_idx] * w[i_idx]

            gammas = np.full(self.p, np.inf)
            actives_np = np.array(self.actives, dtype=np.int32)
            gammas = compute_stepsize_gamma(self.corr, a, A_active, C, self.tiny32, gammas, actives_np)
            if self.dropid:
                gammas[np.array(self.dropid, dtype=int)] = np.inf
            gamma = np.min(gammas)

            if self.lasso and step > 0:
                gammas_lasso = np.full(self.p, np.inf)
                for active in self.actives:
                    gammas_lasso[active] = -beta[active] / (d[active] + self.tiny32)
                positive = [val for val in gammas_lasso if val > self.tiny32]
                gamma_lasso = np.min(positive) if positive else np.inf
                gamma_j = np.where(gammas_lasso == gamma_lasso)

                if gamma_lasso < gamma:
                    mu += gamma_lasso * u
                    beta += gamma_lasso * d
                    for j in np.atleast_1d(gamma_j).flatten():
                        if j in self.actives:
                            self.actives.remove(int(j))
                            self.dropid.append(int(j))
                            beta[int(j)] = 0
                else:
                    mu += gamma * u
                    beta += gamma * d
            else:
                mu += gamma * u
                beta += gamma * d

            self.residuals = self.y - mu
            self.beta_path.append(beta.copy())

            # Count newly selected dummy predictors and stop early at T of them.
            for a_idx in list(self.actives):
                if a_idx >= self.p - self.num_dummies and a_idx not in self.selected_dummies:
                    self.selected_dummies.append(a_idx)
                    self.selected_num_dummies += 1
                    if self.selected_num_dummies >= T:
                        if self.verbose:
                            print(f"Stopping early: {T} dummies selected.")
                        self.step = step + 1
                        break

            if self.selected_num_dummies >= T:
                break

        return np.array(self.beta_path) / self.normx
