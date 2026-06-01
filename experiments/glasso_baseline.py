"""Graphical-lasso baseline for the EUSIPCO 2026 experiments.

The proposed methods live in ``dummy_ggm``; this module isolates the comparison
baseline so the simulation driver stays focused on neighborhood selection (NS)
and the symmetry-constrained pseudo-likelihood selector (SCPL).

For each Monte-Carlo replication :func:`run_glasso_baseline` fits the graphical
lasso along a regularisation grid, scores every fit against the ground-truth
edge mask, and additionally tunes the regularisation parameter via 5-fold
cross-validation, AIC and BIC. The returned dictionary is merged into the
per-run result dict that the simulation driver aggregates.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV

__all__ = ["run_glasso_baseline", "evaluate_precision", "TunedPoint"]


class TunedPoint(NamedTuple):
    """One tuned operating point on the graphical-lasso lambda path."""

    alpha: float
    tpp: float
    fpr: float
    fdr: float

    @classmethod
    def nan(cls) -> "TunedPoint":
        """NaN-filled placeholder used by skipped Monte-Carlo runs."""
        return cls(np.nan, np.nan, np.nan, np.nan)


def _edge_to_index(i: int, j: int, p: int) -> int:
    """Linear upper-triangular index of edge ``(i, j)`` with ``i < j``."""
    return i * (p - 1) - i * (i - 1) // 2 + (j - i - 1)


def _gaussian_loglik(S: np.ndarray, theta: np.ndarray, n_samples: int) -> float:
    """Multivariate-normal log-likelihood (up to an additive constant)."""
    sign, logdet = np.linalg.slogdet(theta)
    if sign <= 0:
        return -np.inf
    return 0.5 * n_samples * (logdet - np.trace(S @ theta))


def _count_free_params(theta: np.ndarray, tol: float = 1e-8) -> int:
    """Number of free parameters: ``p`` diagonal + nonzero off-diagonal edges."""
    p = theta.shape[0]
    edges = int((np.abs(np.triu(theta, k=1)) > tol).sum())
    return p + edges


def evaluate_precision(theta: np.ndarray, true_mask: np.ndarray, p: int,
                       tol: float = 1e-6):
    """TPR, FPR and FDR of the support of ``theta`` against ``true_mask``.

    Edges are enumerated in the same linear upper-triangular order as the
    ground-truth mask, so the returned counts line up element-wise.
    """
    est_mask = np.zeros_like(true_mask, dtype=bool)
    for i in range(p):
        for j in range(i + 1, p):
            if abs(theta[i, j]) > tol:
                est_mask[_edge_to_index(i, j, p)] = True

    num_true = true_mask.sum()
    num_false = (~true_mask).sum()
    tp = int((est_mask & true_mask).sum())
    fp = int((est_mask & ~true_mask).sum())

    tpr = tp / num_true if num_true > 0 else np.nan
    fpr = fp / num_false if num_false > 0 else np.nan
    fdr = fp / (tp + fp) if (tp + fp) > 0 else np.nan
    return tpr, fpr, fdr, tp, fp, int(est_mask.sum())


def _tuned_point(criterion, lambdas, tpp, fpr, fdr) -> TunedPoint:
    """Best ``(alpha, TPR, FPR, FDR)`` along the ``lambdas`` grid for one criterion."""
    if np.all(np.isnan(criterion)):
        return TunedPoint.nan()
    idx = int(np.nanargmin(criterion))
    return TunedPoint(float(lambdas[idx]), float(tpp[idx]),
                      float(fpr[idx]), float(fdr[idx]))


def run_glasso_baseline(Xs: np.ndarray, true_mask: np.ndarray,
                        p: int, n: int, lambdas: np.ndarray) -> dict:
    """Fit the graphical-lasso lambda path and tune with CV / AIC / BIC.

    Parameters
    ----------
    Xs : ndarray of shape (n, p)
        Standardised data matrix.
    true_mask : ndarray of bool, length ``p * (p - 1) / 2``
        Ground-truth edge mask in linear upper-triangular order.
    p, n : int
        Number of variables and samples.
    lambdas : ndarray
        Regularisation grid for the graphical lasso.

    Returns
    -------
    dict
        Per-run baseline summary with keys ``glasso_{TPP,FPR,FDR,FP_counts}``
        (the regularisation path) and ``glasso_{cv,aic,bic}`` (each a
        :class:`TunedPoint`).
    """
    S = (Xs.T @ Xs) / float(n)
    tpp, fpr, fdr, fp_counts, aic, bic = [], [], [], [], [], []

    for lam in lambdas:
        gl = GraphicalLasso(alpha=lam, max_iter=1000, assume_centered=False)
        gl.fit(Xs)
        theta = gl.precision_
        tpr_g, fpr_g, fdr_g, _, fp_g, _ = evaluate_precision(theta, true_mask, p)
        tpp.append(tpr_g)
        fpr.append(fpr_g)
        fdr.append(fdr_g)
        fp_counts.append(fp_g)
        ll = _gaussian_loglik(S, theta, n)
        k_params = _count_free_params(theta)
        aic.append(-2.0 * ll + 2.0 * k_params)
        bic.append(-2.0 * ll + np.log(n) * k_params)

    tpp = np.array(tpp)
    fpr = np.array(fpr)
    fdr = np.array(fdr)
    fp_counts = np.array(fp_counts, dtype=int)
    aic = np.array(aic)
    bic = np.array(bic)

    glcv = GraphicalLassoCV(alphas=lambdas, cv=5)
    glcv.fit(Xs)
    idx_cv = int(np.argmin(np.abs(lambdas - glcv.alpha_)))
    cv_point = TunedPoint(float(glcv.alpha_), float(tpp[idx_cv]),
                          float(fpr[idx_cv]), float(fdr[idx_cv]))

    return dict(
        glasso_TPP=tpp, glasso_FPR=fpr, glasso_FDR=fdr,
        glasso_FP_counts=fp_counts,
        glasso_cv=cv_point,
        glasso_aic=_tuned_point(aic, lambdas, tpp, fpr, fdr),
        glasso_bic=_tuned_point(bic, lambdas, tpp, fpr, fdr),
    )
