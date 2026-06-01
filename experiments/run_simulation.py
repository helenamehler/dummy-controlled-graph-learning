"""Monte-Carlo simulation driver for the EUSIPCO 2026 experiments.

For a chosen graph topology this script generates synthetic Gaussian graphical
models with known ground truth and evaluates, along the selection path:

* the global symmetry-constrained pseudo-likelihood selector (SCPL), and
* dummy-augmented neighborhood selection (NS).

The graphical-lasso baseline (lambda path plus CV / AIC / BIC tuning) lives in
:mod:`glasso_baseline` and the npz layout is owned by :mod:`result_schema`;
both are delegated to from here.

The per-run true-positive rate (TPR), false-positive rate (FPR), false-discovery
rate (FDR) and selected-null counts are aggregated over Monte-Carlo replications
and written to a compressed ``.npz`` file that the plotting scripts consume.

Example
-------
    python experiments/run_simulation.py --topology pa --mc 100 \
        --output results/sim_pa.npz
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed

from dummy_ggm import (
    TLarsGraph,
    TLarsNodewise,
    generate_er_graph,
    generate_pa_graph,
    generate_small_world_graph,
)
from dummy_ggm.graph_tlars import index_to_edge, init_index_to_edge
from glasso_baseline import run_glasso_baseline
from result_schema import aggregate_and_save, empty_result

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

TOPOLOGIES = ("er", "pa", "sw")

# Majority-vote threshold used to turn the joint relative occurrence of an
# edge across the K random experiments into a final selection (paper, Eq. 6).
NS_VOTING_THRESHOLD = 0.5


@dataclass
class SimConfig:
    """Parameters shared by all Monte-Carlo runs of one simulation."""

    topology: str = "pa"
    n: int = 150
    p: int = 150
    num_dummies: int = 150
    K: int = 20            # random experiments per run
    mc: int = 100          # Monte-Carlo replications
    seed: int = 4
    t_max: int = 5000       # largest stopping point for the global selector
    n_jobs: int = -1
    value_range: tuple[float, float] = (0.2, 0.8)
    add_min_eigen: float = 0.15
    # Topology-specific structural parameters (paper defaults).
    er_prob: float = 0.05
    pa_power: float = 0.5
    pa_m: int = 5
    sw_nei: int = 5        # 2 * nei = 10 nearest neighbours on the ring
    sw_rewire: float = 0.5
    lambdas_glasso: np.ndarray = field(
        default_factory=lambda: np.linspace(0.01, 1.2, 25)
    )

    @property
    def t_grid_large(self) -> np.ndarray:
        """Stopping points for the global (SCPL) selector."""
        fine = np.arange(1, self.p + 1, 1, dtype=int)
        coarse = np.arange(self.p, self.t_max + 1, 50, dtype=int)
        return np.unique(np.concatenate([fine, coarse])).astype(int)

    @property
    def t_grid_small(self) -> np.ndarray:
        """Stopping points for nodewise neighborhood selection."""
        return np.arange(1, self.p, 1)


def generate_graph(cfg: SimConfig, rng: np.random.Generator):
    """Draw one ground-truth graph and covariance for the chosen topology."""
    common = dict(value_range=cfg.value_range, add_min_eigen=cfg.add_min_eigen,
                  signs="Random", rng=rng)
    if cfg.topology == "er":
        return generate_er_graph(cfg.p, prob=cfg.er_prob, **common)
    if cfg.topology == "pa":
        return generate_pa_graph(cfg.p, power=cfg.pa_power, m=cfg.pa_m, **common)
    if cfg.topology == "sw":
        return generate_small_world_graph(cfg.p, prob=cfg.sw_rewire, nei=cfg.sw_nei, **common)
    raise ValueError(f"unknown topology {cfg.topology!r}; choose from {TOPOLOGIES}")


# --------------------------------------------------------------------------- #
# Edge-index helpers (undirected edges enumerated as i < j)
# --------------------------------------------------------------------------- #


def edge_to_index(i: int, j: int, p: int) -> int:
    """Linear upper-triangular index of edge ``(i, j)`` with ``i < j``."""
    return i * (p - 1) - i * (i - 1) // 2 + (j - i - 1)


def build_full_to_data_map(p: int, num_dummies: int) -> np.ndarray:
    """Map augmented-graph edge indices to data-edge indices (-1 for dummies)."""
    p_tot = p + num_dummies
    init_index_to_edge(p_tot)
    m_tot = p_tot * (p_tot - 1) // 2
    mapping = np.full(m_tot, -1, dtype=int)
    for k in range(m_tot):
        i, j = index_to_edge(k)
        if j < p:
            mapping[k] = edge_to_index(i, j, p)
    return mapping


def true_edge_mask(true_edges, p: int, m_data: int) -> np.ndarray:
    """Boolean mask over data edges marking the true (active) edges."""
    idx = np.array([edge_to_index(min(i, j), max(i, j), p) for (i, j) in true_edges])
    mask = np.zeros(m_data, dtype=bool)
    if idx.size:
        mask[idx] = True
    return mask


# --------------------------------------------------------------------------- #
# Neighborhood selection (NS)
# --------------------------------------------------------------------------- #


def _ns_single_node(X_minus_j, y, num_dummies, t_grid, rng, normalize):
    """Run nodewise T-LARS for one node over the stopping grid."""
    n, p_minus_1 = X_minus_j.shape
    n_t = len(t_grid)
    sel_matrix = np.zeros((p_minus_1, n_t), dtype=bool)
    dummy_counts = np.zeros(n_t, dtype=int)

    dummies = rng.standard_normal(size=(n, num_dummies))
    XD = np.hstack([X_minus_j, dummies])
    model = TLarsNodewise(XD, y=y.copy(), num_dummies=num_dummies,
                          max_steps=None, lasso=False, normalize=normalize)

    for t_idx, T in enumerate(t_grid):
        if T > num_dummies:
            continue
        model.run(T=T)
        beta = model.beta_path[-1]
        sel_matrix[:, t_idx] = np.abs(beta[:p_minus_1]) > 0
        dummy_counts[t_idx] = int(np.sum(np.abs(beta[p_minus_1:p_minus_1 + num_dummies]) > 0))
    return sel_matrix, dummy_counts


def neighborhood_selection(X, t_grid, K, num_dummies, rng_seed, normalize=True):
    """Dummy-augmented neighborhood selection aggregated over K experiments.

    Returns the per-edge joint relative occurrence (each undirected edge can be
    counted from either endpoint, hence the ``2 * K`` denominator) and the mean
    number of selected dummy edges at each stopping point.
    """
    n, p = X.shape
    n_t = len(t_grid)
    m_data = p * (p - 1) // 2

    edge_counts = np.zeros((n_t, m_data), dtype=np.int32)
    dummy_counts_total = np.zeros(n_t)
    rng_master = np.random.default_rng(rng_seed)

    for k_rep in range(K):
        rng_rep = np.random.default_rng(rng_master.integers(2 ** 30) + k_rep)
        counts_rep = np.zeros((n_t, m_data), dtype=np.int16)

        for j in range(p):
            cols = np.arange(p)[np.arange(p) != j]
            sel_matrix, node_dummy_counts = _ns_single_node(
                X[:, cols], X[:, j], num_dummies, t_grid, rng_rep, normalize
            )
            dummy_counts_total += node_dummy_counts
            for t_idx in range(n_t):
                sel_pos = np.nonzero(sel_matrix[:, t_idx])[0]
                if sel_pos.size == 0:
                    continue
                neighbors = cols[sel_pos]
                lo = np.minimum(neighbors, j).astype(np.int64)
                hi = np.maximum(neighbors, j).astype(np.int64)
                edge_idx = lo * (p - 1) - (lo * (lo - 1) // 2) + (hi - lo - 1)
                counts_rep[t_idx, edge_idx] += 1

        edge_counts += counts_rep

    edge_sel_frac = edge_counts.astype(float) / float(2 * K)
    return edge_sel_frac, dummy_counts_total / K


# --------------------------------------------------------------------------- #
# Metric helpers and per-method evaluation
# --------------------------------------------------------------------------- #


def _classification_metrics(tp, fp, num_true, num_false):
    """Vectorised TPR, FPR and FDR from true/false-positive counts.

    ``tp`` and ``fp`` may be scalars or arrays of the same shape. Divisions by
    zero (no positives, or empty selection) become NaN.
    """
    tp = np.asarray(tp)
    fp = np.asarray(fp)
    num_sel = tp + fp
    with np.errstate(invalid="ignore", divide="ignore"):
        tpr = np.where(num_true > 0, tp / num_true, np.nan)
        fpr = np.where(num_false > 0, fp / num_false, np.nan)
        fdr = np.where(num_sel > 0, fp / num_sel, np.nan)
    return tpr, fpr, fdr


def _run_scpl_path(cfg: SimConfig, Xs: np.ndarray, map_full_to_data: np.ndarray,
                   m_data: int, rng: np.random.Generator) -> np.ndarray:
    """Aggregate the SCPL relative occurrences over ``cfg.K`` random experiments.

    Returns ``Phi_T(i, j) / K`` as an array of shape ``(m_data, n_T_large)``.
    """
    t_grid = cfg.t_grid_large
    phi = np.zeros((m_data, len(t_grid)), dtype=int)

    for _ in range(cfg.K):
        dummies = rng.standard_normal(size=(cfg.n, cfg.num_dummies))
        dummies = (dummies - dummies.mean(axis=0)) / dummies.std(axis=0, ddof=1)
        XD = np.hstack([Xs, dummies])
        model = TLarsGraph(XD, num_dummies=cfg.num_dummies)

        for t_idx, t in enumerate(t_grid):
            model.run(T=t)
            actives = np.array(model.actives, dtype=int)
            data_idx = map_full_to_data[actives]
            data_idx = data_idx[data_idx >= 0]
            if data_idx.size:
                phi[data_idx, t_idx] += 1

    return phi / cfg.K


def _evaluate_scpl(phi_frac: np.ndarray, true_mask: np.ndarray,
                   voting_levels, t_grid_large: np.ndarray) -> dict:
    """Per-voting-level TPR/FPR/FDR plus the dummy-based FDR proxy (DDP)."""
    num_true = int(true_mask.sum())
    num_false = int((~true_mask).sum())

    tpp_levels, fpr_levels, fdr_levels, selected_nulls = [], [], [], []
    num_sel_last = None
    for v in voting_levels:
        selected = phi_frac > v
        num_sel = selected.sum(axis=0)
        tp = (selected & true_mask[:, None]).sum(axis=0)
        fp = num_sel - tp
        tpr, fpr, fdr = _classification_metrics(tp, fp, num_true, num_false)
        tpp_levels.append(tpr)
        fpr_levels.append(fpr)
        fdr_levels.append(fdr)
        selected_nulls.append(fp)
        num_sel_last = num_sel

    # Dummy-based FDR proxy (DDP) at the last voting level: T / (T + |E_hat|).
    t_arr = np.asarray(t_grid_large)
    fdr_est = t_arr / (num_sel_last + t_arr)

    return {
        "TPP_levels":     np.array(tpp_levels),
        "FPR_levels":     np.array(fpr_levels),
        "FDR_levels":     np.array(fdr_levels),
        "selected_nulls": selected_nulls,
        "FDR_est":        fdr_est,
    }


def _evaluate_ns(edge_sel_frac: np.ndarray, ns_dummy_counts: np.ndarray,
                 true_mask: np.ndarray) -> dict:
    """Per-T TPR/FPR/FDR for NS plus the dummy-based FDR proxy."""
    num_true = int(true_mask.sum())
    num_false = int((~true_mask).sum())

    est_masks = edge_sel_frac > NS_VOTING_THRESHOLD                     # (n_T, m_data)
    tp = (est_masks & true_mask[None, :]).sum(axis=1)
    fp = (est_masks & ~true_mask[None, :]).sum(axis=1)
    tpr, fpr, fdr = _classification_metrics(tp, fp, num_true, num_false)

    denom = tp + fp + ns_dummy_counts
    with np.errstate(invalid="ignore", divide="ignore"):
        fdr_est_ns = np.where(denom > 0, ns_dummy_counts / denom, np.nan)

    return {
        "ns_TPP":              tpr.astype(float),
        "ns_FPR":              fpr.astype(float),
        "ns_FDR":              fdr.astype(float),
        "ns_selected_dummies": ns_dummy_counts,
        "ns_selected_nulls":   fp.astype(float),
        "fdr_est_ns":          fdr_est_ns,
    }


# --------------------------------------------------------------------------- #
# One Monte-Carlo replication
# --------------------------------------------------------------------------- #


def _empty_result(cfg: SimConfig, n_voting: int) -> dict:
    return empty_result(n_voting,
                        len(cfg.t_grid_large),
                        len(cfg.t_grid_small),
                        len(cfg.lambdas_glasso))


def run_one_mc(mc_id: int, cfg: SimConfig, map_full_to_data: np.ndarray,
               m_data: int, voting_levels) -> dict:
    """Evaluate all methods on a single synthetic data set."""
    try:
        rng = np.random.default_rng(cfg.seed + mc_id)
        true_edges, _, sigma = generate_graph(cfg, rng)
        X = rng.multivariate_normal(np.zeros(cfg.p), sigma, size=cfg.n)
        Xs = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        true_mask = true_edge_mask(true_edges, cfg.p, m_data)

        # Proposed methods.
        phi_frac = _run_scpl_path(cfg, Xs, map_full_to_data, m_data, rng)
        scpl = _evaluate_scpl(phi_frac, true_mask, voting_levels, cfg.t_grid_large)

        edge_sel_frac, ns_dummy_counts = neighborhood_selection(
            X, cfg.t_grid_small, cfg.K, cfg.num_dummies,
            rng_seed=cfg.seed + mc_id, normalize=True,
        )
        ns = _evaluate_ns(edge_sel_frac, ns_dummy_counts, true_mask)

        # Comparison baseline (delegated).
        glasso = run_glasso_baseline(Xs, true_mask, cfg.p, cfg.n, cfg.lambdas_glasso)

        return {"skipped": False, **scpl, **ns, **glasso}

    except Exception as exc:  # pragma: no cover - defensive guard for long runs
        print(f"[run_one_mc] mc_id={mc_id} skipped: {type(exc).__name__}: {exc}")
        return _empty_result(cfg, len(voting_levels))


# --------------------------------------------------------------------------- #
# Command-line entry point
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topology", choices=TOPOLOGIES, default="pa",
                        help="graph topology to simulate (default: pa)")
    parser.add_argument("--output", default=None,
                        help="output .npz path (default: results/sim_<topology>.npz)")
    parser.add_argument("--n", type=int, default=150, help="number of samples")
    parser.add_argument("--p", type=int, default=150, help="number of variables")
    parser.add_argument("--num-dummies", type=int, default=None,
                        help="number of dummy nodes (default: p)")
    parser.add_argument("--K", type=int, default=20, help="random experiments per run")
    parser.add_argument("--mc", type=int, default=100, help="Monte-Carlo replications")
    parser.add_argument("--seed", type=int, default=4, help="base random seed")
    parser.add_argument("--t-max", type=int, default=5000,
                        help="largest stopping point for the global selector")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="parallel workers for joblib (-1 = all cores)")
    return parser.parse_args()


def _sim_metadata(cfg: SimConfig, voting_levels, n_valid: int) -> dict:
    """Static metadata fields included verbatim in the output archive."""
    return {
        "topology": cfg.topology, "n": cfg.n, "p": cfg.p,
        "num_dummies": cfg.num_dummies, "K": cfg.K, "MC": n_valid,
        "seed": cfg.seed,
        "T_grid_large": cfg.t_grid_large, "T_grid_small": cfg.t_grid_small,
        "voting_levels": np.asarray(voting_levels),
        "lambdas_glasso": cfg.lambdas_glasso,
    }


def main() -> None:
    args = parse_args()
    cfg = SimConfig(
        topology=args.topology, n=args.n, p=args.p,
        num_dummies=args.num_dummies if args.num_dummies is not None else args.p,
        K=args.K, mc=args.mc, seed=args.seed, t_max=args.t_max, n_jobs=args.n_jobs,
    )
    output = args.output or f"results/sim_{cfg.topology}.npz"

    voting_levels = [0.5]
    m_data = cfg.p * (cfg.p - 1) // 2
    map_full_to_data = build_full_to_data_map(cfg.p, cfg.num_dummies)

    raw = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(run_one_mc)(mc_id, cfg, map_full_to_data, m_data, voting_levels)
        for mc_id in range(cfg.mc)
    )
    results = [r for r in raw if not r.get("skipped", False)]
    if not results:
        raise RuntimeError("All Monte-Carlo runs were skipped.")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    aggregate_and_save(results, _sim_metadata(cfg, voting_levels, len(results)), output)


if __name__ == "__main__":
    main()
