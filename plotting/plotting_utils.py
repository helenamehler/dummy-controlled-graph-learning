"""Shared helpers for the EUSIPCO 2026 paper figures."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import scienceplots  # noqa: F401
    _HAS_SCIENCE = True
except ImportError:
    _HAS_SCIENCE = False

PANEL_TITLES = ["ER Graph", "Preferential Attachment Graph", "Small-World Graph"]
MARKERS = ["o", "s", "D", "v", "^", "P", "*"]
C_SCPL, C_NS, C_GLASSO, C_GFC, C_BH = "tab:red", "tab:orange", "tab:green", "tab:cyan", "tab:blue"

_PGF_RCPARAMS = {
    "pgf.texsystem": "pdflatex", "text.usetex": True, "pgf.rcfonts": False,
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 7,
    "legend.fontsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
}


def apply_style(save_format, figsize):
    plt.style.use(["science", "grid"] if _HAS_SCIENCE else ["default"])
    if save_format == "pgf":
        matplotlib.use("pgf")
        matplotlib.rcParams.update({**_PGF_RCPARAMS, "figure.figsize": figsize})


def save_or_show(fig, save_path):
    if save_path is None:
        plt.show()
        return
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved figure -> {save_path}")


def prepend_zero(a):
    """Prepend a leading zero (origin) along the stopping-point axis."""
    a = np.asarray(a)
    if a.ndim == 1: return np.r_[0, a]
    if a.ndim == 2: return np.c_[np.zeros((a.shape[0], 1)), a]
    if a.ndim == 3: return np.concatenate([np.zeros((a.shape[0], a.shape[1], 1)), a], axis=2)
    return a


def _ensure_3d(arr):
    if arr is None: return None
    a = np.asarray(arr)
    if a.ndim == 3: return a
    if a.ndim == 2: return a[:, np.newaxis, :]
    if a.ndim == 1: return a[np.newaxis, np.newaxis, :]
    raise ValueError(f"unexpected array ndim={a.ndim}")


def _mean_point(x, y):
    if x is None or y is None: return None
    x, y = np.asarray(x), np.asarray(y)
    valid = ~np.isnan(x) & ~np.isnan(y)
    return (float(np.nanmean(x[valid])), float(np.nanmean(y[valid]))) if np.any(valid) else None


def extract_means(D):
    """Mean curves and tuned-baseline points from one simulation archive."""
    return {
        "TPP_mean": D["TPP_TLARS_MC"].mean(axis=0),
        "FPR_mean": D["FPR_TLARS_MC"].mean(axis=0),
        "FDR_mean": D["FDR_TLARS_MC"].mean(axis=0),
        "TPP_glasso_mean": D["TPP_glasso_mean"],
        "FPR_glasso_mean": D["FPR_glasso_mean"],
        "FDR_glasso_mean": D["FDR_glasso_mean"],
        "TPP_glasso_MC": D["TPP_glasso_MC"],
        "ns_TPP": D["TPP_ns_mean"], "ns_FPR": D["FPR_ns_mean"], "ns_FDR": D["FDR_ns_mean"],
        "glasso_cv_fpr_point": (D["glasso_cv_best_fprs"].mean(axis=0), D["glasso_cv_best_tpps"].mean(axis=0)),
        "glasso_cv_fdr_point": (D["glasso_cv_best_fdrs"].mean(axis=0), D["glasso_cv_best_tpps"].mean(axis=0)),
        "glasso_aic_fpr_point": _mean_point(D["glasso_aic_best_fprs"], D["glasso_aic_best_tpps"]),
        "glasso_bic_fpr_point": _mean_point(D["glasso_bic_best_fprs"], D["glasso_bic_best_tpps"]),
        "glasso_aic_fdr_point": _mean_point(D["glasso_aic_best_fdrs"], D["glasso_aic_best_tpps"]),
        "glasso_bic_fdr_point": _mean_point(D["glasso_bic_best_fdrs"], D["glasso_bic_best_tpps"]),
        "glasso_FP_counts": D["glasso_FP_counts"],
        "glasso_bic_best_alphas": D["glasso_bic_best_alphas"],
    }


def find_and_collect(estimator, thresholds, true_arr, tpp_arr):
    """For each threshold, mean (metric, TPR) at the last T where estimator <= threshold."""
    thr_list = np.asarray(list(thresholds))
    true_arr, tpp_arr = _ensure_3d(true_arr), _ensure_3d(tpp_arr)
    n_mc, n_voting, n_t_true = true_arr.shape

    est = np.asarray(estimator)
    if est.ndim == 1:
        n_t_est = est.shape[0]
    elif est.ndim == 2:
        n_t_est = est.shape[1]
        if est.shape[0] == 1: est = np.repeat(est, n_mc, axis=0)
    else:
        n_t_est = est.shape[2]
    n_t = min(n_t_est, n_t_true)

    def last_leq(x, thr):
        m = x <= thr
        return int(np.where(m)[0][-1]) if m.any() else -1

    t_idx = np.full((len(thr_list), n_mc, n_voting), -1, dtype=int)
    for i_thr, thr in enumerate(thr_list):
        if est.ndim == 1:
            t = last_leq(est[:n_t], thr)
            if t >= 0: t_idx[i_thr, :, :] = t
        elif est.ndim == 2:
            t = last_leq(np.nanmean(est[:, :n_t], axis=0), thr)
            if t >= 0: t_idx[i_thr, :, :] = t
        else:
            for mc in range(n_mc):
                for lvl in range(n_voting):
                    t_idx[i_thr, mc, lvl] = last_leq(est[mc, lvl, :n_t], thr)

    mean_x = np.full((len(thr_list), n_voting), np.nan)
    mean_y = np.full((len(thr_list), n_voting), np.nan)
    for i_thr in range(len(thr_list)):
        for lvl in range(n_voting):
            xs, ys = [], []
            for mc in range(n_mc):
                t = t_idx[i_thr, mc, lvl]
                if 0 <= t < n_t_true:
                    xs.append(true_arr[mc, lvl, t]); ys.append(tpp_arr[mc, lvl, t])
            if xs:
                mean_x[i_thr, lvl] = np.nanmean(xs)
                mean_y[i_thr, lvl] = np.nanmean(ys)
    return mean_x, mean_y


def compute_estimator_points(D, fpr_targets, fdr_targets):
    """DSR / DDP proxy stopping points for SCPL and NS."""
    def add0(key):
        return prepend_zero(D[key]) if key in D.files else None

    p, num_dummies = D["p"], D["num_dummies"]
    n_dummy_edges = p * num_dummies + num_dummies * (num_dummies - 1) / 2
    dsr_pl = add0("T_grid_large") / n_dummy_edges
    dsr_ns = (prepend_zero(np.nanmean(D["ns_selected_dummies"], axis=0) / (p * num_dummies))
              if "ns_selected_dummies" in D.files else None)

    def safe_find(est, targets, true_arr, tpp_arr):
        if est is None or true_arr is None or tpp_arr is None: return None
        try:
            mx, my = find_and_collect(est, targets, true_arr, tpp_arr)
            return targets, mx, my
        except Exception:
            return None

    return {
        "pl_fpr": safe_find(dsr_pl, fpr_targets, add0("FPR_TLARS_MC"), add0("TPP_TLARS_MC")),
        "ns_fpr": safe_find(dsr_ns, fpr_targets, add0("FPR_ns_MC"), add0("TPP_ns_MC")),
        "pl_fdr": safe_find(add0("FDR_est"), fdr_targets, add0("FDR_TLARS_MC"), add0("TPP_TLARS_MC")),
        "ns_fdr": safe_find(add0("fdr_est_ns"), fdr_targets, add0("FDR_ns_MC"), add0("TPP_ns_MC")),
    }


def extract_bic_glasso_point(bic_alphas_list, value_list, lambdas):
    """Average graphical-lasso value at the BIC-selected lambda per graph."""
    lambdas = np.asarray(lambdas)
    n_graphs, n_mc = len(bic_alphas_list), int(np.asarray(value_list[0]).shape[0])
    out = np.full((n_graphs, n_mc), np.nan)
    for gi in range(n_graphs):
        alphas, values = np.asarray(bic_alphas_list[gi]), np.asarray(value_list[gi])
        for mc in range(n_mc):
            out[gi, mc] = values[mc, int(np.argmin(np.abs(lambdas - alphas[mc])))]
    return np.nanmean(out, axis=1)


def import_r_baseline(csv_path):
    """BH / GFC-L curves from an R results CSV; returns ``{"fpr","fdr","fp"}`` lists."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[plotting] baseline CSV not found, skipping: {csv_path}")
        return {"fpr": [], "fdr": [], "fp": []}
    color_map = {"BH": C_BH, "Liu": C_GFC}
    out = {"fpr": [], "fdr": [], "fp": []}
    for mode in df["mode"].unique():
        label = "GFC-L" if str(mode).lower() == "liu" else mode
        dfm = df[df["mode"] == mode].sort_values("alpha")
        common = {"label": label, "y": dfm["tpp_mean"].values,
                  "color": color_map.get(mode), "alpha": dfm["alpha"].values, "linestyle": "-"}
        out["fpr"].append({**common, "x": dfm["fpr_mean"].values})
        out["fdr"].append({**common, "x": dfm["fdp_mean"].values})
        out["fp"].append({**common, "x_fp": dfm["fp_mean"].values})
    return out


def load_baselines(csv_paths, metric):
    if not csv_paths: return [None, None, None]
    return [import_r_baseline(p)[metric] if p else None for p in csv_paths]
