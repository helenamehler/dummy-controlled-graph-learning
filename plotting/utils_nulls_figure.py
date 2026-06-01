"""Three-panel V-vs-TPR figure (Fig. 1) for the EUSIPCO 2026 paper."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plotting.plotting_utils import (
    C_GFC, C_GLASSO, C_SCPL, MARKERS, PANEL_TITLES,
    apply_style, extract_bic_glasso_point, extract_means,
    import_r_baseline, save_or_show,
)

GFC_ALPHA_TARGET = 0.1
DEFAULT_MARK_T = (10, 20, 50)


def _build_panel(D, csv):
    e = extract_means(D)
    lambdas = D["lambdas_glasso"]
    bic_fp = extract_bic_glasso_point([e["glasso_bic_best_alphas"]],
                                       [e["glasso_FP_counts"]], lambdas)[0]
    bic_tpp = extract_bic_glasso_point([e["glasso_bic_best_alphas"]],
                                        [e["TPP_glasso_MC"]], lambdas)[0]
    return {
        "x": np.nanmean(np.asarray(D["selected_nulls_MC"])[:, 0, :], axis=0),
        "y": np.nanmean(np.asarray(D["TPP_TLARS_MC"])[:, 0, :], axis=0),
        "T_grid": np.asarray(D["T_grid_large"]),
        "glasso": (e["glasso_FP_counts"].mean(axis=0), e["TPP_glasso_mean"]),
        "baselines": import_r_baseline(csv)["fp"] if csv else None,
        "bic_fp": bic_fp, "bic_tpp": bic_tpp,
    }


def _draw_panel(ax, panel, mark_T, title):
    # SCPL curve.
    ax.plot(panel["x"], panel["y"], lw=1.5, color=C_SCPL, linestyle="--", zorder=10)

    # Glasso lambda-path FP curve.
    if panel["glasso"] is not None:
        gx, gy = np.asarray(panel["glasso"][0]), np.asarray(panel["glasso"][1])
        valid = ~np.isnan(gx) & ~np.isnan(gy)
        if np.any(valid):
            o = np.argsort(gx[valid])
            ax.plot(gx[valid][o], gy[valid][o], lw=1.5, color=C_GLASSO,
                    linestyle="dashdot", alpha=0.7, zorder=4)

    # GFC-L baseline curve + FDR-target marker.
    for meth in panel["baselines"] or []:
        if not any(s in str(meth.get("label", "")).lower() for s in ("gfc", "liu")):
            continue
        x = np.asarray(meth.get("x_fp", meth.get("x")), dtype=float)
        y = np.asarray(meth["y"], dtype=float)
        alphas = np.asarray(meth.get("alpha", np.full_like(x, np.nan)), dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid): continue
        if valid.sum() >= 2:
            o = np.argsort(x[valid])
            ax.plot(x[valid][o], y[valid][o], lw=1.8,
                    color=meth.get("color", C_GFC), zorder=3)
        idxs = np.where(valid & np.isfinite(alphas))[0]
        if idxs.size:
            close = np.isclose(alphas[idxs], GFC_ALPHA_TARGET, atol=1e-8, rtol=1e-6)
            i = (idxs[np.where(close)[0][0]] if close.any()
                 else idxs[np.argmin(np.abs(alphas[idxs] - GFC_ALPHA_TARGET))])
            ax.scatter(x[i], y[i], s=50, marker="p", edgecolors=C_GFC,
                       facecolors="none", linewidths=1.5, zorder=4)

    # BIC tuned point.
    bic_fp, bic_tpp = panel["bic_fp"], panel["bic_tpp"]
    if bic_fp is not None and np.isfinite(bic_fp) and np.isfinite(bic_tpp):
        ax.scatter(float(bic_fp), float(bic_tpp), s=50, marker="v", facecolors="none",
                   edgecolor=C_GLASSO, linewidth=1.5, alpha=0.7, zorder=4)

    # T markers on the SCPL curve.
    t_h, t_l = [], []
    for i, T in enumerate(mark_T):
        idxs = np.where(panel["T_grid"] == T)[0]
        if not idxs.size: continue
        idx = int(idxs[0])
        m = MARKERS[i % len(MARKERS)]
        ax.scatter(panel["x"][idx], panel["y"][idx], s=50, marker=m,
                   edgecolors=C_SCPL, facecolors="none", lw=1.5, zorder=5)
        t_h.append(Line2D([0], [0], marker=m, color="k", markerfacecolor="none", linestyle=""))
        t_l.append(f"T = {T}")

    # Axes.
    ax.set_xlabel("V", fontsize=9); ax.set_ylabel("TPR", fontsize=9)
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    if title: ax.set_title(title, fontsize=9)

    # Legend handles (per figure; populated from any panel).
    handles = [Line2D([0], [0], color=C_SCPL, linestyle="--", lw=2.2)]
    labels = ["SCPL (Proposed)"]
    if panel["glasso"] is not None:
        handles.append(Line2D([0], [0], color=C_GLASSO, lw=2.0, linestyle="dashdot", alpha=0.7))
        labels.append("Glasso")
    handles.append(Line2D([0], [0], color=C_GFC, lw=1.8, linestyle="-"))
    labels.append("GFC-L")
    handles += t_h; labels += t_l
    if bic_fp is not None:
        handles.append(Line2D([0], [0], marker="v", markeredgecolor=C_GLASSO,
                              markerfacecolor="none", linestyle="", alpha=0.7))
        labels.append("BIC")
    handles.append(Line2D([0], [0], marker="p", color=C_GFC, markerfacecolor="none", linestyle=""))
    labels.append("FDR Target = 0.1")
    return handles, labels


def plot_nulls_figure(npz_paths, baseline_csvs=None, save_path=None,
                      save_format="pgf", mark_T=DEFAULT_MARK_T,
                      xlim=(0, 150), figsize=(10.5, 2.2)):
    """Render the three-panel V-vs-TPR figure (Fig. 1)."""
    apply_style(save_format, figsize)
    Ds = [np.load(p) for p in npz_paths]
    csvs = baseline_csvs or [None, None, None]
    panels = [_build_panel(D, c) for D, c in zip(Ds, csvs)]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=False)
    handles = labels = []
    for ax, panel, title in zip(axes, panels, PANEL_TITLES):
        handles, labels = _draw_panel(ax, panel, mark_T, title)
        ax.set_xlim(xlim)

    fig.canvas.draw()
    fig.add_artist(fig.legend(handles, labels, loc="upper center",
                              bbox_to_anchor=(0.5, 0.98), ncol=max(1, len(handles)),
                              frameon=False, fontsize=9))
    fig.subplots_adjust(top=0.92)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    save_or_show(fig, save_path)
