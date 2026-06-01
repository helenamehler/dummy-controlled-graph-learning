"""Three-panel ROC figures: TPR vs FPR (Fig. 2) and TPR vs FDR (Fig. 3)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plotting.plotting_utils import (
    C_GLASSO, C_NS, C_SCPL, MARKERS, PANEL_TITLES,
    apply_style, compute_estimator_points, extract_means,
    import_r_baseline, prepend_zero, save_or_show,
)

# Z-order layers (higher = drawn on top).
Z = {"chance": 1, "other": 2, "glasso": 3, "ns": 4, "scpl": 10}
FPR_TARGETS = (0.03, 0.05, 0.1)
FDR_TARGETS = (0.10, 0.2, 0.3)
_AXES = {
    "fpr": dict(label="FPR", x_lim=(-0.01, 0.5), y_lim=(-0.025, 1),
                targets=FPR_TARGETS, proxy="DSR"),
    "fdr": dict(label="FDR", x_lim=(-0.03, 0.8), y_lim=(-0.05, 0.85),
                targets=FDR_TARGETS, proxy="DDP"),
}


def _build_panel(D, metric, csv):
    e = extract_means(D)
    est = compute_estimator_points(D, FPR_TARGETS, FDR_TARGETS)
    fpr = metric == "fpr"
    pick = lambda a, b: e[a] if fpr else e[b]
    return {
        "x": prepend_zero(pick("FPR_mean", "FDR_mean")),
        "y": prepend_zero(e["TPP_mean"]),
        "pl": est["pl_fpr" if fpr else "pl_fdr"],
        "ns_pts": est["ns_fpr" if fpr else "ns_fdr"],
        "glasso": (pick("FPR_glasso_mean", "FDR_glasso_mean"), e["TPP_glasso_mean"]),
        "cv":  pick("glasso_cv_fpr_point",  "glasso_cv_fdr_point"),
        "aic": pick("glasso_aic_fpr_point", "glasso_aic_fdr_point"),
        "bic": pick("glasso_bic_fpr_point", "glasso_bic_fdr_point"),
        "ns_curve": (prepend_zero(pick("ns_FPR", "ns_FDR")), prepend_zero(e["ns_TPP"])),
        "baselines": import_r_baseline(csv)[metric] if csv else None,
    }


def _draw_proxy_points(ax, points, color, size, zorder):
    """Ringed markers for DSR / DDP proxy stopping points."""
    if points is None: return
    _, mx, my = points
    for i_thr in range(mx.shape[0]):
        for lvl in range(mx.shape[1]):
            x, y = mx[i_thr, lvl], my[i_thr, lvl]
            if not (np.isnan(x) or np.isnan(y)):
                ax.scatter(x, y, s=size, marker=MARKERS[i_thr], edgecolor=color,
                           facecolor="none", linewidths=1.5, zorder=zorder)


def _draw_glasso(ax, curve, cv, aic, bic):
    """Graphical-lasso lambda path plus its CV / AIC / BIC tuned points."""
    if curve is not None:
        gx, gy = curve
        o = np.argsort(gx)
        ax.plot(gx[o], gy[o], lw=1.5, color=C_GLASSO, alpha=0.7,
                linestyle="dashdot", zorder=Z["glasso"])
    for pt, m, fc in ((cv, "x", None), (aic, "^", "none"), (bic, "v", "none")):
        if pt is None: continue
        kw = dict(s=50, marker=m, alpha=0.7, linewidths=1.5, zorder=Z["glasso"] + 2)
        kw.update({"color": C_GLASSO} if fc is None
                  else {"facecolors": fc, "edgecolors": C_GLASSO})
        ax.scatter(*pt, **kw)


def _draw_baselines(ax, baselines, metric, targets, alpha_target=0.1):
    """BH / GFC-L curves and the per-metric alpha-target markers."""
    handles, labels = [], []
    for meth in baselines or []:
        x = np.asarray(meth.get("x", []))
        y = np.asarray(meth["y"])
        alphas = np.asarray(meth.get("alpha", np.full_like(x, np.nan)), dtype=float)
        valid = ~np.isnan(x) & ~np.isnan(y)
        if not np.any(valid): continue
        color = meth.get("color")
        if valid.sum() >= 2:
            ax.plot(x[valid], y[valid], lw=1.5, color=color,
                    linestyle=meth.get("linestyle", "-"), zorder=Z["other"])

        label = str(meth.get("label", "")).lower()
        if metric == "fpr" and any(s in label for s in ("gfc", "liu", "bh")):
            idxs = np.where(valid & np.isfinite(alphas))[0]
            if idxs.size:
                close = np.isclose(alphas[idxs], alpha_target, atol=1e-8, rtol=1e-6)
                i = (idxs[np.where(close)[0][0]] if close.any()
                     else idxs[np.argmin(np.abs(alphas[idxs] - alpha_target))])
                if np.isfinite(x[i]) and np.isfinite(y[i]):
                    ax.scatter(x[i], y[i], s=50, marker="p", edgecolors=color or "k",
                               facecolors="none", linewidths=1.5, zorder=Z["other"] + 2)
        elif metric == "fdr":
            for i_thr, thr in enumerate(targets or ()):
                close = valid & np.isfinite(alphas) & np.isclose(alphas, float(thr), atol=1e-8, rtol=1e-6)
                for i in np.where(close)[0]:
                    if np.isfinite(x[i]) and np.isfinite(y[i]):
                        ax.scatter(x[i], y[i], s=50, marker=MARKERS[i_thr % len(MARKERS)],
                                   edgecolors=color or "k", facecolors="none",
                                   linewidths=1.5, zorder=Z["other"] + 0.5)

        handles.append(Line2D([0], [0], color=color or "k", lw=1.8,
                              linestyle=meth.get("linestyle", "-")))
        labels.append(meth.get("label", "Other"))
    return handles, labels


def _draw_panel(ax, panel, voting_levels, metric, cfg, title):
    if metric == "fpr":
        xs = np.linspace(0, 1, 200)
        ax.plot(xs, xs, color="gray", lw=1.0, alpha=0.6, zorder=Z["chance"])
    for lvl in range(len(voting_levels)):
        ax.plot(panel["x"][lvl], panel["y"][lvl], lw=1.5,
                color=C_SCPL, linestyle="--", zorder=Z["scpl"])
    _draw_proxy_points(ax, panel["pl"], C_SCPL, 50, Z["scpl"])
    _draw_glasso(ax, panel["glasso"], panel["cv"], panel["aic"], panel["bic"])
    nx, ny = panel["ns_curve"]
    ax.plot(nx, ny, lw=1.5, color=C_NS, linestyle="--", zorder=Z["ns"])
    _draw_proxy_points(ax, panel["ns_pts"], C_NS, 70, Z["ns"])
    mh, ml = _draw_baselines(ax, panel["baselines"], metric, cfg["targets"])
    ax.set_xlabel(cfg["label"], fontsize=8); ax.set_ylabel("TPR", fontsize=8)
    ax.set_xlim(cfg["x_lim"]); ax.set_ylim(cfg["y_lim"])
    if title: ax.set_title(title, fontsize=9, pad=4)
    ax.grid(True, alpha=0.25); ax.tick_params(axis="both", which="major", labelsize=7)
    return mh, ml


def _legend_handles(method_h, method_l, targets, proxy, metric):
    """Top: SCPL/NS + proxy markers. Bottom: baselines + glasso family (+FDR target)."""
    top_h = [Line2D([0], [0], color=C_SCPL, lw=2, linestyle="--"),
             Line2D([0], [0], color=C_NS,   lw=2, linestyle="--")]
    top_l = ["SCPL (Proposed)", "NS"]
    for i, thr in enumerate(targets or ()):
        top_h.append(Line2D([0], [0], marker=MARKERS[i], color="k",
                            markerfacecolor="none", markersize=7, linestyle=""))
        top_l.append(f"{proxy} = {thr:g}")
    bot_h = list(method_h) + [
        Line2D([0], [0], color=C_GLASSO, lw=2, linestyle="dashdot", alpha=0.7),
        Line2D([0], [0], color=C_GLASSO, marker="x", linestyle="", markersize=9, alpha=0.7),
        Line2D([0], [0], linestyle="", marker="^", markersize=8, markeredgewidth=1.2,
               markeredgecolor=C_GLASSO, markerfacecolor="none", alpha=0.7),
        Line2D([0], [0], linestyle="", marker="v", markersize=8, markeredgewidth=1.2,
               markeredgecolor=C_GLASSO, markerfacecolor="none", alpha=0.7),
    ]
    bot_l = list(method_l) + ["Glasso", "CV", "AIC", "BIC"]
    if metric == "fpr":
        bot_h.append(Line2D([0], [0], linestyle="", marker="p", markersize=8,
                            markeredgewidth=1.2, markeredgecolor="k", markerfacecolor="none"))
        bot_l.append("FDR Target = 0.1")
    return top_h, top_l, bot_h, bot_l


def plot_roc_figure(npz_paths, metric, baseline_csvs=None, save_path=None,
                    save_format="pgf", figsize=(10.5, 2.2)):
    """Render the three-panel ROC figure (``metric`` in {'fpr', 'fdr'})."""
    if metric not in _AXES:
        raise ValueError(f"metric must be 'fpr' or 'fdr', got {metric!r}")
    apply_style(save_format, figsize)
    Ds = [np.load(p) for p in npz_paths]
    csvs = baseline_csvs or [None, None, None]
    panels = [_build_panel(D, metric, c) for D, c in zip(Ds, csvs)]
    cfg, voting = _AXES[metric], Ds[0]["voting_levels"]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=False)
    mh, ml = [], []
    for ax, panel, title in zip(axes, panels, PANEL_TITLES):
        mh, ml = _draw_panel(ax, panel, voting, metric, cfg, title)

    top_h, top_l, bot_h, bot_l = _legend_handles(mh, ml, cfg["targets"], cfg["proxy"], metric)
    fig.add_artist(fig.legend(top_h, top_l, loc="upper center",
                              bbox_to_anchor=(0.5, 0.96), ncol=max(1, len(top_l)),
                              frameon=False, fontsize=9))
    fig.canvas.draw()
    fig.subplots_adjust(top=0.72)
    fig.legend(bot_h, bot_l, loc="upper center", bbox_to_anchor=(0.5, 0.89),
               ncol=max(1, len(bot_l)), frameon=False, fontsize=9)
    save_or_show(fig, save_path)
