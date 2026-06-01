"""npz layout and persistence for the EUSIPCO 2026 simulation driver.

A single :data:`SCHEMA` list declares every non-metadata field of the output
archive as a ``(npz_key, source_field, transform)`` triple. This schema drives:

* :func:`empty_result` -- the NaN-filled placeholder returned for skipped
  Monte-Carlo runs (so the aggregation can still stack across replications);
* :func:`aggregate_and_save` -- stacking the per-run result dicts and writing
  the compressed ``.npz`` consumed by the plotting scripts.

Adding a new metric or method is therefore a localised change to the schema
(plus the matching per-MC field-shape entry in :func:`_field_shapes`).
"""

from __future__ import annotations

import numpy as np

from glasso_baseline import TunedPoint

__all__ = ["SCHEMA", "aggregate_and_save", "empty_result"]


# Transform sentinels used in SCHEMA. ``("tuned", <field>)`` extracts one
# NamedTuple component (TunedPoint.alpha / .tpp / .fpr / .fdr) across MC runs.
_STACK = "stack"
_MEAN = "mean"
_TUNING_CRITERIA = ("cv", "aic", "bic")
_TUNED_COMPONENTS = (("alpha", "alphas"), ("tpp", "tpps"),
                     ("fpr", "fprs"), ("fdr", "fdrs"))


def _field_shapes(n_voting: int, n_large: int, n_small: int,
                  n_lam: int) -> dict[str, tuple[int, ...]]:
    """Per-MC field name -> array shape, used to build empty (skipped) results."""
    scpl_shape = (n_voting, n_large)
    shapes: dict[str, tuple[int, ...]] = {
        "TPP_levels": scpl_shape, "FPR_levels": scpl_shape,
        "FDR_levels": scpl_shape, "selected_nulls": scpl_shape,
        "FDR_est": (n_large,),
    }
    for name in ("glasso_TPP", "glasso_FPR", "glasso_FDR", "glasso_FP_counts"):
        shapes[name] = (n_lam,)
    for name in ("ns_TPP", "ns_FPR", "ns_FDR",
                 "ns_selected_dummies", "ns_selected_nulls", "fdr_est_ns"):
        shapes[name] = (n_small,)
    return shapes


def _build_schema() -> list[tuple[str, str, object]]:
    """All non-metadata npz fields as ``(npz_key, src, transform)`` triples."""
    schema: list[tuple[str, str, object]] = []
    # SCPL selector (asymmetric source naming, MC only).
    for metric in ("TPP", "FPR", "FDR"):
        schema.append((f"{metric}_TLARS_MC", f"{metric}_levels", _STACK))
    schema += [
        ("selected_nulls_MC", "selected_nulls", _STACK),
        ("FDR_est",           "FDR_est",        _STACK),
    ]
    # Glasso and NS share the same "MC + axis-0 mean" layout for TPP/FPR/FDR.
    for method in ("glasso", "ns"):
        for metric in ("TPP", "FPR", "FDR"):
            src = f"{method}_{metric}"
            schema.append((f"{metric}_{method}_MC",   src, _STACK))
            schema.append((f"{metric}_{method}_mean", src, _MEAN))
    # Method-specific extras (one stacked field each).
    for src in ("glasso_FP_counts", "ns_selected_dummies",
                "ns_selected_nulls", "fdr_est_ns"):
        schema.append((src, src, _STACK))
    # Graphical-lasso tuning: 3 criteria x 4 TunedPoint components.
    for crit in _TUNING_CRITERIA:
        for tp_field, npz_field in _TUNED_COMPONENTS:
            schema.append((f"glasso_{crit}_best_{npz_field}",
                           f"glasso_{crit}", ("tuned", tp_field)))
    return schema


SCHEMA: list[tuple[str, str, object]] = _build_schema()


def empty_result(n_voting: int, n_large: int, n_small: int, n_lam: int) -> dict:
    """NaN-filled per-MC result used when a Monte-Carlo run is skipped."""
    out: dict = {
        name: np.full(shape, np.nan)
        for name, shape in _field_shapes(n_voting, n_large, n_small, n_lam).items()
    }
    nan_tp = TunedPoint.nan()
    for crit in _TUNING_CRITERIA:
        out[f"glasso_{crit}"] = nan_tp
    out["skipped"] = True
    return out


def _apply_transform(results, src: str, transform):
    """Stack one per-MC field across runs according to ``transform``."""
    if transform == _STACK:
        return np.array([r[src] for r in results])
    if transform == _MEAN:
        return np.array([r[src] for r in results]).mean(axis=0)
    kind, tp_field = transform  # ("tuned", <NamedTuple field name>)
    if kind == "tuned":
        return np.array([getattr(r[src], tp_field) for r in results])
    raise ValueError(f"unknown transform: {transform!r}")


def aggregate_and_save(results, metadata: dict, output: str) -> None:
    """Stack per-run results next to ``metadata`` and write the npz archive."""
    payload: dict = dict(metadata)
    for npz_key, src, transform in SCHEMA:
        payload[npz_key] = _apply_transform(results, src, transform)
    np.savez_compressed(output, **payload)
    print(f"Saved {len(results)} Monte-Carlo runs -> {output}")
