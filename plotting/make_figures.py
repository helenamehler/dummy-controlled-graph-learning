"""Reproduce the EUSIPCO 2026 figures from simulation archives.

The three figures expect one ``.npz`` archive per topology, in the order
ER, PA, SW (as produced by ``experiments/run_simulation.py``). The R baseline
curves (BH, GFC-L) are read from CSV files; if they are missing the baseline
curves are simply omitted.

Examples
--------
    # All three figures as PGF (requires a LaTeX installation):
    python plotting/make_figures.py --figure all \
        --data results/sim_er.npz results/sim_pa.npz results/sim_sw.npz

    # Quick PNG preview of the FPR figure (no LaTeX needed):
    python plotting/make_figures.py --figure fpr --format png \
        --data results/sim_er.npz results/sim_pa.npz results/sim_sw.npz
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow `python plotting/make_figures.py` (script invocation) by putting the
# repo root on sys.path so the ``plotting`` package becomes importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plotting.utils_nulls_figure import plot_nulls_figure
from plotting.utils_roc_figure import plot_roc_figure

DEFAULT_BASELINES = [
    "baselines_in_R/r_results_sim12_er.csv",
    "baselines_in_R/r_results_sim12_pa.csv",
    "baselines_in_R/r_results_sim12_sw.csv",
]

FIGURES = {
    "nulls": ("tpr_vs_selected_nulls", "Fig. 1: selected null edges V vs TPR"),
    "fpr": ("roc_fpr", "Fig. 2: TPR vs FPR"),
    "fdr": ("roc_fdr", "Fig. 3: TPR vs FDR"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figure", choices=[*FIGURES, "all"], default="all",
                        help="which figure(s) to produce")
    parser.add_argument("--data", nargs=3, metavar=("ER", "PA", "SW"),
                        default=["results/sim_er.npz", "results/sim_pa.npz", "results/sim_sw.npz"],
                        help="simulation archives for the ER, PA and SW topologies")
    parser.add_argument("--baselines", nargs=3, metavar=("ER", "PA", "SW"),
                        default=DEFAULT_BASELINES, help="R baseline CSVs (ER, PA, SW)")
    parser.add_argument("--format", choices=["pgf", "png"], default="png",
                        help="output format (png needs no LaTeX)")
    parser.add_argument("--outdir", default="figures", help="output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    selected = list(FIGURES) if args.figure == "all" else [args.figure]

    for fig in selected:
        stem, description = FIGURES[fig]
        save_path = os.path.join(args.outdir, f"{stem}.{args.format}")
        print(f"Rendering {description} ...")
        if fig == "nulls":
            plot_nulls_figure(args.data, baseline_csvs=args.baselines,
                              save_path=save_path, save_format=args.format)
        else:
            plot_roc_figure(args.data, metric=fig, baseline_csvs=args.baselines,
                            save_path=save_path, save_format=args.format)


if __name__ == "__main__":
    main()
