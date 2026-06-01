# Dummy-Controlled Model Selection for High-Dimensional Gaussian Graph Learning

Reference implementation for the paper

> H. Mehler, T. Koka, M. Muma,
> *"Dummy-Controlled Model Selection for High-Dimensional Gaussian Graph Learning"*,
> EUSIPCO 2026.

Gaussian graphical models (GGMs) encode conditional dependencies in multivariate
data. This repository implements a dummy-augmented randomized selection framework
that uses synthetic null variables (*dummies*) as observable references along the
edge-selection path. The observed dummy selections yield simple, intuitive
stopping criteria that act as proxies for

* the number of false selections (via the number of selected dummy edges),
* the false positive rate (via the **dummy selection rate**, DSR), and
* the false discovery rate (via the **dummy discovery proportion**, DDP).

Two selection methodologies are provided:

* **NS** — dummy-augmented *neighborhood selection*: one forward-selection
  regression per node ([`dummy_ggm.tlars.TLarsNodewise`](src/dummy_ggm/tlars.py)).
* **SCPL** — a global, *symmetry-constrained pseudo-likelihood* formulation that
  performs selection directly at the level of undirected edges
  ([`dummy_ggm.graph_tlars.TLarsGraph`](src/dummy_ggm/graph_tlars.py)).

Both build on the Terminating-Random Experiments (T-Rex) idea: forward selection
on a dummy-augmented design, terminated once `T` dummy variables/edges enter the
active set.

## Repository layout

```
src/dummy_ggm/        Importable package (the proposed methods)
  tlars.py             Nodewise T-LARS (neighborhood selection)
  graph_tlars.py       Global SCPL T-LARS (edge-level selection)
  graph_generators.py           Synthetic GGM generators (ER, PA, small-world)
  _graph_utils.pyx    Cython kernels for the edge-level path
  _helpers.pyx        Cython kernels for the nodewise path
experiments/
  run_simulation.py   Monte-Carlo driver -> writes a results .npz
plotting/
  plotting_utils.py   Shared figure helpers
  make_figures.py     CLI to render Figures 1-3
baselines_in_R/       R baseline results (BH, GFC-L) used in the figures
```

## Installation

The package contains two Cython extension modules, so a C/C++ compiler is
required. With the build dependencies available (`numpy`, `Cython`,
`setuptools`):

```bash
pip install -e .            # builds the Cython extensions and installs the package
pip install -e ".[plotting]"  # additionally installs matplotlib / pandas / scienceplots
```

If you prefer not to use build isolation (e.g. dependencies are already
installed), build the extensions in place instead:

```bash
python setup.py build_ext --inplace
export PYTHONPATH=src        # so `import dummy_ggm` finds the package
```

## Reproducing the experiments

Each figure is a row of three panels (ER, preferential-attachment, small-world).
First generate one simulation archive per topology, then render the figures.

```bash
# 1) Monte-Carlo simulations (paper settings: p = n = 150, 100 replications).
python experiments/run_simulation.py --topology er --output results/sim_er.npz
python experiments/run_simulation.py --topology pa --output results/sim_pa.npz
python experiments/run_simulation.py --topology sw --output results/sim_sw.npz

# 2) Figures (PGF for the paper; use --format png for a quick LaTeX-free preview).
python plotting/make_figures.py --figure all \
    --data results/sim_er.npz results/sim_pa.npz results/sim_sw.npz
```

`run_simulation.py --help` lists all options (`--n`, `--p`, `--K`, `--mc`,
`--t-max`, `--seed`, `--n-jobs`). The full paper run (100 Monte-Carlo
replications with `p = 150`) is computationally heavy; reduce `--mc`, `--p` and
`--t-max` for a quick smoke test.

## Baselines

The graphical-lasso baseline (lambda path, plus CV / AIC / BIC tuning) is
computed inside `run_simulation.py` via scikit-learn. The BH and GFC-L baselines
are produced by external R code and are provided as pre-aggregated CSV files in
[`data/baselines/`](data/baselines); the plotting scripts read them directly and
simply omit those curves if the files are absent.

## Notes on reproducibility

* The selectors are deterministic given the random seed; simulation results are
  averaged over Monte-Carlo replications. Regenerated results are statistically
  equivalent to the published figures but not bit-identical to any particular
  earlier run.
* PGF output requires a working LaTeX installation (`pdflatex`). Use
  `--format png` to preview figures without LaTeX.

## License

Released under the MIT License (see [`LICENSE`](LICENSE)).
