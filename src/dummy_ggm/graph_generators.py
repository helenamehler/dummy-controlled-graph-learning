"""Synthetic Gaussian graphical models with known ground-truth structure.

Each generator returns the triple ``(edges, Omega, Sigma)``:

* ``edges`` -- list of undirected edges ``(i, j)`` with ``i < j``;
* ``Omega`` -- the precision matrix (inverse covariance);
* ``Sigma`` -- the covariance matrix, ready to sample from via
  :meth:`numpy.random.Generator.multivariate_normal`.

The construction follows the EUSIPCO 2026 paper (Sec. IV): off-diagonal entries
of ``Omega`` on the graph edges are drawn from
``U([-0.8, -0.2] U [0.2, 0.8])`` and ``Omega`` is then shifted by
``(|lambda_min| + add_min_eigen) * I`` to guarantee positive definiteness.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

__all__ = [
    "generate_er_graph",
    "generate_pa_graph",
    "generate_small_world_graph",
]


def _sample_sign(signs: str, rng: np.random.Generator) -> int:
    """Sign of an edge weight: ``+1``, ``-1`` or random per ``signs``."""
    if signs == "Positive":
        return 1
    if signs == "Negative":
        return -1
    if signs == "Random":
        return int(rng.choice([1, -1]))
    raise ValueError("signs must be one of {'Positive', 'Negative', 'Random'}")


def _shift_to_positive_definite(omega: np.ndarray, add_min_eigen: float) -> np.ndarray:
    """Shift ``omega`` by ``(|lambda_min| + add_min_eigen) * I``.

    The shift is applied unconditionally (even when ``omega`` is already
    positive definite), matching the reference R implementation.
    """
    min_eig = float(np.min(np.linalg.eigvalsh(omega)))
    shift = abs(min_eig) + float(add_min_eigen)
    return omega + shift * np.eye(omega.shape[0])


def _precision_from_edges(
    p: int,
    edges: list[tuple[int, int]],
    value_range: tuple[float, float],
    add_min_eigen: float,
    signs: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(Omega, Sigma)`` from an edge list with random weights."""
    omega = np.eye(p)
    min_val, max_val = float(value_range[0]), float(value_range[1])
    for i, j in edges:
        weight = rng.uniform(min_val, max_val) * _sample_sign(signs, rng)
        omega[i, j] = omega[j, i] = weight
    omega = _shift_to_positive_definite(omega, add_min_eigen)
    sigma = np.linalg.inv(omega)
    return omega, sigma


def generate_er_graph(
    p: int,
    prob: float,
    value_range: tuple[float, float] = (0.5, 1.0),
    add_min_eigen: float = 0.1,
    signs: str = "Random",
    rng: np.random.Generator | None = None,
):
    """Erdos-Renyi graph with independent edge probability ``prob``."""
    if rng is None:
        rng = np.random.default_rng()
    seed = int(rng.integers(0, 1_000_000_000))
    g = nx.erdos_renyi_graph(n=p, p=prob, seed=seed, directed=False)
    edges = list(g.edges())
    omega, sigma = _precision_from_edges(p, edges, value_range, add_min_eigen, signs, rng)
    return edges, omega, sigma


def generate_small_world_graph(
    p: int,
    prob: float,
    value_range: tuple[float, float] = (0.5, 1.0),
    nei: int = 1,
    add_min_eigen: float = 0.1,
    signs: str = "Random",
    rng: np.random.Generator | None = None,
):
    """Watts-Strogatz small-world graph.

    Each node is connected to its ``2 * nei`` nearest neighbours on a ring
    lattice, and every edge is rewired independently with probability ``prob``.
    """
    if rng is None:
        rng = np.random.default_rng()

    k = 2 * int(nei)
    if k <= 0:
        raise ValueError("nei must be >= 1 (so that k = 2 * nei > 0).")
    # networkx requires an even k strictly smaller than the number of nodes.
    if k >= p:
        k = p - (1 if (p % 2 == 1) else 0)

    if k == 0:
        g = nx.Graph()
        g.add_nodes_from(range(p))
    else:
        seed = int(rng.integers(0, 1_000_000_000))
        g = nx.watts_strogatz_graph(n=p, k=k, p=prob, seed=seed)

    edges = list(g.edges())
    omega, sigma = _precision_from_edges(p, edges, value_range, add_min_eigen, signs, rng)
    return edges, omega, sigma


def generate_pa_graph(
    p: int,
    value_range: tuple[float, float] = (0.5, 1.0),
    power: float = 1.0,
    m: int = 1,
    add_min_eigen: float = 0.1,
    signs: str = "Random",
    rng: np.random.Generator | None = None,
):
    """Barabasi-Albert preferential-attachment graph with tunable ``power``.

    Starting from a clique of ``m`` nodes, every new node attaches to ``m``
    existing nodes chosen without replacement with probability proportional to
    ``degree ** power``. ``power = 1`` recovers linear preferential attachment;
    the paper uses sublinear attachment with ``power = 0.5``.
    """
    if rng is None:
        rng = np.random.default_rng()

    p = int(p)
    m = int(m)
    if p <= 0:
        raise ValueError("p must be positive")
    if not 0 <= m < p:
        raise ValueError("m must satisfy 0 <= m < p")

    g = nx.Graph()
    g.add_nodes_from(range(p))

    if m >= 1:
        # Start from a clique on the first m nodes.
        for i in range(m):
            for j in range(i + 1, m):
                g.add_edge(i, j)

    for new_node in range(m, p):
        candidates = [node for node in g.nodes() if node < new_node]
        if not candidates:
            continue
        degrees = np.array([max(0.0, g.degree(node)) for node in candidates], dtype=float)
        if power == 0.0 or np.all(degrees == 0):
            weights = np.ones_like(degrees)
        else:
            weights = np.power(degrees, float(power))
        probs = weights / weights.sum()
        num_to_pick = min(m, len(candidates))
        targets = rng.choice(candidates, size=num_to_pick, replace=False, p=probs)
        for target in targets:
            g.add_edge(new_node, int(target))

    edges = list(g.edges())
    omega, sigma = _precision_from_edges(p, edges, value_range, add_min_eigen, signs, rng)
    return edges, omega, sigma
