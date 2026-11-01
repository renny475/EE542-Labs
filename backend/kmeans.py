"""Hand-written K-means (Lloyd's algorithm with k-means++ seeding), numpy only.

Used to cluster Word2Vec word vectors into the K bins of the histogram
embedding (Lab 9 Part 5). No scikit-learn, gensim or FAISS clustering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROW_BLOCK = 2048  # rows of x handled per distance computation, bounds memory to ~ROW_BLOCK * k floats


@dataclass
class KMeansResult:
    centroids: np.ndarray  # (k, d)
    labels: np.ndarray  # (n,) index of the nearest centroid per row
    iterations: int
    inertia: float  # sum of squared distances to the assigned centroid


def _sq_dists(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, shape (n, k), computed block by block."""
    out = np.empty((x.shape[0], centroids.shape[0]), dtype=np.float64)
    c_sq = (centroids**2).sum(axis=1)
    for start in range(0, x.shape[0], ROW_BLOCK):
        block = x[start : start + ROW_BLOCK]
        d = (block**2).sum(axis=1)[:, None] - 2.0 * block @ centroids.T + c_sq[None, :]
        np.maximum(d, 0.0, out=d)  # rounding can make tiny negatives
        out[start : start + ROW_BLOCK] = d
    return out


def kmeans_plusplus_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Pick k starting centroids: the first uniformly, each next one with probability ∝ D²."""
    n = x.shape[0]
    chosen = [int(rng.integers(n))]
    best = _sq_dists(x, x[chosen[-1]][None, :])[:, 0]
    while len(chosen) < k:
        total = best.sum()
        if total > 0:
            idx = int(rng.choice(n, p=best / total))
        else:
            # Every remaining point coincides with a chosen one: pick an unchosen index.
            remaining = np.setdiff1d(np.arange(n), chosen)
            idx = int(rng.choice(remaining)) if remaining.size else int(rng.integers(n))
        chosen.append(idx)
        best = np.minimum(best, _sq_dists(x, x[idx][None, :])[:, 0])
    return x[chosen].astype(np.float64, copy=True)


def kmeans(x: np.ndarray, k: int, *, seed: int = 42, max_iter: int = 100) -> KMeansResult:
    x = np.asarray(x, dtype=np.float64)
    if k <= 0 or k > x.shape[0]:
        raise ValueError(f"k must be between 1 and the number of rows ({x.shape[0]}), got {k}")
    rng = np.random.default_rng(seed)
    centroids = kmeans_plusplus_init(x, k, rng)
    labels = np.full(x.shape[0], -1, dtype=np.int64)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        dists = _sq_dists(x, centroids)
        new_labels = dists.argmin(axis=1)
        for j in range(k):
            members = x[new_labels == j]
            if members.shape[0]:
                centroids[j] = members.mean(axis=0)
            else:
                # Empty cluster: re-seed it at the point farthest from its own centroid.
                farthest = int(dists.min(axis=1).argmax())
                centroids[j] = x[farthest]
                new_labels[farthest] = j
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
    inertia = float(_sq_dists(x, centroids)[np.arange(x.shape[0]), labels].sum())
    return KMeansResult(centroids=centroids, labels=labels, iterations=iterations, inertia=inertia)


def nearest_centroids(x: np.ndarray, centroids: np.ndarray, t: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Indices (n, t) and Euclidean distances (n, t) of the t nearest centroids, ascending."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    t = min(t, centroids.shape[0])
    d = _sq_dists(x, centroids)
    idx = np.argpartition(d, t - 1, axis=1)[:, :t] if t < centroids.shape[0] else np.argsort(d, axis=1)
    part = np.take_along_axis(d, idx, axis=1)
    order = np.argsort(part, axis=1, kind="stable")
    idx = np.take_along_axis(idx, order, axis=1)
    return idx, np.sqrt(np.take_along_axis(part, order, axis=1))
