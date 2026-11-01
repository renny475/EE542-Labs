import numpy as np

from kmeans import KMeansResult, kmeans, kmeans_plusplus_init, nearest_centroids


def blobs(rng):
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    x = np.vstack([c + rng.normal(scale=0.5, size=(50, 2)) for c in centres])
    truth = np.repeat(np.arange(3), 50)
    return x, truth


def test_kmeans_recovers_separable_clusters():
    x, truth = blobs(np.random.default_rng(0))
    r = kmeans(x, 3)
    assert isinstance(r, KMeansResult) and r.centroids.shape == (3, 2) and r.iterations <= 100
    groups = [set(r.labels[truth == g].tolist()) for g in range(3)]
    assert all(len(g) == 1 for g in groups) and len(set.union(*groups)) == 3
    assert r.inertia < 100


def test_kmeans_is_deterministic():
    x, _ = blobs(np.random.default_rng(1))
    assert np.array_equal(kmeans(x, 3).centroids, kmeans(x, 3).centroids)


def test_kmeans_plusplus_picks_distinct_points():
    x = np.array([[0, 0], [0, 0], [5, 5], [9, 9]], float)
    init = kmeans_plusplus_init(x, 3, np.random.default_rng(0))
    assert init.shape == (3, 2) and len({tuple(r) for r in init}) == 3


def test_nearest_centroids_orders_by_distance():
    c = np.array([[0, 0], [1, 0], [5, 0]], float)
    idx, dist = nearest_centroids(np.array([[0.9, 0]]), c, t=2)
    assert idx.tolist() == [[1, 0]] and np.allclose(dist, [[0.1, 0.9]])


def test_kmeans_handles_k_greater_than_unique_points_without_nan():
    x = np.array([[0, 0], [0, 0], [1, 1]], float)
    r = kmeans(x, 3)
    assert not np.isnan(r.centroids).any() and r.labels.shape == (3,)
