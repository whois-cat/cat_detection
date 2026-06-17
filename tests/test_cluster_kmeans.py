import numpy as np

from training.build_cluster_manifest import assign_chunked, kmeans


def _dense_assign(x, centers):
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        d = (
            np.sum(x * x, axis=1, keepdims=True)
            + np.sum(centers * centers, axis=1)[None, :]
            - 2.0 * (x @ centers.T)
        )
    lb = np.argmin(d, axis=1)
    return lb, np.maximum(d[np.arange(len(x)), lb], 0.0)


def test_chunked_assignment_matches_dense():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 16)).astype(np.float32)
    centers = x[rng.choice(500, size=12, replace=False)]
    lb_full, d_full = _dense_assign(x, centers)
    # Tiny chunk must give identical labels regardless of N×K never being built.
    lb_chunk, d_chunk = assign_chunked(x, centers, chunk=37)
    assert np.array_equal(lb_chunk, lb_full)
    assert d_chunk.dtype == np.float32
    assert np.allclose(d_chunk, d_full, atol=1e-3)


def test_kmeans_chunk_size_invariant_and_deterministic():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 8)).astype(np.float32)
    a_lab, a_dist = kmeans(x, 10, seed=42, max_iter=20, chunk=4096)
    b_lab, b_dist = kmeans(x, 10, seed=42, max_iter=20, chunk=13)
    # Same seed + same data → identical clustering whatever the chunk size.
    assert np.array_equal(a_lab, b_lab)
    assert np.allclose(a_dist, b_dist, atol=1e-3)
    assert a_dist.dtype == np.float32
    assert (a_dist >= 0).all()
