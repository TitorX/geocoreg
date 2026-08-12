"""MLS / IDW / TPS interpolation of the displacement field."""

from __future__ import annotations

import numpy as np
import pytest

from geocoreg import config, interp

H = W = 64


def _scatter(n=120, seed=3, shape=(H, W)):
    """Random anchor positions on a grid, guaranteed distinct."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(shape[0] * shape[1], size=n, replace=False)
    return np.column_stack(np.unravel_index(idx, shape))  # (n, 2) rows, cols


def _state_from(fn, pts, shape=(H, W), conf=1.0):
    """Build (D, F, C) with anchors at ``pts`` carrying ``fn(col, row)``."""
    D = np.zeros(shape + (2,), dtype=np.float64)
    F = np.zeros(shape, dtype=np.uint8)
    C = np.zeros(shape, dtype=np.float32)
    for r, c in pts:
        D[r, c, :] = fn(float(c), float(r))
        F[r, c] = 1
        C[r, c] = conf
    return D, F, C


def _linear(a=(0.30, -0.12, 7.0), b=(-0.05, 0.21, -3.0)):
    def fn(x, y):
        return (a[0] * x + a[1] * y + a[2], b[0] * x + b[1] * y + b[2])
    return fn


def _all_targets(F):
    ys, xs = np.where(F == 0)
    return list(zip(xs.tolist(), ys.tolist()))


# --------------------------------------------------------------------------
# MLS (interp.fill)
# --------------------------------------------------------------------------

def test_constant_field_stays_constant():
    pts = _scatter()
    D, F, C = _state_from(lambda x, y: (4.25, -9.5), pts)
    out = interp.fill(D, F, C=C)
    assert np.allclose(out[..., 0], 4.25, atol=1e-4)
    assert np.allclose(out[..., 1], -9.5, atol=1e-4)


def test_anchor_pixels_are_left_untouched():
    pts = _scatter()
    D, F, C = _state_from(_linear(), pts)
    out = interp.fill(D, F, C=C)
    rows, cols = pts[:, 0], pts[:, 1]
    assert np.array_equal(out[rows, cols, :], D[rows, cols, :])


@pytest.mark.parametrize("k", [8, 12, 20, 32])
@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_mls_reproduces_a_linear_field_exactly(k, alpha):
    """MLS with an affine local model is exact on affine data, for any k/alpha.

    Tolerance is float32 round-off: ``interp.fill`` accumulates in float32.
    """
    fn = _linear()
    pts = _scatter(n=160, seed=11)
    D, F, C = _state_from(fn, pts)
    config.INTERP_K, config.INTERP_ALPHA = k, alpha
    out = interp.fill(D, F, C=C)

    ys, xs = np.mgrid[0:H, 0:W]
    ex, ey = fn(xs.astype(float), ys.astype(float))
    assert np.abs(out[..., 0] - ex).max() < 5e-3
    assert np.abs(out[..., 1] - ey).max() < 5e-3


def test_chunk_size_does_not_change_the_result():
    """No chunk-boundary seams: chunking is a pure batching detail."""
    fn = _linear()
    pts = _scatter(n=90, seed=5)
    D, F, C = _state_from(fn, pts)

    config.INTERP_CHUNK_SIZE = 10000
    ref = interp.fill(D.copy(), F.copy(), C=C.copy())
    for chunk in (1, 7, 13, 512, 100000):
        config.INTERP_CHUNK_SIZE = chunk
        got = interp.fill(D.copy(), F.copy(), C=C.copy())
        assert np.array_equal(got, ref), f"chunk_size={chunk} changed the result"


def test_targets_restricts_the_pixels_written():
    fn = _linear()
    pts = _scatter(n=80, seed=7)
    D, F, C = _state_from(fn, pts)
    targets = [(3, 4), (10, 11), (60, 61)]
    out = interp.fill(D.copy(), F, targets=targets, C=C)
    for c, r in targets:
        assert not np.array_equal(out[r, c], D[r, c])
    touched = np.any(out != D, axis=-1)
    assert touched.sum() == len(targets)


def test_targets_skips_pixels_that_are_already_anchors():
    fn = _linear()
    pts = _scatter(n=80, seed=7)
    D, F, C = _state_from(fn, pts)
    anchor = (int(pts[0][1]), int(pts[0][0]))  # (col, row)
    out = interp.fill(D.copy(), F, targets=[anchor], C=C)
    assert np.array_equal(out, D)


def test_no_anchors_returns_the_input_unchanged():
    D = np.zeros((8, 8, 2))
    F = np.zeros((8, 8), np.uint8)
    out = interp.fill(D, F)
    assert np.array_equal(out, D)


def test_confidence_weighting_pulls_towards_the_trusted_anchors():
    """A high-confidence anchor should dominate a co-located low-confidence one."""
    D = np.zeros((32, 32, 2))
    F = np.zeros((32, 32), np.uint8)
    C = np.zeros((32, 32), np.float32)
    for (r, c), val, conf in [((5, 5), 10.0, 1.0), ((5, 9), -10.0, 1.0),
                              ((25, 5), 10.0, 1.0), ((25, 9), -10.0, 1.0)]:
        D[r, c, 0] = val
        F[r, c] = 1
        C[r, c] = conf
    config.INTERP_K = 4
    balanced = interp.fill(D.copy(), F.copy(), C=C.copy())[15, 7, 0]

    C_biased = C.copy()
    C_biased[5, 5] = C_biased[25, 5] = 1.0
    C_biased[5, 9] = C_biased[25, 9] = 0.01
    biased = interp.fill(D.copy(), F.copy(), C=C_biased)[15, 7, 0]
    assert biased > balanced


# --------------------------------------------------------------------------
# IDW
# --------------------------------------------------------------------------

def test_idw_is_exact_on_a_constant_field():
    """Inverse-distance weights form a partition of unity."""
    pts = _scatter(n=100, seed=13).astype(float)[:, ::-1]  # (x, y)
    vals = np.tile(np.array([[3.0, -7.0]]), (len(pts), 1))
    targets = np.array([[10.5, 20.5], [40.0, 5.0], [1.0, 63.0]])
    got = interp._predict_idw(pts, vals, None, targets, k=20, alpha=2.0)
    assert np.allclose(got, vals[0], atol=1e-9)


def test_idw_error_on_a_linear_field_is_bounded_by_the_neighbourhood():
    """IDW has no affine term, so it smooths a ramp — but only locally.

    The bound is the field's variation over the k-neighborhood radius, which
    is what "IDW underfits gradients" means quantitatively.
    """
    fn = _linear()
    pts = _scatter(n=200, seed=17).astype(float)[:, ::-1]
    vals = np.stack(fn(pts[:, 0], pts[:, 1]), axis=-1)
    ys, xs = np.mgrid[8:56, 8:56]
    targets = np.column_stack([xs.ravel().astype(float), ys.ravel().astype(float)])
    got = interp._predict_idw(pts, vals, None, targets, k=20, alpha=2.0)
    exact = np.stack(fn(targets[:, 0], targets[:, 1]), axis=-1)
    err = np.linalg.norm(got - exact, axis=1)

    from scipy.spatial import cKDTree
    radius = cKDTree(pts).query(targets, k=20)[0][:, -1]
    grad = np.hypot(0.30, 0.12) + np.hypot(0.05, 0.21)
    assert np.all(err <= grad * radius + 1e-6)
    assert np.median(err) < 0.5 * grad * np.median(radius)


# --------------------------------------------------------------------------
# TPS
# --------------------------------------------------------------------------

def test_tps_reproduces_a_linear_field():
    """Thin-plate splines have an affine null space, so a ramp is (nearly) free.

    ``_predict_tps`` uses smoothing=1.0, so the fit is a regularized one — the
    bound below is loose enough to allow that but tight enough to catch a
    genuinely wrong interpolant.
    """
    fn = _linear()
    pts = _scatter(n=200, seed=19).astype(float)[:, ::-1]
    vals = np.stack(fn(pts[:, 0], pts[:, 1]), axis=-1)
    ys, xs = np.mgrid[8:56:2, 8:56:2]
    targets = np.column_stack([xs.ravel().astype(float), ys.ravel().astype(float)])
    got = interp._predict_tps(pts, vals, targets, neighbors=32)
    exact = np.stack(fn(targets[:, 0], targets[:, 1]), axis=-1)
    err = np.linalg.norm(got - exact, axis=1)
    assert np.median(err) < 0.05
    assert err.max() < 0.5


def test_tps_chunking_is_transparent():
    fn = _linear()
    pts = _scatter(n=120, seed=23).astype(float)[:, ::-1]
    vals = np.stack(fn(pts[:, 0], pts[:, 1]), axis=-1)
    targets = np.column_stack([np.arange(0.0, 60.0), np.arange(0.0, 60.0)])
    a = interp._predict_tps(pts, vals, targets, neighbors=32, workers=1)
    b = interp._predict_tps(pts, vals, targets, neighbors=32, workers=4)
    assert np.allclose(a, b, atol=1e-12)


# --------------------------------------------------------------------------
# autotune
# --------------------------------------------------------------------------

def test_autotune_falls_back_to_plain_fill_below_50_points():
    fn = _linear()
    pts = _scatter(n=30, seed=29)
    D, F, C = _state_from(fn, pts)
    tuned = interp.autotune_fill(D.copy(), F.copy(), C=C.copy())
    plain = interp.fill(D.copy(), F.copy(), C=C.copy())
    assert np.array_equal(tuned, plain)


def test_autotune_recovers_a_linear_field_and_is_deterministic():
    fn = _linear()
    pts = _scatter(n=180, seed=31)
    D, F, C = _state_from(fn, pts)
    a = interp.autotune_fill(D.copy(), F.copy(), C=C.copy())
    b = interp.autotune_fill(D.copy(), F.copy(), C=C.copy())
    assert np.array_equal(a, b), "autotune must be seeded, not random"

    ys, xs = np.mgrid[0:H, 0:W]
    ex, ey = fn(xs.astype(float), ys.astype(float))
    err = np.hypot(a[..., 0] - ex, a[..., 1] - ey)
    assert np.median(err) < 0.05
