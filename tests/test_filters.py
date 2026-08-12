"""The outlier-rejection chain (global MAD, local affine LOO, NMS)."""

from __future__ import annotations

import numpy as np
import pytest

from geocoreg import config, filters

H = W = 96


def _grid_points(n_side=10, step=9, offset=3):
    return [(offset + i * step, offset + j * step)
            for i in range(n_side) for j in range(n_side)]


def _make(points, value_fn, conf=1.0, shape=(H, W)):
    D = np.zeros(shape + (2,), dtype=np.float64)
    F = np.zeros(shape, dtype=np.uint8)
    C = np.zeros(shape, dtype=np.float32)
    for r, c in points:
        D[r, c, :] = value_fn(float(c), float(r))
        F[r, c] = 1
        C[r, c] = conf
    return D, F, C


def _matched(F):
    return {tuple(p) for p in np.argwhere(F == 1)}


# --------------------------------------------------------------------------
# global magnitude MAD filter
# --------------------------------------------------------------------------

def test_global_mad_removes_exactly_the_injected_outliers():
    rng = np.random.default_rng(0)
    pts = _grid_points()
    D, F, C = _make(pts, lambda x, y: (5.0, 5.0))
    for r, c in pts:  # small jitter so the MAD is non-degenerate
        D[r, c, :] += rng.normal(0, 0.05, 2)

    injected = {(pts[i][0], pts[i][1]) for i in (7, 23, 61, 88)}
    for r, c in injected:
        D[r, c, :] = (400.0, -300.0)

    before = _matched(F)
    n = filters.global_outlier_filter(D, F)
    removed = before - _matched(F)

    assert removed == injected
    assert n == len(injected)
    for r, c in injected:
        assert np.array_equal(D[r, c], [0.0, 0.0])


def test_global_mad_keeps_a_clean_field_intact():
    rng = np.random.default_rng(1)
    pts = _grid_points()
    D, F, C = _make(pts, lambda x, y: (3.0, -2.0))
    for r, c in pts:
        D[r, c, :] += rng.normal(0, 0.1, 2)
    assert filters.global_outlier_filter(D, F) == 0
    assert int(F.sum()) == len(pts)


def test_global_mad_is_one_sided_on_magnitude():
    """The gate rejects unusually *large* displacements only."""
    pts = _grid_points(n_side=6, step=15)
    D, F, C = _make(pts, lambda x, y: (10.0, 0.0))
    small = (pts[3][0], pts[3][1])
    D[small[0], small[1], :] = (0.0, 0.0)
    assert filters.global_outlier_filter(D, F) == 0
    assert F[small] == 1


def test_mad_k_is_read_at_call_time():
    """Turning the knob down must reject more, read at call time."""
    rng = np.random.default_rng(2)
    pts = _grid_points()
    D, F, C = _make(pts, lambda x, y: (5.0, 0.0))
    for r, c in pts:
        D[r, c, 0] += rng.normal(0, 1.0)

    config.MAD_K = 3.0
    loose = filters.global_outlier_filter(D.copy(), F.copy())
    config.MAD_K = 0.25
    strict = filters.global_outlier_filter(D.copy(), F.copy())
    assert strict > loose


def test_global_mad_needs_at_least_four_points():
    D, F, C = _make([(1, 1), (2, 2), (3, 3)], lambda x, y: (1000.0, 0.0))
    assert filters.global_outlier_filter(D, F) == 0
    assert int(F.sum()) == 3


# --------------------------------------------------------------------------
# leave-one-out local affine filter
# --------------------------------------------------------------------------

def _affine_field(a=0.11, b=-0.07, c=0.05, d=0.13, e=2.0, f=-3.0):
    def fn(x, y):
        return (a * x + b * y + e, c * x + d * y + f)
    return fn


def test_local_affine_removes_nothing_on_a_purely_affine_field():
    """Every point is predicted exactly by its neighbors' affine model."""
    pts = _grid_points()
    D, F, C = _make(pts, _affine_field())
    n = filters.local_affine_filter(D, F, C)
    assert n == 0
    assert int(F.sum()) == len(pts)


def test_local_affine_removes_nothing_on_a_constant_field():
    pts = _grid_points()
    D, F, C = _make(pts, lambda x, y: (-4.0, 6.0))
    assert filters.local_affine_filter(D, F, C) == 0


def test_local_affine_removes_an_injected_gross_outlier():
    rng = np.random.default_rng(3)
    pts = _grid_points()
    D, F, C = _make(pts, _affine_field())
    for r, c in pts:
        D[r, c, :] += rng.normal(0, 0.02, 2)

    injected = {(pts[i][0], pts[i][1]) for i in (34, 55)}
    for r, c in injected:
        D[r, c, :] += (60.0, -45.0)

    before = _matched(F)
    n = filters.local_affine_filter(D, F, C)
    removed = before - _matched(F)
    assert injected <= removed, f"missed {injected - removed}"
    assert n == len(removed)
    for r, c in injected:
        assert np.array_equal(D[r, c], [0.0, 0.0])
        assert C[r, c] == 0.0


def test_local_affine_needs_more_than_k_plus_three_points():
    pts = _grid_points(n_side=3, step=20)  # 9 points, k defaults to 12
    D, F, C = _make(pts, lambda x, y: (1.0, 1.0))
    D[pts[0][0], pts[0][1], :] = (500.0, 500.0)
    assert filters.local_affine_filter(D, F, C) == 0
    assert int(F.sum()) == len(pts)


def test_local_affine_tolerates_a_smooth_gradient_a_median_filter_would_cut():
    """The point of this filter: a valid ramp is not an outlier."""
    pts = _grid_points()
    steep = _affine_field(a=1.5, b=0.0, c=0.0, d=1.5, e=0.0, f=0.0)
    D, F, C = _make(pts, steep)
    assert filters.local_affine_filter(D, F, C) == 0


# --------------------------------------------------------------------------
# NMS
# --------------------------------------------------------------------------

def test_nms_keeps_only_the_best_point_inside_the_radius():
    D = np.zeros((32, 32, 2))
    F = np.zeros((32, 32), np.uint8)
    C = np.zeros((32, 32), np.float32)
    cluster = [((10, 10), 0.9), ((10, 12), 0.5), ((12, 10), 0.7)]
    for (r, c), conf in cluster:
        D[r, c, :] = (float(c), float(r))
        F[r, c] = 1
        C[r, c] = conf

    removed = filters.nms_filter(D, F, C, radius=5)
    assert removed == 2
    assert _matched(F) == {(10, 10)}
    assert np.array_equal(D[10, 12], [0.0, 0.0])
    assert np.array_equal(D[12, 10], [0.0, 0.0])


def test_nms_keeps_points_further_apart_than_the_radius():
    D = np.zeros((64, 64, 2))
    F = np.zeros((64, 64), np.uint8)
    C = np.zeros((64, 64), np.float32)
    pts = [(4, 4), (4, 30), (30, 4), (30, 30), (58, 58)]
    for i, (r, c) in enumerate(pts):
        F[r, c] = 1
        C[r, c] = 0.5 + 0.01 * i
    assert filters.nms_filter(D, F, C, radius=10) == 0
    assert _matched(F) == set(pts)


@pytest.mark.parametrize("radius,expected_kept", [(1, 5), (11, 3), (40, 1)])
def test_nms_radius_controls_how_many_survive(radius, expected_kept):
    D = np.zeros((64, 64, 2))
    F = np.zeros((64, 64), np.uint8)
    C = np.zeros((64, 64), np.float32)
    for i, c in enumerate([4, 14, 24, 34, 44]):
        F[10, c] = 1
        C[10, c] = 0.9 - 0.05 * i
    filters.nms_filter(D, F, C, radius=radius)
    assert int(F.sum()) == expected_kept


def test_nms_is_a_no_op_below_two_points():
    D = np.zeros((8, 8, 2))
    F = np.zeros((8, 8), np.uint8)
    C = np.zeros((8, 8), np.float32)
    F[3, 3] = 1
    C[3, 3] = 0.5
    assert filters.nms_filter(D, F, C, radius=4) == 0
    assert int(F.sum()) == 1


def test_nms_suppression_follows_confidence_not_position():
    D = np.zeros((32, 32, 2))
    F = np.zeros((32, 32), np.uint8)
    C = np.zeros((32, 32), np.float32)
    F[5, 5], C[5, 5] = 1, 0.2
    F[5, 8], C[5, 8] = 1, 0.95
    filters.nms_filter(D, F, C, radius=6)
    assert _matched(F) == {(5, 8)}
