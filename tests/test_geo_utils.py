"""Coordinate conversions, CRS transforms, overlap and GSD ratio."""

from __future__ import annotations

import _synth
import numpy as np
import pytest

from geocoreg import geo_utils

GT = (500000.0, 10.0, 0.0, 4800000.0, 0.0, -10.0)
ROTATED_GT = (500000.0, 9.5, 1.5, 4800000.0, 2.0, -9.0)


# --------------------------------------------------------------------------
# pixel <-> geo
# --------------------------------------------------------------------------

@pytest.mark.parametrize("gt", [GT, ROTATED_GT], ids=["north_up", "rotated"])
def test_pix2geo_geo2pix_round_trip(gt):
    cols = np.array([0.0, 0.5, 1.0, 37.25, 1023.0])
    rows = np.array([0.0, 0.5, 2.0, 512.75, 99.5])
    for c, r in zip(cols, rows):
        x, y = geo_utils.pix2geo(gt, c, r)
        c2, r2 = geo_utils.geo2pix(gt, x, y)
        assert c2 == pytest.approx(c, abs=1e-9)
        assert r2 == pytest.approx(r, abs=1e-9)


@pytest.mark.parametrize("gt", [GT, ROTATED_GT], ids=["north_up", "rotated"])
def test_array_helpers_match_scalar_helpers(gt):
    cols = np.linspace(0, 500, 17)
    rows = np.linspace(0, 300, 17)
    xs, ys = geo_utils.pix2geo_array(gt, cols, rows)
    for i, (c, r) in enumerate(zip(cols, rows)):
        x, y = geo_utils.pix2geo(gt, c, r)
        assert xs[i] == pytest.approx(x, rel=0, abs=1e-9)
        assert ys[i] == pytest.approx(y, rel=0, abs=1e-9)
    c2, r2 = geo_utils.geo2pix_array(gt, xs, ys)
    assert np.allclose(c2, cols, atol=1e-9)
    assert np.allclose(r2, rows, atol=1e-9)


def test_pix2geo_index_zero_is_the_pixel_corner():
    """Integer indices are corners; centers are at ``index + 0.5``.

    The whole pipeline's half-pixel bookkeeping rests on this.
    """
    x, y = geo_utils.pix2geo(GT, 0, 0)
    assert (x, y) == (GT[0], GT[3])
    xc, yc = geo_utils.pix2geo(GT, 0.5, 0.5)
    assert xc == pytest.approx(GT[0] + 0.5 * GT[1])
    assert yc == pytest.approx(GT[3] + 0.5 * GT[5])


def test_geo2pix_is_not_axis_swapped():
    """Moving +1 column changes x only; +1 row changes y only (north-up)."""
    x0, y0 = geo_utils.pix2geo(GT, 10, 20)
    x1, y1 = geo_utils.pix2geo(GT, 11, 20)
    x2, y2 = geo_utils.pix2geo(GT, 10, 21)
    assert x1 - x0 == pytest.approx(GT[1])
    assert y1 == pytest.approx(y0)
    assert y2 - y0 == pytest.approx(GT[5])
    assert x2 == pytest.approx(x0)


# --------------------------------------------------------------------------
# CRS transforms
# --------------------------------------------------------------------------

def test_make_transform_returns_none_for_identical_crs():
    wkt = _synth.wkt_for(_synth.EPSG_UTM11N)
    assert geo_utils.make_transform(wkt, wkt) is None


def test_transform_point_is_identity_when_ct_is_none():
    assert geo_utils.transform_point(None, 1.0, 2.0) == (1.0, 2.0)


def test_cross_crs_round_trip_within_a_micrometre():
    """UTM -> WGS84 -> UTM must return the same easting/northing.

    1e-6 m is well under any resampling or matching tolerance; anything worse
    would mean the axis-order strategy is wrong.
    """
    utm = _synth.wkt_for(_synth.EPSG_UTM11N)
    wgs = _synth.wkt_for(_synth.EPSG_WGS84)
    fwd = geo_utils.make_transform(utm, wgs)
    bwd = geo_utils.make_transform(wgs, utm)
    assert fwd is not None and bwd is not None

    xs = np.array([500000.0, 512345.678, 487654.321, 500000.0])
    ys = np.array([4800000.0, 4812345.678, 4787654.321, 4750000.0])
    for x, y in zip(xs, ys):
        lon, lat = geo_utils.transform_point(fwd, x, y)
        # Traditional GIS order: x is longitude, y is latitude.
        assert -180.0 <= lon <= 180.0
        assert -90.0 <= lat <= 90.0
        x2, y2 = geo_utils.transform_point(bwd, lon, lat)
        assert abs(x2 - x) < 1e-6
        assert abs(y2 - y) < 1e-6


def test_transform_points_matches_transform_point():
    utm = _synth.wkt_for(_synth.EPSG_UTM11N)
    wgs = _synth.wkt_for(_synth.EPSG_WGS84)
    ct = geo_utils.make_transform(utm, wgs)
    xs = np.array([500000.0, 501000.0, 502000.0])
    ys = np.array([4800000.0, 4801000.0, 4802000.0])
    bx, by = geo_utils.transform_points(ct, xs, ys)
    for i in range(len(xs)):
        px, py = geo_utils.transform_point(ct, xs[i], ys[i])
        assert bx[i] == pytest.approx(px, abs=1e-12)
        assert by[i] == pytest.approx(py, abs=1e-12)


def test_transform_points_none_copies_rather_than_aliases():
    xs = np.array([1.0, 2.0])
    ys = np.array([3.0, 4.0])
    ox, oy = geo_utils.transform_points(None, xs, ys)
    ox[0] = 99.0
    assert xs[0] == 1.0
    assert np.array_equal(oy, ys)


# --------------------------------------------------------------------------
# bbox / overlap / crop
# --------------------------------------------------------------------------

def test_bbox_in_crs_north_up():
    xmin, xmax, ymin, ymax = geo_utils.bbox_in_crs(GT, 100, 50)
    assert xmin == pytest.approx(500000.0)
    assert xmax == pytest.approx(500000.0 + 100 * 10.0)
    assert ymax == pytest.approx(4800000.0)
    assert ymin == pytest.approx(4800000.0 - 50 * 10.0)


def test_compute_overlap_intersection_and_disjoint():
    a = (0.0, 10.0, 0.0, 10.0)
    b = (5.0, 20.0, -5.0, 4.0)
    assert geo_utils.compute_overlap(a, b) == (5.0, 10.0, 0.0, 4.0)
    assert geo_utils.compute_overlap(a, (20.0, 30.0, 0.0, 10.0)) is None
    # Touching edges are not an overlap.
    assert geo_utils.compute_overlap(a, (10.0, 20.0, 0.0, 10.0)) is None


def test_overlap_to_crop_matches_the_pixel_window():
    overlap = (500100.0, 500300.0, 4799700.0, 4799900.0)
    col_off, row_off, w, h = geo_utils.overlap_to_crop(GT, 100, 100, overlap)
    assert (col_off, row_off) == (10, 10)
    assert (w, h) == (20, 20)


def test_overlap_to_crop_clamps_to_the_raster():
    overlap = (499000.0, 501500.0, 4798000.0, 4801000.0)
    col_off, row_off, w, h = geo_utils.overlap_to_crop(GT, 100, 100, overlap)
    assert (col_off, row_off) == (0, 0)
    assert (w, h) == (100, 100)


def test_shifted_geotransform_moves_only_the_origin():
    gt2 = geo_utils.shifted_geotransform(GT, 7, 3)
    assert gt2[1:3] == GT[1:3]
    assert gt2[4:] == GT[4:]
    assert gt2[0] == pytest.approx(GT[0] + 7 * GT[1])
    assert gt2[3] == pytest.approx(GT[3] + 3 * GT[5])
    # A pixel index in the cropped grid names the same ground point.
    assert geo_utils.pix2geo(gt2, 2, 5) == pytest.approx(geo_utils.pix2geo(GT, 9, 8))


# --------------------------------------------------------------------------
# GSD ratio
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ref_gsd,mov_gsd", [(10.0, 10.0), (10.0, 30.0),
                                             (30.0, 10.0), (10.0, 33.3)])
def test_gsd_ratio_equals_the_analytic_value_same_crs(ref_gsd, mov_gsd):
    """One REF pixel spans ``ref_gsd / mov_gsd`` MOV pixels."""
    gt_ref = _synth.north_up_gt(500000.0, 4800000.0, ref_gsd)
    gt_mov = _synth.north_up_gt(500000.0, 4800000.0, mov_gsd)
    rx, ry = geo_utils.compute_gsd_ratio(gt_ref, gt_mov, 17, 23, None)
    assert rx == pytest.approx(ref_gsd / mov_gsd, rel=1e-12)
    assert ry == pytest.approx(ref_gsd / mov_gsd, rel=1e-12)


def test_gsd_ratio_anisotropic_pixels():
    gt_ref = _synth.north_up_gt(500000.0, 4800000.0, 20.0, 5.0)
    gt_mov = _synth.north_up_gt(500000.0, 4800000.0, 10.0, 10.0)
    rx, ry = geo_utils.compute_gsd_ratio(gt_ref, gt_mov, 3, 4, None)
    assert rx == pytest.approx(2.0, rel=1e-12)
    assert ry == pytest.approx(0.5, rel=1e-12)


def test_gsd_ratio_across_crs_is_close_to_the_metric_ratio():
    """A UTM->UTM neighboring-zone transform is near-conformal locally, so the
    ratio still equals the metric one to a fraction of a percent."""
    gt_ref = _synth.north_up_gt(500000.0, 4800000.0, 10.0)
    wkt_ref = _synth.wkt_for(_synth.EPSG_UTM11N)
    wkt_mov = _synth.wkt_for(_synth.EPSG_UTM12N)
    x, y = _synth.transform_xy(wkt_ref, wkt_mov,
                               np.array([[500000.0]]), np.array([[4800000.0]]))
    gt_mov = _synth.north_up_gt(float(x[0, 0]), float(y[0, 0]), 30.0)
    ct = geo_utils.make_transform(wkt_ref, wkt_mov)
    rx, ry = geo_utils.compute_gsd_ratio(gt_ref, gt_mov, 5, 5, ct)
    assert rx == pytest.approx(1.0 / 3.0, rel=2e-3)
    assert ry == pytest.approx(1.0 / 3.0, rel=2e-3)
