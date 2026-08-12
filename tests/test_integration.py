"""Small end-to-end runs of the public API.

Each case builds a (REF, MOV) pair whose true displacement field is known
analytically and scores the exported control points against it. Thresholds are
deliberately loose — the question is whether the pipeline is wired up correctly
for that kind of input, not how accurate the method is.
"""

from __future__ import annotations

import json
import os

import _synth
import numpy as np
import pytest
from osgeo import gdal

import geocoreg
from geocoreg import config, control_points, init, refine
from geocoreg.matchers import RootSIFTMatcher

gdal.UseExceptions()

pytestmark = pytest.mark.slow

G = 10.0  # the reference GSD used by most cases

# name -> (make_pair kwargs, median-EPE ceiling in MOV pixels, min control points)
SCENARIOS = {
    "translation": (dict(size=256, d_fn=_synth.d_constant(3.5 * G, -2.5 * G)),
                    1.0, 20),
    "affine": (dict(size=256, d_fn=_synth.d_affine(2.0 * G, -1.0 * G)),
               1.0, 20),
    "nonrigid": (dict(size=256, d_fn=_synth.d_nonrigid(amp=1.5 * G, period=2000.0)),
                 1.5, 20),
    "coarse_mov_2x": (dict(size=256, mov_gsd=20.0,
                           d_fn=_synth.d_constant(40.0, -30.0)), 1.0, 20),
    "coarse_mov_3.33x": (dict(size=256, mov_gsd=33.3,
                              d_fn=_synth.d_constant(50.0, -33.3)), 1.0, 20),
    "coarse_ref_3x": (dict(size=256, ref_gsd=30.0, mov_gsd=10.0,
                           d_fn=_synth.d_constant(3.0 * G, -2.0 * G)), 1.5, 10),
    "cross_crs": (dict(size=256, mov_epsg=_synth.EPSG_UTM12N,
                       d_fn=_synth.d_constant(3.0 * G, -2.0 * G)), 1.5, 20),
    "multiband": (dict(size=256, bands=3,
                       d_fn=_synth.d_constant(3.5 * G, -2.5 * G)), 1.0, 20),
    "with_nodata": (dict(size=256, nodata=0, mov_nodata_frac=0.12,
                         d_fn=_synth.d_constant(3.0 * G, -2.0 * G)), 1.0, 15),
}

_CACHE: dict = {}


@pytest.fixture(scope="session")
def run_scenario(session_tmp):
    """Build a scenario's pair, run the full pipeline once, cache the result."""
    def _run(name):
        if name not in _CACHE:
            kwargs, _, _ = SCENARIOS[name]
            work = session_tmp / f"scn_{name}"
            work.mkdir(exist_ok=True)
            pair = _synth.make_pair(str(work), name=name, **kwargs)
            out_dir = str(work / "out")
            geocoreg.run_match(pair.ref_path, pair.mov_path, out_dir)
            _CACHE[name] = (pair, out_dir)
        return _CACHE[name]
    return _run


def _control_point_epe(pair, out_dir):
    """End-point error at every exported control point, in MOV pixels."""
    points, crs = control_points.read_control_points(
        os.path.join(out_dir, "control_points.geojson"))
    assert points, "the run exported no control points"
    x = np.array([p["mov_geo_x"] for p in points])
    y = np.array([p["mov_geo_y"] for p in points])
    dx = np.array([p["dx"] for p in points])
    dy = np.array([p["dy"] for p in points])
    true_dx, true_dy = pair.gt_at(x, y)
    return np.hypot(dx - true_dx, dy - true_dy) / pair.mov_gsd, points, crs


# --------------------------------------------------------------------------
# the coverage matrix
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(SCENARIOS))
def test_run_match_recovers_the_known_displacement(run_scenario, name):
    _, ceiling, min_points = SCENARIOS[name]
    pair, out_dir = run_scenario(name)
    epe, points, _ = _control_point_epe(pair, out_dir)

    assert len(points) >= min_points, f"only {len(points)} control points"
    assert np.median(epe) < ceiling, (
        f"median EPE {np.median(epe):.3f} px exceeds the loose ceiling {ceiling}")


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_run_match_writes_the_expected_artifacts(run_scenario, name):
    pair, out_dir = run_scenario(name)
    for artifact in ("matched.tif", "control_points.geojson",
                     "control_points.points", "log.txt"):
        path = os.path.join(out_dir, artifact)
        assert os.path.isfile(path), f"{artifact} missing"
        assert os.path.getsize(path) > 0, f"{artifact} is empty"


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_output_raster_follows_mov_geometry_and_radiometry(run_scenario, name):
    """Output geometry follows the MOV overlap; bands/dtype/nodata follow MOV."""
    pair, out_dir = run_scenario(name)
    mov = gdal.Open(pair.mov_path)
    out = gdal.Open(os.path.join(out_dir, "matched.tif"))

    assert out.RasterCount == mov.RasterCount
    assert out.GetRasterBand(1).DataType == mov.GetRasterBand(1).DataType
    assert out.GetRasterBand(1).GetNoDataValue() == mov.GetRasterBand(1).GetNoDataValue()
    out_gt = out.GetGeoTransform()
    assert out_gt[1] == pytest.approx(pair.mov_gt[1])
    assert out_gt[5] == pytest.approx(pair.mov_gt[5])
    assert out.GetProjection() == mov.GetProjection()
    # The whole of MOV overlaps REF in every scenario, so nothing is cropped.
    assert (out.RasterXSize, out.RasterYSize) == (mov.RasterXSize, mov.RasterYSize)
    out = mov = None


def test_control_points_carry_the_mov_crs(run_scenario):
    pair, out_dir = run_scenario("cross_crs")
    _, points, crs = _control_point_epe(pair, out_dir)
    assert crs == pair.mov_wkt
    doc = json.loads(open(os.path.join(out_dir, "control_points.geojson")).read())
    assert doc["properties"]["mov_path"] == pair.mov_path


def test_the_dense_field_is_accurate_between_the_control_points(session_tmp):
    """The exported points are only the anchors; scoring is done on the filled
    field, so the interpolation is checked too."""
    work = session_tmp / "dense"
    work.mkdir(exist_ok=True)
    pair = _synth.make_pair(str(work), size=256, name="dense",
                            d_fn=_synth.d_nonrigid(amp=1.5 * G, period=2000.0))
    ctx, st = init.run(pair.ref_path, pair.mov_path)
    matcher = RootSIFTMatcher(config.SIFT_NFEATURES, config.RATIO_THRESH,
                              config.RANSAC_REPROJ_PATCH,
                              config.MULTICHAN_DEDUP_RADIUS)
    out = refine.run(ctx, st, matcher, diag_folder=None)

    truth = pair.gt_field(ctx.gt_mov_overlap, ctx.mov_crop[2], ctx.mov_crop[3])
    epe = _synth.epe_pixels(out.D, truth, pair.mov_gsd)
    assert np.median(epe) < 1.5
    assert np.percentile(epe, 95) < 3.0
    assert np.isfinite(out.D).all()


def test_a_single_band_and_a_three_band_run_agree_on_the_geometry(run_scenario):
    """The extra bands must sharpen the match, not move it."""
    one_pair, one_dir = run_scenario("translation")
    three_pair, three_dir = run_scenario("multiband")
    one = np.median(_control_point_epe(one_pair, one_dir)[0])
    three = np.median(_control_point_epe(three_pair, three_dir)[0])
    assert abs(one - three) < 0.5


# --------------------------------------------------------------------------
# the apply path
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["translation", "nonrigid", "cross_crs",
                                  "coarse_mov_2x", "with_nodata"])
def test_run_apply_reproduces_run_match(run_scenario, tmp_path, name):
    """Replaying a run's own control points reproduces that run's raster.

    Both paths select their final interpolator the same way, so this is an
    equality test: a difference here means the two entry points have drifted
    apart, which is exactly the bug this pins down (``run_apply`` used to fill
    with a fixed MLS while ``run_match`` cross-validated, and nearest-neighbor
    resampling turned that sub-pixel difference into flipped samples).
    """
    pair, match_dir = run_scenario(name)
    cp_path = os.path.join(match_dir, "control_points.geojson")
    apply_dir = str(tmp_path / f"apply_{name}")
    geocoreg.run_apply(pair.ref_path, pair.mov_path, apply_dir, cp_path)

    applied = _synth.read_raster(os.path.join(apply_dir, "matched.tif"))[0]
    matched = _synth.read_raster(os.path.join(match_dir, "matched.tif"))[0]
    assert applied.shape == matched.shape
    agree = (applied == matched).mean()
    assert agree == 1.0, f"only {agree:.6%} of samples agree with run_match"


def test_run_apply_output_has_the_same_geometry_as_run_match(run_scenario, tmp_path):
    pair, match_dir = run_scenario("multiband")
    apply_dir = str(tmp_path / "apply_geom")
    geocoreg.run_apply(pair.ref_path, pair.mov_path, apply_dir,
                       os.path.join(match_dir, "control_points.geojson"))

    a = gdal.Open(os.path.join(apply_dir, "matched.tif"))
    b = gdal.Open(os.path.join(match_dir, "matched.tif"))
    assert a.GetGeoTransform() == pytest.approx(b.GetGeoTransform())
    assert (a.RasterXSize, a.RasterYSize) == (b.RasterXSize, b.RasterYSize)
    assert a.RasterCount == b.RasterCount
    assert a.GetProjection() == b.GetProjection()
    a = b = None


def test_run_apply_ignores_control_points_outside_the_overlap(run_scenario,
                                                              tmp_path):
    pair, match_dir = run_scenario("translation")
    points, crs = control_points.read_control_points(
        os.path.join(match_dir, "control_points.geojson"))
    far = dict(points[0])
    far["mov_geo_x"] += 5.0e5
    far["corrected_x"] += 5.0e5
    cp_path = str(tmp_path / "with_outsider.geojson")
    control_points.write_control_points(cp_path, points + [far], crs,
                                        pair.ref_path, pair.mov_path)

    apply_dir = str(tmp_path / "apply_outsider")
    geocoreg.run_apply(pair.ref_path, pair.mov_path, apply_dir, cp_path)
    assert os.path.isfile(os.path.join(apply_dir, "matched.tif"))


# --------------------------------------------------------------------------
# failure paths
# --------------------------------------------------------------------------

def test_disjoint_rasters_raise_a_message_that_says_so(session_tmp):
    wkt = _synth.wkt_for(_synth.EPSG_UTM11N)
    img = _synth.to_dtype(_synth.texture(128, 128, seed=1))
    a = _synth.write_raster(session_tmp / "far_a.tif", img,
                            _synth.north_up_gt(500000.0, 4800000.0, G), wkt)
    b = _synth.write_raster(session_tmp / "far_b.tif", img,
                            _synth.north_up_gt(900000.0, 4300000.0, G), wkt)
    with pytest.raises(RuntimeError, match="No overlap between REF and MOV"):
        geocoreg.run_match(a, b, str(session_tmp / "no_overlap_out"))


def test_a_missing_input_file_is_reported_by_path(session_tmp):
    wkt = _synth.wkt_for(_synth.EPSG_UTM11N)
    ok = _synth.write_raster(session_tmp / "present.tif",
                             _synth.to_dtype(_synth.texture(64, 64)),
                             _synth.north_up_gt(500000.0, 4800000.0, G), wkt)
    missing = str(session_tmp / "definitely_absent.tif")
    with pytest.raises((RuntimeError, FileNotFoundError)) as excinfo:
        geocoreg.run_match(missing, ok, str(session_tmp / "missing_out"))
    assert "definitely_absent" in str(excinfo.value)


def test_an_all_nodata_moving_image_yields_no_correction_rather_than_garbage(
        session_tmp):
    """Nothing can be measured, so the field must stay at zero displacement.

    It does not raise and it does not warn; the point of the test is that the
    output is inert, not wrong.
    """
    wkt = _synth.wkt_for(_synth.EPSG_UTM11N)
    blank = np.zeros((256, 256), np.uint16)
    mov = _synth.write_raster(session_tmp / "blank_mov.tif", blank,
                              _synth.north_up_gt(500000.0, 4800000.0, G), wkt,
                              nodata=0)
    ref = _synth.write_raster(session_tmp / "blank_ref.tif",
                              _synth.to_dtype(_synth.texture(320, 320, seed=3)),
                              _synth.north_up_gt(499700.0, 4800300.0, G), wkt)

    ctx, st = init.run(ref, mov)
    out = refine.run(ctx, st, RootSIFTMatcher(config.SIFT_NFEATURES,
                                              config.RATIO_THRESH,
                                              config.RANSAC_REPROJ_PATCH,
                                              config.MULTICHAN_DEDUP_RADIUS),
                     diag_folder=None)
    assert np.isfinite(out.D).all()
    assert np.array_equal(out.D, np.zeros_like(out.D))
    assert int(out.F.sum()) <= 1, "matched points found in an all-nodata image"


def test_a_textureless_pair_yields_no_correction_rather_than_garbage(session_tmp):
    pair = _synth.make_pair(str(session_tmp), size=256, name="flat", flat=True,
                            d_fn=_synth.d_constant(3.0 * G, -2.0 * G))
    ctx, st = init.run(pair.ref_path, pair.mov_path)
    out = refine.run(ctx, st, RootSIFTMatcher(config.SIFT_NFEATURES,
                                              config.RATIO_THRESH,
                                              config.RANSAC_REPROJ_PATCH,
                                              config.MULTICHAN_DEDUP_RADIUS),
                     diag_folder=None)
    assert np.isfinite(out.D).all()
    assert np.array_equal(out.D, np.zeros_like(out.D))


def test_a_textureless_pair_still_produces_a_readable_output(session_tmp):
    """No match is available, so the output must be MOV placed as-is."""
    pair = _synth.make_pair(str(session_tmp), size=256, name="flat2", flat=True,
                            d_fn=_synth.d_constant(3.0 * G, -2.0 * G))
    out_dir = str(session_tmp / "flat_out")
    geocoreg.run_match(pair.ref_path, pair.mov_path, out_dir)

    result = _synth.read_raster(os.path.join(out_dir, "matched.tif"))[0]
    original = _synth.read_raster(pair.mov_path)[0]
    assert result.shape == original.shape
    assert np.array_equal(result, original)


def test_a_raster_without_a_crs_is_rejected(session_tmp):
    """Currently a raw GDAL/OGR error; pinned here so the behavior cannot
    silently become a wrong warp."""
    nocrs = _synth.write_raster(session_tmp / "nocrs.tif",
                                _synth.to_dtype(_synth.texture(128, 128, seed=4)),
                                _synth.north_up_gt(500000.0, 4800000.0, G), "")
    with pytest.raises(Exception):
        geocoreg.run_match(nocrs, nocrs, str(session_tmp / "nocrs_out"))
