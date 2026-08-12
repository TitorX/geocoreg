"""The GDAL geolocation-array warp.

A half-pixel or channel-order mistake here produces a plausible-looking raster
that is systematically wrong, and nothing downstream raises, so two conventions
are pinned behaviorally rather than structurally:

* the geolocation arrays hold ``X`` in band X and ``Y`` in band Y — a swap
  would send every displacement down the wrong axis;
* ``PIXEL_OFFSET`` / ``LINE_OFFSET`` are ``0.5``, because the arrays store the
  corrected position of each source pixel's *center*. With ``0`` the whole
  output slides half a pixel on both axes.
"""

from __future__ import annotations

import _synth
import numpy as np
import pytest
from osgeo import gdal

from geocoreg import config, geo_utils, init, state, warp

gdal.UseExceptions()

SIZE = 64
GSD = 10.0
ORIGIN_X, ORIGIN_Y = 500000.0, 4800000.0
MOV_GT = _synth.north_up_gt(ORIGIN_X, ORIGIN_Y, GSD)


def _ramp_image(size=SIZE):
    """Every pixel a distinct value, so a shift is unambiguous."""
    r, c = np.mgrid[0:size, 0:size]
    return (r * size + c + 1).astype(np.uint16)


@pytest.fixture(scope="module")
def warp_ctx(tmp_path_factory):
    """A real ``InitContext`` whose MOV overlap is the whole MOV raster."""
    d = tmp_path_factory.mktemp("warp")
    wkt = _synth.wkt_for(_synth.EPSG_UTM11N)
    img = _ramp_image()
    mov_path = _synth.write_raster(d / "mov.tif", img, MOV_GT, wkt)
    ref_path = _synth.write_raster(d / "ref.tif", img, MOV_GT, wkt)
    ctx, _ = init.run(ref_path, mov_path)
    assert ctx.mov_crop == (0, 0, SIZE, SIZE)
    return ctx, img, str(d)


def _warp_with(ctx, d_geo, out_path):
    st = state.DisplacementState(
        D=np.asarray(d_geo, dtype=np.float64),
        F=np.ones(np.shape(d_geo)[:2], np.uint8),
        C=np.ones(np.shape(d_geo)[:2], np.float32))
    warp.run(ctx, st, out_path)
    return _synth.read_raster(out_path)[0]


def _const_field(dx, dy, size=SIZE):
    d = np.empty((size, size, 2), dtype=np.float64)
    d[..., 0] = dx
    d[..., 1] = dy
    return d


def _direct_gdal_translation(mov_path, dx, dy, out_path, gt=MOV_GT, size=SIZE):
    """The same geometry expressed as a plain affine shift, warped by GDAL.

    An independent path through GDAL: move the source's geotransform by
    ``(dx, dy)`` and resample it back onto the original grid.
    """
    src = gdal.Open(mov_path)
    vrt_path = out_path + ".src.vrt"
    vrt = gdal.Translate(vrt_path, src, format="VRT")
    vrt.SetGeoTransform((gt[0] + dx, gt[1], gt[2], gt[3] + dy, gt[4], gt[5]))
    vrt.FlushCache()
    vrt = None
    bounds = (gt[0], gt[3] + size * gt[5], gt[0] + size * gt[1], gt[3])
    gdal.Warp(out_path, vrt_path, format="GTiff", outputBounds=bounds,
              width=size, height=size, resampleAlg="near")
    return _synth.read_raster(out_path)[0]


# --------------------------------------------------------------------------
# geolocation arrays: contents and channel order
# --------------------------------------------------------------------------

def test_geolocation_arrays_hold_x_in_x_and_y_in_y(tmp_path):
    """The single most damaging possible typo in this package.

    Built with an asymmetric geotransform and an asymmetric displacement, so a
    swap of either the two output files or the two D channels is detected.
    """
    gt = (500000.0, 10.0, 0.0, 4800000.0, 0.0, -20.0)  # non-square pixels
    h, w = 5, 7
    D = np.empty((h, w, 2))
    D[..., 0] = 3.0    # x-geo displacement
    D[..., 1] = -111.0  # y-geo displacement
    x_path, y_path = str(tmp_path / "gx.tif"), str(tmp_path / "gy.tif")
    warp._write_geolocation_arrays(x_path, y_path, D, gt,
                                   _synth.wkt_for(_synth.EPSG_UTM11N))

    xs = _synth.read_raster(x_path)[0]
    ys = _synth.read_raster(y_path)[0]
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    want_x, want_y = geo_utils.pix2geo_array(gt, cols, rows)

    assert np.allclose(xs, want_x + 3.0)
    assert np.allclose(ys, want_y - 111.0)
    # And structurally: x must vary along columns, y along rows.
    assert np.allclose(np.diff(xs, axis=1), gt[1])
    assert np.allclose(np.diff(xs, axis=0), 0.0)
    assert np.allclose(np.diff(ys, axis=0), gt[5])
    assert np.allclose(np.diff(ys, axis=1), 0.0)


def test_geolocation_arrays_use_pixel_centres(tmp_path):
    """Index (0, 0) of the array is the *center* of source pixel (0, 0)."""
    gt = _synth.north_up_gt(1000.0, 2000.0, 4.0)
    D = np.zeros((3, 3, 2))
    x_path, y_path = str(tmp_path / "cx.tif"), str(tmp_path / "cy.tif")
    warp._write_geolocation_arrays(x_path, y_path, D, gt, "")
    xs = _synth.read_raster(x_path)[0]
    ys = _synth.read_raster(y_path)[0]
    assert xs[0, 0] == pytest.approx(1000.0 + 0.5 * 4.0)
    assert ys[0, 0] == pytest.approx(2000.0 - 0.5 * 4.0)


def test_non_finite_displacements_become_nodata(tmp_path):
    gt = _synth.north_up_gt(0.0, 0.0, 1.0)
    D = np.zeros((4, 4, 2))
    D[1, 2, 0] = np.nan
    D[3, 0, 1] = np.inf
    x_path, y_path = str(tmp_path / "nx.tif"), str(tmp_path / "ny.tif")
    warp._write_geolocation_arrays(x_path, y_path, D, gt, "")
    xs = _synth.read_raster(x_path)[0]
    ys = _synth.read_raster(y_path)[0]
    assert np.isnan(xs[1, 2]) and np.isnan(ys[1, 2])
    assert np.isnan(xs[3, 0]) and np.isnan(ys[3, 0])
    assert np.isfinite(xs[0, 0])

    ds = gdal.Open(x_path)
    assert np.isnan(ds.GetRasterBand(1).GetNoDataValue())
    ds = None


def test_chunked_writing_does_not_seam(tmp_path):
    """``WARP_CHUNK_ROWS`` is a memory knob and nothing else."""
    gt = _synth.north_up_gt(0.0, 0.0, 1.0)
    rng = np.random.default_rng(0)
    D = rng.normal(0, 3, (37, 11, 2))
    ref = None
    for rows in (1, 4, 37, 256):
        config.WARP_CHUNK_ROWS = rows
        x_path = str(tmp_path / f"x{rows}.tif")
        y_path = str(tmp_path / f"y{rows}.tif")
        warp._write_geolocation_arrays(x_path, y_path, D, gt, "")
        got = np.stack([_synth.read_raster(x_path)[0],
                        _synth.read_raster(y_path)[0]], axis=-1)
        if ref is None:
            ref = got
        else:
            assert np.array_equal(got, ref), f"WARP_CHUNK_ROWS={rows} changed output"


# --------------------------------------------------------------------------
# the GEOLOCATION metadata contract
# --------------------------------------------------------------------------

def test_pixel_and_line_offset_are_half_a_pixel(tmp_path, warp_ctx):
    """``0.5`` because the arrays store pixel *centers*.

    Setting these to ``0`` shifts every warped output by half a pixel on both
    axes; the functional consequence is pinned by the tests below.
    """
    ctx, _, _ = warp_ctx
    vrt_path = str(tmp_path / "src.vrt")
    warp._build_mov_overlap_vrt(vrt_path, ctx.mov, ctx.mov_crop)
    warp._attach_geolocation_metadata(vrt_path, "/nonexistent/x.tif",
                                      "/nonexistent/y.tif", ctx.mov.crs_wkt)

    ds = gdal.Open(vrt_path)
    meta = ds.GetMetadata("GEOLOCATION")
    ds = None
    assert meta["PIXEL_OFFSET"] == "0.5"
    assert meta["LINE_OFFSET"] == "0.5"
    assert meta["PIXEL_STEP"] == "1"
    assert meta["LINE_STEP"] == "1"
    assert meta["X_BAND"] == "1"
    assert meta["Y_BAND"] == "1"
    assert meta["X_DATASET"].endswith("x.tif")
    assert meta["Y_DATASET"].endswith("y.tif")


@pytest.mark.parametrize("frac,expected_shift", [(0.0, 0), (0.3, 0), (0.7, 1),
                                                 (1.0, 1), (-0.3, 0), (-0.7, -1)])
def test_fractional_displacements_snap_to_the_nearest_whole_pixel(warp_ctx, frac,
                                                                  expected_shift):
    """The behavioral signature of ``PIXEL_OFFSET=0.5``.

    Nearest-neighbor resampling of a constant field of ``frac`` pixels must
    land on ``round(frac)``. With ``PIXEL_OFFSET=0`` it lands on
    ``round(frac + 0.5)`` instead — the exact symptom recorded in warp.py.
    """
    ctx, img, workdir = warp_ctx
    config.WARP_INTERP_METHOD = "nearest"
    out = _warp_with(ctx, _const_field(frac * GSD, 0.0),
                     f"{workdir}/snap_{frac}.tif")

    s = expected_shift
    if s == 0:
        assert np.array_equal(out, img)
    elif s > 0:
        assert np.array_equal(out[:, s:], img[:, :-s])
    else:
        assert np.array_equal(out[:, :s], img[:, -s:])


# --------------------------------------------------------------------------
# end-to-end geometry against an independent GDAL path
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kx,ky", [(3, 0), (0, 2), (3, -2), (-4, 5)])
def test_constant_shift_matches_a_direct_gdal_translation(warp_ctx, kx, ky):
    """An exact whole-pixel displacement, two independent ways."""
    ctx, img, workdir = warp_ctx
    config.WARP_INTERP_METHOD = "nearest"
    dx, dy = kx * GSD, ky * GSD

    got = _warp_with(ctx, _const_field(dx, dy), f"{workdir}/shift_{kx}_{ky}.tif")
    want = _direct_gdal_translation(ctx.mov.path, dx, dy,
                                    f"{workdir}/direct_{kx}_{ky}.tif")
    assert np.array_equal(got, want)


def test_a_pure_x_displacement_moves_only_columns(warp_ctx):
    """Channel order, end to end: dx must not leak into rows."""
    ctx, img, workdir = warp_ctx
    config.WARP_INTERP_METHOD = "nearest"
    out = _warp_with(ctx, _const_field(5 * GSD, 0.0), f"{workdir}/only_x.tif")
    assert np.array_equal(out[:, 5:], img[:, :-5])
    assert np.all(out[:, :5] == 0)


def test_a_pure_y_displacement_moves_only_rows(warp_ctx):
    """A positive y-geo displacement moves content *up* (north-up grid)."""
    ctx, img, workdir = warp_ctx
    config.WARP_INTERP_METHOD = "nearest"
    out = _warp_with(ctx, _const_field(0.0, 4 * GSD), f"{workdir}/only_y.tif")
    assert np.array_equal(out[:-4, :], img[4:, :])
    assert np.all(out[-4:, :] == 0)


def test_linear_stretch_matches_a_rescaled_geotransform(warp_ctx):
    """A linear displacement gradient is a pixel-size change; GDAL agrees."""
    ctx, img, workdir = warp_ctx
    config.WARP_INTERP_METHOD = "nearest"
    scale = 1.25

    cols, rows = np.meshgrid(np.arange(SIZE) + 0.5, np.arange(SIZE) + 0.5)
    x, y = geo_utils.pix2geo_array(MOV_GT, cols, rows)
    D = np.stack([(x - ORIGIN_X) * (scale - 1.0),
                  (y - ORIGIN_Y) * (scale - 1.0)], axis=-1)
    got = _warp_with(ctx, D, f"{workdir}/stretch.tif")

    stretched_gt = (ORIGIN_X, GSD * scale, 0.0, ORIGIN_Y, 0.0, -GSD * scale)
    vrt_path = f"{workdir}/stretch_src.vrt"
    vrt = gdal.Translate(vrt_path, gdal.Open(ctx.mov.path), format="VRT")
    vrt.SetGeoTransform(stretched_gt)
    vrt.FlushCache()
    vrt = None
    bounds = (ORIGIN_X, ORIGIN_Y - SIZE * GSD, ORIGIN_X + SIZE * GSD, ORIGIN_Y)
    gdal.Warp(f"{workdir}/stretch_direct.tif", vrt_path, format="GTiff",
              outputBounds=bounds, width=SIZE, height=SIZE, resampleAlg="near")
    want = _synth.read_raster(f"{workdir}/stretch_direct.tif")[0]

    # Whole-pixel agreement everywhere except where a sample sits exactly on a
    # cell boundary and the two transformers break the tie differently.
    agree = (got == want).mean()
    assert agree > 0.98, f"only {agree:.3%} of samples agree"


# --------------------------------------------------------------------------
# output framing, radiometry and failure modes
# --------------------------------------------------------------------------

def test_output_geometry_follows_the_mov_overlap(warp_ctx):
    ctx, img, workdir = warp_ctx
    out_path = f"{workdir}/geom.tif"
    _warp_with(ctx, _const_field(0.0, 0.0), out_path)
    arr, gt, wkt = _synth.read_raster(out_path)
    assert arr.shape == (SIZE, SIZE)
    assert gt == pytest.approx(ctx.gt_mov_overlap)
    assert wkt == ctx.mov.crs_wkt


def test_output_radiometry_follows_mov(tmp_path):
    """Band count, dtype and nodata come from MOV, not from REF."""
    wkt = _synth.wkt_for(_synth.EPSG_UTM11N)
    rng = np.random.default_rng(0)
    mov_img = rng.integers(1, 300, (SIZE, SIZE, 3)).astype(np.uint16)
    ref_img = rng.integers(1, 60, (SIZE, SIZE)).astype(np.uint8)
    mov_path = _synth.write_raster(tmp_path / "m.tif", mov_img, MOV_GT, wkt,
                                   nodata=0)
    ref_path = _synth.write_raster(tmp_path / "r.tif", ref_img, MOV_GT, wkt)

    ctx, _ = init.run(ref_path, mov_path)
    out_path = str(tmp_path / "out.tif")
    _warp_with(ctx, _const_field(0.0, 0.0), out_path)

    ds = gdal.Open(out_path)
    assert ds.RasterCount == 3
    assert ds.GetRasterBand(1).DataType == gdal.GDT_UInt16
    assert ds.GetRasterBand(1).GetNoDataValue() == 0
    arr = np.transpose(ds.ReadAsArray(), (1, 2, 0))
    ds = None
    assert np.array_equal(arr, mov_img)


def test_warp_overwrites_an_existing_output(warp_ctx):
    ctx, img, workdir = warp_ctx
    out_path = f"{workdir}/overwrite.tif"
    _warp_with(ctx, _const_field(2 * GSD, 0.0), out_path)
    again = _warp_with(ctx, _const_field(0.0, 0.0), out_path)
    assert np.array_equal(again, img)


def test_interp_method_is_read_from_config(warp_ctx):
    """Bilinear must blend where nearest copies source values verbatim."""
    ctx, img, workdir = warp_ctx
    half = _const_field(0.5 * GSD, 0.5 * GSD)

    config.WARP_INTERP_METHOD = "nearest"
    near = _warp_with(ctx, half, f"{workdir}/near.tif")
    config.WARP_INTERP_METHOD = "linear"
    lin = _warp_with(ctx, half, f"{workdir}/lin.tif")
    assert not np.array_equal(near, lin)
    assert set(np.unique(near)) <= set(np.unique(img)) | {0}


def test_rotated_overlap_grid_is_refused(warp_ctx):
    """A skewed geotransform is rejected loudly rather than warped wrongly."""
    ctx, _, workdir = warp_ctx
    rotated = (ORIGIN_X, GSD, 1.5, ORIGIN_Y, 2.0, -GSD)
    skewed = type(ctx)(**{**ctx.__dict__, "gt_mov_overlap": rotated})
    with pytest.raises(NotImplementedError, match="north-up"):
        _warp_with(skewed, _const_field(0.0, 0.0), f"{workdir}/rot.tif")
