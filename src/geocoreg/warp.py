"""Warp MOV through GDAL geolocation arrays derived from D_geo."""

import logging
import os
import tempfile
import time

import numpy as np
from osgeo import gdal

from geocoreg import config, geo_utils

log = logging.getLogger("geocoreg.warp")

gdal.UseExceptions()


_RESAMPLE_ALG = {
    "nearest": "near",
    "near": "near",
    "linear": "bilinear",
    "bilinear": "bilinear",
    "cubic": "cubic",
}


def run(ctx, state, output_path):
    """Execute Phase 2 with GDAL geolocation arrays.

    D_geo is defined on the MOV overlap grid and stores
    corrected_mov_geo - original_mov_geo. GDAL geolocation arrays describe
    where each source pixel lives after correction, so they use original + D.
    """
    t0 = time.time()
    mov = ctx.mov
    H, W = state.D.shape[:2]
    gt_ov = ctx.gt_mov_overlap

    if (gt_ov[2] != 0.0) or (gt_ov[4] != 0.0):
        raise NotImplementedError(
            "GDAL geolocation warp currently expects north-up MOV overlap grids "
            f"(got geotransform={gt_ov})"
        )

    resample_alg = _RESAMPLE_ALG.get(
        config.WARP_INTERP_METHOD, config.WARP_INTERP_METHOD)
    creation_options = ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"]
    nodata_val = mov.nodata if mov.nodata is not None else None

    log.info("Warp backend: GDAL geolocation arrays")
    log.info(f"Output raster: {W}×{H} px, {mov.bands} bands, "
             f"dtype={gdal.GetDataTypeName(mov.dtype)}")
    log.info(f"GeoTransform: [{', '.join(f'{v}' for v in gt_ov)}]")
    log.info(f"CRS: {mov.crs_name}")
    log.info(f"Resampling: {resample_alg}")

    _remove_existing_output(output_path)

    temp_parent = _temp_parent_for(output_path)
    with tempfile.TemporaryDirectory(prefix="coreg_warp_", dir=temp_parent) as tmpdir:
        x_path = os.path.join(tmpdir, "geoloc_x.tif")
        y_path = os.path.join(tmpdir, "geoloc_y.tif")
        src_vrt_path = os.path.join(tmpdir, "mov_overlap_geoloc.vrt")

        _write_geolocation_arrays(x_path, y_path, state.D, gt_ov, mov.crs_wkt)
        _build_mov_overlap_vrt(src_vrt_path, mov, ctx.mov_crop)
        _attach_geolocation_metadata(src_vrt_path, x_path, y_path, mov.crs_wkt)

        bounds = _output_bounds(gt_ov, W, H)
        warp_kwargs = dict(
            format="GTiff",
            geoloc=True,
            srcSRS=mov.crs_wkt,
            dstSRS=mov.crs_wkt,
            outputBounds=bounds,
            width=W,
            height=H,
            resampleAlg=resample_alg,
            multithread=True,
            warpOptions=["NUM_THREADS=ALL_CPUS"],
            creationOptions=creation_options,
            callback=_warp_progress,
        )
        if nodata_val is not None:
            warp_kwargs["srcNodata"] = nodata_val
            warp_kwargs["dstNodata"] = nodata_val
        warp_options = gdal.WarpOptions(**warp_kwargs)

        out_ds = gdal.Warp(output_path, src_vrt_path, options=warp_options)
        if out_ds is None:
            raise RuntimeError(f"GDAL geolocation warp failed: {output_path}")
        out_ds.FlushCache()
        out_ds = None

    log.info(f"Warp complete. Output: {output_path}")
    log.info(f"Elapsed: {time.time() - t0:.1f}s")


def _write_geolocation_arrays(x_path, y_path, D, gt, crs_wkt):
    """Write corrected source-pixel coordinates for GDAL's geoloc transformer."""
    H, W = D.shape[:2]
    driver = gdal.GetDriverByName("GTiff")
    options = ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"]
    x_ds = driver.Create(x_path, W, H, 1, gdal.GDT_Float64, options=options)
    y_ds = driver.Create(y_path, W, H, 1, gdal.GDT_Float64, options=options)
    if x_ds is None or y_ds is None:
        raise RuntimeError("Failed to create GDAL geolocation arrays")

    x_ds.SetProjection(crs_wkt)
    y_ds.SetProjection(crs_wkt)
    x_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    y_ds.GetRasterBand(1).SetNoDataValue(np.nan)

    chunk_rows = max(1, int(getattr(config, "WARP_CHUNK_ROWS", 256)))
    n_chunks = (H + chunk_rows - 1) // chunk_rows
    progress_every = max(1, n_chunks // 10)
    log.info(f"Writing geolocation arrays: {W}×{H}, {n_chunks} chunks")

    cols = np.arange(W, dtype=np.float64)
    for i, r0 in enumerate(range(0, H, chunk_rows), start=1):
        r1 = min(H, r0 + chunk_rows)
        rows = np.arange(r0, r1, dtype=np.float64)
        CC, RR = np.meshgrid(cols, rows)
        orig_x, orig_y = geo_utils.pix2geo_array(gt, CC + 0.5, RR + 0.5)
        D_chunk = D[r0:r1, :, :]

        x_geo = orig_x + D_chunk[:, :, 0]
        y_geo = orig_y + D_chunk[:, :, 1]
        valid = np.isfinite(x_geo) & np.isfinite(y_geo)
        if not valid.all():
            x_geo = np.where(valid, x_geo, np.nan)
            y_geo = np.where(valid, y_geo, np.nan)

        x_ds.GetRasterBand(1).WriteArray(x_geo, 0, r0)
        y_ds.GetRasterBand(1).WriteArray(y_geo, 0, r0)

        if i == 1 or i == n_chunks or i % progress_every == 0:
            log.info(f"Geolocation chunk {i}/{n_chunks}")

    x_ds.FlushCache()
    y_ds.FlushCache()
    x_ds = None
    y_ds = None


def _build_mov_overlap_vrt(vrt_path, mov, mov_crop):
    """Create a VRT view of the MOV overlap, matching D's pixel grid."""
    src_win = [int(v) for v in mov_crop]
    ds = gdal.Translate(vrt_path, mov.ds, format="VRT", srcWin=src_win)
    if ds is None:
        raise RuntimeError("Failed to create MOV overlap VRT")
    ds.SetGeoTransform((0, 1, 0, 0, 0, 1))
    ds.SetProjection("")
    ds.FlushCache()
    ds = None


def _attach_geolocation_metadata(vrt_path, x_path, y_path, crs_wkt):
    """Attach GDAL GEOLOCATION domain metadata to the source VRT."""
    ds = gdal.Open(vrt_path, gdal.GA_Update)
    if ds is None:
        raise RuntimeError(f"Failed to reopen VRT for geolocation metadata: {vrt_path}")
    ds.SetMetadata({
        "SRS": crs_wkt,
        "X_DATASET": x_path,
        "X_BAND": "1",
        "Y_DATASET": y_path,
        "Y_BAND": "1",
        # The arrays store the corrected position of each source pixel's
        # CENTER (warp writes pix2geo(col+0.5, row+0.5) + D). GDAL's geoloc
        # transformer maps pixel coordinate P (corner units) to array index
        # (P - PIXEL_OFFSET)/PIXEL_STEP, so center values need OFFSET=0.5 —
        # with OFFSET=0 the whole output shifts by +0.5 px on both axes
        # (verified empirically on GDAL 3.12: fractional constant-D fields
        # NN-snap to round(d+0.5) instead of round(d)).
        "PIXEL_OFFSET": "0.5",
        "LINE_OFFSET": "0.5",
        "PIXEL_STEP": "1",
        "LINE_STEP": "1",
    }, "GEOLOCATION")
    ds.FlushCache()
    ds = None


def _output_bounds(gt, W, H):
    """Return north-up output bounds as (minX, minY, maxX, maxY)."""
    xs, ys = [], []
    for c, r in [(0, 0), (W, 0), (W, H), (0, H)]:
        x, y = geo_utils.pix2geo(gt, c, r)
        xs.append(x)
        ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def _warp_progress(complete, message, unused):
    pct = int(round(complete * 100.0))
    if pct in {0, 25, 50, 75, 100}:
        log.info(f"GDAL warp progress: {pct}%")
    return 1


def _temp_parent_for(output_path):
    """Use the output folder for temp files when it is a normal filesystem path."""
    if output_path.startswith("/vsi"):
        return None
    parent = os.path.dirname(os.path.abspath(output_path))
    return parent if parent and os.path.isdir(parent) else None


def _remove_existing_output(output_path):
    """Remove an existing output before gdal.Warp creates a new dataset."""
    if output_path.startswith("/vsi"):
        try:
            gdal.Unlink(output_path)
        except RuntimeError:
            pass
    elif os.path.exists(output_path):
        os.remove(output_path)
