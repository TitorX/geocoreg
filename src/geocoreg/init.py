"""Initialization: raster I/O, overlap, coordinate setup."""

import logging
import os
import time
import numpy as np
from osgeo import gdal, osr

from geocoreg import geo_utils
from geocoreg.state import InitContext, DisplacementState

log = logging.getLogger("geocoreg.init")

gdal.UseExceptions()


class RasterInfo:
    """Container for raster metadata and handles."""

    def __init__(self, path):
        self.path = path
        self.filename = os.path.basename(path)
        self.ds = gdal.Open(path, gdal.GA_ReadOnly)
        if self.ds is None:
            raise FileNotFoundError(f"Cannot open raster: {path}")

        self.w = self.ds.RasterXSize
        self.h = self.ds.RasterYSize
        self.bands = self.ds.RasterCount
        self.gt = self.ds.GetGeoTransform()
        self.crs_wkt = self.ds.GetProjection()
        self.dtype = self.ds.GetRasterBand(1).DataType
        self.nodata = self.ds.GetRasterBand(1).GetNoDataValue()

        srs = osr.SpatialReference()
        srs.ImportFromWkt(self.crs_wkt)
        self.crs_name = srs.GetAuthorityCode(None)
        if self.crs_name:
            self.crs_name = f"EPSG:{self.crs_name}"
        else:
            self.crs_name = srs.GetName() or "Unknown"

        self.pixel_size = abs(self.gt[1])

    def read_overlap(self, col_off, row_off, w, h):
        """Read a crop from the raster as numpy array (H, W, bands)."""
        data = self.ds.ReadAsArray(col_off, row_off, w, h)
        if data is None:
            raise RuntimeError(f"Failed to read from {self.path}")
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        else:
            data = np.transpose(data, (1, 2, 0))  # (bands, H, W) → (H, W, bands)
        return data

    def read_full_downsampled(self, max_side=2000):
        """Read the full raster, downsampled for display."""
        scale = min(1.0, max_side / max(self.w, self.h))
        out_w = max(1, int(self.w * scale))
        out_h = max(1, int(self.h * scale))
        data = self.ds.ReadAsArray(buf_xsize=out_w, buf_ysize=out_h)
        if data is None:
            return np.zeros((out_h, out_w), dtype=np.uint8)
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        else:
            data = np.transpose(data, (1, 2, 0))
        return data


def run(ref_path, mov_path):
    """Execute Phase 0.

    Returns:
        (InitContext, DisplacementState)
    """
    t0 = time.time()

    # Step 1: Open rasters
    ref = RasterInfo(ref_path)
    mov = RasterInfo(mov_path)

    log.info(f"REF raster: {ref.w}×{ref.h} px, CRS={ref.crs_name}, "
             f"pixel size={ref.pixel_size:.6g} [CRS units]")
    log.info(f"MOV raster: {mov.w}×{mov.h} px, CRS={mov.crs_name}, "
             f"pixel size={mov.pixel_size:.6g} [CRS units]")

    # Step 2: CRS transforms
    ct_mov2ref = geo_utils.make_transform(mov.crs_wkt, ref.crs_wkt)
    ct_ref2mov = geo_utils.make_transform(ref.crs_wkt, mov.crs_wkt)

    if ct_mov2ref is not None:
        log.info("CRS differ — coordinate transforms created")
    else:
        log.info("Same CRS — no coordinate transform needed")

    # Step 3: Overlap in REF CRS
    ref_bbox = geo_utils.bbox_in_crs(ref.gt, ref.w, ref.h, ct=None)
    mov_bbox = geo_utils.bbox_in_crs(mov.gt, mov.w, mov.h, ct=ct_mov2ref)

    overlap_bbox = geo_utils.compute_overlap(ref_bbox, mov_bbox)
    if overlap_bbox is None:
        raise RuntimeError("No overlap between REF and MOV rasters")

    log.info(f"Overlap extent in REF CRS: "
             f"x=[{overlap_bbox[0]:.1f}, {overlap_bbox[1]:.1f}] "
             f"y=[{overlap_bbox[2]:.1f}, {overlap_bbox[3]:.1f}]")

    # Step 4: Pixel crop bounds
    ref_crop = geo_utils.overlap_to_crop(ref.gt, ref.w, ref.h, overlap_bbox)
    mov_crop = geo_utils.overlap_to_crop(mov.gt, mov.w, mov.h, overlap_bbox
                                          if ct_ref2mov is None else
                                          _overlap_in_mov_crs(overlap_bbox, ct_ref2mov))

    log.info(f"REF crop: col={ref_crop[0]}, row={ref_crop[1]}, "
             f"w={ref_crop[2]}, h={ref_crop[3]}")
    log.info(f"MOV crop: col={mov_crop[0]}, row={mov_crop[1]}, "
             f"w={mov_crop[2]}, h={mov_crop[3]}")

    # GeoTransform for the MOV overlap region
    gt_mov_overlap = geo_utils.shifted_geotransform(mov.gt, mov_crop[0], mov_crop[1])

    # Step 5: GSD ratio
    center_col = ref_crop[2] // 2
    center_row = ref_crop[3] // 2
    gsd_ratio_x, gsd_ratio_y = geo_utils.compute_gsd_ratio(
        ref.gt, mov.gt,
        ref_crop[0] + center_col, ref_crop[1] + center_row,
        ct_ref2mov
    )

    log.info(f"GSD ratio at center: 1 REF pixel ≈ {gsd_ratio_x:.1f} MOV pixels (x), "
             f"{gsd_ratio_y:.1f} MOV pixels (y)")

    # Read overlap data
    ref_overlap_data = ref.read_overlap(*ref_crop)
    mov_overlap_data = mov.read_overlap(*mov_crop)

    ctx = InitContext(
        ref=ref,
        mov=mov,
        ct_mov2ref=ct_mov2ref,
        ct_ref2mov=ct_ref2mov,
        overlap_bbox=overlap_bbox,
        ref_crop=ref_crop,
        mov_crop=mov_crop,
        gt_mov_overlap=gt_mov_overlap,
        gsd_ratio_x=gsd_ratio_x,
        gsd_ratio_y=gsd_ratio_y,
        ref_overlap_data=ref_overlap_data,
        mov_overlap_data=mov_overlap_data
    )

    # Step 6: Allocate D, F, C on the MOV overlap grid
    H, W = mov_crop[3], mov_crop[2]
    D = np.zeros((H, W, 2), dtype=np.float64)
    F = np.zeros((H, W), dtype=np.uint8)
    C = np.zeros((H, W), dtype=np.float32)

    # Seed 1 point at center with a zero-displacement initial guess
    cr, cc = H // 2, W // 2
    F[cr, cc] = 1
    C[cr, cc] = 0.1
    log.info(f"D seeded at center ({cc}, {cr}): "
             f"dx={D[cr, cc, 0]:.4f}, dy={D[cr, cc, 1]:.4f}")

    state = DisplacementState(D=D, F=F, C=C)

    log.info(f"Init complete ({time.time() - t0:.1f}s)")

    return ctx, state


def _overlap_in_mov_crs(overlap_ref, ct_ref2mov):
    """Transform the overlap bbox from REF CRS to MOV CRS."""
    corners = [
        (overlap_ref[0], overlap_ref[2]),
        (overlap_ref[1], overlap_ref[2]),
        (overlap_ref[1], overlap_ref[3]),
        (overlap_ref[0], overlap_ref[3]),
    ]
    xs, ys = [], []
    for cx, cy in corners:
        tx, ty = geo_utils.transform_point(ct_ref2mov, cx, cy)
        xs.append(tx)
        ys.append(ty)
    return min(xs), max(xs), min(ys), max(ys)
