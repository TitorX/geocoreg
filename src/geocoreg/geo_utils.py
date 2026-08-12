"""CRS transforms, pixel↔geo conversions, GSD computation."""

import logging
import numpy as np
from osgeo import osr

log = logging.getLogger("geocoreg.geo_utils")


def make_transform(src_wkt: str, dst_wkt: str):
    """Create an OGR CoordinateTransformation from WKT strings.

    Returns None if the CRS are identical.
    """
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src_wkt)
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromWkt(dst_wkt)

    if src_srs.IsSame(dst_srs):
        return None

    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return osr.CoordinateTransformation(src_srs, dst_srs)


def transform_point(ct, x, y):
    """Transform a single (x, y) point. If ct is None, return unchanged."""
    if ct is None:
        return x, y
    tx, ty, _ = ct.TransformPoint(x, y)
    return tx, ty


def transform_points(ct, xs, ys):
    """Transform arrays of points. If ct is None, return unchanged."""
    if ct is None:
        return xs.copy(), ys.copy()
    coords = [(float(xs[i]), float(ys[i])) for i in range(len(xs))]
    transformed = ct.TransformPoints(coords)
    out = np.array(transformed, dtype=np.float64)
    return out[:, 0], out[:, 1]


def pix2geo(gt, col, row):
    """Convert pixel coordinates to geographic coordinates using a GeoTransform."""
    x = gt[0] + col * gt[1] + row * gt[2]
    y = gt[3] + col * gt[4] + row * gt[5]
    return x, y


def geo2pix(gt, x, y):
    """Convert geographic coordinates to fractional pixel coordinates."""
    det = gt[1] * gt[5] - gt[2] * gt[4]
    col = (gt[5] * (x - gt[0]) - gt[2] * (y - gt[3])) / det
    row = (-gt[4] * (x - gt[0]) + gt[1] * (y - gt[3])) / det
    return col, row


def pix2geo_array(gt, cols, rows):
    """Vectorized pixel-to-geo conversion."""
    xs = gt[0] + cols * gt[1] + rows * gt[2]
    ys = gt[3] + cols * gt[4] + rows * gt[5]
    return xs, ys


def geo2pix_array(gt, xs, ys):
    """Vectorized geo-to-pixel conversion."""
    det = gt[1] * gt[5] - gt[2] * gt[4]
    cols = (gt[5] * (xs - gt[0]) - gt[2] * (ys - gt[3])) / det
    rows = (-gt[4] * (xs - gt[0]) + gt[1] * (ys - gt[3])) / det
    return cols, rows


def bbox_in_crs(gt, w, h, ct=None):
    """Compute bounding box of a raster in a (possibly different) CRS.

    Returns (xmin, xmax, ymin, ymax) in the target CRS.
    """
    corners_col = [0, w, w, 0]
    corners_row = [0, 0, h, h]
    xs, ys = [], []
    for c, r in zip(corners_col, corners_row):
        gx, gy = pix2geo(gt, c, r)
        gx, gy = transform_point(ct, gx, gy)
        xs.append(gx)
        ys.append(gy)
    return min(xs), max(xs), min(ys), max(ys)


def compute_overlap(ref_bbox, mov_bbox):
    """Compute intersection rectangle of two bounding boxes.

    Returns (xmin, xmax, ymin, ymax) or None if no overlap.
    """
    xmin = max(ref_bbox[0], mov_bbox[0])
    xmax = min(ref_bbox[1], mov_bbox[1])
    ymin = max(ref_bbox[2], mov_bbox[2])
    ymax = min(ref_bbox[3], mov_bbox[3])
    if xmin >= xmax or ymin >= ymax:
        return None
    return xmin, xmax, ymin, ymax


def overlap_to_crop(gt, w, h, overlap_bbox):
    """Convert an overlap bounding box to pixel crop bounds.

    Returns (col_off, row_off, crop_w, crop_h) clamped to raster edges.
    """
    c0, r0 = geo2pix(gt, overlap_bbox[0], overlap_bbox[3])  # top-left
    c1, r1 = geo2pix(gt, overlap_bbox[1], overlap_bbox[2])  # bottom-right

    col_off = max(0, int(np.floor(min(c0, c1))))
    row_off = max(0, int(np.floor(min(r0, r1))))
    col_end = min(w, int(np.ceil(max(c0, c1))))
    row_end = min(h, int(np.ceil(max(r0, r1))))

    return col_off, row_off, col_end - col_off, row_end - row_off


def compute_gsd_ratio(gt_ref, gt_mov, ref_col, ref_row, ct_ref2mov):
    """Compute how many MOV pixels correspond to one REF pixel at (ref_col, ref_row).

    Returns (ratio_x, ratio_y).
    """
    # REF pixel center and neighbors
    gx0, gy0 = pix2geo(gt_ref, ref_col, ref_row)
    gx1, gy1 = pix2geo(gt_ref, ref_col + 1, ref_row)
    gx2, gy2 = pix2geo(gt_ref, ref_col, ref_row + 1)

    # Transform to MOV CRS
    mx0, my0 = transform_point(ct_ref2mov, gx0, gy0)
    mx1, my1 = transform_point(ct_ref2mov, gx1, gy1)
    mx2, my2 = transform_point(ct_ref2mov, gx2, gy2)

    # Convert to MOV pixel coords
    mc0, mr0 = geo2pix(gt_mov, mx0, my0)
    mc1, mr1 = geo2pix(gt_mov, mx1, my1)
    mc2, mr2 = geo2pix(gt_mov, mx2, my2)

    ratio_x = abs(mc1 - mc0)
    ratio_y = abs(mr2 - mr0)

    return ratio_x, ratio_y


def shifted_geotransform(gt, col_off, row_off):
    """Create a GeoTransform shifted to a crop origin."""
    x0, y0 = pix2geo(gt, col_off, row_off)
    return (x0, gt[1], gt[2], y0, gt[4], gt[5])
