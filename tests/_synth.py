"""Procedural test-data generators — the repository ships no test imagery.

A test raster is a procedural texture (plenty of corners for SIFT to lock onto)
rendered into an arbitrary georeferenced grid, so one generator covers every
pair type the suite exercises: same/cross CRS, same/cross GSD, single- and
multi-band, with or without nodata.

The ground truth is analytic rather than measured. A moving raster is built by
sampling the shared base texture at ``P_mov(y, x) + D_geo(y, x)`` — the
position the pixel's content *should* report — so a perfect method recovers
exactly ``D_geo = corrected_mov_geo - original_mov_geo`` in MOV CRS units.
"""

from __future__ import annotations

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

# Two adjacent northern-hemisphere UTM zones plus WGS84: enough to exercise
# cross-CRS handling.
EPSG_UTM11N = 32611
EPSG_UTM12N = 32612
EPSG_WGS84 = 4326

# A base-texture origin inside UTM zone 11N, far from the zone edges.
BASE_ORIGIN_X = 500000.0
BASE_ORIGIN_Y = 4800000.0


def wkt_for(epsg: int) -> str:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(epsg))
    return srs.ExportToWkt()


def north_up_gt(x0: float, y0: float, gsd_x: float, gsd_y: float | None = None):
    """A north-up geotransform: index (0, 0) is the top-left *corner*."""
    if gsd_y is None:
        gsd_y = gsd_x
    return (float(x0), float(gsd_x), 0.0, float(y0), 0.0, -float(gsd_y))


# --------------------------------------------------------------------------
# procedural texture
# --------------------------------------------------------------------------

def texture(h: int, w: int, seed: int = 0, bands: int = 1) -> np.ndarray:
    """A deterministic texture with many corners, as ``(h, w)`` or ``(h, w, bands)``.

    Smooth low-frequency background + hard-edged rectangles and discs (corners
    for the feature matcher) + fine speckle (structure for phase correlation).
    """
    import cv2

    rng = np.random.default_rng(seed)
    lo_h, lo_w = max(2, h // 16), max(2, w // 16)
    img = rng.normal(0.0, 1.0, (lo_h, lo_w)).astype(np.float32)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

    n_shapes = max(24, (h * w) // 2000)
    for _ in range(n_shapes):
        val = float(rng.uniform(-2.5, 2.5))
        if rng.random() < 0.6:
            x0 = int(rng.integers(0, max(1, w - 8)))
            y0 = int(rng.integers(0, max(1, h - 8)))
            ww = int(rng.integers(6, max(7, w // 8)))
            hh = int(rng.integers(6, max(7, h // 8)))
            cv2.rectangle(img, (x0, y0), (min(w - 1, x0 + ww), min(h - 1, y0 + hh)),
                          val, thickness=-1)
        else:
            cx = int(rng.integers(0, w))
            cy = int(rng.integers(0, h))
            rad = int(rng.integers(4, max(5, min(h, w) // 12)))
            cv2.circle(img, (cx, cy), rad, val, thickness=-1)

    speckle = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    img = img + 0.35 * cv2.GaussianBlur(speckle, (0, 0), sigmaX=0.8)

    img -= img.min()
    peak = float(img.max())
    img = img / (peak if peak > 0 else 1.0)

    if bands == 1:
        return img
    out = np.empty((h, w, bands), dtype=np.float32)
    for b in range(bands):
        # Band-dependent tone curve: correlated but not identical channels.
        out[..., b] = np.clip(img ** (0.8 + 0.25 * b) * (0.85 + 0.1 * b), 0.0, 1.0)
    return out


def to_dtype(img01: np.ndarray, dtype=np.uint16, scale: float | None = None):
    """Scale a 0-1 float texture into an integer sample range."""
    dtype = np.dtype(dtype)
    if scale is None:
        scale = 8000.0 if dtype == np.uint16 else 250.0
    out = np.clip(img01 * scale, 0, np.iinfo(dtype).max if dtype.kind == "u" else None)
    return out.astype(dtype)


# --------------------------------------------------------------------------
# raster I/O
# --------------------------------------------------------------------------

def write_raster(path, arr, gt, crs_wkt, nodata=None):
    """Write ``arr`` (``(h, w)`` or ``(h, w, bands)``) as a GeoTIFF."""
    from osgeo import gdal_array

    a = arr if arr.ndim == 3 else arr[:, :, None]
    h, w, bands = a.shape
    gdt = gdal_array.NumericTypeCodeToGDALTypeCode(a.dtype)
    if gdt is None:
        raise TypeError(f"no GDAL type for dtype {a.dtype}")
    ds = gdal.GetDriverByName("GTiff").Create(str(path), w, h, bands, gdt)
    ds.SetGeoTransform(tuple(float(v) for v in gt))
    ds.SetProjection(crs_wkt)
    for b in range(bands):
        band = ds.GetRasterBand(b + 1)
        band.WriteArray(a[:, :, b])
        if nodata is not None:
            band.SetNoDataValue(float(nodata))
    ds.FlushCache()
    ds = None
    return str(path)


def read_raster(path):
    """Read a raster as ``(arr(h, w[, bands]), geotransform, wkt)``."""
    ds = gdal.Open(str(path))
    arr = ds.ReadAsArray()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    out = (arr, ds.GetGeoTransform(), ds.GetProjection())
    ds = None
    return out


# --------------------------------------------------------------------------
# grid rendering
# --------------------------------------------------------------------------

def grid_centres(gt, w, h):
    """Geographic coordinates of every pixel *center* of a grid."""
    cols, rows = np.meshgrid(np.arange(w, dtype=np.float64) + 0.5,
                             np.arange(h, dtype=np.float64) + 0.5)
    x = gt[0] + cols * gt[1] + rows * gt[2]
    y = gt[3] + cols * gt[4] + rows * gt[5]
    return x, y


def transform_xy(src_wkt, dst_wkt, x, y):
    """Transform coordinate arrays between two CRS (identity when equal)."""
    src = osr.SpatialReference()
    src.ImportFromWkt(src_wkt)
    dst = osr.SpatialReference()
    dst.ImportFromWkt(dst_wkt)
    if src.IsSame(dst):
        return x, y
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(src, dst)
    pts = ct.TransformPoints(np.column_stack([x.ravel(), y.ravel()]).tolist())
    arr = np.asarray(pts, dtype=np.float64)
    return arr[:, 0].reshape(x.shape), arr[:, 1].reshape(y.shape)


def render(base, base_gt, base_wkt, out_gt, out_w, out_h, out_wkt=None,
           d_geo=None, presmooth=0.0):
    """Sample ``base`` onto an arbitrary georeferenced grid.

    ``d_geo`` (``(out_h, out_w, 2)``, in the *output* CRS) offsets the sampling
    position, which is exactly the package's displacement convention: the
    content that ends up at output pixel (y, x) truly belongs at
    ``P_out(y, x) + d_geo[y, x]``.
    """
    import cv2

    if out_wkt is None:
        out_wkt = base_wkt
    x, y = grid_centres(out_gt, out_w, out_h)
    if d_geo is not None:
        x = x + d_geo[..., 0]
        y = y + d_geo[..., 1]
    bx, by = transform_xy(out_wkt, base_wkt, x, y)

    det = base_gt[1] * base_gt[5] - base_gt[2] * base_gt[4]
    col = (base_gt[5] * (bx - base_gt[0]) - base_gt[2] * (by - base_gt[3])) / det
    row = (-base_gt[4] * (bx - base_gt[0]) + base_gt[1] * (by - base_gt[3])) / det

    src = base
    if presmooth > 0:
        src = cv2.GaussianBlur(np.asarray(base, np.float32), (0, 0), sigmaX=presmooth)
    # cv2 maps are pixel-center indexed; GDAL columns are corner-indexed.
    map_x = (col - 0.5).astype(np.float32)
    map_y = (row - 0.5).astype(np.float32)
    out = cv2.remap(np.asarray(src, np.float32), map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    if out.ndim == 2 and base.ndim == 3:
        out = out[:, :, None]
    return out


# --------------------------------------------------------------------------
# displacement models (analytic ground truth, in MOV CRS units)
# --------------------------------------------------------------------------

def d_constant(dx, dy):
    def f(x, y):
        return np.full_like(x, float(dx)), np.full_like(y, float(dy))
    return f


def d_affine(dx0, dy0, a=2.0e-3, b=-1.0e-3, c=1.0e-3, d=2.0e-3, x0=None, y0=None):
    """A linear (affine) displacement gradient about a reference position."""
    def f(x, y):
        px = x0 if x0 is not None else float(np.mean(x))
        py = y0 if y0 is not None else float(np.mean(y))
        u = x - px
        v = y - py
        return dx0 + a * u + b * v, dy0 + c * u + d * v
    return f


def d_nonrigid(amp=18.0, period=1200.0, dx0=0.0, dy0=0.0):
    """A smooth, locally varying field no global affine can represent."""
    def f(x, y):
        px = float(np.mean(x))
        py = float(np.mean(y))
        u = (x - px) / period
        v = (y - py) / period
        return (dx0 + amp * np.sin(2 * np.pi * v) * np.cos(np.pi * u),
                dy0 + amp * np.cos(2 * np.pi * u) * np.sin(np.pi * v))
    return f


def eval_field(fn, gt, w, h):
    """Evaluate a displacement model on a grid's pixel centers -> ``(h, w, 2)``."""
    x, y = grid_centres(gt, w, h)
    dx, dy = fn(x, y)
    return np.stack([np.broadcast_to(dx, x.shape),
                     np.broadcast_to(dy, y.shape)], axis=-1).astype(np.float64)


# --------------------------------------------------------------------------
# pair construction
# --------------------------------------------------------------------------

class Pair:
    """A (REF, MOV) pair on disk with its analytic ground-truth field."""

    def __init__(self, ref_path, mov_path, mov_gt, mov_wkt, mov_w, mov_h,
                 d_fn, ref_gsd, mov_gsd):
        self.ref_path = ref_path
        self.mov_path = mov_path
        self.mov_gt = mov_gt
        self.mov_wkt = mov_wkt
        self.mov_w = mov_w
        self.mov_h = mov_h
        self.d_fn = d_fn
        self.ref_gsd = ref_gsd
        self.mov_gsd = mov_gsd

    def gt_field(self, gt=None, w=None, h=None):
        return eval_field(self.d_fn,
                          self.mov_gt if gt is None else gt,
                          self.mov_w if w is None else w,
                          self.mov_h if h is None else h)

    def gt_at(self, x, y):
        dx, dy = self.d_fn(np.asarray(x, float), np.asarray(y, float))
        return np.broadcast_to(dx, np.shape(x)), np.broadcast_to(dy, np.shape(y))


def make_pair(tmpdir, size=256, ref_gsd=10.0, mov_gsd=10.0, d_fn=None,
              mov_epsg=EPSG_UTM11N, ref_epsg=EPSG_UTM11N, bands=1,
              dtype=np.uint16, nodata=None, seed=0, mov_nodata_frac=0.0,
              flat=False, name="pair"):
    """Build a (REF, MOV) pair on disk plus its analytic ground truth.

    ``size`` is the MOV side length in pixels; REF is sized to cover the same
    ground extent at ``ref_gsd`` plus a margin, so the overlap is the whole of
    MOV and every MOV pixel has a REF partner.
    """
    import cv2

    if d_fn is None:
        d_fn = d_constant(0.0, 0.0)

    ref_wkt = wkt_for(ref_epsg)
    mov_wkt = wkt_for(mov_epsg)

    # Base texture: finer than both grids so resampling into either is a
    # downsample, which is what a real sensor pair looks like.
    base_gsd = min(ref_gsd, mov_gsd) / 2.0
    extent = size * mov_gsd
    # Generous enough that REF still covers MOV once a cross-CRS grid rotation
    # (UTM zone convergence is several degrees) tilts the footprint.
    margin = max(6 * max(ref_gsd, mov_gsd), 0.12 * extent)
    base_n = int(np.ceil((extent + 4 * margin) / base_gsd))
    base_gt = north_up_gt(BASE_ORIGIN_X - 2 * margin, BASE_ORIGIN_Y + 2 * margin,
                          base_gsd)
    if flat:
        base = np.full((base_n, base_n) + ((bands,) if bands > 1 else ()),
                       0.5, dtype=np.float32)
    else:
        base = texture(base_n, base_n, seed=seed, bands=bands)

    mov_gt = north_up_gt(BASE_ORIGIN_X, BASE_ORIGIN_Y, mov_gsd)
    if mov_epsg != EPSG_UTM11N:
        cx = BASE_ORIGIN_X + extent / 2.0
        cy = BASE_ORIGIN_Y - extent / 2.0
        tx, ty = transform_xy(wkt_for(EPSG_UTM11N), mov_wkt,
                              np.array([[cx]]), np.array([[cy]]))
        mov_gt = north_up_gt(float(tx[0, 0]) - extent / 2.0,
                             float(ty[0, 0]) + extent / 2.0, mov_gsd)

    d_geo = eval_field(d_fn, mov_gt, size, size)
    mov = render(base, base_gt, wkt_for(EPSG_UTM11N), mov_gt, size, size,
                 out_wkt=mov_wkt, d_geo=d_geo,
                 presmooth=max(0.0, 0.5 * (mov_gsd / base_gsd - 1.0)))

    # REF covers MOV's extent with a margin, on its own grid and CRS.
    ref_n = int(np.ceil((extent + 2 * margin) / ref_gsd))
    ref_gt = north_up_gt(BASE_ORIGIN_X - margin, BASE_ORIGIN_Y + margin, ref_gsd)
    if ref_epsg != EPSG_UTM11N:
        cx = BASE_ORIGIN_X + extent / 2.0
        cy = BASE_ORIGIN_Y - extent / 2.0
        tx, ty = transform_xy(wkt_for(EPSG_UTM11N), ref_wkt,
                              np.array([[cx]]), np.array([[cy]]))
        ref_gt = north_up_gt(float(tx[0, 0]) - (extent / 2.0 + margin),
                             float(ty[0, 0]) + (extent / 2.0 + margin), ref_gsd)
    ref = render(base, base_gt, wkt_for(EPSG_UTM11N), ref_gt, ref_n, ref_n,
                 out_wkt=ref_wkt,
                 presmooth=max(0.0, 0.5 * (ref_gsd / base_gsd - 1.0)))

    ref_arr = to_dtype(ref, dtype)
    mov_arr = to_dtype(mov, dtype)

    if nodata is not None:
        # Keep valid samples off the sentinel so nodata means nodata.
        ref_arr = np.where(ref_arr == nodata, nodata + 1, ref_arr).astype(dtype)
        mov_arr = np.where(mov_arr == nodata, nodata + 1, mov_arr).astype(dtype)
    if mov_nodata_frac > 0:
        if nodata is None:
            raise ValueError("mov_nodata_frac needs an explicit nodata value")
        edge = int(round(size * mov_nodata_frac))
        mov_arr[:edge, :] = nodata
        mov_arr[:, :edge] = nodata

    ref_path = write_raster(f"{tmpdir}/{name}_ref.tif", ref_arr, ref_gt, ref_wkt,
                            nodata=nodata)
    mov_path = write_raster(f"{tmpdir}/{name}_mov.tif", mov_arr, mov_gt, mov_wkt,
                            nodata=nodata)
    del cv2
    return Pair(ref_path, mov_path, mov_gt, mov_wkt, size, size, d_fn,
                ref_gsd, mov_gsd)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def epe_pixels(d_pred, d_true, gsd):
    """End-point error per pixel, in MOV pixels."""
    diff = np.asarray(d_pred, float) - np.asarray(d_true, float)
    return np.linalg.norm(diff, axis=-1) / float(gsd)
