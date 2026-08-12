"""Read/write control point files in GeoJSON format."""

import json
import logging
import numpy as np
from datetime import datetime, timezone

from geocoreg import geo_utils

log = logging.getLogger("geocoreg.control_points")


def write_control_points(path, points, mov_crs_wkt, ref_path, mov_path):
    """Write matched control points to a GeoJSON file.

    Args:
        path: output file path
        points: list of dicts with keys:
            mov_geo_x, mov_geo_y, corrected_x, corrected_y,
            dx, dy, confidence
        mov_crs_wkt: WKT string of the MOV CRS
        ref_path: path to the reference raster
        mov_path: path to the moving raster
    """
    features = []
    for pt in points:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [pt["mov_geo_x"], pt["mov_geo_y"]],
            },
            "properties": {
                "corrected_x": pt["corrected_x"],
                "corrected_y": pt["corrected_y"],
                "dx": pt["dx"],
                "dy": pt["dy"],
                "confidence": pt["confidence"],
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "crs_wkt": mov_crs_wkt,
        "properties": {
            "ref_path": str(ref_path),
            "mov_path": str(mov_path),
            "created": datetime.now(timezone.utc).isoformat(),
        },
        "features": features,
    }

    with open(path, "w") as f:
        json.dump(geojson, f, indent=2)

    log.info(f"Control points saved: {len(features)} points → {path}")


def write_points_file(path, points, mov_crs_wkt):
    """Write control points in QGIS .points GCP format.

    Args:
        path: output file path (.points)
        points: list of dicts with keys:
            mov_geo_x, mov_geo_y, corrected_x, corrected_y
        mov_crs_wkt: WKT string of the MOV CRS
    """
    lines = [f"#CRS: {mov_crs_wkt}",
             "mapX,mapY,sourceX,sourceY,enable,dX,dY,residual"]
    for pt in points:
        map_x = pt["corrected_x"]
        map_y = pt["corrected_y"]
        src_x = pt["mov_geo_x"]
        src_y = pt["mov_geo_y"]

        lines.append(
            f"{map_x:.17g},{map_y:.17g},{src_x:.17g},{src_y:.17g},1,0,0,0"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    log.info(f"QGIS .points file saved: {len(points)} points → {path}")


def read_control_points(path):
    """Read control points from a GeoJSON file.

    Returns:
        points: list of dicts with keys matching write_control_points
        crs_wkt: CRS WKT string from the file
    """
    with open(path) as f:
        geojson = json.load(f)

    crs_wkt = geojson.get("crs_wkt", "")
    features = geojson.get("features", [])

    points = []
    for feat in features:
        coords = feat["geometry"]["coordinates"]
        props = feat["properties"]
        points.append({
            "mov_geo_x": coords[0],
            "mov_geo_y": coords[1],
            "corrected_x": props["corrected_x"],
            "corrected_y": props["corrected_y"],
            "dx": props["dx"],
            "dy": props["dy"],
            "confidence": props["confidence"],
        })

    log.info(f"Control points loaded: {len(points)} points from {path}")
    return points, crs_wkt


def rasterize_control_points(points, gt_mov_overlap, H, W, ct_cp2mov=None):
    """Rasterize control points onto the MOV overlap pixel grid.

    Args:
        points: list of dicts from read_control_points
        gt_mov_overlap: GeoTransform of the MOV overlap region
        H, W: dimensions of the overlap grid
        ct_cp2mov: optional CRS transform if control point CRS differs from MOV

    Returns:
        D: (H, W, 2) displacement field
        F: (H, W) uint8 flag array
        C: (H, W) float32 confidence array
        n_inside: number of points that fell within the grid
    """
    D = np.zeros((H, W, 2), dtype=np.float64)
    F = np.zeros((H, W), dtype=np.uint8)
    C = np.zeros((H, W), dtype=np.float32)

    n_inside = 0
    for pt in points:
        mx, my = pt["mov_geo_x"], pt["mov_geo_y"]
        cx, cy = pt["corrected_x"], pt["corrected_y"]
        if ct_cp2mov is not None:
            mx, my = geo_utils.transform_point(ct_cp2mov, mx, my)
            cx, cy = geo_utils.transform_point(ct_cp2mov, cx, cy)

        col, row = geo_utils.geo2pix(gt_mov_overlap, mx, my)
        # floor, not round: pixel i spans [i, i+1) in corner-convention pixel
        # coords, and round() would banker's-round the exact .5 centers that
        # pixel-center control points land on.
        c, r = int(np.floor(col)), int(np.floor(row))

        if 0 <= c < W and 0 <= r < H:
            D[r, c, 0] = cx - mx
            D[r, c, 1] = cy - my
            F[r, c] = 1
            C[r, c] = pt["confidence"]
            n_inside += 1

    return D, F, C, n_inside
