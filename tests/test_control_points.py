"""Control-point serialization (GeoJSON round trip, GCP text format)."""

from __future__ import annotations

import json

import _synth
import numpy as np
import pytest

from geocoreg import control_points, geo_utils

POINTS = [
    {"mov_geo_x": 500123.456789, "mov_geo_y": 4799876.543211,
     "corrected_x": 500128.456789, "corrected_y": 4799869.543211,
     "dx": 5.0, "dy": -7.0, "confidence": 0.875},
    {"mov_geo_x": 501000.0, "mov_geo_y": 4799000.0,
     "corrected_x": 500999.25, "corrected_y": 4799002.5,
     "dx": -0.75, "dy": 2.5, "confidence": 0.125},
    {"mov_geo_x": 502222.2222222222, "mov_geo_y": 4798888.888888888,
     "corrected_x": 502222.2222222222, "corrected_y": 4798888.888888888,
     "dx": 0.0, "dy": 0.0, "confidence": 1.0},
]


# --------------------------------------------------------------------------
# GeoJSON
# --------------------------------------------------------------------------

def test_geojson_round_trip_is_identity(tmp_path):
    wkt = _synth.wkt_for(_synth.EPSG_UTM11N)
    path = tmp_path / "cp.geojson"
    control_points.write_control_points(str(path), POINTS, wkt, "ref.tif", "mov.tif")
    back, crs = control_points.read_control_points(str(path))

    assert crs == wkt
    assert len(back) == len(POINTS)
    for got, want in zip(back, POINTS):
        assert set(got) == set(want)
        for key in want:
            assert got[key] == want[key], key


def test_geojson_layout_is_a_valid_feature_collection(tmp_path):
    wkt = _synth.wkt_for(_synth.EPSG_UTM11N)
    path = tmp_path / "cp.geojson"
    control_points.write_control_points(str(path), POINTS, wkt, "r.tif", "m.tif")
    doc = json.loads(path.read_text())

    assert doc["type"] == "FeatureCollection"
    assert doc["crs_wkt"] == wkt
    assert doc["properties"]["ref_path"] == "r.tif"
    assert doc["properties"]["mov_path"] == "m.tif"
    assert "created" in doc["properties"]
    assert len(doc["features"]) == len(POINTS)

    feat = doc["features"][0]
    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] == "Point"
    # Geometry holds the ORIGINAL moving position; the corrected one is a
    # property. Swapping the two would invert every displacement.
    assert feat["geometry"]["coordinates"] == [POINTS[0]["mov_geo_x"],
                                               POINTS[0]["mov_geo_y"]]
    assert feat["properties"]["corrected_x"] == POINTS[0]["corrected_x"]
    assert feat["properties"]["dx"] == POINTS[0]["dx"]


def test_geojson_preserves_a_geographic_crs_verbatim(tmp_path):
    wkt = _synth.wkt_for(_synth.EPSG_WGS84)
    path = tmp_path / "cp_wgs84.geojson"
    control_points.write_control_points(str(path), POINTS[:1], wkt, "r", "m")
    _, crs = control_points.read_control_points(str(path))
    assert crs == wkt


def test_empty_point_list_round_trips(tmp_path):
    path = tmp_path / "empty.geojson"
    control_points.write_control_points(str(path), [], "", "r", "m")
    back, crs = control_points.read_control_points(str(path))
    assert back == []
    assert crs == ""


# --------------------------------------------------------------------------
# QGIS Georeferencer .points GCP format
# --------------------------------------------------------------------------

def test_points_file_header_and_column_order(tmp_path):
    wkt = _synth.wkt_for(_synth.EPSG_UTM11N)
    path = tmp_path / "cp.points"
    control_points.write_points_file(str(path), POINTS, wkt)
    lines = path.read_text().splitlines()

    assert lines[0] == f"#CRS: {wkt}"
    assert lines[1] == "mapX,mapY,sourceX,sourceY,enable,dX,dY,residual"
    assert len(lines) == 2 + len(POINTS)

    for line, pt in zip(lines[2:], POINTS):
        fields = line.split(",")
        assert len(fields) == 8
        # map* is the CORRECTED position, source* the original moving one.
        assert float(fields[0]) == pt["corrected_x"]
        assert float(fields[1]) == pt["corrected_y"]
        assert float(fields[2]) == pt["mov_geo_x"]
        assert float(fields[3]) == pt["mov_geo_y"]
        assert fields[4:] == ["1", "0", "0", "0"]


def test_points_file_keeps_full_double_precision(tmp_path):
    path = tmp_path / "precision.points"
    pt = {"mov_geo_x": 500000.123456789012, "mov_geo_y": 4799999.987654321,
          "corrected_x": 500000.223456789012, "corrected_y": 4799999.887654321}
    control_points.write_points_file(str(path), [pt], "")
    fields = path.read_text().splitlines()[2].split(",")
    assert float(fields[0]) == pt["corrected_x"]
    assert float(fields[3]) == pt["mov_geo_y"]


def test_points_file_ends_with_a_newline(tmp_path):
    path = tmp_path / "nl.points"
    control_points.write_points_file(str(path), POINTS, "")
    assert path.read_text().endswith("\n")


# --------------------------------------------------------------------------
# rasterization onto the MOV overlap grid
# --------------------------------------------------------------------------

def test_rasterize_places_points_and_computes_the_displacement():
    gt = _synth.north_up_gt(500000.0, 4800000.0, 10.0)
    pts = []
    for col, row, dx, dy in [(0, 0, 1.5, -2.5), (7, 3, -4.0, 8.0), (19, 19, 0.0, 0.0)]:
        x, y = geo_utils.pix2geo(gt, col + 0.5, row + 0.5)
        pts.append({"mov_geo_x": x, "mov_geo_y": y,
                    "corrected_x": x + dx, "corrected_y": y + dy,
                    "dx": dx, "dy": dy, "confidence": 0.5})

    D, F, C, n_inside = control_points.rasterize_control_points(pts, gt, 20, 20)
    assert n_inside == 3
    assert int(F.sum()) == 3
    assert D[0, 0] == pytest.approx([1.5, -2.5])
    assert D[3, 7] == pytest.approx([-4.0, 8.0])
    assert F[19, 19] == 1
    assert np.allclose(C[F == 1], 0.5)


def test_rasterize_drops_points_outside_the_grid():
    gt = _synth.north_up_gt(500000.0, 4800000.0, 10.0)
    inside_x, inside_y = geo_utils.pix2geo(gt, 5.5, 5.5)
    outside_x, outside_y = geo_utils.pix2geo(gt, 500.5, 500.5)
    pts = [
        {"mov_geo_x": inside_x, "mov_geo_y": inside_y,
         "corrected_x": inside_x + 1, "corrected_y": inside_y, "dx": 1.0,
         "dy": 0.0, "confidence": 1.0},
        {"mov_geo_x": outside_x, "mov_geo_y": outside_y,
         "corrected_x": outside_x, "corrected_y": outside_y, "dx": 0.0,
         "dy": 0.0, "confidence": 1.0},
    ]
    D, F, C, n_inside = control_points.rasterize_control_points(pts, gt, 20, 20)
    assert n_inside == 1
    assert int(F.sum()) == 1


def test_rasterize_uses_floor_so_pixel_centres_land_in_their_own_pixel():
    """A control point at a pixel center must not round into the next pixel."""
    gt = _synth.north_up_gt(0.0, 0.0, 1.0)
    pts = []
    for col, row in [(0, 0), (1, 1), (2, 2), (3, 3)]:
        x, y = geo_utils.pix2geo(gt, col + 0.5, row + 0.5)
        pts.append({"mov_geo_x": x, "mov_geo_y": y, "corrected_x": x,
                    "corrected_y": y, "dx": 0.0, "dy": 0.0, "confidence": 1.0})
    _, F, _, n = control_points.rasterize_control_points(pts, gt, 8, 8)
    assert n == 4
    assert set(map(tuple, np.argwhere(F == 1))) == {(0, 0), (1, 1), (2, 2), (3, 3)}


def test_rasterize_applies_a_crs_transform_when_the_points_come_from_elsewhere():
    """Points saved in another CRS must land in the right MOV pixels."""
    mov_wkt = _synth.wkt_for(_synth.EPSG_UTM11N)
    cp_wkt = _synth.wkt_for(_synth.EPSG_WGS84)
    gt = _synth.north_up_gt(500000.0, 4800000.0, 10.0)

    x, y = geo_utils.pix2geo(gt, 5.5, 7.5)
    to_wgs = geo_utils.make_transform(mov_wkt, cp_wkt)
    lon, lat = geo_utils.transform_point(to_wgs, x, y)
    lon2, lat2 = geo_utils.transform_point(to_wgs, x + 30.0, y - 20.0)
    pts = [{"mov_geo_x": lon, "mov_geo_y": lat, "corrected_x": lon2,
            "corrected_y": lat2, "dx": 0.0, "dy": 0.0, "confidence": 1.0}]

    ct = geo_utils.make_transform(cp_wkt, mov_wkt)
    D, F, C, n = control_points.rasterize_control_points(pts, gt, 20, 20, ct)
    assert n == 1
    assert F[7, 5] == 1
    assert D[7, 5, 0] == pytest.approx(30.0, abs=1e-3)
    assert D[7, 5, 1] == pytest.approx(-20.0, abs=1e-3)
