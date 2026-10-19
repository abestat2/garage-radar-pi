"""Unit tests: point/frame models, unit helpers, garage geometry and coordinate transforms."""
import numpy as np
import pytest

from app.geometry.coordinate_transform import (boresight_vector, fov_edge_rays, garage_to_radar, in_fov,
                                               radar_angles, radar_to_garage, rotation)
from app.geometry.garage import Door, Garage, default_doors, default_stalls
from app.geometry.parking_zones import IgnoreZone, ParkingStall, default_car_zone, points_in_polygon
from app.geometry.radar_pose import RadarPose
from app.geometry.units import FT, fmt_len, from_m, to_m
from app.models.radar_frame import RadarFrame
from app.models.radar_object import RadarTrack
from app.models.radar_point import empty_points, ranges, xyz


def test_empty_points_defaults_to_nan_snr():
    assert len(empty_points()) == 0
    p = empty_points(3)
    assert len(p) == 3 and np.isnan(p["snr"]).all() and np.isnan(p["noise"]).all()


def test_xyz_and_ranges():
    p = empty_points(2)
    p["x"], p["y"], p["z"] = [3, 0], [4, 0], [0, 2]
    assert xyz(p).shape == (2, 3) and xyz(p).dtype == np.float64
    assert ranges(p).tolist() == [5.0, 2.0]


def test_radar_frame_point_count_and_defaults():
    f = RadarFrame(frame_number=7, timestamp=1.5)
    assert f.num_points == 0 and f.tracks == [] and not f.damaged
    f.points = empty_points(4)
    assert f.num_points == 4
    assert RadarTrack(1, 0.0, 1.0, 2.0).confidence == 0.0


def test_unit_conversions_round_trip():
    assert to_m(1, "ft") == FT and from_m(FT, "ft") == pytest.approx(1)
    assert to_m(12, "in") == pytest.approx(FT)
    assert from_m(to_m(3.3, "in"), "in") == pytest.approx(3.3)
    assert fmt_len(FT * 2) == "2.0 ft" and fmt_len(1.234, "m", 2) == "1.23 m"


def test_points_in_polygon():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
    got = points_in_polygon(np.array([[1, 1], [3, 1], [-1, -1]], float), sq)
    assert got.tolist() == [True, False, False]
    assert len(points_in_polygon(np.zeros((0, 2)), sq)) == 0


def test_parking_stall_geometry_and_round_trip():
    s = ParkingStall("A", left=0, right=2, front=0, rear=5)
    assert s.area() == pytest.approx(10) and s.centroid() == (1.0, 2.5)
    assert s.contains([[1, 1], [3, 1]]).tolist() == [True, False]
    zone = s.car_zone_vertices()
    assert zone.shape == (4, 2) and s.car_zone_contains([[1, 2.5], [0.001, 2.5]]).tolist() == [True, False]
    assert s.to_dict()["name"] == "A"
    again = ParkingStall.from_dict(s.to_dict())
    assert again == s


def test_stall_custom_polygon_and_car_zone_override():
    tri = [(0, 0), (4, 0), (0, 4)]
    zone = [(0, 0), (1, 0), (1, 1), (0, 1)]
    s = ParkingStall("T", 0, 4, 0, 4, polygon=tri, car_zone=zone)
    assert s.vertices().shape == (3, 2) and s.area() == pytest.approx(8)
    assert s.car_zone_vertices().tolist() == [list(p) for p in zone]
    assert default_car_zone(ParkingStall("n", 0, 1, 0, 1))  # narrow stall: width clamps to stall width


def test_ignore_zone():
    z = IgnoreZone("clutter", [(0, 0), (1, 0), (1, 1), (0, 1)])
    assert z.contains([[0.5, 0.5], [2, 2]]).tolist() == [True, False]
    assert IgnoreZone("none", []).contains([[0, 0], [1, 1]]).tolist() == [False, False]
    assert IgnoreZone.from_dict(z.to_dict()) == z


def test_default_layouts():
    doors = default_doors(32 * FT)
    assert len(doors) == 3 and doors[0].x_left < doors[1].x_left < doors[2].x_left
    assert default_doors(1.0)[0].x_left == 0.0  # too narrow for the gaps
    stalls = default_stalls(9.0, 7.0)
    assert [s.name for s in stalls] == ["Stall 1", "Stall 2", "Stall 3"]
    assert Door.from_dict(doors[0].to_dict()) == doors[0]


def test_garage_defaults_and_persistence(tmp_path):
    g = Garage()
    assert len(g.doors) == 3 and len(g.stalls) == 3
    path = tmp_path / "g.json"
    g.save(path)
    g2 = Garage.load(path)
    assert g2.to_dict() == g.to_dict()


def test_garage_outline_and_contains():
    g = Garage(width=4, depth=3)
    assert g.outline_vertices().tolist() == [[0, 0], [4, 0], [4, 3], [0, 3]]
    assert g.contains_xy([[2, 1.5], [5, 1.5]]).tolist() == [True, False]
    assert g.contains_xy([[4.03, 1.5]], tol=0.05).tolist() == [True]  # on the wall, within tolerance
    assert g.contains_xy([[4.03, 1.5]], tol=0).tolist() == [False]
    g.outline = [(0, 0), (4, 0), (0, 3)]
    assert g.outline_vertices().shape == (3, 2)
    assert g.contains_xy([[3.5, 2.5]], tol=0).tolist() == [False]


def test_stall_fov_coverage_and_coverage_map():
    g = Garage(width=6, depth=6, radar=RadarPose(x=3, y=0.2, z=2.5, azimuth_deg=0, elevation_deg=-20))
    inside = ParkingStall("in", 2, 4, 2, 4)
    outside = ParkingStall("out", 10, 12, 10, 12)
    assert g.stall_fov_coverage(inside) > 0.9
    assert g.stall_fov_coverage(outside) == 0.0
    degenerate = ParkingStall("d", 1, 1.01, 1, 1.01, polygon=[(1, 1), (1.01, 1), (1.01, 1.01)])
    assert g.stall_fov_coverage(degenerate, step=1.0) == 0.0  # no sample point falls inside
    m = g.coverage_map(height=1.0, step=0.5)
    assert m.shape == (12, 12) and m.any()


def test_garage_warnings():
    good = Garage(width=6, depth=6, radar=RadarPose(x=3, y=0.2, z=2.5, azimuth_deg=0, elevation_deg=-20),
                  stalls=[ParkingStall("S", 2, 4, 2, 4)], doors=[Door(2, 2, 1)])
    assert good.warnings() == []
    bad = Garage(width=6, depth=6, radar=RadarPose(x=30, y=0.2, z=2.5),
                 stalls=[ParkingStall("S", 5, 9, 2, 4)], doors=[Door(2, 2, 5)])
    text = " ".join(bad.warnings())
    assert "Radar position" in text and "extends outside" in text and "FOV" in text and "Door 1" in text


def test_radar_pose_round_trip_ignores_unknown_keys():
    p = RadarPose(x=1, flip_azimuth=True)
    d = p.to_dict()
    d["bogus"] = 5
    assert RadarPose.from_dict(d) == p
    assert RadarPose.from_dict({"x": "2"}).x == 2.0


@pytest.mark.parametrize("flip", [False, True])
def test_transform_round_trip(flip):
    pose = RadarPose(x=1, y=2, z=3, azimuth_deg=30, elevation_deg=-25, roll_deg=5, flip_azimuth=flip)
    pts = np.array([[0.5, 4.0, -1.0], [-2.0, 6.0, 0.5]])
    assert garage_to_radar(radar_to_garage(pts, pose), pose) == pytest.approx(pts)
    R = rotation(pose)
    assert R @ R.T == pytest.approx(np.eye(3))


def test_radar_faces_door_wall_by_default_and_boresight():
    pose = RadarPose(x=0, y=5, z=0, azimuth_deg=180, elevation_deg=0)
    assert boresight_vector(pose) == pytest.approx([0, -1, 0], abs=1e-9)
    assert radar_to_garage([[0, 2, 0]], pose)[0] == pytest.approx([0, 3, 0], abs=1e-9)


def test_radar_angles_and_fov():
    rng, az, el = radar_angles(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]))
    assert rng == pytest.approx([2 ** 0.5, 2 ** 0.5])
    assert az == pytest.approx([45, 0]) and el == pytest.approx([0, 45])
    pose = RadarPose(x=0, y=0, z=0, azimuth_deg=0, elevation_deg=0, fov_azimuth_deg=60, fov_elevation_deg=40, max_range=5)
    pts = np.array([[0, 3, 0], [0, -3, 0], [3, 3, 0], [0, 3, 3], [0, 9, 0]], float)
    assert in_fov(pts, pose).tolist() == [True, False, False, False, False]


def test_fov_edge_rays():
    rays = fov_edge_rays(RadarPose(), length=2.0)
    assert set(rays) == {"left", "right", "down", "up", "boresight"}
    assert all(r.shape == (2, 3) for r in rays.values())
    assert np.linalg.norm(rays["boresight"][1] - rays["boresight"][0]) == pytest.approx(2.0)
    assert fov_edge_rays(RadarPose())["boresight"].shape == (2, 3)  # default length = max_range
