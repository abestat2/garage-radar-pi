"""Unit tests: detection parameters, filters, background model, clustering, tracking, vehicle flags, pipeline."""
import json
from dataclasses import fields

import numpy as np
import pytest

from app.detection.clustering import cluster_points, dbscan, describe_cluster, merge_close_clusters
from app.detection.filters import GPOINT_DTYPE, BackgroundModel, filter_mask
from app.detection.parameters import DetectionParams, RadarSideParams
from app.detection.pipeline import DetectionPipeline, cluster_at, stall_results_rows, to_garage_points
from app.detection.tracking import ClusterTracker, _fit_velocity
from app.detection.vehicle_detector import displacement, is_moving, size_checks, update_vehicle_flags
from app.geometry.garage import Garage
from app.geometry.radar_pose import RadarPose
from app.models.garage_state import Cluster, Track
from app.models.radar_frame import RadarFrame
from app.models.radar_object import RadarTrack
from app.models.radar_point import empty_points

ROW_KEYS = ("gx", "gy", "gz", "range", "doppler", "snr", "noise")


def gpoints(rows):
    a = np.zeros(len(rows), dtype=GPOINT_DTYPE)
    for i, r in enumerate(rows):
        a[i] = tuple(r) + (0.0,)
    return a


def good_row(**kw):
    d = dict(gx=2.0, gy=3.0, gz=1.0, range=4.0, doppler=0.0, snr=20.0, noise=10.0)
    d.update(kw)
    return [d[k] for k in ROW_KEYS]


GARAGE = Garage(width=6.0, depth=6.0)


# ---------------------------------------------------------------- parameters
def test_params_round_trip_and_type_coercion(tmp_path):
    p = DetectionParams()
    p.min_range = 1.5
    path = tmp_path / "p.json"
    p.save(path)
    assert DetectionParams.load(path).min_range == 1.5
    q = DetectionParams.from_dict({"cluster_min_points": "12", "clip_to_garage": 0, "unknown": 1})
    assert q.cluster_min_points == 12 and isinstance(q.cluster_min_points, int) and q.clip_to_garage is False
    assert json.loads(path.read_text())["min_range"] == 1.5


def test_radar_side_params_round_trip():
    r = RadarSideParams.from_dict({"frame_period_ms": "50", "clutter_removal": 1, "nope": 3})
    assert r.frame_period_ms == 50.0 and r.clutter_removal is True
    assert RadarSideParams.from_dict(r.to_dict()) == r


def test_param_metadata_present():
    for cls in (DetectionParams, RadarSideParams):
        for f in fields(cls):
            assert {"label", "unit", "lo", "hi", "group"} <= set(f.metadata), f.name


# ---------------------------------------------------------------- filter_mask
@pytest.mark.parametrize("override,expected", [
    ({}, True),
    ({"range": 0.1}, False),  # below min_range
    ({"range": 99.0}, False),  # above max_range
    ({"snr": 1.0}, False),  # below min_snr
    ({"snr": np.nan}, True),  # firmware without SNR is kept
    ({"doppler": 11.0}, False),
    ({"gz": 0.0}, False),  # floor reject
    ({"gz": 2.5}, False),  # ceiling reject
    ({"gx": -1.0}, False),  # outside garage outline
    ({"gy": 7.0}, False),
])
def test_filter_mask_single_point(override, expected):
    pts = gpoints([good_row(**override)])
    assert filter_mask(pts, GARAGE, DetectionParams(), None).tolist() == [expected]


def test_filter_mask_empty_and_optional_filters():
    assert len(filter_mask(gpoints([]), GARAGE, DetectionParams(), None)) == 0
    pts = gpoints([good_row(snr=10, noise=2), good_row(snr=np.nan, noise=np.nan), good_row(snr=10, noise=30)])
    assert filter_mask(pts, GARAGE, DetectionParams(min_signal_db=20.0), None).tolist() == [False, True, True]
    assert not filter_mask(gpoints([good_row(doppler=0.1)]), GARAGE, DetectionParams(min_abs_doppler=0.5), None)[0]
    p = DetectionParams(clip_to_garage=False, x_min=0, x_max=1)
    assert not filter_mask(gpoints([good_row(gx=2.0)]), GARAGE, p, None)[0]
    assert filter_mask(gpoints([good_row(gx=-1.0)]), GARAGE, DetectionParams(clip_to_garage=False), None)[0]


def test_filter_mask_uses_background_when_ready():
    pts = gpoints([good_row()])
    bg = BackgroundModel(0.2)
    bg.learning = True
    for _ in range(5):
        bg.learn(pts)
    bg.learning = False
    assert filter_mask(pts, GARAGE, DetectionParams(background_removal=True), bg).tolist() == [False]
    assert filter_mask(pts, GARAGE, DetectionParams(background_removal=False), bg).tolist() == [True]
    far = gpoints([good_row(gx=5.0, gy=5.0)])
    assert filter_mask(far, GARAGE, DetectionParams(background_removal=True), bg).tolist() == [True]


# ---------------------------------------------------------------- background model
def test_background_model_lifecycle():
    bg = BackgroundModel(0.5)
    assert not bg.ready
    bg.learning = True
    bg.learn(gpoints([]))  # an empty frame still counts
    assert bg.frames == 1 and not bg.ready
    bg.learn(gpoints([good_row()]))
    bg.learning = False
    assert bg.ready
    assert bg.is_background(gpoints([]), 0.5).shape == (0,)
    assert bg.background_voxels(0.5).shape == (1, 3)  # 1 of 2 frames >= 0.5
    assert bg.background_voxels(0.9).shape == (0, 3)
    near = gpoints([good_row(gx=2.4)])  # adjacent voxel
    assert bg.is_background(near, 0.5).tolist() == [True]
    assert BackgroundModel(0.5).is_background(near, 0.5).tolist() == [False]  # nothing learned
    bg.clear(voxel=0.25)
    assert bg.voxel == 0.25 and bg.frames == 0 and not bg.counts
    bg.clear()
    assert bg.voxel == 0.25


def test_background_model_serialisation():
    bg = BackgroundModel(0.3)
    bg.learn(gpoints([good_row(), good_row(gx=4.0)]))
    # voxel keys are np.int32, which json cannot encode by itself
    again = BackgroundModel.from_dict(json.loads(json.dumps(bg.to_dict(), default=int)))
    assert again.voxel == 0.3 and again.frames == 1 and again.counts == bg.counts
    assert BackgroundModel.from_dict({}).voxel == 0.2


# ---------------------------------------------------------------- clustering
def blob(center, n=20, spread=0.1, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(center) + rng.normal(0, spread, (n, 3))


def test_dbscan_separates_blobs_and_marks_noise():
    xyz = np.vstack([blob((0, 0, 1)), blob((5, 5, 1), seed=1), [[20, 20, 1]]])
    labels = dbscan(xyz, eps=0.5, min_points=5)
    assert len(set(labels[:20])) == 1 and len(set(labels[20:40])) == 1 and labels[0] != labels[20]
    assert labels[-1] == -1
    assert len(dbscan(np.zeros((0, 3)), 0.5, 3)) == 0


def test_merge_close_clusters():
    xyz = np.vstack([blob((0, 0, 1)), blob((0.9, 0, 1), seed=1), blob((8, 8, 1), seed=2)])
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    merged = merge_close_clusters(xyz, labels, 1.0)
    assert len(set(merged)) == 2 and merged[0] == merged[20] != merged[40]
    assert merge_close_clusters(xyz, labels, 0).tolist() == labels.tolist()
    assert merge_close_clusters(xyz[:20], labels[:20], 1.0).tolist() == labels[:20].tolist()


def test_describe_cluster_dimensions():
    rng = np.random.default_rng(3)
    xyz = np.column_stack([rng.uniform(-0.9, 0.9, 200), rng.uniform(-2, 2, 200), rng.uniform(0.2, 1.4, 200)])
    snr = np.full(200, 15.0)
    c = describe_cluster(0, np.arange(200), xyz, snr)
    assert c.num_points == 200 and c.mean_snr == 15.0
    assert 3.2 < c.depth < 4.1 and 1.4 < c.width < 1.9  # 5-95 percentile extents
    tiny = describe_cluster(1, np.array([0]), xyz, snr)
    assert tiny.width == 0.0 and tiny.depth == 0.0
    assert np.isnan(describe_cluster(2, np.arange(5), xyz, np.full(200, np.nan)).mean_snr)


def test_cluster_points_filters_by_size():
    p = DetectionParams(cluster_min_points=5, cluster_max_points=50)
    xyz = np.vstack([blob((0, 0, 1), n=20), blob((6, 6, 1), n=80, seed=1), blob((12, 0, 1), n=3, seed=2)])
    labels, clusters = cluster_points(xyz, np.full(len(xyz), 10.0), p)
    assert len(clusters) == 1 and clusters[0].num_points == 20
    assert (labels[20:] == -1).all() and (labels[:20] == 0).all()


# ---------------------------------------------------------------- tracking
def cl(x, y, **kw):
    d = dict(cluster_id=0, point_indices=[], centroid=(x, y, 1.0), width=1.8, depth=4.5, height=1.5, z_min=0.1,
             area=8.0, heading_deg=0.0, num_points=30, mean_snr=10.0)
    d.update(kw)
    return Cluster(**d)


def test_fit_velocity():
    assert _fit_velocity([(0, 0, 0), (1, 1, 1)]) == (0.0, 0.0)  # too short
    assert _fit_velocity([(0, 0, 0), (0.1, 1, 1), (0.2, 2, 2)]) == (0.0, 0.0)  # span < 0.5 s
    vx, vy = _fit_velocity([(t, 2 * t, -t) for t in (0.0, 0.5, 1.0, 1.5)])
    assert vx == pytest.approx(2.0) and vy == pytest.approx(-1.0)


def test_tracker_associates_creates_and_expires():
    p = DetectionParams(track_gate=1.0, cluster_disappearance_s=2.0)
    tr = ClusterTracker()
    tr.update([cl(0, 0)], 0.0, p)
    tr.update([cl(0.3, 0), cl(5, 5)], 0.5, p)
    assert [t.track_id for t in tr.tracks] == [1, 2] and tr.tracks[0].hits == 2
    assert len(tr.visible(0.5)) == 2
    for k in range(1, 6):
        tr.update([cl(0.3 + 0.5 * k, 0)], 0.5 + 0.5 * k, p)  # moving along +x
    assert [t.track_id for t in tr.tracks] == [1]
    assert tr.tracks[0].velocity[0] == pytest.approx(1.0, abs=0.05)
    assert len(tr.visible(3.0)) == 1 and tr.visible(99) == []
    tr.reset()
    assert tr.tracks == []
    tr.update([cl(1, 1)], 10.0, p)
    assert tr.tracks[0].track_id == 1


# ---------------------------------------------------------------- vehicle detector
def make_track(**kw):
    c = kw.pop("cluster", cl(0, 0))
    t = Track(1, c, first_seen=0.0, last_seen=kw.pop("last_seen", 5.0))
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def test_size_checks():
    p = DetectionParams()
    assert all(ok for ok, _ in size_checks(make_track(), p))
    bad = size_checks(make_track(cluster=cl(0, 0, width=0.3, depth=0.5, height=0.2, area=0.1)), p)
    assert [ok for ok, _ in bad] == [True, False, False, False, False]
    assert not size_checks(make_track(last_seen=0.1), p)[0][0]
    assert bad[1][1].startswith("✗")


def test_displacement_and_is_moving():
    p = DetectionParams()
    t = make_track()
    assert displacement(t) == 0.0
    t.history = [(0, 0, 0), (2, 3, 4)]
    assert displacement(t) == 5.0
    t.velocity = (1.0, 0.0)
    assert is_moving(t, p)
    t.velocity = (0.0, 0.0)
    assert not is_moving(t, p)


def test_vehicle_flag_hysteresis():
    p = DetectionParams(vehicle_persistence_s=2.0, departure_persistence_s=3.0, cluster_persistence_s=0.0)
    t = make_track(last_seen=0.0)
    update_vehicle_flags([t], 0.0, p)
    assert t.candidate_since == 0.0 and not t.is_vehicle
    t.last_seen = 2.5
    update_vehicle_flags([t], 2.5, p)
    assert t.is_vehicle and t.vehicle_since == 2.5
    # no longer seen: stays a vehicle until the departure persistence has passed
    update_vehicle_flags([t], 4.0, p)
    assert t.is_vehicle
    update_vehicle_flags([t], 6.0, p)
    assert not t.is_vehicle and t.vehicle_since is None and t.candidate_since is None
    assert any("MOVING" in r or "stationary" in r for r in t.reasons)
    # a failed candidacy before persistence resets the timer
    t2 = make_track(cluster=cl(0, 0, width=0.1), last_seen=1.0)
    t2.candidate_since = 0.5
    update_vehicle_flags([t2], 1.0, p)
    assert t2.candidate_since is None


# ---------------------------------------------------------------- pipeline
POSE = RadarPose(x=3, y=0.3, z=2.0, azimuth_deg=0, elevation_deg=-20)


def radar_frame(n, t, doppler=0.0, number=1):
    f = RadarFrame(frame_number=number, timestamp=t)
    f.points = empty_points(n)
    rng = np.random.default_rng(int(t * 10) + 1)
    f.points["x"] = rng.uniform(-0.8, 0.8, n)
    f.points["y"] = rng.uniform(2.0, 4.0, n)
    f.points["z"] = rng.uniform(-0.5, 0.2, n)
    f.points["doppler"] = doppler
    f.points["snr"] = 20.0
    f.points["noise"] = 5.0
    return f


def test_to_garage_points():
    g = Garage(width=6, depth=6, radar=RadarPose(x=3, y=0.5, z=2.0, azimuth_deg=0, elevation_deg=0))
    assert len(to_garage_points(RadarFrame(1, 0.0), g)) == 0
    f = RadarFrame(1, 12.5)
    f.points = empty_points(1)
    f.points["y"], f.points["x"], f.points["doppler"] = 4.0, 1.0, -0.5
    out = to_garage_points(f, g)
    assert (out["gx"][0], out["gy"][0], out["gz"][0]) == pytest.approx((4.0, 4.5, 2.0))
    assert out["range"][0] == pytest.approx(np.hypot(1, 4)) and out["t"][0] == 12.5 and out["doppler"][0] == -0.5


def test_pipeline_processes_frames_and_firmware_tracks():
    g = Garage(width=6, depth=6, radar=POSE)
    pipe = DetectionPipeline(g, DetectionParams())
    assert pipe.occupancy is None and len(pipe.coverage) == 3
    res = None
    for i in range(12):
        f = radar_frame(40, i * 0.1, doppler=-0.6, number=i)
        f.tracks = [RadarTrack(9, 0.0, 3.0, 0.0)]
        res = pipe.process(f)
    assert res.state.frame_number == 11 and len(res.state.stalls) == 3
    assert len(res.points) == 40 and res.keep.shape == (40,)
    assert res.fw_tracks[0].track_id == 9
    assert pipe.last_data_time == pytest.approx(1.1)
    rows = stall_results_rows(res.state)
    assert len(rows) == 3 and rows[0]["stall"] == 1 and {"state", "confidence", "track_id"} <= set(rows[0])
    assert cluster_at(res.state.clusters, 99, 99) is None


def test_pipeline_learns_background_and_times_out():
    pipe = DetectionPipeline(Garage(width=6, depth=6, radar=POSE), DetectionParams(accumulation_s=0.3))
    pipe.background.learning = True
    for i in range(5):
        pipe.process(radar_frame(30, i * 0.1, number=i))
    pipe.background.learning = False
    assert pipe.background.frames == 5 and pipe.background.ready
    assert pipe.check_timeout(0.5) is None  # data is fresh
    stalls = pipe.check_timeout(60.0)
    assert stalls is not None and len(stalls) == 3
    pipe.reset()
    assert pipe.last_data_time is None
    assert pipe.check_timeout(1.0) is not None  # never received data


def test_pipeline_setters_and_reset():
    g = Garage(width=6, depth=6, radar=POSE)
    pipe = DetectionPipeline(g)
    pipe.background.learn(gpoints([good_row()]))
    pipe.set_params(DetectionParams(background_voxel=0.4))
    assert pipe.background.voxel == 0.4 and pipe.background.frames == 0  # cleared on voxel change
    g2 = Garage(width=8, depth=6, radar=POSE)
    pipe.set_garage(g2)
    assert pipe.garage is g2
    pipe.set_occupancy(None)
    assert pipe.occupancy is None


def test_cluster_at_picks_nearest_within_radius():
    a, b = cl(0, 0), cl(0.3, 0)
    assert cluster_at([a, b], 0.25, 0) is b
    assert cluster_at([a, b], 5, 5) is None
