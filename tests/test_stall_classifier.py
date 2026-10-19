"""Unit tests: the per-stall state machine (heatmap evidence, motion, fallbacks, events, confidence)."""
import numpy as np
import pytest

from app.detection.heatmap_occupancy import OccupancyReading, StallCalibration
from app.detection.parameters import DetectionParams
from app.detection.stall_classifier import MotionEvidence, StallClassifier
from app.geometry.parking_zones import ParkingStall
from app.models.garage_state import Cluster, StallState as S, Track

P = DetectionParams(occupancy_persistence_s=5.0, empty_persistence_s=5.0, sensor_timeout_s=2.0, motion_min_points=4,
                    entering_timeout_s=15.0)
STALL = ParkingStall("Stall 1", 0, 2, 0, 5)
CAL = StallCalibration(threshold=0.2, empty_level=0.05, occupied_level=0.4)
NO_XY = np.zeros((0, 2))


def reading(changed, ready=True, health=None, reason=""):
    return OccupancyReading([changed], [0.0], [CAL.threshold], ready, reason, health or [""], [10])


class Rig:
    """Drives one stall through time."""

    def __init__(self, params=P, coverage=1.0):
        self.c, self.p, self.cov, self.t = StallClassifier(1), params, coverage, 100.0

    def step(self, dt=1.0, changed=None, motion=None, tracks=(), xy=NO_XY, last_data=None, sensor_error=None, occ="auto"):
        self.t += dt
        if occ == "auto":
            occ = reading(changed) if changed is not None else None
        res, ev = self.c.classify([STALL], [self.cov], xy, list(tracks), self.t,
                                  self.t if last_data is None else last_data, self.p, sensor_error=sensor_error,
                                  occupancy=occ, calibration=[CAL] if occ is not None else None,
                                  motion=[motion] if motion else None)
        return res[0], (ev[0] if ev else None)

    def hold(self, seconds, **kw):
        out = None
        for _ in range(int(seconds)):
            out = self.step(1.0, **kw)
        return out


def test_sensor_error_when_no_data_or_stale_or_flagged():
    c = StallClassifier(1)
    res = c.classify([STALL], [1.0], NO_XY, [], 10.0, None, P)[0][0]
    assert res.state == S.SENSOR_ERROR and res.explanation == ["no radar data received"]
    res = c.classify([STALL], [1.0], NO_XY, [], 10.0, 5.0, P)[0][0]
    assert res.state == S.SENSOR_ERROR and res.explanation == ["no radar data for 5.0 s"]
    res = c.classify([STALL], [1.0], NO_XY, [], 10.0, 10.0, P, sensor_error="cable unplugged")[0][0]
    assert res.explanation == ["cable unplugged"] and res.confidence == 0.0


def test_unknown_when_stall_is_mostly_outside_the_radar_view():
    res, _ = Rig(coverage=0.2).step(changed=0.0)
    assert res.state == S.UNKNOWN and "20%" in res.explanation[0]


def test_empty_then_occupied_with_persistence_and_entered_event():
    r = Rig()
    assert r.step(changed=0.0)[0].state == S.UNKNOWN  # nothing settled yet
    res = r.hold(6, changed=0.0)[0]
    assert res.state == S.EMPTY and res.changed == 0.0 and res.threshold == CAL.threshold
    assert res.confidence > 0.5 and any("Changed car-zone" in line for line in res.explanation)
    events = []
    for _ in range(8):
        res, ev = r.step(changed=0.45)
        events += [ev] if ev else []
        if res.state == S.OCCUPIED:
            break
        assert any("pending" in line for line in res.explanation)  # persistence not yet satisfied
    assert res.state == S.OCCUPIED and [e.kind for e in events] == ["ENTERED"]
    assert "changed area 45%" in events[0].detail and res.persistence == 0.0
    assert res.confidence > 0.9


def test_occupied_to_empty_fires_exited_event_and_hysteresis_holds_borderline_readings():
    r = Rig()
    r.hold(6, changed=0.0)
    r.hold(6, changed=0.45)
    assert r.step(changed=0.19)[0].state == S.OCCUPIED  # inside the hysteresis band: stays occupied
    events = []
    for _ in range(8):
        res, ev = r.step(changed=0.0)
        events += [ev] if ev else []
    assert res.state == S.EMPTY and [e.kind for e in events] == ["EXITED"]


def test_first_settling_from_unknown_is_not_an_event():
    r = Rig()
    events = [ev for _ in range(8) for ev in [r.step(changed=0.45)[1]] if ev]
    assert events == [] and r.step(changed=0.45)[0].state == S.OCCUPIED


def test_pending_candidate_resets_when_evidence_flips_back():
    r = Rig()
    r.hold(6, changed=0.0)
    r.hold(3, changed=0.45)
    r.hold(1, changed=0.0)  # flipped back before 5 s
    res = r.hold(4, changed=0.45)[0]
    assert res.state == S.EMPTY  # the 5 s clock restarted


def test_approaching_motion_reports_entering_then_occupied_when_heatmap_confirms():
    r = Rig()
    r.hold(6, changed=0.0)
    res, _ = r.step(changed=0.0, motion=MotionEvidence(approaching=6, receding=0))
    assert res.state == S.ENTERING and res.motion_in == 6 and 0 < res.confidence <= 1
    res, _ = r.step(changed=0.0)  # motion stopped, heatmap still empty: car still arriving
    assert res.state == S.ENTERING
    r.hold(6, changed=0.45)
    assert r.step(changed=0.45)[0].state == S.OCCUPIED
    assert r.step(changed=0.0, motion=MotionEvidence(approaching=9, receding=0))[0].state == S.OCCUPIED  # parked: ignores


def test_entering_without_confirmation_times_out_to_empty():
    r = Rig()
    r.hold(6, changed=0.0)
    r.step(changed=0.0, motion=MotionEvidence(5, 0))
    res = r.hold(25, changed=0.0)[0]
    assert res.state == S.EMPTY


def test_receding_motion_from_an_occupied_stall_reports_exiting():
    r = Rig()
    r.hold(6, changed=0.0)
    r.hold(6, changed=0.45)
    res, _ = r.step(changed=0.45, motion=MotionEvidence(approaching=0, receding=8))
    assert res.state == S.EXITING and res.motion_out == 8
    res = r.hold(12, changed=0.0)[0]  # the car has left: heatmap back to empty
    assert res.state == S.EMPTY


def test_mixed_motion_is_ignored_unless_one_direction_dominates():
    r = Rig()
    r.hold(6, changed=0.0)
    res, _ = r.step(changed=0.0, motion=MotionEvidence(approaching=5, receding=4))
    assert res.state == S.EMPTY
    res, _ = r.step(changed=0.0, motion=MotionEvidence(approaching=2, receding=0))  # below motion_min_points
    assert res.state == S.EMPTY


def test_heatmap_health_warning_is_explained():
    r = Rig()
    res, _ = r.step(occ=reading(0.1, health=["reference looks stale"]))
    assert any(line.startswith("⚠") and "stale" in line for line in res.explanation)


def test_heatmap_not_ready_or_missing_explains_why():
    r = Rig()
    res, _ = r.step(occ=reading(0.1, ready=False, reason="filling averaging window"))
    assert any("unavailable: filling averaging window" in line for line in res.explanation)
    res, _ = r.step(occ=None)
    assert any("not configured" in line for line in res.explanation) and res.state == S.UNKNOWN


def vehicle_track(velocity=(0.0, 0.0), n_points=40):
    c = Cluster(0, list(range(n_points)), (1.0, 2.5, 1.0), 1.8, 4.5, 1.5, 0.1, 8.0, 0.0, n_points, 10.0)
    t = Track(3, c, first_seen=0.0, last_seen=1e9, velocity=velocity, is_vehicle=True)
    return t


def test_point_cloud_fallback_without_heatmap():
    r = Rig()
    xy = np.tile([[1.0, 2.5]], (40, 1))
    res, _ = r.step(occ=None, tracks=[vehicle_track()], xy=xy)
    assert res.state == S.UNKNOWN and res.track_id == 3 and res.points_inside == 40  # pending persistence
    assert res.width == pytest.approx(1.8)
    res = r.hold(6, occ=None, tracks=[vehicle_track()], xy=xy)[0]
    assert res.state == S.OCCUPIED and res.confidence == pytest.approx(0.5, abs=0.01)
    res, _ = r.step(occ=None)  # track gone, no heatmap: not evidence of EMPTY
    assert res.state == S.UNKNOWN and res.confidence == 0.0


def test_fallback_uses_track_velocity_for_direction():
    r = Rig()
    xy = np.tile([[1.0, 2.5]], (40, 1))
    moving = vehicle_track(velocity=(0.0, 1.0))
    moving.history = [(0, 0, 0), (2, 0, 2)]
    res, _ = r.step(occ=None, tracks=[moving], xy=xy)
    assert res.state == S.ENTERING
    leaving = vehicle_track(velocity=(0.0, -1.0))
    leaving.history = [(0, 0, 2), (2, 0, 0)]
    r2 = Rig()
    r2.hold(6, occ=None, tracks=[vehicle_track()], xy=xy)
    assert r2.step(occ=None, tracks=[leaving], xy=xy)[0].state == S.EXITING


def test_best_track_needs_enough_points_inside_and_current_visibility():
    c = StallClassifier(1)
    inside = np.array([True] * 5 + [False] * 35)
    t = vehicle_track()
    assert c._best_track([t], inside, 1.0, P) is None  # only 5 of 40 points inside
    assert c._best_track([t], np.ones(40, bool), 1.0, P) is t
    stale = vehicle_track()
    stale.last_seen = 0.0
    assert c._best_track([stale], np.ones(40, bool), 1.0, P) is None
    not_vehicle = vehicle_track()
    not_vehicle.is_vehicle = False
    assert c._best_track([not_vehicle], np.ones(40, bool), 1.0, P) is None
    assert c._best_track([t], np.zeros(0, bool), 1.0, P) is None
    empty = vehicle_track(n_points=0)
    empty.cluster.point_indices = []
    assert c._best_track([empty], np.ones(40, bool), 1.0, P) is None


def test_stall_count_change_resets_memory_and_reset_clears_state():
    c = StallClassifier(3)
    res, _ = c.classify([STALL], [1.0], NO_XY, [], 10.0, 10.0, P)
    assert len(res) == 1 and len(c.mem) == 1
    c.reset()
    assert len(c.mem) == 1 and c.mem[0].state == S.UNKNOWN
    c.reset(4)
    assert len(c.mem) == 4
