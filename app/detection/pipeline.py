"""The full PC-side detection pipeline.

    radar frame (radar coords)
      -> transform to garage frame
      -> range / SNR / Doppler / XYZ / background filtering
      -> time accumulation
      -> clustering
      -> tracking + persistence
      -> vehicle-candidate size checks
      -> three-stall classification

It is deliberately free of Qt and serial code so it can run identically on
live data, on replayed recordings and in batch A/B comparisons.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from app.geometry.coordinate_transform import radar_to_garage
from app.geometry.garage import Garage
from app.models.garage_state import Cluster, GarageState, StallResult
from app.models.radar_frame import RadarFrame
from app.models.radar_object import RadarTrack

from .clustering import cluster_points
from .filters import GPOINT_DTYPE, BackgroundModel, filter_mask
from .heatmap_occupancy import HeatmapOccupancy, OccupancyReading
from .parameters import DetectionParams
from .stall_classifier import MotionEvidence, StallClassifier
from .tracking import ClusterTracker
from .vehicle_detector import update_vehicle_flags

MAX_ACCUMULATED = 4000


@dataclass
class FrameResult:
    frame: RadarFrame
    points: np.ndarray  # this frame, GPOINT_DTYPE (garage coords)
    keep: np.ndarray  # filter mask for `points`
    accumulated: np.ndarray  # filtered points in the accumulation window
    labels: np.ndarray  # cluster label per accumulated point (-1 = noise)
    state: GarageState
    fw_tracks: list[RadarTrack] = field(default_factory=list)  # firmware tracks, garage coords
    occupancy: OccupancyReading | None = None


def to_garage_points(frame: RadarFrame, garage: Garage) -> np.ndarray:
    pts = frame.points
    out = np.zeros(len(pts), dtype=GPOINT_DTYPE)
    if len(pts) == 0:
        return out
    xyz_r = np.column_stack([pts["x"], pts["y"], pts["z"]]).astype(np.float64)
    g = radar_to_garage(xyz_r, garage.radar)
    out["gx"], out["gy"], out["gz"] = g.T
    out["range"] = np.linalg.norm(xyz_r, axis=1)
    out["doppler"] = pts["doppler"]
    out["snr"] = pts["snr"]
    out["noise"] = pts["noise"]
    out["t"] = frame.timestamp
    return out


class DetectionPipeline:
    def __init__(self, garage: Garage, params: DetectionParams | None = None):
        self.params = params or DetectionParams()
        self.background = BackgroundModel(self.params.background_voxel)
        self.tracker = ClusterTracker()
        self.classifier = StallClassifier(len(garage.stalls))
        self._acc: deque[tuple[float, np.ndarray]] = deque()
        self.last_data_time: float | None = None
        self.garage = garage
        self.coverage = [garage.stall_fov_coverage(s) for s in garage.stalls]
        self.occupancy: HeatmapOccupancy | None = None  # set by the owner (needs cfg + reference)
        self._motion: deque[tuple[float, np.ndarray]] = deque()  # (t, moving garage-frame points)
        self._last_stalls = None

    # ------------------------------------------------------------ configuration
    def set_garage(self, garage: Garage) -> None:
        self.garage = garage
        self.coverage = [garage.stall_fov_coverage(s) for s in garage.stalls]
        if self.occupancy is not None:
            o = self.occupancy
            p = self.params
            self.occupancy = HeatmapOccupancy(garage, o.cfg, o.reference, o.calibration, o.change_db,
                                              o._images.maxlen, p.occ_assumed_height, p.entry_approach_m,
                                              p.occ_edge_margin, p.occ_adapt_tau_s, p.occ_adapt_empty_hold_s)
        self.reset()

    def set_occupancy(self, occupancy: HeatmapOccupancy | None) -> None:
        self.occupancy = occupancy

    def set_params(self, params: DetectionParams) -> None:
        if params.background_voxel != self.background.voxel and self.background.frames:
            self.background.clear(params.background_voxel)
        self.background.voxel = params.background_voxel
        self.params = params

    def reset(self) -> None:
        self._acc.clear()
        self._motion.clear()
        self._last_stalls = None
        if self.occupancy is not None:
            self.occupancy.reset()
        self.tracker.reset()
        self.classifier.reset(len(self.garage.stalls))
        self.last_data_time = None

    # ------------------------------------------------------------ processing
    def process(self, frame: RadarFrame) -> FrameResult:
        p = self.params
        now = frame.timestamp
        self.last_data_time = now
        pts = to_garage_points(frame, self.garage)

        if self.background.learning:
            # learn from everything inside the height/range limits, before background removal
            learn_p = DetectionParams.from_dict({**p.to_dict(), "background_removal": False})
            self.background.learn(pts[filter_mask(pts, self.garage, learn_p, None)])

        keep = filter_mask(pts, self.garage, p, self.background)
        self._acc.append((now, pts[keep]))
        while self._acc and self._acc[0][0] < now - p.accumulation_s:
            self._acc.popleft()
        acc = np.concatenate([a for _, a in self._acc])
        if len(acc) > MAX_ACCUMULATED:
            acc = acc[-MAX_ACCUMULATED:]

        xyz = np.column_stack([acc["gx"], acc["gy"], acc["gz"]]).astype(np.float64)
        labels, clusters = cluster_points(xyz, acc["snr"], p)
        tracks = self.tracker.update(clusters, now, p)
        update_vehicle_flags(tracks, now, p)

        occ, cal = None, None
        if self.occupancy is not None:
            self.occupancy.change_db = p.occ_change_db
            occ = self.occupancy.update(frame.az_heatmap, self._last_stalls, now)
            cal = self.occupancy.thresholds()
        motion = self._motion_evidence(pts, keep, now, p)
        stalls, events = self.classifier.classify(
            self.garage.stalls, self.coverage, xyz[:, :2], tracks, now, self.last_data_time, p,
            occupancy=occ, calibration=cal, motion=motion,
        )
        self._last_stalls = stalls
        state = GarageState(now, frame.frame_number, stalls, list(tracks), clusters, events)
        return FrameResult(frame, pts, keep, acc, labels, state, self._fw_tracks(frame), occ)

    def _motion_evidence(self, pts: np.ndarray, keep: np.ndarray, now: float, p: DetectionParams) -> list[MotionEvidence]:
        """Moving points per stall (inside the stall or in the approach zone in front of its door)."""
        mv = pts[keep & (np.abs(pts["doppler"]) >= p.motion_min_speed)] if len(pts) else pts
        self._motion.append((now, mv))
        while self._motion and self._motion[0][0] < now - p.motion_window_s:
            self._motion.popleft()
        allm = np.concatenate([m for _, m in self._motion]) if self._motion else mv
        out = []
        xy = np.column_stack([allm["gx"], allm["gy"]]) if len(allm) else np.zeros((0, 2))
        for s in self.garage.stalls:
            if not len(allm):
                out.append(MotionEvidence())
                continue
            v = s.vertices()
            x0, x1, front = v[:, 0].min(), v[:, 0].max(), v[:, 1].min()
            door_y = min((d.y for d in self.garage.doors
                          if min(x1, d.x_left + d.width) > max(x0, d.x_left)), default=front)
            approach = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= door_y - p.entry_approach_m) & (xy[:, 1] < front)
            zone = s.contains(xy) | approach
            d = allm["doppler"][zone]
            out.append(MotionEvidence(int((d < 0).sum()), int((d > 0).sum())))
        return out

    def check_timeout(self, now: float) -> list[StallResult] | None:
        """Stall results when no frames are arriving (SENSOR ERROR after the timeout)."""
        if self.last_data_time is not None and now - self.last_data_time <= self.params.sensor_timeout_s:
            return None
        stalls, _ = self.classifier.classify(
            self.garage.stalls, self.coverage, np.zeros((0, 2)), [], now, self.last_data_time, self.params
        )
        return stalls

    def _fw_tracks(self, frame: RadarFrame) -> list[RadarTrack]:
        out = []
        for t in frame.tracks:
            g = radar_to_garage(np.array([[t.x, t.y, t.z]]), self.garage.radar)[0]
            out.append(RadarTrack(t.track_id, *g, t.vx, t.vy, t.vz, t.confidence))
        return out


def stall_results_rows(state: GarageState) -> list[dict]:
    """Flat rows for detections.csv."""
    rows = []
    for s in state.stalls:
        rows.append(
            {
                "timestamp": f"{state.timestamp:.3f}",
                "frame": state.frame_number,
                "stall": s.stall_index + 1,
                "state": s.state.value,
                "confidence": f"{s.confidence:.3f}",
                "points_inside": s.points_inside,
                "fraction_inside": f"{s.fraction_inside:.3f}",
                "width_m": f"{s.width:.3f}",
                "depth_m": f"{s.depth:.3f}",
                "height_m": f"{s.height:.3f}",
                "persistence_s": f"{s.persistence:.2f}",
                "changed_pct": "" if s.changed is None else f"{s.changed * 100:.1f}",
                "threshold_pct": "" if s.threshold is None else f"{s.threshold * 100:.1f}",
                "motion_in": s.motion_in,
                "motion_out": s.motion_out,
                "track_id": s.track_id if s.track_id is not None else "",
                "num_clusters": len(state.clusters),
                "num_vehicle_tracks": sum(1 for t in state.tracks if t.is_vehicle),
            }
        )
    return rows


def cluster_at(clusters: list[Cluster], x: float, y: float, radius: float = 0.5) -> Cluster | None:
    best, bd = None, radius
    for c in clusters:
        d = np.hypot(c.centroid[0] - x, c.centroid[1] - y)
        if d <= bd:
            best, bd = c, d
    return best
