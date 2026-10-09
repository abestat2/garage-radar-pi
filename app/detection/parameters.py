"""Editable detection parameters.

Each field carries UI metadata (label, unit, range, group).  `restart=True`
marks parameters that live in the radar .cfg and only take effect after the
configuration is re-sent to the radar (radar restart).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def P(default, label, unit="", lo=0.0, hi=100.0, step=0.1, group="", restart=False, tip=""):
    return field(
        default=default,
        metadata=dict(label=label, unit=unit, lo=lo, hi=hi, step=step, group=group, restart=restart, tip=tip),
    )


G_FILTER = "Point filtering"
G_CLUSTER = "Clustering / tracking"
G_VEHICLE = "Vehicle candidates"
G_STALL = "Stall classification"
G_RADAR = "Radar configuration"
G_OCC = "Parked-car occupancy (heatmap) and entry detection"


@dataclass
class DetectionParams:
    # ---------------------------------------------------------------- point filtering
    min_range: float = P(0.3, "Minimum range", "m", 0, 20, 0.1, G_FILTER)
    max_range: float = P(12.0, "Maximum range", "m", 0.5, 50, 0.5, G_FILTER)
    min_snr_db: float = P(6.0, "Minimum SNR", "dB", 0, 60, 0.5, G_FILTER, tip="Ignored when firmware sends no SNR")
    min_signal_db: float = P(0.0, "Minimum signal strength (SNR+noise)", "dB", 0, 120, 1, G_FILTER, tip="0 = off")
    background_removal: bool = P(False, "Static-clutter removal (PC background map)", group=G_FILTER,
                                 tip="Removes points in voxels that were occupied while learning an EMPTY garage")
    background_threshold: float = P(0.15, "Background voxel occupancy threshold", "", 0.01, 1, 0.01, G_FILTER)
    background_voxel: float = P(0.2, "Background voxel size", "m", 0.05, 1, 0.05, G_FILTER)
    min_abs_doppler: float = P(0.0, "Minimum |Doppler|", "m/s", 0, 10, 0.05, G_FILTER, tip="0 keeps static points")
    max_abs_doppler: float = P(10.0, "Maximum |Doppler|", "m/s", 0, 20, 0.1, G_FILTER)
    clip_to_garage: bool = P(True, "Limit X/Y to garage outline", group=G_FILTER)
    garage_margin: float = P(0.2, "Garage outline margin", "m", -1, 3, 0.05, G_FILTER)
    x_min: float = P(-50.0, "X minimum", "m", -50, 50, 0.1, G_FILTER)
    x_max: float = P(50.0, "X maximum", "m", -50, 50, 0.1, G_FILTER)
    y_min: float = P(-50.0, "Y minimum", "m", -50, 50, 0.1, G_FILTER)
    y_max: float = P(50.0, "Y maximum", "m", -50, 50, 0.1, G_FILTER)
    z_min: float = P(0.1, "Height minimum (floor reject)", "m", -2, 5, 0.05, G_FILTER)
    z_max: float = P(2.0, "Height maximum (ceiling/opener/open-door reject)", "m", 0, 6, 0.05, G_FILTER,
                  tip="Open overhead-door panels sit at ~2.1-2.3 m")
    accumulation_s: float = P(1.0, "Point accumulation window", "s", 0.1, 10, 0.1, G_FILTER,
                              tip="Sparse radar frames are accumulated before clustering")

    # ---------------------------------------------------------------- clustering / tracking
    cluster_eps: float = P(0.5, "Point-to-point distance", "m", 0.05, 3, 0.05, G_CLUSTER)
    cluster_z_weight: float = P(0.5, "Height weight in distance", "", 0, 2, 0.1, G_CLUSTER)
    cluster_min_points: int = P(8, "Minimum points", "", 1, 500, 1, G_CLUSTER)
    cluster_max_points: int = P(3000, "Maximum points", "", 10, 20000, 10, G_CLUSTER)
    cluster_merge_distance: float = P(0.6, "Cluster merge distance", "m", 0, 3, 0.05, G_CLUSTER)
    track_gate: float = P(1.5, "Track association gate", "m", 0.1, 5, 0.1, G_CLUSTER)
    cluster_persistence_s: float = P(0.5, "Cluster persistence (confirm)", "s", 0, 30, 0.1, G_CLUSTER)
    cluster_disappearance_s: float = P(3.0, "Cluster disappearance time", "s", 0, 60, 0.5, G_CLUSTER)

    # ---------------------------------------------------------------- vehicle candidates
    veh_min_width: float = P(1.0, "Minimum width", "m", 0, 5, 0.05, G_VEHICLE)
    veh_max_width: float = P(2.6, "Maximum width", "m", 0, 6, 0.05, G_VEHICLE)
    veh_min_depth: float = P(1.5, "Minimum depth", "m", 0, 10, 0.05, G_VEHICLE)
    veh_max_depth: float = P(6.5, "Maximum depth", "m", 0, 12, 0.05, G_VEHICLE)
    veh_min_height: float = P(0.7, "Minimum height", "m", 0, 4, 0.05, G_VEHICLE)
    veh_max_height: float = P(2.5, "Maximum height", "m", 0, 5, 0.05, G_VEHICLE)
    veh_min_area: float = P(2.0, "Minimum occupied area", "m²", 0, 30, 0.1, G_VEHICLE)
    movement_threshold: float = P(0.2, "Movement threshold", "m/s", 0, 5, 0.01, G_VEHICLE)
    movement_min_displacement: float = P(0.3, "Movement minimum displacement (2 s window)", "m", 0, 5, 0.05,
                                         G_VEHICLE, tip="Rejects centroid jitter of parked vehicles")
    vehicle_persistence_s: float = P(2.0, "Required persistence", "s", 0, 60, 0.5, G_VEHICLE)
    departure_persistence_s: float = P(3.0, "Departure persistence", "s", 0, 60, 0.5, G_VEHICLE)

    # ---------------------------------------------------------------- stall classification
    stall_min_fraction: float = P(0.5, "Minimum fraction of points inside stall", "", 0, 1, 0.05, G_STALL)
    stall_min_points: int = P(10, "Minimum point count inside stall", "", 1, 1000, 1, G_STALL)
    occupancy_persistence_s: float = P(5.0, "Occupancy persistence", "s", 0, 120, 0.5, G_STALL)
    empty_persistence_s: float = P(5.0, "Empty persistence", "s", 0, 120, 0.5, G_STALL)
    min_fov_coverage: float = P(0.5, "Minimum stall FOV coverage (else UNKNOWN)", "", 0, 1, 0.05, G_STALL)
    sensor_timeout_s: float = P(2.0, "No-data timeout (sensor error)", "s", 0.2, 30, 0.1, G_STALL)

    # ---------------------------------------------------------------- heatmap occupancy / entry
    occ_change_db: float = P(6.0, "Echo change counted as 'changed'", "dB", 1, 30, 0.5, G_OCC,
                             tip="Cell echo vs the empty-garage reference")
    occ_window_frames: int = P(8, "Heatmap averaging window", "frames", 1, 60, 1, G_OCC)
    occ_assumed_height: float = P(0.8, "Assumed reflector height (heatmap -> floor plan)", "m", 0, 2.5, 0.05, G_OCC)
    occ_hysteresis: float = P(0.2, "Threshold hysteresis (fraction of threshold)", "", 0, 0.9, 0.05, G_OCC)
    entry_approach_m: float = P(3.5, "Approach zone outside each door", "m", 0, 10, 0.25, G_OCC)
    motion_min_speed: float = P(0.3, "Motion: minimum |Doppler|", "m/s", 0, 5, 0.05, G_OCC)
    motion_min_points: int = P(4, "Motion: points needed in the window", "", 1, 200, 1, G_OCC)
    motion_window_s: float = P(2.0, "Motion: time window", "s", 0.25, 20, 0.25, G_OCC)
    entering_timeout_s: float = P(15.0, "ENTERING without occupancy reverts after", "s", 1, 120, 1, G_OCC)
    occ_edge_margin: float = P(0.25, "Car-zone edge margin (fraction of beam half-width)", "", 0, 1, 0.05, G_OCC,
                               tip="1 = use only cells whose whole 3 dB beam blur is inside the car zone")
    occ_adapt_tau_s: float = P(300.0, "Reference re-learning time constant (0 = off)", "s", 0, 7200, 30, G_OCC,
                               tip="Clutter that builds up or is cleared out becomes normal over this time")
    occ_adapt_empty_hold_s: float = P(120.0, "Re-learn inside a car zone after EMPTY for", "s", 10, 7200, 10, G_OCC)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DetectionParams":
        p = cls()
        for f in fields(cls):
            if f.name in d:
                setattr(p, f.name, type(getattr(p, f.name))(d[f.name]))
        return p

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "DetectionParams":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class RadarSideParams:
    """Values that live in the radar .cfg (see RadarConfig.apply_radar_params)."""

    frame_period_ms: float = P(100.0, "Frame period", "ms", 20, 1000, 5, G_RADAR, restart=True)
    clutter_removal: bool = P(False, "Radar static clutter removal (clutterRemoval)", group=G_RADAR, restart=True,
                              tip="Removes zero-Doppler returns on the radar - this also removes PARKED vehicles")
    cfar_range_threshold_db: float = P(15.0, "CFAR range threshold", "dB", 0, 40, 0.5, G_RADAR, restart=True)
    cfar_doppler_threshold_db: float = P(15.0, "CFAR Doppler threshold", "dB", 0, 40, 0.5, G_RADAR, restart=True)
    cfar_max_range_m: float = P(10.0, "CFAR maximum range", "m", 1, 50, 0.5, G_RADAR, restart=True)
    aoa_az_limit_deg: float = P(90.0, "Azimuth FOV limit (±)", "deg", 1, 90, 1, G_RADAR, restart=True)
    aoa_el_limit_deg: float = P(90.0, "Elevation FOV limit (±)", "deg", 1, 90, 1, G_RADAR, restart=True)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RadarSideParams":
        p = cls()
        for f in fields(cls):
            if f.name in d:
                setattr(p, f.name, type(getattr(p, f.name))(d[f.name]))
        return p
