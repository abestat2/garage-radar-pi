"""Output types of the detection pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StallState(str, Enum):
    EMPTY = "EMPTY"
    OCCUPIED = "OCCUPIED"
    ENTERING = "VEHICLE ENTERING"
    EXITING = "VEHICLE EXITING"
    UNKNOWN = "UNKNOWN"
    SENSOR_ERROR = "SENSOR/CONFIGURATION ERROR"


@dataclass
class Cluster:
    cluster_id: int
    point_indices: list[int]
    centroid: tuple[float, float, float]  # garage frame, metres
    width: float  # extent across the principal axis, metres
    depth: float  # extent along the principal axis, metres
    height: float  # top of cluster above floor, metres
    z_min: float
    area: float  # oriented bounding-box area, m^2
    heading_deg: float  # orientation of the principal axis in the garage frame
    num_points: int
    mean_snr: float


@dataclass
class Track:
    """PC-side tracked cluster (persists across frames)."""

    track_id: int
    cluster: Cluster
    first_seen: float
    last_seen: float
    velocity: tuple[float, float] = (0.0, 0.0)  # garage-frame m/s, smoothed
    hits: int = 1
    is_vehicle: bool = False
    vehicle_since: float | None = None
    history: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y) centroids
    candidate_since: float | None = None  # when the size checks started passing
    last_candidate: float | None = None  # last time the size checks passed
    reasons: list[str] = field(default_factory=list)  # why it is / is not a vehicle

    @property
    def age(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def speed(self) -> float:
        return (self.velocity[0] ** 2 + self.velocity[1] ** 2) ** 0.5


@dataclass
class StallResult:
    stall_index: int
    name: str
    state: StallState
    confidence: float  # 0..1
    points_inside: int = 0
    fraction_inside: float = 0.0
    width: float = 0.0
    depth: float = 0.0
    height: float = 0.0
    persistence: float = 0.0  # seconds the current state has been held
    track_id: int | None = None
    changed: float | None = None  # heatmap changed fraction (None = heatmap occupancy unavailable)
    threshold: float | None = None
    motion_in: int = 0  # approaching moving points in the stall/approach zone (window)
    motion_out: int = 0  # receding moving points
    explanation: list[str] = field(default_factory=list)


@dataclass
class StallEvent:
    timestamp: float
    stall_index: int
    name: str
    kind: str  # "ENTERED" (became OCCUPIED) or "EXITED" (became EMPTY)
    detail: str = ""


@dataclass
class GarageState:
    timestamp: float
    frame_number: int
    stalls: list[StallResult]
    tracks: list[Track]
    clusters: list[Cluster]
    events: list[StallEvent] = field(default_factory=list)
