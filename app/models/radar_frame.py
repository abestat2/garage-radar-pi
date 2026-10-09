"""A single parsed radar frame."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .radar_object import RadarTrack
from .radar_point import empty_points


@dataclass
class RadarFrame:
    frame_number: int
    timestamp: float  # host time (time.time()) when the frame was received
    points: np.ndarray = field(default_factory=empty_points)  # POINT_DTYPE, radar frame
    tracks: list[RadarTrack] = field(default_factory=list)
    platform: int = 0
    version: int = 0
    tlv_types: list[int] = field(default_factory=list)
    raw: bytes = b""  # complete packet bytes, kept for recording
    az_heatmap: np.ndarray | None = None  # TLV 4, complex64 flat (range bins x azimuth antennas)
    damaged: bool = False  # packet lost bytes in transit; only verified leading TLVs were decoded

    @property
    def num_points(self) -> int:
        return int(len(self.points))
