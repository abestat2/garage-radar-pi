"""Tracked objects reported by the radar firmware (e.g. TI 3D people tracking)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RadarTrack:
    track_id: int
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    confidence: float = 0.0
