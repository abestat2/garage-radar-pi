"""Radar mounting pose and field of view.

Garage frame (all metres):
    origin  = inside corner where the door wall meets the left side wall
              (left as seen on the top-down plan, door wall at the bottom)
    +X      = along the door wall, toward the right wall
    +Y      = from the door wall toward the back wall
    +Z      = up from the floor

Pose angles (degrees):
    azimuth   = boresight heading measured from +Y toward +X
                (0 = facing the back wall, 180 = facing the doors)
    elevation = boresight tilt, positive up (a high mount pointing
                down at the cars is typically -20 .. -45)
    roll      = rotation about the boresight
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RadarPose:
    x: float = 4.88
    y: float = 7.2
    z: float = 2.6
    azimuth_deg: float = 180.0
    elevation_deg: float = -30.0
    roll_deg: float = 0.0
    # IWR6843LEVM nominal field of view (TI: ~120 deg az x 80 deg el)
    fov_azimuth_deg: float = 120.0
    fov_elevation_deg: float = 80.0
    max_range: float = 12.0
    # The firmware's azimuth sign: True when the board reports +X on its LEFT (looking out along the
    # boresight).  Measured on the user's IWR6843LEVM + SDK 3.05 OOB demo (2026-10-08 drive-in test).
    flip_azimuth: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RadarPose":
        fields = cls.__dataclass_fields__
        return cls(**{k: (bool(v) if k == "flip_azimuth" else float(v)) for k, v in d.items() if k in fields})
