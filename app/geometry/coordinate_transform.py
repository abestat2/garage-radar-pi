"""Radar-frame <-> garage-frame transforms (see radar_pose.py for conventions)."""
from __future__ import annotations

import numpy as np

from .radar_pose import RadarPose


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation(pose: RadarPose) -> np.ndarray:
    """Rotation matrix taking radar-frame vectors into the garage frame."""
    return (
        _rot_z(-np.radians(pose.azimuth_deg))
        @ _rot_x(np.radians(pose.elevation_deg))
        @ _rot_y(np.radians(pose.roll_deg))
    )


def _sign(pose: RadarPose) -> np.ndarray:
    """Per-axis sign applied to firmware coordinates (azimuth mirror for boards that report +X left)."""
    return np.array([-1.0 if pose.flip_azimuth else 1.0, 1.0, 1.0])


def radar_to_garage(xyz_radar: np.ndarray, pose: RadarPose) -> np.ndarray:
    """(N, 3) radar-frame points (as the firmware reports them) -> (N, 3) garage-frame points."""
    xyz_radar = np.asarray(xyz_radar, dtype=np.float64).reshape(-1, 3) * _sign(pose)
    t = np.array([pose.x, pose.y, pose.z])
    return xyz_radar @ rotation(pose).T + t


def garage_to_radar(xyz_garage: np.ndarray, pose: RadarPose) -> np.ndarray:
    """Inverse of radar_to_garage: returns coordinates as the firmware would report them."""
    xyz_garage = np.asarray(xyz_garage, dtype=np.float64).reshape(-1, 3)
    t = np.array([pose.x, pose.y, pose.z])
    return ((xyz_garage - t) @ rotation(pose)) * _sign(pose)


def radar_angles(xyz_radar: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (range, azimuth_deg, elevation_deg) of radar-frame points."""
    x, y, z = xyz_radar[:, 0], xyz_radar[:, 1], xyz_radar[:, 2]
    rng = np.sqrt(x * x + y * y + z * z)
    az = np.degrees(np.arctan2(x, y))
    el = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y)))
    return rng, az, el


def in_fov(xyz_garage: np.ndarray, pose: RadarPose) -> np.ndarray:
    """Boolean mask: which garage-frame points the radar can see (nominal FOV)."""
    pr = garage_to_radar(xyz_garage, pose)
    rng, az, el = radar_angles(pr)
    return (
        (pr[:, 1] > 0)
        & (np.abs(az) <= pose.fov_azimuth_deg / 2)
        & (np.abs(el) <= pose.fov_elevation_deg / 2)
        & (rng <= pose.max_range)
    )


def boresight_vector(pose: RadarPose) -> np.ndarray:
    return rotation(pose) @ np.array([0.0, 1.0, 0.0])


def fov_edge_rays(pose: RadarPose, length: float | None = None) -> dict[str, np.ndarray]:
    """Garage-frame segments (2x3) along the FOV edges and boresight, for drawing."""
    length = pose.max_range if length is None else length
    ha, he = np.radians(pose.fov_azimuth_deg / 2), np.radians(pose.fov_elevation_deg / 2)
    rays = {}
    for name, az, el in [
        ("left", -ha, 0.0),
        ("right", ha, 0.0),
        ("down", 0.0, -he),
        ("up", 0.0, he),
        ("boresight", 0.0, 0.0),
    ]:
        d = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
        rays[name] = radar_to_garage(np.vstack([np.zeros(3), d * length]), pose)
    return rays
