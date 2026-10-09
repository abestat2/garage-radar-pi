"""Point-cloud representation.

Points are held in a NumPy structured array so a whole frame can be filtered
and transformed without Python-level loops.  Coordinates in this dtype are in
the *radar* frame (TI convention: x = right, y = boresight, z = up, metres).
"""
from __future__ import annotations

import numpy as np

POINT_DTYPE = np.dtype(
    [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("doppler", "f4"),  # radial velocity, m/s (negative = approaching)
        ("snr", "f4"),  # dB (NaN when the firmware does not supply it)
        ("noise", "f4"),  # dB (NaN when the firmware does not supply it)
    ]
)


def empty_points(n: int = 0) -> np.ndarray:
    pts = np.zeros(n, dtype=POINT_DTYPE)
    if n:
        pts["snr"] = np.nan
        pts["noise"] = np.nan
    return pts


def xyz(points: np.ndarray) -> np.ndarray:
    """Return an (N, 3) float array of x, y, z."""
    return np.column_stack([points["x"], points["y"], points["z"]]).astype(np.float64)


def ranges(points: np.ndarray) -> np.ndarray:
    return np.sqrt(points["x"] ** 2 + points["y"] ** 2 + points["z"] ** 2)
