"""Point filtering (garage-frame points) and the learned static-background map."""
from __future__ import annotations

import numpy as np

from app.geometry.garage import Garage

from .parameters import DetectionParams

# Garage-frame point array: radar POINT_DTYPE fields plus garage coordinates
GPOINT_DTYPE = np.dtype(
    [
        ("gx", "f4"),
        ("gy", "f4"),
        ("gz", "f4"),
        ("range", "f4"),
        ("doppler", "f4"),
        ("snr", "f4"),
        ("noise", "f4"),
        ("t", "f8"),
    ]
)


def filter_mask(pts: np.ndarray, garage: Garage, p: DetectionParams, background: "BackgroundModel | None") -> np.ndarray:
    """Boolean mask of GPOINT_DTYPE points that pass every enabled filter."""
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    m = (pts["range"] >= p.min_range) & (pts["range"] <= p.max_range)
    snr = pts["snr"]
    has_snr = ~np.isnan(snr)
    m &= ~has_snr | (snr >= p.min_snr_db)
    if p.min_signal_db > 0:
        sig = snr + pts["noise"]
        m &= np.isnan(sig) | (sig >= p.min_signal_db)
    ad = np.abs(pts["doppler"])
    m &= (ad >= p.min_abs_doppler) & (ad <= p.max_abs_doppler)
    m &= (pts["gx"] >= p.x_min) & (pts["gx"] <= p.x_max) & (pts["gy"] >= p.y_min) & (pts["gy"] <= p.y_max)
    if p.clip_to_garage:
        g = p.garage_margin
        m &= (pts["gx"] >= -g) & (pts["gx"] <= garage.width + g) & (pts["gy"] >= -g) & (pts["gy"] <= garage.depth + g)
    m &= (pts["gz"] >= p.z_min) & (pts["gz"] <= p.z_max)
    if p.background_removal and background is not None and background.ready:
        m &= ~background.is_background(pts, p.background_threshold)
    return m


class BackgroundModel:
    """Voxel occupancy map learned from an empty garage.

    While learning, each frame marks which voxels contain at least one point.
    A voxel whose occupied-frame fraction exceeds the threshold is treated as
    static background (walls, shelving, door hardware) and filtered out.
    """

    def __init__(self, voxel: float = 0.2):
        self.voxel = voxel
        self.counts: dict[tuple[int, int, int], int] = {}
        self.frames = 0
        self.learning = False

    @property
    def ready(self) -> bool:
        return self.frames > 0 and not self.learning

    def clear(self, voxel: float | None = None) -> None:
        if voxel:
            self.voxel = voxel
        self.counts.clear()
        self.frames = 0

    def _keys(self, pts: np.ndarray) -> np.ndarray:
        return np.floor(np.column_stack([pts["gx"], pts["gy"], pts["gz"]]) / self.voxel).astype(np.int32)

    def learn(self, pts: np.ndarray) -> None:
        self.frames += 1
        if len(pts) == 0:
            return
        for k in {tuple(r) for r in self._keys(pts)}:
            self.counts[k] = self.counts.get(k, 0) + 1

    def is_background(self, pts: np.ndarray, threshold: float) -> np.ndarray:
        if len(pts) == 0 or self.frames == 0:
            return np.zeros(len(pts), dtype=bool)
        lim = threshold * self.frames
        keys = self._keys(pts)
        out = np.zeros(len(pts), dtype=bool)
        for i, r in enumerate(keys):
            # include the 26-neighbourhood so slightly shifted returns are also rejected
            hit = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if self.counts.get((r[0] + dx, r[1] + dy, r[2] + dz), 0) >= lim:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    break
            out[i] = hit
        return out

    def background_voxels(self, threshold: float) -> np.ndarray:
        """(N, 3) centres of voxels classified as background (for display)."""
        lim = threshold * max(self.frames, 1)
        ks = [k for k, c in self.counts.items() if c >= lim]
        if not ks:
            return np.zeros((0, 3))
        return (np.array(ks, dtype=np.float64) + 0.5) * self.voxel

    def to_dict(self) -> dict:
        return {"voxel": self.voxel, "frames": self.frames, "counts": [[*k, c] for k, c in self.counts.items()]}

    @classmethod
    def from_dict(cls, d: dict) -> "BackgroundModel":
        b = cls(d.get("voxel", 0.2))
        b.frames = int(d.get("frames", 0))
        b.counts = {(int(r[0]), int(r[1]), int(r[2])): int(r[3]) for r in d.get("counts", [])}
        return b
