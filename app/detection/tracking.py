"""Frame-to-frame association of clusters into persistent tracks."""
from __future__ import annotations

import numpy as np

from app.models.garage_state import Cluster, Track

from .parameters import DetectionParams

VELOCITY_WINDOW_S = 2.0  # centroid history used for the velocity fit


def _fit_velocity(history: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Least-squares slope of centroid position over time (robust to per-frame jitter)."""
    if len(history) < 3:
        return 0.0, 0.0
    h = np.asarray(history)
    t = h[:, 0] - h[:, 0].mean()
    if np.ptp(t) < 0.5:
        return 0.0, 0.0
    denom = float((t * t).sum())
    return float((t * (h[:, 1] - h[:, 1].mean())).sum() / denom), float((t * (h[:, 2] - h[:, 2].mean())).sum() / denom)


class ClusterTracker:
    def __init__(self):
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def update(self, clusters: list[Cluster], now: float, p: DetectionParams) -> list[Track]:
        unmatched = list(range(len(clusters)))
        # greedy nearest-neighbour association on XY centroid, closest pairs first
        pairs = []
        for ti, t in enumerate(self.tracks):
            for ci, c in enumerate(clusters):
                d = np.hypot(t.cluster.centroid[0] - c.centroid[0], t.cluster.centroid[1] - c.centroid[1])
                if d <= p.track_gate:
                    pairs.append((d, ti, ci))
        used_t, used_c = set(), set()
        for d, ti, ci in sorted(pairs):
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            t, c = self.tracks[ti], clusters[ci]
            t.history.append((now, c.centroid[0], c.centroid[1]))
            while t.history and t.history[0][0] < now - VELOCITY_WINDOW_S:
                t.history.pop(0)
            t.velocity = _fit_velocity(t.history)
            t.cluster = c
            t.last_seen = now
            t.hits += 1
        unmatched = [ci for ci in range(len(clusters)) if ci not in used_c]
        for ci in unmatched:
            c = clusters[ci]
            t = Track(self._next_id, c, first_seen=now, last_seen=now)
            t.history.append((now, c.centroid[0], c.centroid[1]))
            self.tracks.append(t)
            self._next_id += 1
        self.tracks = [t for t in self.tracks if now - t.last_seen <= p.cluster_disappearance_s]
        return self.tracks

    def visible(self, now: float) -> list[Track]:
        """Tracks updated in this frame."""
        return [t for t in self.tracks if t.last_seen >= now]
