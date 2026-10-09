"""DBSCAN-style clustering of accumulated garage-frame points (NumPy only)."""
from __future__ import annotations

import numpy as np

from app.models.garage_state import Cluster

from .parameters import DetectionParams


def dbscan(xyz: np.ndarray, eps: float, min_points: int, z_weight: float = 1.0) -> np.ndarray:
    """Return a label per point (-1 = noise).  O(N^2) memory; fine for a few thousand points."""
    n = len(xyz)
    labels = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return labels
    q = xyz * np.array([1.0, 1.0, z_weight])
    d2 = ((q[:, None, :] - q[None, :, :]) ** 2).sum(-1)
    nbr = d2 <= eps * eps
    core = nbr.sum(1) >= min_points
    cid = 0
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        labels[i] = cid
        stack = [i]
        while stack:
            j = stack.pop()
            if not core[j]:
                continue
            for k in np.flatnonzero(nbr[j] & (labels == -1)):
                labels[k] = cid
                stack.append(k)
        cid += 1
    return labels


def merge_close_clusters(xyz: np.ndarray, labels: np.ndarray, merge_distance: float) -> np.ndarray:
    """Merge clusters whose XY bounding boxes are within merge_distance of each other."""
    ids = [c for c in np.unique(labels) if c >= 0]
    if len(ids) < 2 or merge_distance <= 0:
        return labels
    boxes = {c: (xyz[labels == c, 0].min(), xyz[labels == c, 0].max(), xyz[labels == c, 1].min(), xyz[labels == c, 1].max()) for c in ids}
    parent = {c: c for c in ids}

    def root(c):
        while parent[c] != c:
            c = parent[c]
        return c

    for a in ids:
        for b in ids:
            if a >= b:
                continue
            ax0, ax1, ay0, ay1 = boxes[a]
            bx0, bx1, by0, by1 = boxes[b]
            gap_x = max(0.0, max(ax0, bx0) - min(ax1, bx1))
            gap_y = max(0.0, max(ay0, by0) - min(ay1, by1))
            if np.hypot(gap_x, gap_y) <= merge_distance:
                parent[root(b)] = root(a)
    out = labels.copy()
    remap = {}
    for c in ids:
        r = root(c)
        remap.setdefault(r, len(remap))
        out[labels == c] = remap[r]
    return out


def describe_cluster(cid: int, idx: np.ndarray, xyz: np.ndarray, snr: np.ndarray) -> Cluster:
    p = xyz[idx]
    xy = p[:, :2]
    c = xy.mean(0)
    if len(p) >= 3:
        cov = np.cov((xy - c).T)
        evals, evecs = np.linalg.eigh(cov)
        major = evecs[:, 1]
    else:
        major = np.array([0.0, 1.0])
    minor = np.array([-major[1], major[0]])
    along = (xy - c) @ major
    across = (xy - c) @ minor
    # 5-95 percentile extents are robust to a few stray points
    depth = float(np.percentile(along, 95) - np.percentile(along, 5)) if len(p) > 1 else 0.0
    width = float(np.percentile(across, 95) - np.percentile(across, 5)) if len(p) > 1 else 0.0
    z = p[:, 2]
    s = snr[idx]
    return Cluster(
        cluster_id=cid,
        point_indices=idx.tolist(),
        centroid=(float(c[0]), float(c[1]), float(z.mean())),
        width=width,
        depth=depth,
        height=float(np.percentile(z, 95)),
        z_min=float(z.min()),
        area=width * depth,
        heading_deg=float(np.degrees(np.arctan2(major[0], major[1]))),
        num_points=len(idx),
        mean_snr=float(np.nanmean(s)) if np.any(~np.isnan(s)) else float("nan"),
    )


def cluster_points(xyz: np.ndarray, snr: np.ndarray, p: DetectionParams) -> tuple[np.ndarray, list[Cluster]]:
    labels = dbscan(xyz, p.cluster_eps, p.cluster_min_points, p.cluster_z_weight)
    orig = merge_close_clusters(xyz, labels, p.cluster_merge_distance)
    labels = orig.copy()
    clusters = []
    for cid in [c for c in np.unique(orig) if c >= 0]:
        idx = np.flatnonzero(orig == cid)
        if len(idx) < p.cluster_min_points or len(idx) > p.cluster_max_points:
            labels[idx] = -1
            continue
        clusters.append(describe_cluster(len(clusters), idx, xyz, snr))
        labels[idx] = clusters[-1].cluster_id
    return labels, clusters
