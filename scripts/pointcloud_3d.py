"""Connect to the Pi's raw radar stream (TCP 7777) and build an aggregated 3D point cloud.

    python scripts/pointcloud_3d.py <pi-host> [--port 7777] [--voxel 0.05] [--min-snr 8]
                                    [--save cloud.ply] [--no-plot]

Every detected point is dropped into a voxel grid; each voxel keeps the centroid of the points that
landed in it and a hit count, so a static scene sharpens over time instead of piling up duplicates.
Coordinates are the radar frame (x = right, y = boresight, z = up, metres).  Ctrl+C (or closing the
plot window) stops it and writes --save (.ply, .csv or .npy) if given.
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.radar.packet_parser import PacketParser  # noqa: E402


class VoxelCloud:
    """Voxel-hashed accumulator: centroid + hit count per occupied voxel."""

    def __init__(self, voxel: float, max_voxels: int = 200_000):
        self.voxel, self.max_voxels = voxel, max_voxels
        self._sum: dict[tuple, np.ndarray] = {}
        self._n: dict[tuple, int] = {}
        self.points_seen = 0
        self.lock = threading.Lock()

    def add(self, xyz: np.ndarray) -> None:
        keys = np.floor(xyz / self.voxel).astype(np.int64)
        with self.lock:
            for k, p in zip(map(tuple, keys), xyz):
                if k in self._sum:
                    self._sum[k] += p
                    self._n[k] += 1
                elif len(self._sum) < self.max_voxels:
                    self._sum[k] = p.astype(np.float64).copy()
                    self._n[k] = 1
            self.points_seen += len(xyz)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """(N, 3) centroids and (N,) hit counts."""
        with self.lock:
            if not self._sum:
                return np.empty((0, 3)), np.empty(0, dtype=int)
            n = np.fromiter(self._n.values(), dtype=int, count=len(self._n))
            s = np.array(list(self._sum.values()))
        return s / n[:, None], n


def receive(host: str, port: int, cloud: VoxelCloud, min_snr: float, stop: threading.Event, stats: dict) -> None:
    parser = PacketParser()
    while not stop.is_set():
        try:
            with socket.create_connection((host, port), timeout=5) as sock:
                sock.settimeout(1.0)
                parser.reset()
                stats["connected"] = True
                while not stop.is_set():
                    try:
                        data = sock.recv(65536)
                    except socket.timeout:
                        continue
                    if not data:
                        break
                    for fr in parser.feed(data):
                        stats["frames"] += 1
                        pts = fr.points
                        if not len(pts):
                            continue
                        if min_snr > 0:  # snr is NaN when the firmware sends no side info: keep those
                            pts = pts[~(pts["snr"] < min_snr)]
                        if len(pts):
                            cloud.add(np.column_stack([pts["x"], pts["y"], pts["z"]]).astype(np.float64))
        except OSError as e:
            stats["connected"] = False
            print(f"connection to {host}:{port} failed ({e}); retrying in 2 s", file=sys.stderr)
            stop.wait(2.0)
        stats["connected"] = False


def save(cloud: VoxelCloud, path: Path) -> None:
    xyz, n = cloud.snapshot()
    ext = path.suffix.lower()
    if ext == ".npy":
        np.save(path, np.column_stack([xyz, n]))
    elif ext == ".csv":
        np.savetxt(path, np.column_stack([xyz, n]), delimiter=",", header="x,y,z,hits", comments="", fmt="%.4f")
    elif ext == ".ply":
        with open(path, "w") as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\n"
                    "property float z\nproperty int hits\nend_header\n")
            for (x, y, z), h in zip(xyz, n):
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {h}\n")
    else:
        raise SystemExit(f"unsupported --save extension {ext!r} (use .ply, .csv or .npy)")
    print(f"saved {len(xyz)} voxels to {path}")


def run_plot(cloud: VoxelCloud, stats: dict, stop: threading.Event, min_hits: int, refresh_s: float) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    fig.canvas.mpl_connect("close_event", lambda _e: stop.set())
    scat = None
    while not stop.is_set() and plt.fignum_exists(fig.number):
        xyz, n = cloud.snapshot()
        keep = n >= min_hits
        xyz, n = xyz[keep], n[keep]
        if scat is not None:
            scat.remove()
        if len(xyz):
            scat = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=np.log1p(n), cmap="viridis", s=4, depthshade=True)
        else:
            scat = None
        ax.set_xlabel("x right (m)")
        ax.set_ylabel("y boresight (m)")
        ax.set_zlabel("z up (m)")
        if len(xyz):  # equal-scale axes so the scene is not distorted
            lo, hi = xyz.min(axis=0), xyz.max(axis=0)
            mid, half = (lo + hi) / 2, max((hi - lo).max() / 2, 0.5)
            ax.set_xlim(mid[0] - half, mid[0] + half)
            ax.set_ylim(mid[1] - half, mid[1] + half)
            ax.set_zlim(mid[2] - half, mid[2] + half)
        state = "connected" if stats["connected"] else "not connected"
        ax.set_title(f"{state} | {stats['frames']} frames | {cloud.points_seen} points -> {len(xyz)} voxels")
        plt.pause(refresh_s)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("host", help="Pi hostname or IP")
    ap.add_argument("--port", type=int, default=7777)
    ap.add_argument("--voxel", type=float, default=0.05, help="voxel size in metres (default 0.05)")
    ap.add_argument("--min-snr", type=float, default=0.0, help="drop points below this SNR in dB (default: keep all)")
    ap.add_argument("--min-hits", type=int, default=1, help="plot only voxels hit at least this many times")
    ap.add_argument("--refresh", type=float, default=0.5, help="plot refresh interval in seconds")
    ap.add_argument("--save", type=Path, help="write the cloud on exit (.ply, .csv or .npy)")
    ap.add_argument("--no-plot", action="store_true", help="headless: just accumulate and print progress")
    args = ap.parse_args()

    cloud = VoxelCloud(args.voxel)
    stop = threading.Event()
    stats = {"frames": 0, "connected": False}
    rx = threading.Thread(target=receive, args=(args.host, args.port, cloud, args.min_snr, stop, stats), daemon=True)
    rx.start()
    try:
        if args.no_plot:
            while not stop.is_set():
                time.sleep(2.0)
                print(f"frames {stats['frames']}  points {cloud.points_seen}  voxels {len(cloud.snapshot()[1])}")
        else:
            run_plot(cloud, stats, stop, args.min_hits, args.refresh)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        rx.join(timeout=3)
        if args.save:
            save(cloud, args.save)


if __name__ == "__main__":
    main()
