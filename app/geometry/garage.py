"""Garage geometry model (Garage Geometry Mode)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .coordinate_transform import in_fov
from .parking_zones import IgnoreZone, ParkingStall, points_in_polygon
from .radar_pose import RadarPose
from .units import FT


@dataclass
class Door:
    width: float
    height: float
    x_left: float  # horizontal position: X of the door's left edge, m
    bottom: float = 0.0  # optional bottom height, m
    y: float = 0.0  # setback: Y of the wall this door is in (0 = main door wall), m

    def to_dict(self) -> dict:
        return dict(width=self.width, height=self.height, x_left=self.x_left, bottom=self.bottom, y=self.y)

    @classmethod
    def from_dict(cls, d: dict) -> "Door":
        return cls(float(d["width"]), float(d["height"]), float(d["x_left"]), float(d.get("bottom", 0.0)),
                   float(d.get("y", 0.0)))


def default_doors(width: float) -> list[Door]:
    door_w = 8 * FT
    gap = max((width - 3 * door_w) / 4, 0.0)
    return [Door(door_w, 7 * FT, gap + i * (door_w + gap)) for i in range(3)]


def default_stalls(width: float, depth: float) -> list[ParkingStall]:
    w = width / 3
    return [
        ParkingStall(f"Stall {i + 1}", left=i * w + 0.15, right=(i + 1) * w - 0.15, front=0.3, rear=depth - 0.6)
        for i in range(3)
    ]


@dataclass
class Garage:
    """Garage envelope.  `width`/`depth` are the bounding box; `outline` (optional, XY polygon)
    gives the real shape for non-rectangular garages.  See app/geometry/site.py for the
    blueprint-based default."""

    width: float = 32 * FT
    depth: float = 24 * FT
    ceiling_height: float = 9 * FT
    doors: list[Door] = field(default_factory=list)
    stalls: list[ParkingStall] = field(default_factory=list)
    radar: RadarPose = field(default_factory=RadarPose)
    display_unit: str = "ft"
    outline: list[tuple[float, float]] = field(default_factory=list)  # optional floor-plan polygon
    ignore_zones: list[IgnoreZone] = field(default_factory=list)  # clutter areas (no parked-car evidence)

    def __post_init__(self):
        if not self.doors:
            self.doors = default_doors(self.width)
        if not self.stalls:
            self.stalls = default_stalls(self.width, self.depth)

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "coordinate_system": "metres; origin = inside corner of door wall and left wall; "
            "+X along door wall, +Y toward back wall, +Z up",
            "width": self.width,
            "depth": self.depth,
            "ceiling_height": self.ceiling_height,
            "doors": [d.to_dict() for d in self.doors],
            "stalls": [s.to_dict() for s in self.stalls],
            "radar": self.radar.to_dict(),
            "display_unit": self.display_unit,
            "outline": [list(p) for p in self.outline],
            "ignore_zones": [z.to_dict() for z in self.ignore_zones],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Garage":
        return cls(
            width=float(d["width"]),
            depth=float(d["depth"]),
            ceiling_height=float(d["ceiling_height"]),
            doors=[Door.from_dict(x) for x in d.get("doors", [])],
            stalls=[ParkingStall.from_dict(x) for x in d.get("stalls", [])],
            radar=RadarPose.from_dict(d.get("radar", {})),
            display_unit=d.get("display_unit", "ft"),
            outline=[tuple(map(float, p)) for p in d.get("outline", [])],
            ignore_zones=[IgnoreZone.from_dict(z) for z in d.get("ignore_zones", [])],
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Garage":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------ shape
    def outline_vertices(self) -> np.ndarray:
        """Floor-plan polygon (N, 2): the custom outline, or the width x depth rectangle."""
        if len(self.outline) >= 3:
            return np.asarray(self.outline, dtype=np.float64)
        return np.array([[0, 0], [self.width, 0], [self.width, self.depth], [0, self.depth]], dtype=np.float64)

    def contains_xy(self, xy: np.ndarray, tol: float = 0.05) -> np.ndarray:
        """Points inside the floor plan (with a small tolerance for points on the walls)."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        poly = self.outline_vertices()
        inside = points_in_polygon(xy, poly)
        if tol > 0 and not inside.all():
            for dx, dy in ((tol, 0), (-tol, 0), (0, tol), (0, -tol)):
                inside |= points_in_polygon(xy + (dx, dy), poly)
        return inside

    # ------------------------------------------------------------ analysis
    def stall_fov_coverage(self, stall: ParkingStall, heights=(0.3, 1.0), step: float = 0.1) -> float:
        """Fraction of the stall area (sampled at the given heights) inside the radar FOV."""
        v = stall.vertices()
        xs = np.arange(v[:, 0].min() + step / 2, v[:, 0].max(), step)
        ys = np.arange(v[:, 1].min() + step / 2, v[:, 1].max(), step)
        gx, gy = np.meshgrid(xs, ys)
        xy = np.column_stack([gx.ravel(), gy.ravel()])
        xy = xy[stall.contains(xy)]
        if len(xy) == 0:
            return 0.0
        total, visible = 0, 0
        for h in heights:
            pts = np.column_stack([xy, np.full(len(xy), h)])
            m = in_fov(pts, self.radar)
            total += len(m)
            visible += int(m.sum())
        return visible / total

    def coverage_map(self, height: float, step: float = 0.1) -> np.ndarray:
        """Boolean FOV visibility grid (rows = Y, cols = X) over the floor plan at a given height."""
        xs = np.arange(step / 2, self.width, step)
        ys = np.arange(step / 2, self.depth, step)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, height)])
        vis = in_fov(pts, self.radar) & self.contains_xy(pts[:, :2], tol=0)
        return vis.reshape(gx.shape)

    def warnings(self) -> list[str]:
        out = []
        r = self.radar
        if not (self.contains_xy([[r.x, r.y]], tol=0.2)[0] and 0 <= r.z <= self.ceiling_height):
            out.append("Radar position is outside the garage envelope.")
        for s in self.stalls:
            if not self.contains_xy(s.vertices()).all():
                out.append(f"{s.name} extends outside the garage outline.")
            cov = self.stall_fov_coverage(s)
            if cov < 0.999:
                out.append(f"{s.name}: only {cov * 100:.0f}% of the zone is inside the radar FOV.")
        for i, d in enumerate(self.doors):
            if d.x_left < 0 or d.x_left + d.width > self.width:
                out.append(f"Door {i + 1} extends beyond the door wall.")
        return out
