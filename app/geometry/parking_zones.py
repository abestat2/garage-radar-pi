"""Parking-stall definitions and point-in-polygon tests."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ParkingStall:
    name: str
    left: float  # X of left boundary, m
    right: float  # X of right boundary, m
    front: float  # Y of front (door-side) boundary, m
    rear: float  # Y of rear (back-wall-side) boundary, m
    polygon: list[tuple[float, float]] = field(default_factory=list)  # optional override
    # where the CAR BODY sits (parked-car detection uses only this); empty = default strip
    car_zone: list[tuple[float, float]] = field(default_factory=list)

    def vertices(self) -> np.ndarray:
        if len(self.polygon) >= 3:
            return np.asarray(self.polygon, dtype=np.float64)
        return np.array(
            [
                [self.left, self.front],
                [self.right, self.front],
                [self.right, self.rear],
                [self.left, self.rear],
            ],
            dtype=np.float64,
        )

    def contains(self, xy: np.ndarray) -> np.ndarray:
        return points_in_polygon(np.asarray(xy, dtype=np.float64).reshape(-1, 2), self.vertices())

    def car_zone_vertices(self) -> np.ndarray:
        """Car-body zone: the custom polygon, or a CAR_ZONE_WIDTH strip centred in the stall."""
        if len(self.car_zone) >= 3:
            return np.asarray(self.car_zone, dtype=np.float64)
        return np.asarray(default_car_zone(self), dtype=np.float64)

    def car_zone_contains(self, xy: np.ndarray) -> np.ndarray:
        return points_in_polygon(np.asarray(xy, dtype=np.float64).reshape(-1, 2), self.car_zone_vertices())

    def area(self) -> float:
        v = self.vertices()
        x, y = v[:, 0], v[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def centroid(self) -> tuple[float, float]:
        v = self.vertices()
        return float(v[:, 0].mean()), float(v[:, 1].mean())

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "left": self.left,
            "right": self.right,
            "front": self.front,
            "rear": self.rear,
            "polygon": [list(p) for p in self.polygon],
            "car_zone": [list(p) for p in self.car_zone],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ParkingStall":
        return cls(
            name=d.get("name", "Stall"),
            left=float(d["left"]),
            right=float(d["right"]),
            front=float(d["front"]),
            rear=float(d["rear"]),
            polygon=[tuple(p) for p in d.get("polygon", [])],
            car_zone=[tuple(map(float, p)) for p in d.get("car_zone", [])],
        )


CAR_ZONE_WIDTH = 6.5 * 0.3048  # m: a typical car body plus a little
CAR_ZONE_END_MARGIN = 1.0 * 0.3048  # m: kept clear at each end of the stall


def default_car_zone(stall: "ParkingStall", width: float = CAR_ZONE_WIDTH, front_margin: float = CAR_ZONE_END_MARGIN,
                     rear_margin: float = CAR_ZONE_END_MARGIN) -> list[tuple[float, float]]:
    v = stall.vertices()
    x0, x1, y0, y1 = v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()
    cx, w = (x0 + x1) / 2, min(width, x1 - x0)
    a, b = y0 + front_margin, max(y0 + front_margin + 0.5, y1 - rear_margin)
    return [(cx - w / 2, a), (cx + w / 2, a), (cx + w / 2, b), (cx - w / 2, b)]


@dataclass
class IgnoreZone:
    """Floor area where clutter collects: never used as parked-car evidence."""

    name: str
    polygon: list[tuple[float, float]]

    def vertices(self) -> np.ndarray:
        return np.asarray(self.polygon, dtype=np.float64)

    def contains(self, xy: np.ndarray) -> np.ndarray:
        if len(self.polygon) < 3:
            return np.zeros(len(np.asarray(xy).reshape(-1, 2)), dtype=bool)
        return points_in_polygon(np.asarray(xy, dtype=np.float64).reshape(-1, 2), self.vertices())

    def to_dict(self) -> dict:
        return {"name": self.name, "polygon": [list(p) for p in self.polygon]}

    @classmethod
    def from_dict(cls, d: dict) -> "IgnoreZone":
        return cls(d.get("name", "Clutter"), [tuple(map(float, p)) for p in d.get("polygon", [])])


def points_in_polygon(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray casting.  xy: (N, 2), poly: (M, 2)."""
    if len(xy) == 0:
        return np.zeros(0, dtype=bool)
    x, y = xy[:, 0][:, None], xy[:, 1][:, None]
    x1, y1 = poly[:, 0][None, :], poly[:, 1][None, :]
    x2, y2 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    straddles = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    crossings = straddles & (x < x_cross)
    return (np.count_nonzero(crossings, axis=1) % 2) == 1
