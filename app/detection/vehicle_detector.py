"""Explainable geometric vehicle-candidate test applied to each track."""
from __future__ import annotations

import math

from app.models.garage_state import Track

from .parameters import DetectionParams


def _check(ok: bool, text: str) -> tuple[bool, str]:
    return ok, ("✓ " if ok else "✗ ") + text


def size_checks(t: Track, p: DetectionParams) -> list[tuple[bool, str]]:
    c = t.cluster
    return [
        _check(t.age >= p.cluster_persistence_s, f"seen {t.age:.1f} s (≥ {p.cluster_persistence_s:.1f} s)"),
        _check(p.veh_min_width <= c.width <= p.veh_max_width,
               f"width {c.width:.2f} m in [{p.veh_min_width:.2f}, {p.veh_max_width:.2f}]"),
        _check(p.veh_min_depth <= c.depth <= p.veh_max_depth,
               f"depth {c.depth:.2f} m in [{p.veh_min_depth:.2f}, {p.veh_max_depth:.2f}]"),
        _check(p.veh_min_height <= c.height <= p.veh_max_height,
               f"height {c.height:.2f} m in [{p.veh_min_height:.2f}, {p.veh_max_height:.2f}]"),
        _check(c.area >= p.veh_min_area, f"area {c.area:.1f} m² ≥ {p.veh_min_area:.1f}"),
    ]


def displacement(t: Track) -> float:
    """Net centroid displacement over the track's velocity-history window."""
    if len(t.history) < 2:
        return 0.0
    (_, x0, y0), (_, x1, y1) = t.history[0], t.history[-1]
    return math.hypot(x1 - x0, y1 - y0)


def is_moving(t: Track, p: DetectionParams) -> bool:
    return t.speed > p.movement_threshold and displacement(t) >= p.movement_min_displacement


def update_vehicle_flags(tracks: list[Track], now: float, p: DetectionParams) -> None:
    """Set Track.is_vehicle with required-persistence / departure-persistence hysteresis."""
    for t in tracks:
        checks = size_checks(t, p)
        seen_now = t.last_seen >= now - 1e-9
        candidate = seen_now and all(ok for ok, _ in checks)
        if candidate:
            if t.candidate_since is None:
                t.candidate_since = now
            t.last_candidate = now
        elif not t.is_vehicle:
            t.candidate_since = None
        held = now - t.candidate_since if t.candidate_since is not None else 0.0
        if not t.is_vehicle and candidate and held >= p.vehicle_persistence_s:
            t.is_vehicle = True
            t.vehicle_since = now
        elif t.is_vehicle and not candidate and now - (t.last_candidate or now) > p.departure_persistence_s:
            t.is_vehicle = False
            t.vehicle_since = None
            t.candidate_since = None
        heading = math.degrees(math.atan2(t.velocity[0], t.velocity[1]))
        t.reasons = [txt for _, txt in checks]
        t.reasons.append(_check(t.is_vehicle, f"candidate held {held:.1f} s (need {p.vehicle_persistence_s:.1f} s)")[1])
        t.reasons.append(
            f"speed {t.speed:.2f} m/s, moved {displacement(t):.2f} m → "
            f"{'MOVING' if is_moving(t, p) else 'stationary'} (heading {heading:.0f}°)"
        )
