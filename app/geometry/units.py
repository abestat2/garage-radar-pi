"""Length-unit helpers.  Everything is stored internally in metres."""
from __future__ import annotations

FT = 0.3048
UNITS = {"m": 1.0, "ft": FT, "in": FT / 12.0}


def to_m(value: float, unit: str) -> float:
    return value * UNITS[unit]


def from_m(value: float, unit: str) -> float:
    return value / UNITS[unit]


def fmt_len(value_m: float, unit: str = "ft", digits: int = 1) -> str:
    return f"{from_m(value_m, unit):.{digits}f} {unit}"
