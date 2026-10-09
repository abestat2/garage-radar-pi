"""Digital outputs to the ESP32: one pin per stall + a "system OK" pin.

Contract (see deploy/README_PI.md):
    stall pin HIGH  = a car is in the stall  (OCCUPIED or VEHICLE EXITING; optionally ENTERING)
    stall pin LOW   = the stall is empty     (EMPTY)
    UNKNOWN / SENSOR ERROR: the stall pin HOLDS its last level, and the OK pin goes LOW
    OK pin HIGH     = radar running and parked-car detection ready -> stall pins are trustworthy
A rising edge on a stall pin is therefore a confirmed car entry, a falling edge a confirmed exit.
(`active_high: false` inverts every pin.)
"""
from __future__ import annotations

import logging

from app.models.garage_state import StallState

log = logging.getLogger("garage_pi.gpio")

PRESENT = {StallState.OCCUPIED, StallState.EXITING}
ABSENT = {StallState.EMPTY}


class MockPin:
    """Stand-in when gpiozero / a Pi is not available (and in tests)."""

    def __init__(self, pin: int):
        self.pin, self.value, self.history = pin, 0, []

    def on(self):
        self._set(1)

    def off(self):
        self._set(0)

    def _set(self, v):
        if v != self.value or not self.history:
            self.history.append(v)
        self.value = v

    def close(self):
        pass


def _make_pin(pin: int, use_hardware: bool):
    if use_hardware:
        try:
            from gpiozero import DigitalOutputDevice

            return DigitalOutputDevice(pin, active_high=True, initial_value=False)
        except Exception as e:  # not a Pi, or gpiozero/lgpio missing
            log.warning("GPIO %s unavailable (%s) - using a mock pin", pin, e)
    return MockPin(pin)


def desired_level(state: StallState, previous: bool | None, entering_is_present: bool) -> bool | None:
    """True = car present, False = absent, None = hold the previous level."""
    if state in PRESENT or (entering_is_present and state == StallState.ENTERING):
        return True
    if state in ABSENT:
        return False
    if state == StallState.ENTERING:
        return previous if previous is not None else False
    return previous  # UNKNOWN / SENSOR ERROR: hold


class StallOutputs:
    def __init__(self, settings: dict, use_hardware: bool = True):
        self.enabled = bool(settings.get("enabled", True))
        self.active_high = bool(settings.get("active_high", True))
        self.entering_is_present = bool(settings.get("entering_is_present", False))
        hw = use_hardware and self.enabled
        self.pin_numbers = list(settings.get("stall_pins", []))
        self.ok_pin_number = settings.get("ok_pin")
        self.pins = [_make_pin(p, hw) for p in self.pin_numbers]
        ok = settings.get("ok_pin")
        self.ok_pin = _make_pin(ok, hw) if ok is not None else None
        self.levels: list[bool | None] = [None] * len(self.pins)
        self.ok = False
        for p in self.pins + ([self.ok_pin] if self.ok_pin else []):
            self._write(p, False)

    def _write(self, pin, logical: bool) -> None:
        (pin.on if logical == self.active_high else pin.off)()

    def update(self, stall_states: list[StallState], system_ok: bool) -> list[tuple[int, bool]]:
        """Apply new states; returns [(stall index, new level)] for edges that occurred."""
        edges = []
        for i, (pin, st) in enumerate(zip(self.pins, stall_states)):
            lvl = desired_level(st, self.levels[i], self.entering_is_present)
            if lvl is not None and lvl != self.levels[i]:
                self._write(pin, lvl)
                if self.levels[i] is not None:
                    edges.append((i, lvl))
                self.levels[i] = lvl
        if self.ok_pin is not None and system_ok != self.ok:
            self._write(self.ok_pin, system_ok)
        self.ok = system_ok
        return edges

    def describe(self) -> list[dict]:
        nums = self.pin_numbers
        return [{"stall": i + 1, "gpio": nums[i] if i < len(nums) else None,
                 "level": None if lvl is None else ("HIGH" if lvl == self.active_high else "LOW"),
                 "car_present": lvl} for i, lvl in enumerate(self.levels)]

    def close(self) -> None:
        for p in self.pins + ([self.ok_pin] if self.ok_pin else []):
            try:
                p.off()
                p.close()
            except Exception:
                pass
