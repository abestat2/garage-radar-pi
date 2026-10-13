"""Digital outputs to the ESP32: one pin per stall + a "system OK" pin.

Contract (see deploy/README_PI.md):
    stall pin HIGH  = a car is in the stall  (OCCUPIED or VEHICLE EXITING; optionally ENTERING)
    stall pin LOW   = the stall is empty     (EMPTY)
    UNKNOWN / SENSOR ERROR: the stall pin HOLDS its last level, and the OK pin goes LOW
    OK pin HIGH     = radar running and parked-car detection ready -> stall pins are trustworthy
A rising edge on a stall pin is therefore a confirmed car entry, a falling edge a confirmed exit.
(`active_high: false` inverts every pin.)

Test: simulate_arrival() drives one stall pin LOW (empty) and then HIGH (car) with the OK pin HIGH, so the
stall's indicator guides a car in without one. Only the pins are faked: levels, edges and events stay real,
and the pin returns to its real level when the simulation ends.
"""
from __future__ import annotations

import logging
import time

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
        self.levels: list[bool | None] = [None] * len(self.pins)  # real (radar) levels
        self.ok = False  # real system OK
        # What is on the pins, which differs from the real levels while simulating
        self._written: list[bool] = [False] * len(self.pins)
        self._ok_written = False
        # stall index -> (time the simulated car arrives, time the simulation ends)
        self.simulations: dict[int, tuple[float, float]] = {}
        for p in self.pins + ([self.ok_pin] if self.ok_pin else []):
            self._write(p, False)

    def _write(self, pin, logical: bool) -> None:
        (pin.on if logical == self.active_high else pin.off)()

    def update(self, stall_states: list[StallState], system_ok: bool, now: float | None = None) -> list[tuple[int, bool]]:
        """Apply new states; returns [(stall index, new level)] for edges that occurred."""
        edges = []
        for i, st in enumerate(stall_states[:len(self.pins)]):
            lvl = desired_level(st, self.levels[i], self.entering_is_present)
            if lvl is not None and lvl != self.levels[i]:
                if self.levels[i] is not None:
                    edges.append((i, lvl))
                self.levels[i] = lvl
        self.ok = system_ok
        self.refresh(now)
        return edges

    def refresh(self, now: float | None = None) -> None:
        """Write the pins: the real levels, or a simulation's while one runs. Call regularly so a
        simulation's steps happen on time even without new radar states."""
        now = time.time() if now is None else now
        self.simulations = {i: span for i, span in self.simulations.items() if now < span[1]}
        for i, pin in enumerate(self.pins):
            if i in self.simulations:
                level = now >= self.simulations[i][0]
            elif self.levels[i] is not None:
                level = self.levels[i]
            else:
                continue  # nothing known yet: stays low
            if level != self._written[i]:
                self._write(pin, level)
                self._written[i] = level
        ok = self.ok or bool(self.simulations)
        if self.ok_pin is not None and ok != self._ok_written:
            self._write(self.ok_pin, ok)
            self._ok_written = ok

    def simulate_arrival(self, i: int, empty_s: float, present_s: float, now: float | None = None) -> None:
        """Fake a car arriving in stall i on its pin: empty for empty_s (so the indicator sees the
        stall empty first), then a car for present_s, then back to the real level."""
        if not 0 <= i < len(self.pins):
            raise ValueError(f"no stall {i + 1}")
        now = time.time() if now is None else now
        self.simulations[i] = (now + empty_s, now + empty_s + present_s)
        self.refresh(now)

    def stop_simulations(self, now: float | None = None) -> None:
        self.simulations.clear()
        self.refresh(now)

    def simulation(self, i: int, now: float | None = None) -> dict | None:
        """The running simulation of stall i: its phase and seconds left, or None."""
        now = time.time() if now is None else now
        span = self.simulations.get(i)
        if span is None or now >= span[1]:
            return None
        arriving = now < span[0]
        return {"phase": "empty" if arriving else "car", "remaining_s": round((span[0] if arriving else span[1]) - now)}

    def describe(self) -> list[dict]:
        nums = self.pin_numbers
        return [{"stall": i + 1, "gpio": nums[i] if i < len(nums) else None,
                 "level": None if lvl is None else ("HIGH" if lvl == self.active_high else "LOW"),
                 "car_present": lvl, "simulation": self.simulation(i)} for i, lvl in enumerate(self.levels)]

    def close(self) -> None:
        for p in self.pins + ([self.ok_pin] if self.ok_pin else []):
            try:
                p.off()
                p.close()
            except Exception:
                pass
