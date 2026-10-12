"""Battery and contact of the stall indicators, reported by the detector ESP32 over a UART.

The detector serves the battery-powered indicators over ESP-NOW and writes one line per event on its
UART (detector GPIO21 -> Pi RXD, header pin 10, 115200 baud):

    IND stall=1 state=Present battery_mv=3850       an indicator completed an exchange
    IND stall=1 contact=lost                        it is overdue
    IND stall=1 contact=ok                          it is back in contact
    IND stall=1 contact=ok battery_mv=3850 age_s=12 periodic summary (battery_mv 0 / age_s -1 until
                                                    the indicator's first exchange)

A battery_mv of 0 means "not measured". Other lines (boot noise) are ignored.
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger("garage_pi.indicators")

# Resting voltage of a 1-cell Li-ion / LiPo against state of charge (approximate, light load)
_LIION_CURVE = [(3300, 0), (3450, 5), (3680, 10), (3740, 20), (3770, 30), (3790, 40), (3820, 50),
                (3870, 60), (3920, 70), (4000, 80), (4100, 90), (4200, 100)]


def battery_percent(mv: int) -> int | None:
    """Estimated charge of a 1-cell Li-ion battery from its voltage; None if not measured."""
    if mv <= 0:
        return None
    if mv <= _LIION_CURVE[0][0]:
        return 0
    for (v0, p0), (v1, p1) in zip(_LIION_CURVE, _LIION_CURVE[1:]):
        if mv <= v1:
            return round(p0 + (p1 - p0) * (mv - v0) / (v1 - v0))
    return 100


def parse_line(line: str) -> dict | None:
    """Fields of one detector report line, with "stall" as an int; None for anything else."""
    parts = line.strip().split()
    if not parts or parts[0] != "IND":
        return None
    fields = {}
    for part in parts[1:]:
        key, sep, value = part.partition("=")
        if not sep:
            return None
        fields[key] = value
    try:
        fields["stall"] = int(fields["stall"])
        for key in ("battery_mv", "age_s"):
            if key in fields:
                fields[key] = int(fields[key])
    except (KeyError, ValueError):
        return None
    return fields


class IndicatorStatus:
    """What is known about each stall's indicator, from the detector's reports."""

    def __init__(self, n_stalls: int, clock=time.time):
        self.n = n_stalls
        self._clock = clock
        self._lock = threading.Lock()
        self._stalls = [{"contact": "unknown", "battery_mv": None, "last_state": None, "last_exchange": None}
                        for _ in range(n_stalls)]
        self.last_line_wall: float | None = None

    def apply(self, report: dict) -> bool:
        """Apply one parsed report; True if the stall's published values changed."""
        i = report["stall"] - 1
        if not 0 <= i < self.n:
            return False
        now = self._clock()
        with self._lock:
            self.last_line_wall = now
            s = self._stalls[i]
            before = self._published(s)
            if "contact" in report:
                s["contact"] = report["contact"]
            if "state" in report:  # an exchange just completed, so the indicator is in contact
                s["contact"] = "ok"
                s["last_state"] = report["state"]
                s["last_exchange"] = now
            if report.get("battery_mv", 0) > 0:
                s["battery_mv"] = report["battery_mv"]
            if report.get("age_s", -1) >= 0:
                s["last_exchange"] = now - report["age_s"]
            return self._published(s) != before

    @staticmethod
    def _published(s: dict) -> tuple:
        return s["contact"], s["battery_mv"]

    def snapshot(self) -> list[dict]:
        now = self._clock()
        with self._lock:
            out = []
            for i, s in enumerate(self._stalls):
                mv = s["battery_mv"]
                out.append({
                    "stall": i + 1,
                    "contact": s["contact"],
                    "battery_v": None if mv is None else round(mv / 1000, 2),
                    "battery_pct": None if mv is None else battery_percent(mv),
                    "last_state": s["last_state"],
                    "last_exchange_age_s": None if s["last_exchange"] is None else round(now - s["last_exchange"]),
                })
            return out


class IndicatorLink:
    """Reads the detector's report lines from a serial port on a background thread."""

    def __init__(self, settings: dict, status: IndicatorStatus, on_change=None):
        self.port = settings.get("port", "/dev/serial0")
        self.baud = int(settings.get("baud", 115200))
        self.status = status
        self.on_change = on_change
        self.error = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="indicator-link", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            import serial
        except ImportError:
            self.error = "pyserial is not installed"
            log.warning(self.error)
            return
        while not self._stop.is_set():
            try:
                with serial.Serial(self.port, self.baud, timeout=1) as port:
                    self.error = ""
                    log.info("reading indicator reports on %s", self.port)
                    buf = b""
                    while not self._stop.is_set():
                        buf += port.read(256)
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            self.feed(line.decode("ascii", "replace"))
            except Exception as e:  # port missing (UART not enabled) or unplugged: retry
                if str(e) != self.error:
                    log.warning("indicator report port %s: %s", self.port, e)
                self.error = str(e)
                self._stop.wait(10)

    def feed(self, line: str) -> None:
        report = parse_line(line)
        if report is not None and self.status.apply(report) and self.on_change:
            self.on_change(report["stall"] - 1)

    def close(self) -> None:
        self._stop.set()
