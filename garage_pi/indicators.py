"""Battery and contact of the stall indicators, reported by the detector ESP32 over a UART.

The detector serves the battery-powered indicators over ESP-NOW and writes one line per event on its
UART (detector GPIO21 -> Pi RXD, header pin 10, 115200 baud):

    IND stall=1 state=Present battery_mv=3850       an indicator completed an exchange
    IND stall=1 contact=lost                        it is overdue
    IND stall=1 contact=ok                          it is back in contact
    IND stall=1 contact=ok battery_mv=3850 age_s=12 periodic summary (battery_mv 0 / age_s -1 until
                                                    the indicator's first exchange)

A battery_mv of 0 means "not measured". Other lines (boot noise) are ignored.

The detector also keeps each stall's indicator MAC address in its flash and reports it, and the Pi
can replace it (the detector saves it, echoes it back and restarts to use it):

    MAC stall=1 addr=aa:bb:cc:dd:ee:ff              the saved address ("none": the stall has no indicator)
    ERR what=<reason>                               a command was refused

Commands from the Pi, one line each:

    MAC?                                            report every saved address now
    SETMAC stall=1 addr=aa:bb:cc:dd:ee:ff           save an address ("none" clears it)
"""
from __future__ import annotations

import logging
import re
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


MAC_RE = re.compile(r"^[0-9A-Fa-f]{2}(:[0-9A-Fa-f]{2}){5}$")


def normalize_mac(text: str) -> str:
    """"aa:bb:cc:dd:ee:ff" in lowercase, or "" for blank / "none"; ValueError if malformed."""
    text = str(text or "").strip()
    if text.lower() in ("", "none"):
        return ""
    if not MAC_RE.match(text):
        raise ValueError(f"{text!r} is not a MAC address like aa:bb:cc:dd:ee:ff")
    return text.lower()


def parse_line(line: str) -> dict | None:
    """Fields of one detector report line, with "stall" as an int; None for anything else.

    "kind" is "IND" for an indicator report, "MAC" for a saved address (as "addr", "" when none) and
    "ERR" for a refused command.
    """
    parts = line.strip().split()
    if not parts or parts[0] not in ("IND", "MAC", "ERR"):
        return None
    kind = parts[0]
    fields = {}
    for part in parts[1:]:
        key, sep, value = part.partition("=")
        if not sep:
            return None
        fields[key] = value
    if kind == "ERR":
        return {"kind": kind, "what": fields.get("what", "")} if "what" in fields else None
    try:
        fields["stall"] = int(fields["stall"])
        for key in ("battery_mv", "age_s"):
            if key in fields:
                fields[key] = int(fields[key])
        if kind == "MAC":
            fields["addr"] = normalize_mac(fields["addr"])
    except (KeyError, ValueError):
        return None
    if kind != "IND":
        fields["kind"] = kind
    return fields


class IndicatorStatus:
    """What is known about each stall's indicator, from the detector's reports."""

    def __init__(self, n_stalls: int, clock=time.time):
        self.n = n_stalls
        self._clock = clock
        self._lock = threading.Lock()
        self._stalls = [{"contact": "unknown", "battery_mv": None, "last_state": None, "last_exchange": None,
                         "mac": None} for _ in range(n_stalls)]
        self.last_line_wall: float | None = None
        self.last_error = ""  # the detector's last refusal of a command

    def apply(self, report: dict) -> bool:
        """Apply one parsed report; True if the stall's published values changed."""
        now = self._clock()
        if report.get("kind") == "ERR":
            with self._lock:
                self.last_line_wall = now
                self.last_error = report["what"]
            return False
        i = report["stall"] - 1
        if not 0 <= i < self.n:
            return False
        with self._lock:
            self.last_line_wall = now
            if report.get("kind") == "MAC":  # not published to MQTT
                self._stalls[i]["mac"] = report["addr"]
                return False
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
                    "mac": s["mac"],  # saved in the detector; None until it has reported
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
        self._serial = None  # the open port, while there is one
        self._write_lock = threading.Lock()
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
                    self._serial = port
                    self._send("MAC?")  # learn the saved addresses without waiting for the next summary
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
            finally:
                self._serial = None

    def _send(self, text: str) -> None:
        port = self._serial
        if port is None:
            raise RuntimeError(f"not connected to the detector on {self.port}")
        with self._write_lock:
            port.write(text.encode("ascii") + b"\n")

    def set_mac(self, stall: int, addr: str) -> str:
        """Ask the detector to save the stall's (1-based) indicator MAC address ("" clears it).

        The detector echoes the saved address back and restarts to use it. Returns the normalized address.
        Raises ValueError for a bad stall or address, RuntimeError if the detector is not connected.
        """
        if not 1 <= stall <= self.status.n:
            raise ValueError(f"stall must be 1 to {self.status.n}")
        addr = normalize_mac(addr)
        self._send(f"SETMAC stall={stall} addr={addr or 'none'}")
        return addr

    def feed(self, line: str) -> None:
        report = parse_line(line)
        if report is not None and self.status.apply(report) and self.on_change:
            self.on_change(report["stall"] - 1)

    def close(self) -> None:
        self._stop.set()
