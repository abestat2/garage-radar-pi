"""NeoPixel (WS2812/WS2812B, or SK6812 RGBW) bar driven from the Pi's SPI0 MOSI pin (GPIO10).

WS2812 LEDs need a single-wire signal with ~0.3/0.8 us pulses; I2C cannot produce it, but SPI can:
each LED data bit becomes one SPI byte clocked at ~6.4 MHz (160 ns per SPI bit):
    "0" bit -> 0b11000000  (~0.32 us high)
    "1" bit -> 0b11111000  (~0.80 us high)
followed by >= 300 us of zeros to latch.  Needs `dtparam=spi=on`.  Most WS2812B accept 3.3 V
data, but a 74AHCT125 level shifter to 5 V is recommended; power the strip from 5 V, not the Pi.

`order` is the byte order the LEDs expect: "GRB" for WS2812B, "GRBW" for SK6812 RGBW bars (4 bytes per
LED - with a 3-byte order such a bar shows only 3/4 of its LEDs, in scrambled colours).  For RGBW the
white part of each colour (min of R, G, B) is sent to the dedicated white chip.
"""
from __future__ import annotations

import logging
import threading
import time

from app.models.garage_state import StallState

log = logging.getLogger("garage_pi.leds")

SPI_HZ = 6_400_000
BIT0, BIT1 = 0b11000000, 0b11111000
LATCH_BYTES = 300  # 300 bytes x 8 bits x 156 ns = 375 us low


def encode_ws2812(colors: list[tuple[int, ...]], order: str = "GRB") -> bytes:
    """RGB (or RGBW) tuples -> SPI byte stream.  With a "W" in `order`, an RGB colour's common white
    component is moved to the white chip."""
    idx = {"R": 0, "G": 1, "B": 2, "W": 3}
    rgbw = "W" in order
    out = bytearray()
    for c in colors:
        if rgbw and len(c) == 3:
            w = min(c)
            c = (c[0] - w, c[1] - w, c[2] - w, w)
        for ch in order:
            v = int(c[idx[ch]] if idx[ch] < len(c) else 0) & 0xFF
            for bit in range(7, -1, -1):
                out.append(BIT1 if (v >> bit) & 1 else BIT0)
    out.extend(b"\x00" * LATCH_BYTES)
    return bytes(out)


class _MockStrip:
    def __init__(self, n):
        self.frames = []
        self.last: list = [(0, 0, 0)] * n

    def show(self, colors):
        self.last = list(colors)

    def close(self):
        pass


class _SpiStrip:
    def __init__(self, bus: int, device: int, order: str):
        import spidev  # python3-spidev on Raspberry Pi OS

        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = SPI_HZ
        self.spi.mode = 0
        self.order = order

    def show(self, colors):
        data = encode_ws2812(colors, self.order)
        for i in range(0, len(data), 4096):  # spidev's default transfer limit
            self.spi.writebytes2(data[i:i + 4096])

    def close(self):
        try:
            self.spi.close()
        except Exception:
            pass


class LedBar:
    """Renders stall states and event flashes at ~20 Hz on its own thread."""

    def __init__(self, settings: dict, n_stalls: int, use_hardware: bool = True):
        self.s = settings
        self.enabled = bool(settings.get("enabled", False))
        self.count = int(settings.get("count", 8))
        self.strip = None
        self.problem = "no LED hardware (mock output)"
        if self.enabled and use_hardware:
            try:
                self.strip = _SpiStrip(int(settings.get("spi_bus", 0)), int(settings.get("spi_device", 0)),
                                       settings.get("order", "GRB"))
            except Exception as e:
                log.warning("SPI LED strip unavailable (%s) - LED output disabled", e)
                self.problem = f"SPI unavailable: {e}"
        if self.strip is None:
            self.strip = _MockStrip(self.count)
        self.hardware = isinstance(self.strip, _SpiStrip)
        if not self.enabled:
            self.problem = "LED bar is disabled (Settings -> leds -> enabled, then restart the service)"
            log.info("LED bar disabled in the settings")
        elif self.hardware:
            self.problem = ""
            log.info("LED bar on SPI%s.%s: %d LEDs, order %s", settings.get("spi_bus", 0),
                     settings.get("spi_device", 0), self.count, settings.get("order", "GRB"))
        self.states = [StallState.UNKNOWN] * n_stalls
        self.flash: list[tuple[str, float]] = [("", 0.0)] * n_stalls
        self.system = "starting"  # "ok", "error", "busy", "starting"
        self.test_until = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="led-bar", daemon=True)
        if self.enabled:
            self._thread.start()

    # ------------------------------------------------------------------ inputs
    def set_states(self, states: list[StallState], system: str) -> None:
        with self._lock:
            self.states = list(states)
            self.system = system

    def event(self, stall_index: int, kind: str) -> None:
        with self._lock:
            if stall_index < len(self.flash):
                self.flash[stall_index] = (kind, time.time() + float(self.s.get("event_flash_s", 3.0)))

    def test(self, seconds: float = 5.0) -> None:
        self.test_until = time.time() + seconds

    # ------------------------------------------------------------------ rendering
    def frame(self, now: float | None = None) -> list[tuple[int, int, int]]:
        now = time.time() if now is None else now
        colors = [(0, 0, 0)] * self.count
        bright = float(self.s.get("brightness", 0.25))
        blink_on = int(now * 2) % 2 == 0
        if now < self.test_until:  # chase pattern for wiring checks
            k = int(now * 8) % max(self.count, 1)
            wheel = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
            return [tuple(int(v * bright) for v in wheel[(i + k) % 4]) for i in range(self.count)]
        palette = self.s.get("colors", {})
        blink = set(self.s.get("blink_states", []))
        with self._lock:
            states, flash, system = list(self.states), list(self.flash), self.system
        for i, leds in enumerate(self.s.get("stall_leds", [])):
            if i >= len(states):
                break
            st = states[i]
            col = palette.get(st.value, [0, 0, 0])
            if st.value in blink and not blink_on:
                col = [0, 0, 0]
            kind, until = flash[i]
            if now < until:  # event flash overrides: fast blink green (entered) / amber (exited)
                col = ([0, 255, 0] if kind == "ENTERED" else [255, 120, 0]) if int(now * 6) % 2 == 0 else [0, 0, 0]
            for j in leds:
                if 0 <= j < self.count:
                    colors[j] = tuple(int(v * bright) for v in col)
        sl = self.s.get("status_led")
        if sl is not None and 0 <= sl < self.count:
            col = {"ok": (0, 60, 0), "error": (120, 0, 0) if blink_on else (0, 0, 0), "busy": (0, 0, 120)}.get(system, (40, 40, 0))
            colors[sl] = tuple(int(v * bright) for v in col)
        return colors

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.strip.show(self.frame())
            except Exception as e:
                log.warning("LED update failed: %s", e)
                time.sleep(1)
            time.sleep(0.05)

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1)
        try:
            self.strip.show([(0, 0, 0)] * self.count)
            self.strip.close()
        except Exception:
            pass
