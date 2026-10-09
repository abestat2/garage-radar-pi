"""mmWave CLI configuration (.cfg) parsing, derived parameters and editing.

The configuration is kept as an ordered list of lines so comments and command
order survive editing.  Radar-side parameters (frame period, CFAR thresholds,
clutter removal, FOV limits) are edited here and require the configuration to
be re-sent, i.e. a radar restart.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

C = 299_792_458.0


@dataclass
class DerivedParams:
    start_freq_ghz: float = 0.0
    bandwidth_mhz: float = 0.0
    range_resolution_m: float = 0.0
    max_range_m: float = 0.0
    velocity_resolution_mps: float = 0.0
    max_velocity_mps: float = 0.0
    frame_period_ms: float = 0.0
    frame_rate_hz: float = 0.0
    num_tx: int = 0
    num_rx: int = 0
    chirps_per_frame: int = 0
    adc_samples: int = 0

    def summary(self) -> list[tuple[str, str]]:
        return [
            ("Start frequency", f"{self.start_freq_ghz:.2f} GHz"),
            ("Bandwidth", f"{self.bandwidth_mhz:.0f} MHz"),
            ("TX / RX", f"{self.num_tx} / {self.num_rx}"),
            ("ADC samples", str(self.adc_samples)),
            ("Chirps / frame", str(self.chirps_per_frame)),
            ("Range resolution", f"{self.range_resolution_m * 100:.1f} cm"),
            ("Max range (IF)", f"{self.max_range_m:.1f} m"),
            ("Max velocity", f"±{self.max_velocity_mps:.2f} m/s"),
            ("Velocity resolution", f"{self.velocity_resolution_mps:.3f} m/s"),
            ("Frame period", f"{self.frame_period_ms:.0f} ms ({self.frame_rate_hz:.1f} Hz)"),
        ]


class RadarConfig:
    def __init__(self, text: str = "", name: str = "untitled.cfg"):
        self.name = name
        self.lines: list[str] = [ln.rstrip() for ln in text.splitlines()]

    # ------------------------------------------------------------ I/O
    @classmethod
    def load(cls, path: str | Path) -> "RadarConfig":
        p = Path(path)
        return cls(p.read_text(), name=p.name)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.text())

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"

    def commands(self) -> list[str]:
        """Lines to send to the CLI (comments and blanks removed)."""
        out = []
        for ln in self.lines:
            s = ln.split("%", 1)[0].strip()
            if s:
                out.append(s)
        return out

    # ------------------------------------------------------------ access
    def find(self, cmd: str, *prefix: str) -> list[str] | None:
        """Arguments of the first `cmd` line whose leading args match `prefix`."""
        for ln in self.commands():
            parts = ln.split()
            if parts[0] == cmd and parts[1 : 1 + len(prefix)] == list(prefix):
                return parts[1:]
        return None

    def find_all(self, cmd: str) -> list[list[str]]:
        return [ln.split()[1:] for ln in self.commands() if ln.split()[0] == cmd]

    def set_arg(self, cmd: str, prefix: tuple[str, ...], index: int, value) -> bool:
        """Set argument `index` (0-based, after the command) of the matching line."""
        for i, ln in enumerate(self.lines):
            body, _, comment = ln.partition("%")
            parts = body.split()
            if parts and parts[0] == cmd and parts[1 : 1 + len(prefix)] == list(prefix):
                if index >= len(parts) - 1:
                    return False
                parts[index + 1] = _fmt(value)
                self.lines[i] = " ".join(parts) + (f" %{comment}" if comment else "")
                return True
        return False

    def ensure_line(self, cmd: str, args: list) -> None:
        """Replace the first `cmd` line, or insert one before sensorStart."""
        new = " ".join([cmd] + [_fmt(a) for a in args])
        for i, ln in enumerate(self.lines):
            parts = ln.split("%", 1)[0].split()
            if parts and parts[0] == cmd:
                self.lines[i] = new
                return
        idx = next((i for i, ln in enumerate(self.lines) if ln.strip().startswith("sensorStart")), len(self.lines))
        self.lines.insert(idx, new)

    # ------------------------------------------------------------ radar-side parameters
    def get_radar_params(self) -> dict:
        p = {}
        frame = self.find("frameCfg")
        if frame:
            p["frame_period_ms"] = float(frame[4])
        cr = self.find("clutterRemoval")
        if cr:
            p["clutter_removal"] = bool(int(cr[1]))
        cfar_r = self.find("cfarCfg", "-1", "0")
        if cfar_r:
            p["cfar_range_threshold_db"] = float(cfar_r[7])
        cfar_d = self.find("cfarCfg", "-1", "1")
        if cfar_d:
            p["cfar_doppler_threshold_db"] = float(cfar_d[7])
        fov_r = self.find("cfarFovCfg", "-1", "0")
        if fov_r:
            p["cfar_max_range_m"] = float(fov_r[3])
        aoa = self.find("aoaFovCfg")
        if aoa:
            p["aoa_az_limit_deg"] = max(abs(float(aoa[1])), abs(float(aoa[2])))
            p["aoa_el_limit_deg"] = max(abs(float(aoa[3])), abs(float(aoa[4])))
        return p

    def apply_radar_params(self, p: dict) -> None:
        if "frame_period_ms" in p:
            self.set_arg("frameCfg", (), 4, p["frame_period_ms"])
        if "clutter_removal" in p:
            self.set_arg("clutterRemoval", ("-1",), 1, int(bool(p["clutter_removal"])))
        if "cfar_range_threshold_db" in p:
            self.set_arg("cfarCfg", ("-1", "0"), 7, p["cfar_range_threshold_db"])
        if "cfar_doppler_threshold_db" in p:
            self.set_arg("cfarCfg", ("-1", "1"), 7, p["cfar_doppler_threshold_db"])
        if "cfar_max_range_m" in p:
            self.set_arg("cfarFovCfg", ("-1", "0"), 3, p["cfar_max_range_m"])
        if "aoa_az_limit_deg" in p:
            a = p["aoa_az_limit_deg"]
            self.set_arg("aoaFovCfg", ("-1",), 1, -a)
            self.set_arg("aoaFovCfg", ("-1",), 2, a)
        if "aoa_el_limit_deg" in p:
            e = p["aoa_el_limit_deg"]
            self.set_arg("aoaFovCfg", ("-1",), 3, -e)
            self.set_arg("aoaFovCfg", ("-1",), 4, e)

    # ------------------------------------------------------------ derived values
    def derived(self) -> DerivedParams:
        d = DerivedParams()
        ch = self.find("channelCfg")
        if ch:
            d.num_rx = bin(int(ch[0])).count("1")
            d.num_tx = bin(int(ch[1])).count("1")
        prof = self.find("profileCfg")
        frame = self.find("frameCfg")
        if not prof:
            return d
        start_ghz = float(prof[1])
        idle_us = float(prof[2])
        ramp_end_us = float(prof[4])
        slope_mhz_us = float(prof[7])
        n_adc = int(prof[9])
        fs_ksps = float(prof[10])
        d.start_freq_ghz = start_ghz
        d.adc_samples = n_adc
        slope = slope_mhz_us * 1e12  # Hz/s
        fs = fs_ksps * 1e3
        sampling_time = n_adc / fs
        d.bandwidth_mhz = slope * sampling_time / 1e6
        d.range_resolution_m = C / (2 * slope * sampling_time)
        d.max_range_m = 0.9 * fs * C / (2 * slope)  # 0.9: complex IF bandwidth margin
        if frame:
            c_start, c_end, loops = int(frame[0]), int(frame[1]), int(frame[2])
            tx_per_loop = c_end - c_start + 1
            d.chirps_per_frame = tx_per_loop * loops
            d.frame_period_ms = float(frame[4])
            d.frame_rate_hz = 1000.0 / d.frame_period_ms if d.frame_period_ms else 0.0
            lam = C / ((start_ghz + d.bandwidth_mhz / 2000.0) * 1e9)
            tc = (idle_us + ramp_end_us) * 1e-6
            d.max_velocity_mps = lam / (4 * tc * tx_per_loop)
            d.velocity_resolution_mps = lam / (2 * loops * tx_per_loop * tc)
        return d


def _fmt(v) -> str:
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)
