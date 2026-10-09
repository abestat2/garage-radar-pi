"""Parked-car occupancy from the zero-Doppler range-azimuth heatmap (TLV 4).

Why: the on-chip CFAR masks extended static targets, so a parked car yields only a few points
(sometimes none), but its echo is plainly visible in the raw zero-Doppler heatmap (20-27 dB above
noise).  This module compares the live heatmap with an EMPTY-GARAGE REFERENCE and measures, per
stall, the fraction of the stall's image cells whose echo grew by more than `change_db`.

    TLV 4 (complex, range bins x azimuth antennas)
      -> angle FFT -> |magnitude| (range x angle image, radar frame)
      -> short moving average (noise)
      -> dB change vs the empty reference
      -> per-stall "changed fraction" over the stall's CAR ZONE only (cells mapped onto the floor
         plan assuming reflectors at `assumed_height` - the heatmap has no elevation).  A cell is used
         only if its side-to-side beam blur stays inside the car zone and clear of IGNORE ZONES
         (clutter areas); with 8 azimuth antennas the beam is ~13 deg wide (~1-2 m at 4-8 m).
      -> the reference slowly re-learns (adapts) outside the car zones, and inside a car zone only
         while that stall has been EMPTY for a long time, so clutter that builds up or is cleared
         out becomes 'normal' without ever learning a parked car into the background
      -> compared with per-stall thresholds calibrated from labelled recordings

The reference depends on the radar CHIRP CONFIGURATION and the physical radar mount, not on the
geometry entered in the app; it is stored with a fingerprint of the chirp-defining cfg lines.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from app.geometry.coordinate_transform import radar_to_garage
from app.geometry.garage import Garage
from app.radar.config_parser import RadarConfig

ANGLE_FFT = 64
# Angle processing version - part of the reference fingerprint, so a reference built with a different
# method is rejected.  Hann taper across the azimuth antennas: sidelobes ~-31 dB instead of -13 dB.
# Measured 2026-10-08: cut neighbour-stall cross-talk from ~10 points to ~2-3 when the centre car leaves.
IMAGE_METHOD = "hann-v1"
CHIRP_COMMANDS = ("channelCfg", "adcCfg", "profileCfg", "chirpCfg", "frameCfg", "guiMonitor", "bpmCfg")
FT = 0.3048


def config_fingerprint(cfg: RadarConfig) -> str:
    """Hash of the cfg lines that change the heatmap's meaning (chirp, antennas, frame)."""
    lines = [c for c in cfg.commands() if c.split()[0] in CHIRP_COMMANDS] + [IMAGE_METHOD]
    return hashlib.sha1("\n".join(lines).encode()).hexdigest()[:12]


def heatmap_enabled(cfg: RadarConfig) -> bool:
    g = cfg.find("guiMonitor")
    return bool(g and len(g) > 4 and g[4] == "1")


@dataclass
class HeatmapLayout:
    """How a TLV 4 payload maps to range bins / angles for a given radar configuration."""

    num_range_bins: int
    bin_m: float  # range per bin, metres
    num_azimuth_antennas: int = 8

    @classmethod
    def from_config(cls, cfg: RadarConfig) -> "HeatmapLayout":
        d = cfg.derived()
        nbins = 1 << int(np.ceil(np.log2(max(d.adc_samples, 1))))
        nva = max(d.num_rx * min(max(d.num_tx, 1), 2), 1)  # 3 TX: 2 azimuth TX x 4 RX (TX2 is elevation)
        return cls(nbins, d.range_resolution_m * d.adc_samples / nbins, nva)

    @property
    def beam_half_width_sin(self) -> float:
        """Half of the 3 dB beamwidth in sin(azimuth) units (lambda/2 array, Hann taper: 1.44 bins)."""
        return 1.44 / self.num_azimuth_antennas

    def image(self, az_heatmap: np.ndarray) -> np.ndarray | None:
        """Range x angle magnitude image (angle index 0..63 = sin(az) from -1 to +1)."""
        if az_heatmap is None or len(az_heatmap) % self.num_range_bins:
            return None
        z = az_heatmap.reshape(self.num_range_bins, -1)
        z = z * _taper(z.shape[1])[None, :]
        return np.abs(np.fft.fftshift(np.fft.fft(z, n=ANGLE_FFT, axis=1), axes=1)).astype(np.float32)


def _taper(n: int) -> np.ndarray:
    w = np.hanning(n + 2)[1:-1]  # Hann without its zero end-points
    return (w / w.mean()).astype(np.float32)


def cell_floor_positions(layout: HeatmapLayout, garage: Garage, height: float,
                         sin_offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Garage XY of every (range, angle) cell assuming reflectors at `height`; and a validity mask.
    `sin_offset` shifts the angle (used to trace the edges of each cell's beam blur)."""
    r = np.arange(layout.num_range_bins) * layout.bin_m
    sin_az = np.arange(-ANGLE_FFT // 2, ANGLE_FFT // 2) / (ANGLE_FFT / 2) + sin_offset
    R, S = np.meshgrid(r, sin_az, indexing="ij")
    pose = garage.radar
    dz = height - pose.z
    # elevation relative to the boresight for a point at that height and slant range
    el_abs = np.arcsin(np.clip(dz / np.maximum(R, 1e-3), -1, 1))
    el = el_abs - np.radians(pose.elevation_deg)
    th = np.arcsin(np.clip(S, -1, 1))
    pr = np.stack([R * np.cos(el) * np.sin(th), R * np.cos(el) * np.cos(th), R * np.sin(el)], -1)
    g = radar_to_garage(pr.reshape(-1, 3), pose).reshape(pr.shape)
    valid = (np.abs(S) < 0.97) & (R > 1.0) & (R > abs(dz)) & (np.abs(np.degrees(el)) <= pose.fov_elevation_deg / 2)
    return g[..., :2], valid


@dataclass
class StallZones:
    stall: list[np.ndarray]  # bool masks (range x angle): car-zone cells used as parked-car evidence
    entry: list[np.ndarray]  # bool masks for the approach zone in front of each stall's door
    touch: list[np.ndarray] = field(default_factory=list)  # cells whose beam blur touches the car zone
    ignored: np.ndarray | None = None  # cells whose blur touches an ignore zone

    @classmethod
    def build(cls, layout: HeatmapLayout, garage: Garage, height: float, approach_m: float,
              edge_margin: float = 0.25) -> "StallZones":
        xy, valid = cell_floor_positions(layout, garage, height)
        flat = xy.reshape(-1, 2)
        # sample the cell's beam blur across +/- edge_margin of the 3 dB half-width (and the full width
        # for 'touches'), so a cell is evidence only if its blur lies inside the car zone
        hw = layout.beam_half_width_sin
        core_offsets = [o * hw * edge_margin for o in (-1.0, -0.5, 0.5, 1.0)] if edge_margin > 0 else []
        touch_offsets = [o * hw for o in (-1.0, -0.5, 0.0, 0.5, 1.0)]
        shifted = {o: cell_floor_positions(layout, garage, height, o)[0].reshape(-1, 2)
                   for o in set(core_offsets + touch_offsets)}
        shape = valid.shape

        def inside(test, offsets, combine):
            out = test(flat).reshape(shape)
            for o in offsets:
                out = combine(out, test(shifted[o]).reshape(shape))
            return out

        ignored = np.zeros(shape, dtype=bool)
        for z in garage.ignore_zones:
            ignored |= inside(z.contains, touch_offsets, np.logical_or)
        stalls, entries, touch = [], [], []
        for s in garage.stalls:
            core = inside(s.car_zone_contains, core_offsets, np.logical_and) & valid & ~ignored
            if not core.any():  # zone narrower than the beam here: fall back to cells centred in it
                core = s.car_zone_contains(flat).reshape(shape) & valid & ~ignored
            stalls.append(core)
            touch.append(inside(s.car_zone_contains, touch_offsets, np.logical_or) & valid)
            v = s.vertices()
            x0, x1, front = v[:, 0].min(), v[:, 0].max(), v[:, 1].min()
            door_y = _door_y_for(garage, x0, x1, front)
            m = (xy[..., 0] >= x0) & (xy[..., 0] <= x1) & (xy[..., 1] >= door_y - approach_m) & (xy[..., 1] < front + 1.0)
            entries.append(m & valid)
        return cls(stalls, entries, touch, ignored)


def _door_y_for(garage: Garage, x0: float, x1: float, default: float) -> float:
    best = None
    for d in garage.doors:
        overlap = min(x1, d.x_left + d.width) - max(x0, d.x_left)
        if overlap > 0 and (best is None or overlap > best[0]):
            best = (overlap, d.y)
    return best[1] if best else default


# ---------------------------------------------------------------- reference + calibration files
@dataclass
class EmptyReference:
    image: np.ndarray  # range x angle mean magnitude of the EMPTY garage
    fingerprint: str
    frames: int
    sources: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, image=self.image, fingerprint=self.fingerprint, frames=self.frames,
                            sources=json.dumps(self.sources))

    @classmethod
    def load(cls, path: str | Path) -> "EmptyReference":
        z = np.load(path, allow_pickle=False)
        return cls(z["image"], str(z["fingerprint"]), int(z["frames"]), json.loads(str(z["sources"])))

    @classmethod
    def combine(cls, images: list[np.ndarray], fingerprint: str, frames: int, sources: list[str]) -> "EmptyReference":
        """Element-wise maximum of several empty scenes (e.g. doors open AND doors closed), so neither
        state's door echoes count as 'change'."""
        return cls(np.maximum.reduce(images), fingerprint, frames, sources)


@dataclass
class StallCalibration:
    threshold: float  # changed fraction separating empty from occupied
    empty_level: float  # typical changed fraction when empty
    occupied_level: float  # typical changed fraction when occupied


@dataclass
class NeighbourModel:
    """How much an EMPTY stall's reading rises when each neighbour is occupied (beam leakage), plus the
    occupied levels seen with each neighbour configuration.  Fitted by tools/build_occupancy_model.py.

    Thresholds are then conditional: stall i's threshold = midpoint between its predicted EMPTY level
    (base + spread + sum of leaks from occupied neighbours) and its lowest OCCUPIED level for that
    neighbour configuration."""

    base: list[float]  # empty reading with every neighbour empty
    spread: list[float]  # 99th percentile - median of empty readings
    leak: list[list[float]]  # leak[i][j]: rise of stall i's empty reading when stall j is occupied
    leak_assumed: list[list[bool]]  # True where leak[i][j] was not measured (taken from the mirror stall)
    occ_low: dict  # "i:key" -> 1st percentile occupied reading; key = other stalls' states, e.g. "OE"
    occ_low_any: list[float]  # per stall, lowest occupied level over all configurations
    occ_med: list[float]

    def key(self, i: int, occupied: list[bool]) -> str:
        return "".join("O" if occupied[j] else "E" for j in range(len(occupied)) if j != i)

    def calibration_for(self, i: int, occupied: list[bool]) -> StallCalibration:
        empty = self.base[i] + sum(self.leak[i][j] for j in range(len(occupied)) if j != i and occupied[j])
        empty_hi = empty + self.spread[i]
        occ_lo = self.occ_low.get(f"{i}:{self.key(i, occupied)}", self.occ_low_any[i])
        thr = max((empty_hi + occ_lo) / 2, empty_hi + 0.01)
        return StallCalibration(float(thr), float(empty), float(max(self.occ_med[i], thr + 0.01)))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OccupancyCalibration:
    fingerprint: str
    stalls: list[StallCalibration]
    notes: str = ""
    neighbours: NeighbourModel | None = None

    def save(self, path: str | Path) -> None:
        d = {"fingerprint": self.fingerprint, "notes": self.notes, "stalls": [asdict(s) for s in self.stalls],
             "neighbours": self.neighbours.to_dict() if self.neighbours else None}
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "OccupancyCalibration":
        d = json.loads(Path(path).read_text())
        nb = NeighbourModel(**d["neighbours"]) if d.get("neighbours") else None
        return cls(d["fingerprint"], [StallCalibration(**s) for s in d["stalls"]], d.get("notes", ""), nb)


DEFAULT_STALL_CAL = StallCalibration(threshold=0.15, empty_level=0.05, occupied_level=0.35)


# ---------------------------------------------------------------- runtime estimator
@dataclass
class OccupancyReading:
    changed: list[float]  # per-stall changed fraction (0..1)
    entry_changed: list[float]  # per-stall approach-zone changed fraction
    thresholds: list[float]
    ready: bool
    reason: str = ""
    health: list[str] = field(default_factory=list)  # per-stall warning ('' = fine)
    cells: list[int] = field(default_factory=list)  # evidence cells per stall


class HeatmapOccupancy:
    def __init__(self, garage: Garage, cfg: RadarConfig, reference: EmptyReference | None,
                 calibration: OccupancyCalibration | None, change_db: float = 6.0, window: int = 8,
                 assumed_height: float = 0.8, approach_m: float = 3.5, edge_margin: float = 0.25,
                 adapt_tau_s: float = 300.0, adapt_empty_hold_s: float = 120.0):
        self.garage = garage
        self.cfg = cfg
        self.layout = HeatmapLayout.from_config(cfg)
        self.fingerprint = config_fingerprint(cfg)
        self.reference = reference
        self.calibration = calibration
        self.change_db = change_db
        self._images: deque[np.ndarray] = deque(maxlen=max(1, window))
        self.zones = StallZones.build(self.layout, garage, assumed_height, approach_m, edge_margin)
        self.last: OccupancyReading | None = None
        self.adapt_tau_s = adapt_tau_s
        self.adapt_empty_hold_s = adapt_empty_hold_s
        self.live: np.ndarray | None = None  # adapting copy of the reference
        self._last_t: float | None = None
        self._empty_level: list[float | None] = [None] * len(garage.stalls)
        self.adapted_s = 0.0  # seconds of adaptation applied since the reference was loaded
        background = np.ones(self.layout.num_range_bins * ANGLE_FFT, dtype=bool).reshape(
            self.layout.num_range_bins, ANGLE_FFT)
        for t in self.zones.touch:
            background &= ~t
        self._background = background  # cells unaffected by any car zone: always allowed to adapt

    # ------------------------------------------------------------ status
    def problem(self) -> str:
        if not heatmap_enabled(self.cfg):
            return "radar config does not send the range-azimuth heatmap (guiMonitor 5th value = 1)"
        if self.reference is None:
            return "no empty-garage reference - capture one with the garage empty"
        if self.reference.fingerprint != self.fingerprint:
            return "empty-garage reference was captured with a different chirp configuration"
        if self.reference.image.shape[0] != self.layout.num_range_bins:
            return "empty-garage reference size does not match the radar configuration"
        return ""

    def thresholds(self, changed: list[float] | None = None) -> list[StallCalibration]:
        """Per-stall thresholds.  With a neighbour model they depend on which OTHER stalls look occupied:
        start from the unconditional thresholds, then re-decide each stall with thresholds conditioned on
        its neighbours (a few rounds - the centre stall's reading is unambiguous and anchors the rest)."""
        n = len(self.garage.stalls)
        cal = self.calibration
        if not (cal and cal.fingerprint == self.fingerprint and len(cal.stalls) == n):
            return [DEFAULT_STALL_CAL] * n
        if changed is None and self.last is not None and self.last.ready:
            changed = self.last.changed
        nb = cal.neighbours
        if nb is None or changed is None or len(nb.base) != n:
            return cal.stalls
        occ = [changed[i] >= cal.stalls[i].threshold for i in range(n)]
        out = cal.stalls
        for _ in range(3):
            out = [nb.calibration_for(i, occ) for i in range(n)]
            new = [changed[i] >= out[i].threshold for i in range(n)]
            if new == occ:
                break
            occ = new
        return out

    def reset(self) -> None:
        self._images.clear()
        self.last = None
        self.live = None
        self._last_t = None
        self._empty_level = [None] * len(self.garage.stalls)
        self.adapted_s = 0.0

    # ------------------------------------------------------------ processing
    def update(self, az_heatmap: np.ndarray | None, stalls=None, now: float | None = None) -> OccupancyReading | None:
        """Feed one frame's TLV 4 (or None).  `stalls` = the previous StallResults (for adaptation and
        health), `now` = frame time.  Returns the latest reading."""
        img = self.layout.image(az_heatmap) if az_heatmap is not None else None
        if img is not None:
            self._images.append(img)
        why = self.problem()
        cal = self.thresholds()
        if why or len(self._images) < self._images.maxlen:  # not ready until the average window is full
            self.last = OccupancyReading([0.0] * len(cal), [0.0] * len(cal), [c.threshold for c in cal], False,
                                         why or f"warming up ({len(self._images)}/{self._images.maxlen} heatmaps)")
            return self.last
        if img is None and self.last is not None and self.last.ready:
            return self.last  # damaged frame without a heatmap: keep the previous reading
        mean = np.mean(self._images, axis=0)
        if self.live is None:
            self.live = self.reference.image.astype(np.float32).copy()
        diff = 20 * np.log10((mean + 1.0) / (self.live + 1.0))
        hot = diff > self.change_db
        changed = [float(hot[m].mean()) if m.any() else 0.0 for m in self.zones.stall]
        entry = [float(hot[m].mean()) if m.any() else 0.0 for m in self.zones.entry]
        cal = self.thresholds(changed)
        health = self._health(changed, cal, stalls, now)
        self._adapt(mean, stalls, now)
        self.last = OccupancyReading(changed, entry, [c.threshold for c in cal], True, health=health,
                                     cells=[int(m.sum()) for m in self.zones.stall])
        return self.last

    def _adapt(self, mean: np.ndarray, stalls, now: float | None) -> None:
        """Slowly blend the current scene into the reference: background cells always, a stall's car-zone
        cells only after it has been EMPTY for adapt_empty_hold_s.  Occupied stalls are frozen."""
        if self.adapt_tau_s <= 0 or now is None:
            return
        dt = 0.0 if self._last_t is None else min(max(now - self._last_t, 0.0), 5.0)
        self._last_t = now
        if dt <= 0:
            return
        allowed = self._background.copy()
        if stalls:
            from app.models.garage_state import StallState  # local import: avoid a cycle
            for i, s in enumerate(stalls):
                if i < len(self.zones.touch) and s.state == StallState.EMPTY and s.persistence >= self.adapt_empty_hold_s:
                    allowed |= self.zones.touch[i]
            for i, s in enumerate(stalls):  # never adapt cells touching an occupied-like stall
                if i < len(self.zones.touch) and s.state != StallState.EMPTY:
                    allowed &= ~self.zones.touch[i]
        a = dt / self.adapt_tau_s
        self.live[allowed] += a * (mean[allowed] - self.live[allowed])
        self.adapted_s += dt

    def _health(self, changed, cal, stalls, now) -> list[str]:
        """Warn when an EMPTY stall's reading creeps toward its threshold (reference getting stale)."""
        out = [""] * len(changed)
        if not stalls:
            return out
        from app.models.garage_state import StallState
        for i, s in enumerate(stalls[: len(changed)]):
            if s.state != StallState.EMPTY:
                continue
            prev = self._empty_level[i]
            lvl = changed[i] if prev is None else 0.98 * prev + 0.02 * changed[i]
            self._empty_level[i] = lvl
            thr = cal[i].threshold
            if lvl > 0.6 * thr:
                out[i] = (f"empty reading {lvl * 100:.0f}% is close to the {thr * 100:.0f}% threshold - "
                          "recapture the empty-garage reference or check the clutter zones")
        return out


def mean_image(frames, layout: HeatmapLayout) -> tuple[np.ndarray | None, int]:
    imgs = [layout.image(f.az_heatmap) for f in frames if f.az_heatmap is not None]
    imgs = [i for i in imgs if i is not None]
    return (np.mean(imgs, axis=0), len(imgs)) if imgs else (None, 0)


def load_occupancy(garage: Garage, cfg: RadarConfig, params, config_dir: str | Path) -> HeatmapOccupancy:
    """Build the estimator from configs/empty_reference.npz + configs/occupancy_calibration.json (if present)."""
    config_dir = Path(config_dir)
    ref = cal = None
    if (config_dir / "empty_reference.npz").exists():
        ref = EmptyReference.load(config_dir / "empty_reference.npz")
    if (config_dir / "occupancy_calibration.json").exists():
        cal = OccupancyCalibration.load(config_dir / "occupancy_calibration.json")
    return HeatmapOccupancy(garage, cfg, ref, cal, params.occ_change_db, params.occ_window_frames,
                            params.occ_assumed_height, params.entry_approach_m, params.occ_edge_margin,
                            params.occ_adapt_tau_s, params.occ_adapt_empty_hold_s)
