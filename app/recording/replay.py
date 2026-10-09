"""Loading recordings for replay and running the pipeline offline (A/B testing)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.detection.filters import BackgroundModel
from app.detection.heatmap_occupancy import load_occupancy
from app.detection.parameters import DetectionParams
from app.detection.pipeline import DetectionPipeline
from app.geometry.garage import Garage
from app.models.garage_state import StallState
from app.models.radar_frame import RadarFrame
from app.radar.config_parser import RadarConfig
from app.radar.packet_parser import PacketParser

from .session import load_session_meta, read_points_file, read_raw_file


@dataclass
class Recording:
    path: Path
    meta: dict
    frames: list[RadarFrame]
    config_text: str = ""
    notes: str = ""
    background: BackgroundModel | None = None

    @property
    def duration(self) -> float:
        return self.frames[-1].timestamp - self.frames[0].timestamp if len(self.frames) > 1 else 0.0

    @property
    def garage(self) -> Garage | None:
        g = self.meta.get("garage")
        return Garage.from_dict(g) if g else None

    @property
    def params(self) -> DetectionParams | None:
        p = self.meta.get("detection_params")
        return DetectionParams.from_dict(p) if p else None

    @property
    def expected(self) -> list[str]:
        return list(self.meta.get("expected_states", []))


def load_recording(path: str | Path, source: str = "auto") -> Recording:
    """source = "raw" (re-parse raw UART bytes with the current parser - includes the heatmap),
    "points" (parsed frames from points.bin - fast, no heatmap) or "auto" (raw when available)."""
    path = Path(path)
    meta = load_session_meta(path)
    if source == "auto":
        source = "raw" if (path / "raw_uart.bin").exists() else "points"
    if source == "raw" or not (path / "points.bin").exists():
        parser = PacketParser()
        frames = []
        for ts, chunk in read_raw_file(path / "raw_uart.bin"):
            frames.extend(parser.feed(chunk, ts))
    else:
        frames = read_points_file(path / "points.bin")
    cfg = (path / "radar_config.cfg").read_text() if (path / "radar_config.cfg").exists() else ""
    notes = (path / "notes.txt").read_text() if (path / "notes.txt").exists() else ""
    bg = None
    if (path / "background.json").exists():
        bg = BackgroundModel.from_dict(json.loads((path / "background.json").read_text()))
    return Recording(path, meta, frames, cfg, notes, bg)


# ---------------------------------------------------------------- A/B analysis
OCCUPIED_LIKE = {StallState.OCCUPIED, StallState.ENTERING, StallState.EXITING}


@dataclass
class RunMetrics:
    label: str
    frames: int = 0
    state_changes: list[int] = field(default_factory=list)
    occupied_fraction: list[float] = field(default_factory=list)
    average_confidence: float = 0.0
    false_detections: int | None = None
    missed_vehicles: int | None = None
    stall_accuracy: float | None = None

    def rows(self) -> list[tuple[str, str]]:
        def opt(v, f):
            return "n/a (set expected states)" if v is None else f(v)

        return [
            ("Frames processed", str(self.frames)),
            ("False detections", opt(self.false_detections, str)),
            ("Missed vehicles", opt(self.missed_vehicles, str)),
            ("Stall accuracy", opt(self.stall_accuracy, lambda v: f"{v * 100:.1f}%")),
            ("Average confidence", f"{self.average_confidence * 100:.1f}%"),
            ("State changes per stall", " / ".join(map(str, self.state_changes))),
            ("Time occupied per stall", " / ".join(f"{v * 100:.0f}%" for v in self.occupied_fraction)),
        ]


def run_offline(rec: Recording, garage: Garage, params: DetectionParams, label: str = "",
                background: BackgroundModel | None = None, warmup_s: float | None = None,
                config_dir: str | Path | None = None) -> RunMetrics:
    """Run the current pipeline over a recording.

    If the session has expected per-stall states ("OCCUPIED"/"EMPTY"), accuracy
    metrics are computed over frames after the warm-up period (default: the
    longer of the occupancy/empty persistence times).
    """
    pipe = DetectionPipeline(garage, params)
    if config_dir is not None and rec.config_text:
        pipe.set_occupancy(load_occupancy(garage, RadarConfig(rec.config_text), params, config_dir))
    if background is not None:
        pipe.background = background
    n = len(garage.stalls)
    m = RunMetrics(label=label, state_changes=[0] * n, occupied_fraction=[0.0] * n)
    expected = rec.expected + [""] * (n - len(rec.expected))
    warm = warmup_s if warmup_s is not None else max(params.occupancy_persistence_s, params.empty_persistence_s) + 1
    if warmup_s is None and pipe.occupancy is not None:  # the heatmap average must fill before it is ready
        frame_s = RadarConfig(rec.config_text).derived().frame_period_ms / 1000.0
        warm += params.occ_window_frames * frame_s
    prev = [None] * n
    conf_sum, conf_n, correct, judged = 0.0, 0, 0, 0
    false_det, ever_occupied = 0, [False] * n
    t0 = rec.frames[0].timestamp if rec.frames else 0.0
    for fr in rec.frames:
        res = pipe.process(fr)
        m.frames += 1
        scored = fr.timestamp - t0 >= warm
        for s in res.state.stalls:
            i = s.stall_index
            occ = s.state in OCCUPIED_LIKE
            if prev[i] is not None and s.state != prev[i]:
                m.state_changes[i] += 1
                if occ and prev[i] not in OCCUPIED_LIKE and expected[i] == "EMPTY":
                    false_det += 1
            prev[i] = s.state
            m.occupied_fraction[i] += occ
            if s.state not in (StallState.UNKNOWN, StallState.SENSOR_ERROR):
                conf_sum += s.confidence
                conf_n += 1
            if occ:
                ever_occupied[i] = True
            if scored and expected[i] in ("OCCUPIED", "EMPTY"):
                judged += 1
                correct += (expected[i] == "OCCUPIED") == occ
    if m.frames:
        m.occupied_fraction = [v / m.frames for v in m.occupied_fraction]
    m.average_confidence = conf_sum / conf_n if conf_n else 0.0
    if any(e in ("OCCUPIED", "EMPTY") for e in expected):
        m.false_detections = false_det
        m.missed_vehicles = sum(1 for i in range(n) if expected[i] == "OCCUPIED" and not ever_occupied[i])
        m.stall_accuracy = correct / judged if judged else None
    return m
