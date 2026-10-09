"""Session recorder.  Thread-safe: raw bytes arrive on the serial reader thread."""
from __future__ import annotations

import csv
import json
import struct
import threading
from datetime import datetime
from pathlib import Path

from app.models.garage_state import GarageState
from app.models.radar_frame import RadarFrame

from .session import PTS_MAGIC, RAW_MAGIC, SESSION_FILE, encode_frame, make_session_dir

CSV_FIELDS = [
    "timestamp", "frame", "stall", "state", "confidence", "points_inside", "fraction_inside",
    "width_m", "depth_m", "height_m", "persistence_s", "changed_pct", "threshold_pct", "motion_in", "motion_out",
    "track_id", "num_clusters", "num_vehicle_tracks",
]


class Recorder:
    def __init__(self):
        self.path: Path | None = None
        self._lock = threading.Lock()
        self._raw = self._pts = self._csv_file = self._ev_file = None
        self._csv = self._ev = None
        self.meta: dict = {}
        self.frames = 0
        self.raw_bytes = 0

    @property
    def active(self) -> bool:
        return self.path is not None

    def start(self, root: str | Path, name: str, meta: dict, config_text: str, notes: str = "",
              background: dict | None = None) -> Path:
        if self.active:
            self.stop()
        path = make_session_dir(root, name)
        self.meta = {"name": name, "created": datetime.now().isoformat(timespec="seconds"), **meta}
        (path / "radar_config.cfg").write_text(config_text or "")
        (path / "notes.txt").write_text(notes)
        if background:
            (path / "background.json").write_text(json.dumps(background))
        with self._lock:
            self._raw = open(path / "raw_uart.bin", "wb")
            self._raw.write(RAW_MAGIC)
            self._pts = open(path / "points.bin", "wb")
            self._pts.write(PTS_MAGIC)
            self._csv_file = open(path / "detections.csv", "w", newline="")
            self._csv = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS)
            self._csv.writeheader()
            self._ev_file = open(path / "events.csv", "w", newline="")
            self._ev = csv.writer(self._ev_file)
            self._ev.writerow(["timestamp", "stall", "name", "event", "detail"])
            self.frames = self.raw_bytes = 0
            self.path = path
        self._write_meta()
        return path

    def write_raw(self, data: bytes, ts: float) -> None:
        with self._lock:
            if self._raw:
                self._raw.write(struct.pack("<dI", ts, len(data)) + data)
                self.raw_bytes += len(data)

    def write_frame(self, frame: RadarFrame, state: GarageState | None, rows: list[dict] | None = None) -> None:
        with self._lock:
            if not self._pts:
                return
            self._pts.write(encode_frame(frame))
            if rows and self._csv:
                self._csv.writerows(rows)
            self.frames += 1

    def write_event(self, e) -> None:
        with self._lock:
            if self._ev:
                self._ev.writerow([f"{e.timestamp:.3f}", e.stall_index + 1, e.name, e.kind, e.detail])
                self._ev_file.flush()

    def update_meta(self, **kw) -> None:
        self.meta.update(kw)
        if self.active:
            self._write_meta()

    def set_notes(self, notes: str) -> None:
        if self.path:
            (self.path / "notes.txt").write_text(notes)
            self.update_meta(notes=notes)

    def stop(self) -> Path | None:
        path = self.path
        if not path:
            return None
        self.meta["ended"] = datetime.now().isoformat(timespec="seconds")
        self.meta["frames"] = self.frames
        self.meta["raw_bytes"] = self.raw_bytes
        self._write_meta()
        with self._lock:
            for f in (self._raw, self._pts, self._csv_file, self._ev_file):
                if f:
                    f.close()
            self._raw = self._pts = self._csv_file = self._csv = self._ev_file = self._ev = None
            self.path = None
        return path

    def _write_meta(self) -> None:
        if self.path:
            (self.path / SESSION_FILE).write_text(json.dumps(self.meta, indent=2, default=str))
