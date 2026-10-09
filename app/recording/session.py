"""Recording-session folder layout and binary point format.

    recordings/<YYYY-MM-DD_HHMMSS>_<name>/
        session.json       metadata, serial settings, firmware info, parameters,
                           garage geometry/zones, expected stall states, notes
        radar_config.cfg   configuration sent to the radar
        raw_uart.bin       every data-port chunk as received   (RAW1 records)
        points.bin         parsed frames                        (PTS1 records)
        detections.csv     per-frame, per-stall classification results
        notes.txt          user notes
        background.json    learned background map (if one existed)

raw_uart.bin  : b"GRRAW001" then repeated  <f8 timestamp><u4 nbytes><bytes>
points.bin    : b"GRPTS001" then repeated  <f8 timestamp><u4 frame><u4 npoints>
                npoints x POINT_DTYPE (6 x f4) <u4 ntracks> ntracks x <u4 id><7 f4>
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from app.models.radar_frame import RadarFrame
from app.models.radar_object import RadarTrack
from app.models.radar_point import POINT_DTYPE

RAW_MAGIC = b"GRRAW001"
PTS_MAGIC = b"GRPTS001"
_TRACK_FMT = "<I7f"
_TRACK_LEN = struct.calcsize(_TRACK_FMT)

SESSION_FILE = "session.json"


@dataclass
class SessionInfo:
    name: str
    created: str
    path: Path
    meta: dict = field(default_factory=dict)


def make_session_dir(root: str | Path, name: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()) or "session"
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = Path(root) / f"{stamp}_{slug}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def encode_frame(frame: RadarFrame) -> bytes:
    pts = np.ascontiguousarray(frame.points, dtype=POINT_DTYPE)
    out = [struct.pack("<dII", frame.timestamp, frame.frame_number, len(pts)), pts.tobytes()]
    out.append(struct.pack("<I", len(frame.tracks)))
    for t in frame.tracks:
        out.append(struct.pack(_TRACK_FMT, t.track_id, t.x, t.y, t.z, t.vx, t.vy, t.vz, t.confidence))
    return b"".join(out)


def read_points_file(path: str | Path) -> list[RadarFrame]:
    data = Path(path).read_bytes()
    if not data.startswith(PTS_MAGIC):
        raise ValueError(f"{path} is not a points file")
    frames, off = [], len(PTS_MAGIC)
    while off + 16 <= len(data):
        ts, fn, n = struct.unpack_from("<dII", data, off)
        off += 16
        nbytes = n * POINT_DTYPE.itemsize
        if off + nbytes + 4 > len(data):
            break  # truncated final record (e.g. recording interrupted)
        pts = np.frombuffer(data, dtype=POINT_DTYPE, count=n, offset=off).copy()
        off += nbytes
        (nt,) = struct.unpack_from("<I", data, off)
        off += 4
        tracks = []
        for _ in range(nt):
            if off + _TRACK_LEN > len(data):
                break
            v = struct.unpack_from(_TRACK_FMT, data, off)
            off += _TRACK_LEN
            tracks.append(RadarTrack(v[0], *v[1:]))
        frames.append(RadarFrame(frame_number=fn, timestamp=ts, points=pts, tracks=tracks))
    return frames


def read_raw_file(path: str | Path) -> list[tuple[float, bytes]]:
    data = Path(path).read_bytes()
    if not data.startswith(RAW_MAGIC):
        raise ValueError(f"{path} is not a raw UART file")
    chunks, off = [], len(RAW_MAGIC)
    while off + 12 <= len(data):
        ts, n = struct.unpack_from("<dI", data, off)
        off += 12
        if off + n > len(data):
            break
        chunks.append((ts, data[off : off + n]))
        off += n
    return chunks


def list_sessions(root: str | Path) -> list[SessionInfo]:
    out = []
    root = Path(root)
    if not root.exists():
        return out
    for d in sorted(root.iterdir(), reverse=True):
        f = d / SESSION_FILE
        if f.exists():
            try:
                meta = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            out.append(SessionInfo(meta.get("name", d.name), meta.get("created", ""), d, meta))
    return out


def load_session_meta(path: str | Path) -> dict:
    return json.loads((Path(path) / SESSION_FILE).read_text())
