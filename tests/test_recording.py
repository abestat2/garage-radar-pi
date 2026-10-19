"""Unit tests: recording session files, the recorder, replay loading and offline A/B runs."""
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from app.detection.filters import BackgroundModel
from app.detection.parameters import DetectionParams
from app.geometry.garage import Garage
from app.geometry.radar_pose import RadarPose
from app.models.garage_state import StallEvent
from app.models.radar_frame import RadarFrame
from app.models.radar_object import RadarTrack
from app.models.radar_point import empty_points
from app.radar.packet_parser import build_packet
from app.recording import session as ses
from app.recording.recorder import CSV_FIELDS, Recorder
from app.recording.replay import Recording, RunMetrics, load_recording, run_offline

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "stall3_out_and_in"


def make_frame(n=4, number=1, ts=10.0, tracks=()):
    f = RadarFrame(frame_number=number, timestamp=ts)
    f.points = empty_points(n)
    f.points["x"], f.points["y"], f.points["z"] = np.arange(n), np.arange(n) + 1.0, 0.5
    f.points["doppler"], f.points["snr"], f.points["noise"] = -0.3, 15.0, 4.0
    f.tracks = list(tracks)
    return f


# ---------------------------------------------------------------- session files
def test_make_session_dir_slug_and_collision(tmp_path, monkeypatch):
    import datetime as dt

    class Frozen(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 2, 3, 4, 5)

    monkeypatch.setattr(ses, "datetime", Frozen)
    p = ses.make_session_dir(tmp_path, "  my test/run!  ")
    assert p.is_dir() and p.name == "2026-01-02_030405_my_test_run_"
    assert ses.make_session_dir(tmp_path / "new", "").name.endswith("_session")
    with pytest.raises(FileExistsError):
        ses.make_session_dir(tmp_path, "my test/run!")


def test_encode_and_read_points_round_trip(tmp_path):
    frames = [make_frame(3, 1, 1.5, [RadarTrack(7, 1, 2, 3, 0.1, 0.2, 0.3, 0.8)]), make_frame(0, 2, 2.5)]
    path = tmp_path / "points.bin"
    path.write_bytes(ses.PTS_MAGIC + b"".join(ses.encode_frame(f) for f in frames))
    got = ses.read_points_file(path)
    assert [f.frame_number for f in got] == [1, 2] and got[0].timestamp == 1.5
    assert got[0].points["x"].tolist() == [0.0, 1.0, 2.0] and got[1].num_points == 0
    t = got[0].tracks[0]
    assert t.track_id == 7 and t.confidence == pytest.approx(0.8) and t.vz == pytest.approx(0.3)


def test_read_points_file_validation_and_truncation(tmp_path):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"NOTAPOINTS")
    with pytest.raises(ValueError):
        ses.read_points_file(bad)
    blob = ses.PTS_MAGIC + ses.encode_frame(make_frame(3, 1)) + ses.encode_frame(make_frame(3, 2))
    cut = tmp_path / "cut.bin"
    cut.write_bytes(blob[:-10])  # interrupted mid-record
    assert [f.frame_number for f in ses.read_points_file(cut)] == [1]


@pytest.mark.xfail(strict=True, reason="read_points_file re-parses the leftover bytes of a half-written track "
                                      "record as a new (phantom) frame; fix upstream in mmWave_car_detect_test")
def test_read_points_file_truncated_track_record_adds_no_phantom_frame(tmp_path):
    one = ses.encode_frame(make_frame(1, 3, tracks=[RadarTrack(1, 0, 0, 0)]))
    trunc = tmp_path / "trk.bin"
    trunc.write_bytes(ses.PTS_MAGIC + one[:-4])
    assert [f.frame_number for f in ses.read_points_file(trunc)] == [3]


def test_read_raw_file(tmp_path):
    path = tmp_path / "raw.bin"
    path.write_bytes(ses.RAW_MAGIC + struct.pack("<dI", 1.0, 3) + b"abc" + struct.pack("<dI", 2.0, 100) + b"xy")
    assert ses.read_raw_file(path) == [(1.0, b"abc")]  # second chunk is truncated
    wrong = tmp_path / "wrong.bin"
    wrong.write_bytes(b"nope")
    with pytest.raises(ValueError):
        ses.read_raw_file(wrong)


def test_list_sessions_and_meta(tmp_path):
    assert ses.list_sessions(tmp_path / "missing") == []
    for name, meta in (("a", {"name": "Alpha", "created": "t1"}), ("b", {})):
        d = tmp_path / name
        d.mkdir()
        (d / ses.SESSION_FILE).write_text(json.dumps(meta))
    broken = tmp_path / "c"
    broken.mkdir()
    (broken / ses.SESSION_FILE).write_text("{not json")
    (tmp_path / "d").mkdir()  # folder without session.json
    (tmp_path / "file.txt").write_text("x")
    found = ses.list_sessions(tmp_path)
    assert [s.name for s in found] == ["b", "Alpha"]  # newest (reverse-sorted) first; name falls back to folder
    assert ses.load_session_meta(tmp_path / "a")["created"] == "t1"


# ---------------------------------------------------------------- recorder
def test_recorder_full_session(tmp_path):
    r = Recorder()
    assert not r.active and r.stop() is None
    r.set_notes("ignored when inactive")
    r.update_meta(x=1)
    path = r.start(tmp_path, "garage test", {"garage": {"w": 1}}, "sensorStop\n", notes="hello",
                   background={"voxel": 0.2})
    assert r.active and r.path == path
    r.write_raw(b"\x01\x02\x03", 5.0)
    row = {k: "" for k in CSV_FIELDS}
    row.update(timestamp="5.000", stall=1, state="EMPTY")
    r.write_frame(make_frame(2, 1, 5.0), None, [row])
    r.write_frame(make_frame(2, 2, 5.1), None)  # no csv rows
    r.write_event(StallEvent(5.2, 0, "Stall 1", "ENTERED", "detail"))
    r.update_meta(expected_states=["OCCUPIED"])
    r.set_notes("new notes")
    assert r.stop() == path and not r.active and r.frames == 2

    meta = json.loads((path / "session.json").read_text())
    assert meta["name"] == "garage test" and meta["frames"] == 2 and meta["raw_bytes"] == 3
    assert meta["expected_states"] == ["OCCUPIED"] and meta["notes"] == "new notes" and "ended" in meta
    assert (path / "notes.txt").read_text() == "new notes"
    assert (path / "radar_config.cfg").read_text() == "sensorStop\n"
    assert json.loads((path / "background.json").read_text()) == {"voxel": 0.2}
    assert ses.read_raw_file(path / "raw_uart.bin") == [(5.0, b"\x01\x02\x03")]
    assert [f.frame_number for f in ses.read_points_file(path / "points.bin")] == [1, 2]
    csv_lines = (path / "detections.csv").read_text().splitlines()
    assert csv_lines[0].split(",") == CSV_FIELDS and len(csv_lines) == 2
    assert "ENTERED" in (path / "events.csv").read_text()
    r.write_raw(b"x", 1.0)  # after stop: ignored, no error
    r.write_frame(make_frame(), None)
    r.write_event(StallEvent(1, 0, "S", "EXITED"))
    assert r.frames == 2


def test_recorder_start_while_active_stops_previous(tmp_path, monkeypatch):
    import app.recording.recorder as recorder_mod

    counter = iter(range(10))

    def unique_dir(root, name):
        d = Path(root) / f"{next(counter)}_{name}"
        d.mkdir(parents=True)
        return d

    monkeypatch.setattr(recorder_mod, "make_session_dir", unique_dir)
    r = Recorder()
    first = r.start(tmp_path, "one", {}, "")
    second = r.start(tmp_path, "two", {}, "")
    assert first != second and r.path == second
    assert "ended" in json.loads((first / "session.json").read_text())
    r.stop()


# ---------------------------------------------------------------- replay loading
def test_load_recording_from_real_fixture():
    rec = load_recording(FIXTURE)
    assert len(rec.frames) > 10 and rec.duration > 1.0 and rec.config_text.startswith("%")
    assert rec.meta["name"] == "session" and rec.background is None and rec.notes == ""
    assert isinstance(rec.expected, list)
    assert rec.garage is None or isinstance(rec.garage, Garage)
    _ = rec.params  # property does not raise


def test_load_recording_points_and_background(tmp_path):
    root = tmp_path / "rec"
    root.mkdir()
    (root / "session.json").write_text(json.dumps({"name": "x", "expected_states": ["OCCUPIED", "EMPTY"],
                                                    "garage": Garage(width=5, depth=5).to_dict(),
                                                    "detection_params": {"min_range": 0.9}}))
    (root / "points.bin").write_bytes(ses.PTS_MAGIC + ses.encode_frame(make_frame(2, 1, 1.0))
                                      + ses.encode_frame(make_frame(2, 2, 3.0)))
    bg = BackgroundModel(0.3)
    bg.frames = 2
    (root / "background.json").write_text(json.dumps(bg.to_dict()))
    (root / "notes.txt").write_text("note")
    (root / "radar_config.cfg").write_text("cfg")
    rec = load_recording(root)  # auto: no raw_uart.bin -> points
    assert len(rec.frames) == 2 and rec.duration == 2.0 and rec.notes == "note" and rec.config_text == "cfg"
    assert rec.background.voxel == 0.3 and rec.expected == ["OCCUPIED", "EMPTY"]
    assert rec.garage.width == 5 and rec.params.min_range == 0.9
    assert Recording(root, {}, []).duration == 0.0
    # raw requested but only points exist falls back to ... raw file missing -> explicit error is not expected
    (root / "raw_uart.bin").write_bytes(ses.RAW_MAGIC + struct.pack("<dI", 1.0, 0))
    assert load_recording(root, source="points").frames[0].frame_number == 1
    assert load_recording(root, source="raw").frames == []


def test_load_recording_raw_parses_packets(tmp_path):
    root = tmp_path / "rec"
    root.mkdir()
    (root / "session.json").write_text("{}")
    pkt = build_packet(5, make_frame(3).points)
    (root / "raw_uart.bin").write_bytes(ses.RAW_MAGIC + struct.pack("<dI", 2.0, len(pkt)) + pkt)
    rec = load_recording(root)
    assert [f.frame_number for f in rec.frames] == [5] and rec.frames[0].timestamp == 2.0


# ---------------------------------------------------------------- offline run
POSE = RadarPose(x=3, y=0.3, z=2.0, azimuth_deg=0, elevation_deg=-20)


def synthetic_recording(expected, seconds=30, with_car=True):
    frames = []
    for i in range(seconds * 4):
        f = RadarFrame(frame_number=i, timestamp=100.0 + i * 0.25)
        n = 40 if with_car else 0
        f.points = empty_points(n)
        rng = np.random.default_rng(i)
        f.points["x"] = rng.uniform(-0.8, 0.8, n)
        f.points["y"] = rng.uniform(2.0, 4.0, n)
        f.points["z"] = rng.uniform(-0.5, 0.2, n)
        f.points["snr"], f.points["noise"] = 20.0, 5.0
        frames.append(f)
    return Recording(Path("."), {"expected_states": expected}, frames)


def test_run_offline_metrics_without_expectations():
    rec = synthetic_recording([], seconds=5)
    m = run_offline(rec, Garage(width=6, depth=6, radar=POSE), DetectionParams(), label="A")
    assert m.label == "A" and m.frames == 20 and len(m.state_changes) == 3
    assert m.false_detections is None and m.missed_vehicles is None and m.stall_accuracy is None
    assert 0.0 <= m.average_confidence <= 1.0
    assert all("n/a" in v for k, v in m.rows() if k in ("False detections", "Missed vehicles", "Stall accuracy"))


def test_run_offline_scores_expected_states():
    rec = synthetic_recording(["OCCUPIED", "EMPTY", "EMPTY"], seconds=30)
    m = run_offline(rec, Garage(width=6, depth=6, radar=POSE), DetectionParams(), warmup_s=1.0,
                    background=BackgroundModel(0.2))
    assert m.false_detections is not None and m.missed_vehicles in (0, 1) and 0.0 <= m.stall_accuracy <= 1.0
    assert dict(m.rows())["Stall accuracy"].endswith("%")


def test_run_offline_with_no_frames_and_row_formatting():
    m = run_offline(Recording(Path("."), {}, []), Garage(width=6, depth=6, radar=POSE), DetectionParams())
    assert m.frames == 0 and m.average_confidence == 0.0
    rows = dict(RunMetrics("x", 10, [1, 2, 3], [0.5, 0.25, 0.0], 0.75, 1, 0, 0.9).rows())
    assert rows["State changes per stall"] == "1 / 2 / 3" and rows["Time occupied per stall"] == "50% / 25% / 0%"
    assert rows["Average confidence"] == "75.0%" and rows["False detections"] == "1"
