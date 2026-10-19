"""Unit tests: GarageService internals driven directly (no threads, hardware or network radar)."""
import json
import queue
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import garage_pi.service as service_mod
from app.detection.heatmap_occupancy import EmptyReference, StallCalibration, OccupancyCalibration
from app.models.garage_state import StallEvent, StallState
from app.recording.session import read_raw_file
from garage_pi.config import init_data_dir
from garage_pi.service import CAPTURED_REF, LABELS, LIVE_REF, GarageService

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "stall3_out_and_in"


class FakeLink:
    kind = "fake"

    def __init__(self):
        self.started = self.closed = False
        self.bytes_received = 0
        self.on_bytes = self.on_error = None
        self.finished = None

    def start(self, on_bytes, on_error):
        self.started, self.on_bytes, self.on_error = True, on_bytes, on_error

    def close(self):
        self.closed = True

    def describe(self):
        return "fake radar"


@pytest.fixture
def cfg(tmp_path):
    c = init_data_dir(tmp_path / "gr")
    c.update({"web": {"port": 0}, "raw_stream": {"enabled": False}, "radar": {"frame_timeout_s": 30, "retry_s": 5},
              "indicators": {"enabled": False}})
    return c


@pytest.fixture
def svc(cfg):
    s = GarageService(cfg, lambda settings, radar_cfg: FakeLink(), use_hardware=False)
    yield s
    s.shutdown()


def feed_fixture(svc, n_chunks=250):
    """Push the first chunks of the real recording through the service's own parser/pipeline."""
    chunks = read_raw_file(FIXTURE / "raw_uart.bin")[:n_chunks]
    t0 = time.time()
    for ts, data in chunks:
        svc._on_bytes(data, ts)
    svc._drain()
    while not svc._bytes.empty():
        svc._drain()
    return chunks, t0


# ---------------------------------------------------------------- link management
def test_on_bytes_records_and_drops_when_queue_is_full(cfg):
    s = GarageService(cfg, lambda *a: None, use_hardware=False)
    s._bytes = queue.Queue(maxsize=1)
    s._on_bytes(b"a", 1.0)
    s._on_bytes(b"b", 2.0)  # queue full: silently dropped
    assert s._bytes.get_nowait() == (b"a", 1.0) and s._bytes.empty()
    s.shutdown()


def test_ensure_link_connects_then_retries_after_failure(cfg):
    link = FakeLink()
    attempts = []

    def factory(settings, radar_cfg):
        attempts.append(settings)
        if len(attempts) == 1:
            raise RuntimeError("no radar plugged in")
        return link

    s = GarageService(cfg, factory, use_hardware=False)
    s._ensure_link()
    assert s.link is None and "no radar" in s.link_error and s._next_connect > time.time()
    s._ensure_link()  # too soon: no second attempt
    assert len(attempts) == 1
    s._next_connect = 0
    s._ensure_link()
    assert s.link is link and link.started and s.link_error == "" and s.link_since is not None
    s._ensure_link()  # already connected
    assert len(attempts) == 2
    s.shutdown()
    assert link.closed


def test_drop_link_closes_and_schedules_reconnect(svc):
    link = FakeLink()
    svc.link = link
    svc._drop_link("lost")
    assert link.closed and svc.link is None and svc.link_error == "lost" and svc._next_connect > time.time()
    svc._drop_link("again")  # nothing to close
    broken = FakeLink()
    broken.close = lambda: (_ for _ in ()).throw(OSError("busy"))
    svc.link = broken
    svc._drop_link("close raises")
    assert svc.link is None


def test_link_error_callback_queues_reconnect(svc):
    svc._on_link_error("usb unplugged")
    assert svc.link_error == "usb unplugged" and svc.commands.get_nowait() == ("reconnect",)
    svc.link = FakeLink()
    svc._cmd_reconnect()
    assert svc.link is None and svc.link_error == "usb unplugged"
    svc._cmd_reconnect()
    assert svc.link_error  # keeps a reason even with no error message set
    svc.link_error = ""
    svc._cmd_reconnect()
    assert svc.link_error == "reconnect requested"


def test_handle_commands_reports_failures_and_continues(svc):
    svc.commands.put(("record", "x", [], 5.0, False))  # no radar link yet -> RuntimeError
    svc.commands.put(("nonexistent",))
    svc.commands.put(("stop_simulation",))
    svc._handle_commands()
    assert svc.last_report == ["Simulation stopped"]  # the later command still ran
    svc.commands.put(("record", "x", [], 5.0, False))
    svc._handle_commands()
    assert svc.last_report[0].startswith("ERROR: record") and "radar is not running" in svc.last_report[0]


# ---------------------------------------------------------------- processing
def test_processing_fixture_updates_state_outputs_and_events(svc, cfg):
    svc.link = FakeLink()
    feed_fixture(svc, 600)
    assert svc.frames > 0 and svc.last_result is not None and svc.last_frame_wall is not None
    assert len(svc.frame_times) <= 20
    snap = svc.snapshot()
    assert len(snap["stalls"]) == 3 and snap["stalls"][0]["gpio"]["stall"] == 1
    assert snap["status"]["frames"] == svc.frames and snap["link"]["kind"] == "fake"
    assert snap["reference"]["heatmap_in_config"] is True and snap["calibration"]["fingerprint_ok"] is True
    assert snap["recording"] is None and snap["leds"]["running"] is False and snap["raw_stream"] == {"port": None, "clients": 0}
    json.dumps(snap, default=str)  # serialisable for the web page
    assert svc.status_summary()["health"] in ("ok", "starting")


def test_event_is_logged_published_and_written_to_csv(svc, cfg):
    published = []
    svc.mqtt.publish_event = published.append
    e = StallEvent(time.time(), 1, "Stall 2", "ENTERED", "changed area 40%")
    svc._event(e)
    svc._event(StallEvent(time.time(), 1, "Stall 2", "EXITED", "x"))
    assert [d["event"] for d in svc.events] == ["EXITED", "ENTERED"] and len(published) == 2
    lines = cfg.file("events.csv").read_text().splitlines()
    assert lines[0] == "time,stall,event,detail" and len(lines) == 3 and lines[1].split(",")[1] == "2"
    for _ in range(120):
        svc._event(e)
    assert len(svc.events) == 100


def test_event_survives_unwritable_csv(svc, cfg):
    cfg.file("events.csv").mkdir()  # a directory where the file should be
    svc._event(StallEvent(time.time(), 0, "Stall 1", "ENTERED"))
    assert svc.events[0]["event"] == "ENTERED"


def test_outputs_publish_mqtt_only_on_change(svc):
    sent = []
    svc.mqtt.publish_stall = lambda i, s, present: sent.append((i, s.state, present))
    stalls = [SimpleNamespace(state=StallState.EMPTY) for _ in range(3)]
    svc._outputs(stalls)
    svc._outputs(stalls)
    assert len(sent) == 3
    stalls[1] = SimpleNamespace(state=StallState.OCCUPIED)
    svc._outputs(stalls)
    assert sent[-1][:2] == (1, StallState.OCCUPIED) and len(sent) == 4


def test_system_ok_requires_link_fresh_frames_and_ready_heatmap(svc):
    assert not svc._system_ok()
    svc.link = FakeLink()
    svc.last_frame_wall = time.time()
    svc.pipeline.occupancy.last = SimpleNamespace(ready=True)
    assert svc._system_ok()
    svc.last_frame_wall = time.time() - 999
    assert not svc._system_ok()
    svc.last_frame_wall = time.time()
    svc.pipeline.occupancy.last = SimpleNamespace(ready=False)
    assert not svc._system_ok()
    svc.pipeline.set_occupancy(None)
    assert not svc._system_ok() and svc.status_summary()["occupancy"] == "off"


# ---------------------------------------------------------------- housekeeping
def test_housekeeping_without_link_or_frames_shows_error(svc):
    svc._housekeeping()
    assert [p.value for p in svc.gpio.pins] == [0, 0, 0] and svc.gpio.ok_pin.value == 0
    assert svc.leds.system == "error"


def test_housekeeping_drops_a_silent_link_but_not_a_finished_replay(svc):
    link = FakeLink()
    svc.link, svc.link_since = link, time.time() - 100
    svc._housekeeping()
    assert svc.link is None and link.closed and "no radar frames" in svc.link_error
    replay = FakeLink()
    replay.finished = SimpleNamespace(is_set=lambda: True)
    svc.link, svc.link_since = replay, time.time() - 100
    svc._housekeeping()
    assert svc.link is replay


def test_housekeeping_after_data_loss_holds_pins(svc):
    svc.link = FakeLink()
    feed_fixture(svc, 300)
    before = [p.value for p in svc.gpio.pins]
    svc.last_frame_wall = time.time() - 999
    svc.pipeline.last_data_time = time.time() - 999
    svc._housekeeping()
    assert svc.link is None
    assert [p.value for p in svc.gpio.pins] == before and svc.gpio.ok_pin.value == 0


def test_housekeeping_finishes_timed_recordings_and_saves_status(svc):
    svc.link = FakeLink()
    svc.last_frame_wall = time.time()
    svc._cmd_record("job", [], 0.0, False)
    published = []
    svc.mqtt.publish_status = published.append
    svc._last_status_pub = 0
    svc._housekeeping()
    assert svc.rec_job is None and svc.last_report[0].startswith("saved recording") and len(published) == 1
    svc._last_live_save = 0
    svc._housekeeping()  # periodic reference save is a no-op without adaptation, but must not raise


# ---------------------------------------------------------------- recording and labels
def test_record_validations_and_labelled_recording(svc, cfg):
    svc.link = FakeLink()
    svc._cmd_record("my_run", ["EMPTY", "OCCUPIED", "EMPTY"], 5000, False)
    assert svc.rec_job["until"] - time.time() <= cfg["recording"]["max_seconds"] + 1 and svc.rec_job["labelled"]
    assert "recording my_run" in svc.last_report[0] or "my_run" in svc.last_report[0]
    snap = svc.snapshot()["recording"]
    assert snap["states"] == ["EMPTY", "OCCUPIED", "EMPTY"] and snap["remaining_s"] > 0
    with pytest.raises(RuntimeError, match="already running"):
        svc._cmd_record("again", [], 5, False)
    svc._cmd_stop_record()
    assert svc.rec_job is None and any("added to the calibration set" in m for m in svc.last_report)
    labels = svc._labels()
    assert labels["segments"][0]["states"] == ["EMPTY", "OCCUPIED", "EMPTY"] and labels["reference"] == []
    svc._cmd_stop_record()  # nothing running: no-op
    name = labels["segments"][0]["recording"]
    svc._cmd_remove_label(name)
    assert svc._labels() == {"reference": [], "segments": []}


def test_unlabelled_recording_is_not_added_to_calibration(svc):
    svc.link = FakeLink()
    svc._cmd_record("plain", ["OCCUPIED", "UNKNOWN"], 1, False)
    assert not svc.rec_job["labelled"]
    svc._finish_recording()
    assert svc.last_report[0].startswith("saved recording") and len(svc.last_report) == 1
    assert svc._labels()["segments"] == []
    svc._finish_recording()  # no job and nothing recorded: no-op


def test_reference_recording_rebuilds_empty_reference(svc, cfg):
    svc.link = FakeLink()
    svc._cmd_record("empty_ref", ["EMPTY", "EMPTY", "EMPTY"], 60, True)
    for ts, data in read_raw_file(FIXTURE / "raw_uart.bin")[:300]:
        svc._on_bytes(data, ts)
    cfg.file(LIVE_REF).write_bytes(b"stale")
    svc._cmd_stop_record()
    assert any("reference rebuilt" in m for m in svc.last_report), svc.last_report
    assert not cfg.file(LIVE_REF).exists() and EmptyReference.load(cfg.file(CAPTURED_REF)).sources
    assert svc._labels()["reference"]
    assert svc.link is None  # reload drops the link so it reconnects with the new reference


def test_reference_from_a_different_chirp_is_refused(svc, cfg):
    svc.link = FakeLink()
    svc._cmd_record("odd", ["EMPTY"] * 3, 60, True)
    for ts, data in read_raw_file(FIXTURE / "raw_uart.bin")[:300]:
        svc._on_bytes(data, ts)
    svc.radar_cfg.set_arg("frameCfg", (), 4, 100)  # radar.cfg changed after the recording started
    with pytest.raises(service_mod.CalibrationError, match="different radar configuration"):
        svc._cmd_stop_record()


def test_labels_file_round_trip(svc):
    assert svc._labels() == {"reference": [], "segments": []}
    svc._save_labels({"reference": ["a"], "segments": [{"recording": "a"}]})
    assert json.loads(svc.cfg.file(LABELS).read_text())["reference"] == ["a"]
    svc._cmd_remove_label("a")
    assert svc._labels() == {"reference": [], "segments": []}


# ---------------------------------------------------------------- fit / reload / revert
def test_fit_runs_in_background_and_reloads(svc, cfg, monkeypatch):
    ref = EmptyReference(np.zeros((4, 4), np.float32), "fp", 3, ["a"])
    cal = OccupancyCalibration("fp", [StallCalibration(0.2, 0.05, 0.4)] * 3)
    monkeypatch.setattr(service_mod, "calibrate", lambda labels, d, g, p: (ref, cal, ["fit report"]))
    cfg.file(LIVE_REF).write_bytes(b"x")
    svc._save_labels({"reference": [], "segments": [1, 2]})
    svc._cmd_fit()
    end = time.time() + 5
    while svc.busy and time.time() < end:
        time.sleep(0.01)
    assert svc.busy == "" and svc.last_report == ["fit report", "calibration saved and loaded"]
    assert svc.commands.get_nowait() == ("reload",) and not cfg.file(LIVE_REF).exists()
    assert OccupancyCalibration.load(cfg.file("occupancy_calibration.json")).notes.startswith("fitted on the Pi")


def test_fit_failure_is_reported_and_busy_guard(svc, monkeypatch):
    def boom(*a):
        raise service_mod.CalibrationError("needs at least one EMPTY")

    monkeypatch.setattr(service_mod, "calibrate", boom)
    svc._cmd_fit()
    end = time.time() + 5
    while svc.busy and time.time() < end:
        time.sleep(0.01)
    assert svc.last_report == ["calibration FAILED: needs at least one EMPTY"]
    svc.busy = "fitting thresholds"
    with pytest.raises(RuntimeError, match="busy"):
        svc._cmd_fit()


def test_reload_rereads_files_and_resets_published_state(svc, cfg):
    svc.link = FakeLink()
    svc._published = ["x"] * 3
    shutil.copy(ROOT / "configs" / "occupancy_calibration.json", cfg.file("occupancy_calibration.json"))
    svc._cmd_reload()
    assert svc.link is None and svc._published == [None] * len(svc.garage.stalls)
    assert svc.pipeline.occupancy is not None and svc.reference_source == "captured"


def test_live_reference_is_used_only_when_compatible_and_newer(svc, cfg):
    occ = svc.pipeline.occupancy
    live = EmptyReference(occ.reference.image * 2, occ.fingerprint, 9, ["live"])
    live.save(cfg.file(LIVE_REF))
    cfg.file(LIVE_REF).touch()
    svc.load_models()
    assert svc.reference_source.startswith("self-updated") and svc.pipeline.occupancy.reference.frames == 9
    bad = EmptyReference(np.zeros((2, 2), np.float32), "other", 1, [])
    bad.save(cfg.file(LIVE_REF))
    svc.load_models()
    assert svc.reference_source == "captured"
    cfg.file(LIVE_REF).write_bytes(b"corrupt")
    svc.load_models()
    assert svc.reference_source == "captured"  # unreadable live file is ignored
    cfg.file(LIVE_REF).unlink()
    svc._cmd_revert_reference()
    assert svc.last_report == ["reverted to the captured empty-garage reference"]


def test_load_models_without_any_files(cfg):
    for name in ("radar.cfg", "garage.json", "detection_params.json", "empty_reference.npz", "occupancy_calibration.json"):
        cfg.file(name).unlink(missing_ok=True)
    s = GarageService(cfg, lambda *a: None, use_hardware=False)
    try:
        assert s.reference_source == "none" and len(s.garage.stalls) == 3
        s.save_live_reference()  # nothing to save
        assert s.snapshot()["reference"]["frames"] == 0
        assert s.snapshot()["calibration"]["notes"] == "none"
    finally:
        s.shutdown()


def test_save_live_reference_writes_after_adaptation(svc, cfg):
    occ = svc.pipeline.occupancy
    occ.live = occ.reference.image.copy()
    occ.adapted_s = 120.0
    svc.save_live_reference()
    saved = EmptyReference.load(cfg.file(LIVE_REF))
    assert saved.sources[-1] == "self-updated" and not cfg.file(LIVE_REF + ".tmp.npz").exists()


# ---------------------------------------------------------------- simulation / leds / shutdown
def test_simulation_commands_use_gpio_settings(svc):
    svc._cmd_simulate_arrival(2)
    assert svc.gpio.simulation(1) is not None and "stall 2" in svc.last_report[0]
    svc._cmd_stop_simulation()
    assert svc.gpio.simulation(1) is None


def test_led_test_when_bar_available(svc):
    svc.leds.problem = ""
    svc._cmd_led_test()
    assert "chase" in svc.last_report[0] and svc.leds.test_until > time.time()


def test_publish_indicators_force_and_shutdown_survives_failures(svc):
    sent = []
    svc.mqtt.publish_indicator = lambda i, info: sent.append(i)
    svc._publish_indicators(force=True)
    svc._publish_indicators(force=True)
    assert len(sent) == 6
    svc.rec_job = {"path": Path("."), "until": 0, "states": [], "reference": False, "labelled": False}
    svc.recorder.stop = lambda: (_ for _ in ()).throw(OSError("disk full"))
    svc.shutdown()  # swallows the failure and still closes everything
    svc.gpio.close = lambda: (_ for _ in ()).throw(RuntimeError("x"))
    svc.shutdown()


def test_run_loop_exits_on_stop_and_shuts_down(cfg):
    link = FakeLink()
    s = GarageService(cfg, lambda *a: link, use_hardware=False)
    s.commands.put(("stop_simulation",))
    import threading
    t = threading.Thread(target=s.run, daemon=True)
    t.start()
    end = time.time() + 5
    while not link.started and time.time() < end:
        time.sleep(0.01)
    s.stop()
    t.join(10)
    assert not t.is_alive() and link.started and link.closed
