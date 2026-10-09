"""Tests of the headless Pi service, run on any PC (mock GPIO / LEDs, local sockets)."""
import json
import shutil
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "stall3_out_and_in"  # real garage recording (raw radar bytes)


def test_service_does_not_import_qt():
    """The Pi has no Qt: importing the service must not pull in PySide6."""
    import subprocess

    code = ("import sys; sys.modules['PySide6'] = None\n"
            "import garage_pi.service, garage_pi.web, garage_pi.__main__\n"
            "print('ok')")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


def test_ws2812_encoding():
    from garage_pi.leds import BIT0, BIT1, LATCH_BYTES, encode_ws2812

    data = encode_ws2812([(0x80, 0x01, 0x00)], "GRB")
    bits = data[:24]
    assert list(bits[:8]) == [BIT0] * 7 + [BIT1]  # G = 0x01
    assert list(bits[8:16]) == [BIT1] + [BIT0] * 7  # R = 0x80
    assert list(bits[16:24]) == [BIT0] * 8  # B = 0
    assert len(data) == 24 + LATCH_BYTES and data[24:] == b"\x00" * LATCH_BYTES


def test_gpio_contract():
    from app.models.garage_state import StallState as S
    from garage_pi.gpio_out import StallOutputs

    g = StallOutputs({"stall_pins": [1, 2, 3], "ok_pin": 4}, use_hardware=False)
    assert g.update([S.UNKNOWN, S.UNKNOWN, S.UNKNOWN], False) == []
    assert [p.value for p in g.pins] == [0, 0, 0]  # unknown at start -> held low
    g.update([S.OCCUPIED, S.EMPTY, S.OCCUPIED], True)
    assert [p.value for p in g.pins] == [1, 0, 1] and g.ok_pin.value == 1
    edges = g.update([S.EXITING, S.ENTERING, S.EMPTY], True)  # exiting = still present; entering = hold
    assert [p.value for p in g.pins] == [1, 0, 0] and edges == [(2, False)]
    g.update([S.SENSOR_ERROR, S.UNKNOWN, S.SENSOR_ERROR], False)  # errors hold, OK pin drops
    assert [p.value for p in g.pins] == [1, 0, 0] and g.ok_pin.value == 0
    early = StallOutputs({"stall_pins": [1], "ok_pin": None, "entering_is_present": True}, use_hardware=False)
    early.update([S.EMPTY], True)
    assert early.update([S.ENTERING], True) == [(0, True)]


def test_ha_discovery_payloads():
    from garage_pi.mqtt_ha import MqttPublisher

    class FakeClient:
        def __init__(self):
            self.msgs = []

        def publish(self, topic, payload, qos=0, retain=False):
            self.msgs.append((topic, json.loads(payload) if payload.startswith("{") else payload, retain))

    c = FakeClient()
    m = MqttPublisher({"enabled": True, "device_id": "gr", "base_topic": "garage_radar"}, 3, "t", client=c)
    m.publish_discovery()
    topics = [t for t, _, _ in c.msgs]
    assert "homeassistant/binary_sensor/gr/stall1_occupied/config" in topics
    assert "homeassistant/event/gr/stall3_event/config" in topics
    ev = next(p for t, p, _ in c.msgs if t.endswith("stall2_event/config"))
    assert ev["event_types"] == ["ENTERED", "EXITED"] and ev["state_topic"] == "garage_radar/gr/stall2/event"
    assert all(r for _, _, r in c.msgs)  # discovery is retained


@pytest.fixture
def data_dir(tmp_path):
    from garage_pi.config import init_data_dir

    cfg = init_data_dir(tmp_path / "gr")
    cfg.update({"web": {"port": 0}, "raw_stream": {"port": 0}, "radar": {"frame_timeout_s": 30}})
    return cfg


def test_service_replay_end_to_end(data_dir):
    """Replay the stall-3 out-and-in recording through the whole service at 10x speed."""
    from garage_pi.radar_link import ReplayRadarLink
    from garage_pi.service import GarageService
    from garage_pi.web import WebServer

    link = ReplayRadarLink(FIXTURE, speed=10.0)
    svc = GarageService(data_dir, lambda s, c: link, use_hardware=False)
    web = WebServer(svc, 0, host="127.0.0.1")
    th = threading.Thread(target=svc.run, daemon=True)
    th.start()
    try:
        # raw stream: a client receives TI packets
        sock = socket.create_connection(("127.0.0.1", svc.raw.port), timeout=5)
        got = b""
        t0 = time.time()
        while b"\x02\x01\x04\x03\x06\x05\x08\x07" not in got and time.time() - t0 < 10:
            got += sock.recv(65536)
        sock.close()
        assert b"\x02\x01\x04\x03\x06\x05\x08\x07" in got
        assert link.finished.wait(60), "replay did not finish"
        time.sleep(1.0)
        status = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{web.port}/api/status", timeout=5).read())
        cfgtxt = urllib.request.urlopen(f"http://127.0.0.1:{web.port}/api/radar_config", timeout=5).read().decode()
        history = list(svc.gpio.pins[2].history)  # before shutdown (which switches every pin off)
        levels = [p.value for p in svc.gpio.pins]
    finally:
        svc.stop()
        th.join(timeout=10)
        web.close()
    assert "guiMonitor -1 1 1 0 1 0 1" in cfgtxt
    events = [(e["stall"], e["event"]) for e in reversed(status["events"])]
    assert events == [(3, "EXITED"), (3, "ENTERED")]
    assert history == [0, 1, 0, 1]  # stall 3 pin: initial low, occupied, exited (falling), entered (rising)
    assert levels == [1, 1, 1]
    assert (data_dir.file("events.csv")).read_text().count("ENTERED") == 1


def test_calibration_needs_empty_and_occupied_data(data_dir):
    """Fitting from too little labelled data fails with a clear message instead of a bad model."""
    shutil.copytree(FIXTURE, data_dir.recordings_dir / "fixture")
    from app.detection.calibrate import CalibrationError, calibrate
    from app.geometry.site import default_garage

    labels = {"reference": ["fixture"], "segments": [
        {"recording": "fixture", "t": [72, 96], "states": ["OCCUPIED"] * 3}]}
    with pytest.raises(CalibrationError, match="needs at least one EMPTY and one OCCUPIED"):
        calibrate(labels, data_dir.recordings_dir, default_garage())
