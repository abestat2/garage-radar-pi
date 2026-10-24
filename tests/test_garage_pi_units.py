"""Unit tests: service config, LED bar, MQTT publisher, raw stream, indicator link, radar links, entry point."""
import json
import socket
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import garage_pi.__main__ as entry
from app.models.garage_state import StallEvent, StallResult, StallState
from app.radar.config_parser import RadarConfig
from garage_pi import config as cfgmod
from garage_pi import indicators, leds, mqtt_ha, radar_link, raw_stream
from garage_pi.config import DEFAULTS, ServiceConfig, _merge, init_data_dir

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "stall3_out_and_in"


def wait_for(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


# ---------------------------------------------------------------- config
def test_merge_is_deep_and_non_destructive():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    out = _merge(base, {"a": {"y": 9}, "c": 4})
    assert out == {"a": {"x": 1, "y": 9}, "b": 3, "c": 4} and base["a"]["y"] == 2
    assert _merge(base, None) == base
    assert _merge({"a": {"x": 1}}, {"a": 5}) == {"a": 5}  # a scalar replaces a section


def test_service_config_defaults_update_and_persistence(tmp_path):
    cfg = ServiceConfig(tmp_path / "d")
    assert cfg.values == DEFAULTS and cfg.values is not DEFAULTS
    assert cfg["web"]["port"] == 8080
    cfg.update({"web": {"port": 9000}, "mqtt": {"password": "secret", "host": "h"}})
    again = ServiceConfig(tmp_path / "d")
    assert again["web"]["port"] == 9000 and again["web"]["enabled"] is True  # untouched keys keep defaults
    assert not (tmp_path / "d" / "config.tmp").exists()
    pub = again.public()
    assert pub["mqtt"]["password"] == "********" and again["mqtt"]["password"] == "secret"
    assert pub["web"]["password"] == ""  # empty passwords are not masked
    assert again.file("x.json") == again.data_dir / "x.json" and again.recordings_dir.name == "recordings"


def test_init_data_dir_seeds_files_once(tmp_path):
    cfg = init_data_dir(tmp_path / "gr")
    for name in ("radar.cfg", "empty_reference.npz", "occupancy_calibration.json", "config.json"):
        assert cfg.file(name).exists(), name
    assert cfg.recordings_dir.is_dir()
    cfg.file("radar.cfg").write_text("custom")
    init_data_dir(tmp_path / "gr")
    assert cfg.file("radar.cfg").read_text() == "custom"  # never overwritten
    assert cfgmod.REPO == ROOT


def test_init_data_dir_expands_user(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert init_data_dir("~/gr").data_dir == tmp_path / "gr"


# ---------------------------------------------------------------- LED bar
def led_settings(**kw):
    s = dict(DEFAULTS["leds"])
    s.update(kw)
    return s


def test_encode_ws2812_pads_missing_channels():
    assert len(leds.encode_ws2812([(1,)], "GRB")) == 24 + leds.LATCH_BYTES
    data = leds.encode_ws2812([(255, 255, 255, 0)], "RGBW")  # explicit 4-tuple is not re-derived
    assert len(data) == 32 + leds.LATCH_BYTES and set(data[24:32]) == {leds.BIT0}


def test_led_bar_disabled_uses_mock_and_explains():
    bar = leds.LedBar(led_settings(enabled=False), 3)
    assert not bar.enabled and not bar.hardware and "disabled" in bar.problem
    assert bar.strip.last == [(0, 0, 0)] * 8
    bar.close()


def test_led_bar_without_hardware_flag_or_spi(monkeypatch):
    bar = leds.LedBar(led_settings(enabled=True), 3, use_hardware=False)
    assert bar.enabled and not bar.hardware and "mock" in bar.problem
    bar.close()
    bar2 = leds.LedBar(led_settings(enabled=True), 3, use_hardware=True)  # no spidev on a PC
    assert not bar2.hardware and "SPI unavailable" in bar2.problem
    bar2.close()


def test_led_bar_with_fake_spi_hardware(monkeypatch):
    sent = []

    class FakeSpi:
        def __init__(self):
            self.max_speed_hz = self.mode = None

        def open(self, bus, dev):
            self.addr = (bus, dev)  # pylint: disable=attribute-defined-outside-init

        def writebytes2(self, data):
            sent.append(bytes(data))

        def close(self):
            raise RuntimeError("close errors are swallowed")

    monkeypatch.setitem(__import__("sys").modules, "spidev", SimpleNamespace(SpiDev=FakeSpi))
    bar = leds.LedBar(led_settings(enabled=True, order="GRBW"), 3)
    assert bar.hardware and bar.problem == "" and bar.strip.spi.max_speed_hz == leds.SPI_HZ
    assert wait_for(lambda: len(sent) > 0)
    bar.close()
    assert all(len(chunk) <= 4096 for chunk in sent)


def test_led_frame_colours_blink_flash_and_status():
    s = led_settings(enabled=False, brightness=1.0, status_led=7, stall_leds=[[0, 1], [2], [99]])
    bar = leds.LedBar(s, 3)
    bar.set_states([StallState.OCCUPIED, StallState.ENTERING, StallState.EMPTY], "ok")
    f = bar.frame(now=0.0)  # blink phase "on"
    assert f[0] == f[1] == (255, 0, 0) and f[2] == (0, 255, 0) and f[7] == (0, 60, 0)
    assert bar.frame(now=0.5)[2] == (0, 0, 0)  # blink phase "off"
    bar.event(0, "ENTERED")
    bar.event(1, "EXITED")
    bar.event(9, "ENTERED")  # unknown stall: ignored
    t = time.time()
    flashing = bar.frame(now=t)
    assert flashing[0] in ((0, 255, 0), (0, 0, 0))
    assert bar.frame(now=t + 100)[0] == (255, 0, 0)  # flash expired
    for system, expected in (("error", {(120, 0, 0), (0, 0, 0)}), ("busy", {(0, 0, 120)}), ("starting", {(40, 40, 0)})):
        bar.set_states([StallState.EMPTY] * 3, system)
        assert bar.frame(now=t + 200)[7] in expected
    bar.set_states([StallState.EMPTY], "ok")  # fewer states than stalls: loop stops early
    bar.frame(now=0.0)
    bar.close()


def test_led_test_pattern_and_close_blanks_strip():
    bar = leds.LedBar(led_settings(enabled=False, brightness=1.0), 3)
    bar.test(seconds=10)
    chase = bar.frame()
    assert len(chase) == 8 and set(chase) <= {(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)}
    bar.close()
    assert bar.strip.last == [(0, 0, 0)] * 8


def test_led_render_loop_survives_strip_errors():
    bar = leds.LedBar(led_settings(enabled=True), 3, use_hardware=False)

    class Broken:
        calls = 0

        def show(self, colors):
            Broken.calls += 1
            raise OSError("bus error")

        def close(self):
            pass

    bar.strip = Broken()
    assert wait_for(lambda: Broken.calls >= 1)
    bar.close()


# ---------------------------------------------------------------- MQTT
class FakeClient:
    def __init__(self, fail=False):
        self.msgs, self.fail, self.stopped, self.disconnected = [], fail, False, False

    def publish(self, topic, payload, qos=0, retain=False):
        if self.fail:
            raise RuntimeError("broker gone")
        self.msgs.append((topic, json.loads(payload) if payload.startswith("{") else payload, retain))

    def loop_stop(self):
        self.stopped = True

    def disconnect(self):
        self.disconnected = True


def stall_result(**kw):
    d = dict(stall_index=0, name="Stall 1", state=StallState.OCCUPIED, confidence=0.87654, changed=0.4123,
             threshold=0.2, persistence=12.34, explanation=list("abcdef"))
    d.update(kw)
    return StallResult(**d)


def test_mqtt_enabled_rules():
    assert not mqtt_ha.MqttPublisher({"enabled": False}, 3).enabled
    assert not mqtt_ha.MqttPublisher({"enabled": True, "host": ""}, 3).enabled
    assert mqtt_ha.MqttPublisher({"enabled": True}, 3, client=FakeClient()).enabled


def test_mqtt_publishes_stall_event_status_and_indicator():
    c = FakeClient()
    m = mqtt_ha.MqttPublisher({"enabled": True, "base_topic": "b", "device_id": "d"}, 3, client=c)
    m.publish_stall(1, stall_result(), True)
    m.publish_stall(2, stall_result(changed=None, threshold=None, state=StallState.EMPTY), None)
    m.publish_event(StallEvent(5.0, 0, "Stall 1", "ENTERED", "x"))
    m.publish_status({"health": "ok"})
    m.publish_indicator(0, {"battery_pct": 50, "battery_v": 3.8, "contact": "lost", "last_exchange_age_s": 9})
    got = {(t, r): p for t, p, r in c.msgs}
    assert got[("b/d/stall2/state", True)] == "VEHICLE ENTERING" or got[("b/d/stall2/state", True)] == "OCCUPIED"
    assert got[("b/d/stall2/occupied", True)] == "ON"
    attrs = got[("b/d/stall2/attributes", True)]
    assert attrs["confidence"] == 0.877 and attrs["changed_pct"] == 41.2 and len(attrs["explanation"]) == 4
    assert got[("b/d/stall3/attributes", True)]["changed_pct"] is None
    assert not any(t == "b/d/stall3/occupied" for t, _, _ in c.msgs)  # present=None: no occupied topic
    ev = got[("b/d/stall1/event", False)]  # events are not retained
    assert ev["event_type"] == "ENTERED" and ev["stall"] == 1
    assert got[("b/d/status", True)] == {"health": "ok"}
    assert got[("b/d/stall1/indicator", True)]["connected"] == "OFF"


def test_mqtt_publish_errors_are_recorded_not_raised():
    m = mqtt_ha.MqttPublisher({"enabled": True}, 3, client=FakeClient(fail=True))
    m.publish_status({})
    assert m.last_error == "broker gone"
    off = mqtt_ha.MqttPublisher({"enabled": False}, 3)
    off.publish_status({})  # silently ignored
    off.publish_discovery()


def test_mqtt_connect_callbacks_publish_discovery_and_availability():
    c = FakeClient()
    m = mqtt_ha.MqttPublisher({"enabled": True, "host": "h", "base_topic": "b", "device_id": "d"}, 1, client=c)
    m._on_connect(c, None, {}, 5)
    assert not m.connected and "refused" in m.last_error and not c.msgs
    m._on_connect(c, None, {}, SimpleNamespace(is_failure=False))  # paho 2 reason code object
    assert m.connected and m.last_error == "" and ("b/d/availability", "online", True) in c.msgs
    assert any(t.endswith("/config") for t, _, _ in c.msgs)
    m._on_disconnect()
    assert not m.connected
    m._on_connect(c, None, {}, SimpleNamespace(is_failure=True))
    assert not m.connected
    m._on_connect(c, None, {}, 0)
    assert m.connected


def test_mqtt_close_announces_offline():
    c = FakeClient()
    m = mqtt_ha.MqttPublisher({"enabled": True, "base_topic": "b", "device_id": "d"}, 1, client=c)
    m.close()
    assert ("b/d/availability", "offline", True) in c.msgs and c.stopped and c.disconnected
    mqtt_ha.MqttPublisher({"enabled": False}, 1).close()  # nothing to close
    broken = mqtt_ha.MqttPublisher({"enabled": True}, 1, client=FakeClient())
    broken.client.loop_stop = lambda: (_ for _ in ()).throw(RuntimeError("x"))
    broken.close()


def test_mqtt_make_client_configures_paho(monkeypatch):
    calls = {}

    class PahoClient:
        def __init__(self, *a, **k):
            calls["init"] = (a, k)

        def username_pw_set(self, u, p):
            calls["auth"] = (u, p)

        def will_set(self, *a, **k):
            calls["will"] = (a, k)

        def reconnect_delay_set(self, *a):
            calls["delay"] = a

        def connect_async(self, host, port, keepalive):
            calls["connect"] = (host, port, keepalive)

        def loop_start(self):
            calls["loop"] = True

    import paho.mqtt.client as paho
    monkeypatch.setattr(paho, "Client", PahoClient)
    m = mqtt_ha.MqttPublisher({"enabled": True, "host": "broker", "port": "1884", "username": "u", "password": "",
                               "device_id": "dev"}, 1)
    assert isinstance(m.client, PahoClient) and m.enabled
    assert calls["auth"] == ("u", None) and calls["connect"] == ("broker", 1884, 30) and calls["loop"]
    assert calls["will"][0][:2] == ("garage_radar/dev/availability", "offline")


def test_mqtt_make_client_survives_connect_failure_and_missing_paho(monkeypatch):
    import paho.mqtt.client as paho

    class Failing(paho.Client):
        def connect_async(self, *a, **k):
            raise OSError("dns")

    monkeypatch.setattr(paho, "Client", Failing)
    m = mqtt_ha.MqttPublisher({"enabled": True, "host": "h"}, 1)
    assert m.last_error == "dns" and m.client is not None

    import builtins
    real_import = builtins.__import__

    def no_paho(name, *a, **k):
        if name.startswith("paho"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_paho)
    gone = mqtt_ha.MqttPublisher({"enabled": True, "host": "h"}, 1)
    assert gone.client is None and not gone.enabled and "not installed" in gone.last_error


# ---------------------------------------------------------------- raw stream
def test_raw_stream_broadcasts_to_clients_and_limits_them():
    srv = raw_stream.RawStreamServer(0, host="127.0.0.1", max_clients=1)
    try:
        a = socket.create_connection(("127.0.0.1", srv.port), timeout=3)
        assert wait_for(lambda: srv.clients == 1)
        extra = socket.create_connection(("127.0.0.1", srv.port), timeout=3)
        extra.settimeout(2)
        assert extra.recv(10) == b""  # over the limit: closed by the server
        extra.close()
        srv.broadcast(b"hello")
        a.settimeout(3)
        assert a.recv(10) == b"hello"
        a.close()
        assert wait_for(lambda: (srv.broadcast(b"x" * 1000) or True) and srv.clients == 0)
    finally:
        srv.close()


def test_raw_stream_disconnects_clients_whose_queue_is_full():
    import queue

    srv = raw_stream.RawStreamServer(0, host="127.0.0.1")
    try:
        q = queue.Queue(maxsize=2)
        with srv._lock:
            srv._clients.append((object(), q))
        srv.broadcast(b"1")
        srv.broadcast(b"2")
        assert q.qsize() == 2
        srv.broadcast(b"3")  # full: backlog discarded, client told to stop
        assert q.get_nowait() is None and q.empty()
    finally:
        srv.close()


def test_raw_stream_client_that_hangs_up_is_removed():
    srv = raw_stream.RawStreamServer(0, host="127.0.0.1")
    try:
        sock = socket.create_connection(("127.0.0.1", srv.port), timeout=3)
        assert wait_for(lambda: srv.clients == 1)
        sock.close()
        assert wait_for(lambda: (srv.broadcast(b"x" * 100000) or True) and srv.clients == 0, timeout=5)
    finally:
        srv.close()


# ---------------------------------------------------------------- indicator link
def test_indicator_link_feed_calls_on_change_only_for_changes():
    status = indicators.IndicatorStatus(3)
    seen = []
    link = indicators.IndicatorLink({"port": "/dev/null-nonexistent"}, status, on_change=seen.append)
    try:
        link.feed("boot noise")
        link.feed("IND stall=2 contact=ok battery_mv=3900")
        link.feed("IND stall=2 contact=ok battery_mv=3900")
        link.feed("IND stall=9 contact=ok")
        assert seen == [1]
        assert wait_for(lambda: link.error != "")  # the missing port is reported, not raised
    finally:
        link.close()


def test_indicator_link_reads_lines_from_serial(monkeypatch):
    import serial

    ports = []

    class Port:
        def __init__(self, *a, **k):
            self.chunks = [b"IND stall=1 state=Pre", b"sent battery_mv=4000\nIND stall=3 contact=lost\n"]
            self.written = []
            ports.append(self)

        def write(self, data):
            self.written.append(data)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            time.sleep(0.01)
            return self.chunks.pop(0) if self.chunks else b""

    monkeypatch.setattr(serial, "Serial", Port)
    status = indicators.IndicatorStatus(3)
    link = indicators.IndicatorLink({"port": "p", "baud": "9600"}, status)
    try:
        assert wait_for(lambda: status.snapshot()[0]["battery_v"] == 4.0 and status.snapshot()[2]["contact"] == "lost")
        assert link.error == "" and link.baud == 9600
        assert ports[0].written == [b"MAC?\n", b"DIST?\n"]  # asks for the detector's saved settings on connecting
    finally:
        link.close()


def test_battery_percent_edges():
    assert indicators.battery_percent(0) is None and indicators.battery_percent(3000) == 0
    assert indicators.battery_percent(4200) == 100 and indicators.battery_percent(4500) == 100
    assert indicators.parse_line("IND stall=x") is None and indicators.parse_line("IND stall") is None
    assert indicators.parse_line("IND contact=ok") is None and indicators.parse_line("") is None


# ---------------------------------------------------------------- radar links
class P:
    def __init__(self, device, description="", hwid=""):
        self.device, self.description, self.hwid = device, description, hwid


def patch_ports(monkeypatch, ports):
    from app.radar import serial_manager as sm

    monkeypatch.setattr(radar_link, "scan_ports", lambda: [sm.PortInfo(p.device, p.description, p.hwid,
                                                                     sm.guess_role(p.description)) for p in ports])


def test_find_ports_explicit_descriptor_and_bridge_fallback(monkeypatch):
    assert radar_link.find_ports("A", "B") == ("A", "B")
    patch_ports(monkeypatch, [P("COM3", "CP210x Enhanced COM Port"), P("COM4", "CP210x Standard COM Port")])
    assert radar_link.find_ports() == ("COM3", "COM4")
    assert radar_link.find_ports(cli="/dev/x") == ("/dev/x", "COM4")
    patch_ports(monkeypatch, [P("/dev/ttyUSB1", "CP2105", "USB VID:PID=10C4:EA70 LOCATION=1-1.1:1.1"),
                              P("/dev/ttyUSB0", "CP2105", "USB VID:PID=10C4:EA70 LOCATION=1-1.1:1.0")])
    assert radar_link.find_ports() == ("/dev/ttyUSB0", "/dev/ttyUSB1")
    patch_ports(monkeypatch, [P("COM1", "Communications Port")])
    with pytest.raises(RuntimeError, match="not found.*COM1"):
        radar_link.find_ports()


class FakeSerialManager:
    def __init__(self, config_ok=True):
        self.cli, self.bytes_received, self.opened, self.reader, self.closed = object(), 0, None, None, False

    def open(self, *a):
        self.opened = a

    def start_reader(self, on_bytes, on_error):
        self.reader = (on_bytes, on_error)

    def close(self):
        self.closed = True


def test_serial_link_start_success_and_close(monkeypatch):
    from app.radar.cli_interface import CliResponse
    from app.radar.device_info import DeviceInfo

    sent = []

    class FakeCli:
        def __init__(self, port):
            pass

        def send_config(self, config):
            return [CliResponse("a", True, ["Done"])]

        def stop(self):
            sent.append("stop")

    monkeypatch.setattr(radar_link, "CliInterface", FakeCli)
    monkeypatch.setattr(radar_link, "query_device_info", lambda cli: DeviceInfo(True, {"Platform": "xWR68xx"}))
    link = radar_link.SerialRadarLink({"cli_port": "C", "data_port": "D", "cli_baud": "1", "data_baud": "2"},
                                      RadarConfig("sensorStart"))
    assert link.describe() == "not connected"
    link.serial = FakeSerialManager()
    link.start(lambda b, t: None, lambda e: None)
    assert link.serial.opened == ("C", "D", 1, 2) and link.serial.reader is not None
    assert "C / D" in link.describe() and link.bytes_received == 0
    link.close()
    assert link.serial.closed and sent == ["stop"]


def test_serial_link_rejected_config_closes_and_raises(monkeypatch):
    from app.radar.cli_interface import CliResponse
    from app.radar.device_info import DeviceInfo

    class FakeCli:
        def __init__(self, port):
            pass

        def send_config(self, config):
            return [CliResponse("ok", True, []), CliResponse("badCmd", False, ["Error"])]

        def stop(self):
            raise RuntimeError("stop failures are ignored")

    monkeypatch.setattr(radar_link, "CliInterface", FakeCli)
    monkeypatch.setattr(radar_link, "query_device_info", lambda cli: DeviceInfo())
    link = radar_link.SerialRadarLink({"cli_port": "C", "data_port": "D", "cli_baud": 1, "data_baud": 2}, RadarConfig(""))
    link.serial = FakeSerialManager()
    with pytest.raises(RuntimeError, match="rejected 'badCmd'"):
        link.start(lambda b, t: None, lambda e: None)
    assert link.serial.closed and link.serial.reader is None


def test_replay_link_plays_recording_once_and_finishes():
    link = radar_link.ReplayRadarLink(str(FIXTURE), speed=1000.0)
    got = []
    link.start(lambda data, ts: got.append((data, ts)), lambda e: None)
    assert link.finished.wait(20) and got and link.bytes_received == sum(len(d) for d, _ in got)
    stamps = [t for _, t in got]
    assert stamps == sorted(stamps) and stamps[-1] - stamps[0] > 10  # recorded timing is preserved at any speed
    assert "REPLAY" in link.device.fields["Platform"] and "1000x" in link.describe()
    link.close()


def test_replay_link_loops_until_closed(tmp_path):
    from app.recording import session as ses

    d = tmp_path / "rec"
    d.mkdir()
    (d / "raw_uart.bin").write_bytes(ses.RAW_MAGIC + struct.pack("<dI", 1.0, 2) + b"ab" + struct.pack("<dI", 1.01, 2) + b"cd")
    link = radar_link.ReplayRadarLink(d, speed=1.0, loop=True)
    chunks = []
    link.start(lambda data, ts: chunks.append(data), lambda e: None)
    assert wait_for(lambda: len(chunks) >= 6)
    link.close()
    n = len(chunks)
    time.sleep(0.1)
    assert len(chunks) == n and not link.finished.is_set()
    radar_link.ReplayRadarLink(d).close()  # closing a link that never started is fine


# ---------------------------------------------------------------- entry point
def test_main_live_and_replay_wiring(monkeypatch, tmp_path):
    made = {}

    class FakeService:
        def __init__(self, cfg, factory, use_hardware=True):
            made.update(service=self, factory=factory, hardware=use_hardware)
            self.stopped = False

        def run(self):
            made["ran"] = True

        def stop(self):
            self.stopped = True

    class FakeWeb:
        def __init__(self, service, port):
            made["web_port"] = port
            self.closed = False
            made["web"] = self

        def close(self):
            self.closed = True

    handlers = {}
    monkeypatch.setattr(entry, "GarageService", FakeService)
    monkeypatch.setattr(entry, "WebServer", FakeWeb)
    monkeypatch.setattr(entry.signal, "signal", lambda sig, fn: handlers.setdefault(sig, fn))
    monkeypatch.setattr(entry, "ReplayRadarLink", lambda path, speed, loop: ("replay", path, speed, loop))
    monkeypatch.setattr(entry, "SerialRadarLink", lambda s, c: ("serial", s, c))

    assert entry.main(["--data-dir", str(tmp_path / "live"), "--no-hardware", "-v"]) == 0
    assert made["ran"] and made["hardware"] is False and made["web_port"] == 8080 and made["web"].closed
    assert made["factory"]({"s": 1}, "cfg") == ("serial", {"s": 1}, "cfg")
    handlers[entry.signal.SIGTERM]()
    assert made["service"].stopped

    assert entry.main(["--data-dir", str(tmp_path / "rp"), "--replay", "REC", "--speed", "4", "--loop"]) == 0
    assert made["hardware"] is True and made["factory"]({}, None) == ("replay", "REC", 4.0, True)


def test_main_without_web_server(monkeypatch, tmp_path):
    class FakeService:
        def __init__(self, *a, **k):
            pass

        def run(self):
            pass

    monkeypatch.setattr(entry, "GarageService", FakeService)
    monkeypatch.setattr(entry, "WebServer", lambda *a: pytest.fail("web server must not start"))
    monkeypatch.setattr(entry.signal, "signal", lambda *a: None)
    cfg = init_data_dir(tmp_path / "noweb")
    cfg.update({"web": {"enabled": False}})
    assert entry.main(["--data-dir", str(tmp_path / "noweb")]) == 0
