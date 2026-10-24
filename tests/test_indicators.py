"""Indicator reports from the detector ESP32: parsing, battery estimate, status, MQTT."""
import json

import pytest

from garage_pi.indicators import IndicatorStatus, battery_percent, parse_line


class FakeClient:
    def __init__(self):
        self.msgs = []

    def publish(self, topic, payload, qos=0, retain=False):
        self.msgs.append((topic, json.loads(payload), retain))


@pytest.fixture
def data_dir(tmp_path):
    from garage_pi.config import init_data_dir

    cfg = init_data_dir(tmp_path / "gr")
    cfg.update({"web": {"port": 0}, "raw_stream": {"port": 0}})
    return cfg


def test_report_lines_parse():
    assert parse_line("IND stall=1 state=Present battery_mv=3850\r") == {
        "stall": 1, "state": "Present", "battery_mv": 3850}
    assert parse_line("IND stall=2 contact=lost") == {"stall": 2, "contact": "lost"}
    assert parse_line("IND stall=3 contact=ok battery_mv=0 age_s=-1") == {
        "stall": 3, "contact": "ok", "battery_mv": 0, "age_s": -1}
    for junk in ("", "ESP-ROM:esp32c3-api1-20210207", "IND contact=ok", "IND stall=x contact=ok",
                 "IND stall=1 battery_mv=lots", "IND stall=1 broken"):
        assert parse_line(junk) is None


def test_battery_percent_follows_the_liion_curve():
    assert battery_percent(0) is None  # not measured
    assert battery_percent(3000) == 0
    assert battery_percent(4200) == 100 and battery_percent(4350) == 100
    assert battery_percent(3820) == 50
    assert 50 < battery_percent(3850) < 60
    levels = [battery_percent(mv) for mv in range(3300, 4201, 10)]
    assert levels == sorted(levels)  # never drops as the voltage rises


def test_status_tracks_contact_and_battery():
    now = [1000.0]
    st = IndicatorStatus(3, clock=lambda: now[0])
    assert st.snapshot()[0]["contact"] == "unknown" and st.snapshot()[0]["battery_v"] is None
    assert st.apply(parse_line("IND stall=1 state=Absent battery_mv=3712"))
    s1 = st.snapshot()[0]
    assert (s1["contact"], s1["battery_v"], s1["last_state"]) == ("ok", 3.71, "Absent")
    assert not st.apply(parse_line("IND stall=1 state=Present battery_mv=3712"))  # nothing published changed
    assert st.apply(parse_line("IND stall=1 contact=lost"))
    assert st.snapshot()[0]["contact"] == "lost" and st.snapshot()[0]["battery_v"] == 3.71  # battery kept
    # A summary without a measurement keeps the last battery; its age dates the last exchange
    now[0] = 1100.0
    st.apply(parse_line("IND stall=1 contact=ok battery_mv=0 age_s=40"))
    assert st.snapshot()[0]["battery_v"] == 3.71 and st.snapshot()[0]["last_exchange_age_s"] == 40
    assert not st.apply(parse_line("IND stall=9 contact=ok"))  # no such stall


def test_discovery_only_when_indicators_are_enabled():
    from garage_pi.mqtt_ha import MqttPublisher

    off, on = FakeClient(), FakeClient()
    MqttPublisher({"enabled": True, "device_id": "gr"}, 3, "t", client=off).publish_discovery()
    MqttPublisher({"enabled": True, "device_id": "gr"}, 3, "t", client=on, indicators=True).publish_discovery()
    assert not any("indicator" in t for t, _, _ in off.msgs)
    topics = {t: p for t, p, _ in on.msgs}
    bat = topics["homeassistant/sensor/gr/stall2_indicator_battery/config"]
    assert bat["device_class"] == "battery" and bat["unit_of_measurement"] == "%"
    assert bat["state_topic"] == "garage_radar/gr/stall2/indicator"
    conn = topics["homeassistant/binary_sensor/gr/stall3_indicator_connected/config"]
    assert conn["device_class"] == "connectivity" and conn["value_template"] == "{{ value_json.connected }}"
    assert "homeassistant/sensor/gr/stall1_indicator_voltage/config" in topics


def test_service_publishes_indicator_changes(data_dir):
    from garage_pi.service import GarageService

    svc = GarageService(data_dir, lambda s, c: None, use_hardware=False)
    try:
        assert svc.indicator_link is None  # no serial reader without hardware
        client = FakeClient()
        svc.mqtt.client, svc.mqtt.enabled = client, True
        svc.indicators.apply(parse_line("IND stall=2 state=Present battery_mv=4000"))
        svc._publish_indicators()
        svc._publish_indicators()  # unchanged: not published again
        msgs = [(t, p) for t, p, _ in client.msgs if t.endswith("/indicator")]
        assert len(msgs) == 3  # the first pass publishes every stall once
        stall2 = next(p for t, p in msgs if t.endswith("stall2/indicator"))
        assert stall2["connected"] == "ON" and stall2["battery_pct"] == 80 and stall2["battery_v"] == 4.0
        stall1 = next(p for t, p in msgs if t.endswith("stall1/indicator"))
        assert stall1["connected"] == "OFF" and stall1["battery_pct"] is None

        svc.indicators.apply(parse_line("IND stall=2 contact=lost"))
        svc._publish_indicators()
        last = [p for t, p, _ in client.msgs if t.endswith("stall2/indicator")][-1]
        assert last["connected"] == "OFF" and last["battery_v"] == 4.0
        assert svc.snapshot()["indicators"]["stalls"][1]["contact"] == "lost"
    finally:
        svc.shutdown()


def test_mac_reports_and_errors_parse():
    assert parse_line("MAC stall=2 addr=B4:3A:45:6B:FF:90\r") == {"kind": "MAC", "stall": 2, "addr": "b4:3a:45:6b:ff:90"}
    assert parse_line("MAC stall=3 addr=none") == {"kind": "MAC", "stall": 3, "addr": ""}
    assert parse_line("ERR what=addr") == {"kind": "ERR", "what": "addr"}
    for junk in ("MAC stall=1 addr=nonsense", "MAC stall=1", "MAC addr=aa:bb:cc:dd:ee:ff", "ERR", "ERR broken"):
        assert parse_line(junk) is None


def test_status_keeps_the_detectors_saved_macs_and_errors():
    st = IndicatorStatus(3)
    assert [s["mac"] for s in st.snapshot()] == [None, None, None]  # nothing reported yet
    assert not st.apply(parse_line("MAC stall=1 addr=58:8c:81:04:72:b4"))  # not an MQTT-published change
    assert not st.apply(parse_line("MAC stall=2 addr=none"))
    assert [s["mac"] for s in st.snapshot()] == ["58:8c:81:04:72:b4", "", None]
    assert not st.apply(parse_line("MAC stall=9 addr=none"))  # no such stall
    st.apply(parse_line("ERR what=addr"))
    assert st.last_error == "addr"


def test_set_mac_sends_a_command_to_the_detector():
    from garage_pi.indicators import IndicatorLink

    class Port:
        def __init__(self):
            self.written = []

        def write(self, data):
            self.written.append(data)

    link = IndicatorLink({"port": "/dev/null"}, IndicatorStatus(3))
    link.close()
    with pytest.raises(RuntimeError):
        link.set_mac(1, "aa:bb:cc:dd:ee:ff")  # no open port
    link._serial = port = Port()
    assert link.set_mac(1, "AA:BB:CC:DD:EE:FF") == "aa:bb:cc:dd:ee:ff"
    assert link.set_mac(3, "") == ""
    assert port.written == [b"SETMAC stall=1 addr=aa:bb:cc:dd:ee:ff\n", b"SETMAC stall=3 addr=none\n"]
    for stall, addr in ((0, "aa:bb:cc:dd:ee:ff"), (4, "aa:bb:cc:dd:ee:ff"), (1, "aa:bb:cc:dd:ee"), (1, "zz:bb:cc:dd:ee:ff")):
        with pytest.raises(ValueError):
            link.set_mac(stall, addr)
    assert len(port.written) == 2


def test_web_sets_an_indicator_mac(data_dir):
    import urllib.error
    import urllib.request

    from garage_pi.service import GarageService
    from garage_pi.web import WebServer

    svc = GarageService(data_dir, lambda s, c: None, use_hardware=False)
    web = WebServer(svc, 0, host="127.0.0.1")

    def post(body):
        req = urllib.request.Request(f"http://127.0.0.1:{web.port}/api/indicator_mac", json.dumps(body).encode())
        try:
            r = urllib.request.urlopen(req, timeout=5)
            return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    class FakeLink:
        port = "fake"
        error = ""
        sent = []

        def set_mac(self, stall, addr):
            from garage_pi.indicators import normalize_mac

            self.sent.append((stall, normalize_mac(addr)))
            return normalize_mac(addr)

    try:
        assert post({"stall": 1, "mac": "aa:bb:cc:dd:ee:ff"})[0] == 409  # no link to the detector
        svc.indicator_link = FakeLink()
        code, body = post({"stall": 2, "mac": "B4:3A:45:6B:FF:90"})
        assert code == 200 and body["mac"] == "b4:3a:45:6b:ff:90"
        assert post({"stall": 3, "mac": ""})[0] == 200
        assert post({"stall": 1, "mac": "bogus"})[0] == 400
        assert FakeLink.sent == [(2, "b4:3a:45:6b:ff:90"), (3, "")]
        assert svc.snapshot()["indicators"]["stalls"][0]["mac"] is None
        assert "detector_error" in svc.snapshot()["indicators"]
    finally:
        svc.indicator_link = None
        web.close()
        svc.shutdown()
