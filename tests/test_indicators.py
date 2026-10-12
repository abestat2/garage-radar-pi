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
