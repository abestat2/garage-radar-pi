"""MQTT publishing with Home Assistant MQTT discovery.

Entities created (per stall N, user numbering):
    binary_sensor  <name> Stall N occupied     ON/OFF (device_class occupancy)
    sensor         <name> Stall N state        EMPTY / OCCUPIED / VEHICLE ENTERING / ... (+ attributes)
    event          <name> Stall N              event_types ENTERED / EXITED
and one radar status sensor. With indicators, per stall also:
    sensor         <name> Stall N indicator battery    % (device_class battery)
    sensor         <name> Stall N indicator voltage    V (diagnostic)
    binary_sensor  <name> Stall N indicator connected  (device_class connectivity)
Availability uses a retained LWT on <base>/<id>/availability.

Topics (retained unless noted):
    <base>/<id>/availability           online | offline
    <base>/<id>/stallN/occupied        ON | OFF
    <base>/<id>/stallN/state           state text
    <base>/<id>/stallN/attributes      JSON (confidence, changed, threshold, ...)
    <base>/<id>/stallN/event           JSON {"event_type": "ENTERED"|"EXITED", ...}   (not retained)
    <base>/<id>/stallN/indicator       JSON {"battery_pct", "battery_v", "connected": "ON"|"OFF", ...}
    <base>/<id>/status                 JSON radar/service status
"""
from __future__ import annotations

import json
import logging
import threading

log = logging.getLogger("garage_pi.mqtt")


class MqttPublisher:
    def __init__(self, settings: dict, n_stalls: int, version: str = "", client=None, indicators: bool = False):
        self.s = settings
        self.n = n_stalls
        self.indicators = indicators
        self.version = version
        self.enabled = bool(settings.get("enabled")) and (bool(settings.get("host")) or client is not None)
        self.base = f"{settings.get('base_topic', 'garage_radar')}/{settings.get('device_id', 'garage_radar')}"
        self.connected = False
        self.last_error = ""
        self._lock = threading.Lock()
        self.client = client
        if self.enabled and self.client is None:
            self.client = self._make_client()

    # ------------------------------------------------------------------ connection
    def _make_client(self):
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            self.last_error = "paho-mqtt is not installed"
            log.warning(self.last_error)
            self.enabled = False
            return None
        try:  # paho-mqtt >= 2.0
            c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.s.get("device_id", "garage_radar"))
        except AttributeError:
            c = mqtt.Client(client_id=self.s.get("device_id", "garage_radar"))
        if self.s.get("username"):
            c.username_pw_set(self.s["username"], self.s.get("password") or None)
        c.will_set(f"{self.base}/availability", "offline", qos=1, retain=True)
        c.on_connect = self._on_connect
        c.on_disconnect = self._on_disconnect
        c.reconnect_delay_set(2, 60)
        try:
            c.connect_async(self.s["host"], int(self.s.get("port", 1883)), keepalive=30)
            c.loop_start()
        except Exception as e:
            self.last_error = str(e)
            log.warning("MQTT connect failed: %s", e)
        return c

    def _on_connect(self, client, userdata, flags, rc, *args):
        ok = (rc == 0) if isinstance(rc, int) else not getattr(rc, "is_failure", False)
        self.connected = ok
        if ok:
            self.last_error = ""
            log.info("MQTT connected to %s", self.s.get("host"))
            self.publish_discovery()
            self._pub("availability", "online", retain=True)
        else:
            self.last_error = f"connect refused ({rc})"

    def _on_disconnect(self, *args):
        self.connected = False

    def _pub(self, sub: str, payload, retain: bool = True, topic: str | None = None):
        if not (self.enabled and self.client):
            return
        if not isinstance(payload, str):
            payload = json.dumps(payload)
        try:
            self.client.publish(topic or f"{self.base}/{sub}", payload, qos=1, retain=retain)
        except Exception as e:
            self.last_error = str(e)

    # ------------------------------------------------------------------ discovery
    def discovery_messages(self) -> list[tuple[str, dict]]:
        did, name, pre = self.s.get("device_id", "garage_radar"), self.s.get("device_name", "Garage Radar"), \
            self.s.get("discovery_prefix", "homeassistant")
        device = {"identifiers": [did], "name": name, "model": "IWR6843LEVM garage radar",
                  "manufacturer": "DIY", "sw_version": self.version}
        avail = {"availability_topic": f"{self.base}/availability"}
        out = []
        for i in range(1, self.n + 1):
            st = f"{self.base}/stall{i}"
            out.append((f"{pre}/binary_sensor/{did}/stall{i}_occupied/config", {
                "name": f"Stall {i} occupied", "unique_id": f"{did}_stall{i}_occupied", "device_class": "occupancy",
                "state_topic": f"{st}/occupied", "payload_on": "ON", "payload_off": "OFF",
                "json_attributes_topic": f"{st}/attributes", "device": device, **avail}))
            out.append((f"{pre}/sensor/{did}/stall{i}_state/config", {
                "name": f"Stall {i} state", "unique_id": f"{did}_stall{i}_state", "icon": "mdi:car-info",
                "state_topic": f"{st}/state", "json_attributes_topic": f"{st}/attributes",
                "device": device, **avail}))
            out.append((f"{pre}/event/{did}/stall{i}_event/config", {
                "name": f"Stall {i}", "unique_id": f"{did}_stall{i}_event", "icon": "mdi:garage-variant",
                "state_topic": f"{st}/event", "event_types": ["ENTERED", "EXITED"], "device": device, **avail}))
            if self.indicators:
                out.append((f"{pre}/sensor/{did}/stall{i}_indicator_battery/config", {
                    "name": f"Stall {i} indicator battery", "unique_id": f"{did}_stall{i}_indicator_battery",
                    "device_class": "battery", "unit_of_measurement": "%", "state_class": "measurement",
                    "state_topic": f"{st}/indicator", "value_template": "{{ value_json.battery_pct }}",
                    "json_attributes_topic": f"{st}/indicator", "device": device, **avail}))
                out.append((f"{pre}/sensor/{did}/stall{i}_indicator_voltage/config", {
                    "name": f"Stall {i} indicator voltage", "unique_id": f"{did}_stall{i}_indicator_voltage",
                    "device_class": "voltage", "unit_of_measurement": "V", "state_class": "measurement",
                    "entity_category": "diagnostic", "state_topic": f"{st}/indicator",
                    "value_template": "{{ value_json.battery_v }}", "device": device, **avail}))
                out.append((f"{pre}/binary_sensor/{did}/stall{i}_indicator_connected/config", {
                    "name": f"Stall {i} indicator connected", "unique_id": f"{did}_stall{i}_indicator_connected",
                    "device_class": "connectivity", "entity_category": "diagnostic",
                    "state_topic": f"{st}/indicator", "value_template": "{{ value_json.connected }}",
                    "payload_on": "ON", "payload_off": "OFF", "device": device, **avail}))
        out.append((f"{pre}/sensor/{did}/radar_status/config", {
            "name": "Radar status", "unique_id": f"{did}_radar_status", "icon": "mdi:radar",
            "state_topic": f"{self.base}/status", "value_template": "{{ value_json.health }}",
            "json_attributes_topic": f"{self.base}/status", "entity_category": "diagnostic",
            "device": device, **avail}))
        return out

    def publish_discovery(self) -> None:
        for topic, cfg in self.discovery_messages():
            self._pub("", cfg, retain=True, topic=topic)

    # ------------------------------------------------------------------ state
    def publish_stall(self, i: int, result, present: bool | None) -> None:
        st = f"stall{i + 1}"
        self._pub(f"{st}/state", result.state.value)
        if present is not None:
            self._pub(f"{st}/occupied", "ON" if present else "OFF")
        self._pub(f"{st}/attributes", {
            "confidence": round(result.confidence, 3),
            "changed_pct": None if result.changed is None else round(result.changed * 100, 1),
            "threshold_pct": None if result.threshold is None else round(result.threshold * 100, 1),
            "persistence_s": round(result.persistence, 1), "explanation": result.explanation[:4]})

    def publish_indicator(self, i: int, info: dict) -> None:
        """One stall's indicator: battery and whether the detector is in contact with it."""
        self._pub(f"stall{i + 1}/indicator", {
            "battery_pct": info["battery_pct"], "battery_v": info["battery_v"],
            "connected": "ON" if info["contact"] == "ok" else "OFF", "contact": info["contact"],
            "last_exchange_age_s": info["last_exchange_age_s"]})

    def publish_event(self, e) -> None:
        self._pub(f"stall{e.stall_index + 1}/event",
                  {"event_type": e.kind, "stall": e.stall_index + 1, "detail": e.detail, "timestamp": e.timestamp},
                  retain=False)

    def publish_status(self, status: dict) -> None:
        self._pub("status", status)

    def close(self) -> None:
        if self.enabled and self.client:
            try:
                self._pub("availability", "offline", retain=True)
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
