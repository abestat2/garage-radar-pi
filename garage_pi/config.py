"""Service configuration (data_dir/config.json) and the data-directory layout.

    data_dir/
        config.json                  this file's settings (edited on the web page)
        radar.cfg                    radar CLI configuration sent at start-up
        garage.json                  geometry + car/clutter zones (upload from the laptop app)
        detection_params.json        optional detection parameter overrides
        empty_reference.npz          captured empty-garage reference
        empty_reference_live.npz     self-updated (adapted) reference, saved periodically
        occupancy_calibration.json   per-stall thresholds + neighbour model
        calibration_labels.json      labelled recordings made on the Pi (for re-fitting)
        recordings/                  recordings made on the Pi
        events.csv                   ENTERED / EXITED history
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DEFAULTS: dict = {
    "radar": {
        "cli_port": "auto",  # or e.g. "/dev/ttyUSB0"
        "data_port": "auto",  # or e.g. "/dev/ttyUSB1"
        "cli_baud": 115200,
        "data_baud": 921600,
        "frame_timeout_s": 5.0,  # no frames for this long -> reconnect and reconfigure
        "retry_s": 10.0,
    },
    "gpio": {
        "enabled": True,
        "stall_pins": [17, 27, 22],  # BCM numbers; HIGH = car in stall (user stall numbering 1, 2, 3)
        "ok_pin": 23,  # HIGH = radar running and occupancy ready (stall pins are trustworthy)
        "active_high": True,
        "entering_is_present": False,  # True: raise the pin already on VEHICLE ENTERING (earlier, less certain)
    },
    "leds": {
        "enabled": False,
        "spi_bus": 0,
        "spi_device": 0,  # SPI0 MOSI = GPIO10 carries the WS2812 data
        "count": 8,
        "stall_leds": [[5, 6, 7], [3, 4], [0, 1, 2]],  # LED indexes per stall (stall 1, 2, 3)
        "status_led": None,  # optional LED index showing system health
        "brightness": 0.25,
        "order": "GRB",
        "colors": {
            "OCCUPIED": [255, 0, 0],
            "VEHICLE ENTERING": [0, 255, 0],
            "VEHICLE EXITING": [255, 120, 0],
            "EMPTY": [0, 0, 0],
            "UNKNOWN": [40, 40, 40],
            "SENSOR/CONFIGURATION ERROR": [120, 0, 160],
        },
        "blink_states": ["VEHICLE ENTERING", "VEHICLE EXITING", "SENSOR/CONFIGURATION ERROR"],
        "event_flash_s": 3.0,  # flash on ENTERED (green) / EXITED (amber)
    },
    "mqtt": {
        "enabled": False,
        "host": "",
        "port": 1883,
        "username": "",
        "password": "",
        "base_topic": "garage_radar",
        "discovery_prefix": "homeassistant",
        "device_id": "garage_radar",
        "device_name": "Garage Radar",
    },
    "web": {"enabled": True, "port": 8080, "password": ""},
    "raw_stream": {"enabled": True, "port": 7777},
    "reference": {"save_interval_s": 900.0},
    "recording": {"max_seconds": 900.0},
    "performance": {"max_accumulated_points": 800},  # keeps clustering memory small on a Pi Zero 2 W
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class ServiceConfig:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir).expanduser()
        self.path = self.data_dir / "config.json"
        self.values = copy.deepcopy(DEFAULTS)
        if self.path.exists():
            self.values = _merge(DEFAULTS, json.loads(self.path.read_text()))

    def __getitem__(self, section: str) -> dict:
        return self.values[section]

    def update(self, changes: dict) -> None:
        self.values = _merge(self.values, changes)
        self.save()

    def save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.values, indent=2))
        tmp.replace(self.path)

    def public(self) -> dict:
        """Config for the web page (passwords masked)."""
        v = copy.deepcopy(self.values)
        for sec in ("mqtt", "web"):
            if v[sec].get("password"):
                v[sec]["password"] = "********"
        return v

    # ------------------------------------------------------------------ files
    def file(self, name: str) -> Path:
        return self.data_dir / name

    @property
    def recordings_dir(self) -> Path:
        return self.data_dir / "recordings"


def init_data_dir(data_dir: str | Path) -> ServiceConfig:
    """Create the data directory, seeding it from the repository's tested defaults."""
    cfg = ServiceConfig(data_dir)
    d = cfg.data_dir
    d.mkdir(parents=True, exist_ok=True)
    cfg.recordings_dir.mkdir(exist_ok=True)
    seeds = {
        "radar.cfg": REPO / "configs" / "profiles" / "iwr6843_garage_v2.cfg",
        "empty_reference.npz": REPO / "configs" / "empty_reference.npz",
        "occupancy_calibration.json": REPO / "configs" / "occupancy_calibration.json",
        "garage.json": REPO / "configs" / "garage.json",
        "detection_params.json": REPO / "configs" / "detection_params.json",
    }
    for name, src in seeds.items():
        if not (d / name).exists() and src.exists():
            shutil.copy2(src, d / name)
    if not cfg.path.exists():
        cfg.save()
    return cfg
