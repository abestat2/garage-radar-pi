"""Radar byte sources for the service: the live IWR6843LEVM over USB, or a recording (replay).

Both call on_bytes(data, timestamp) from their own thread, exactly like the laptop app.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from app.radar.cli_interface import CliInterface
from app.radar.config_parser import RadarConfig
from app.radar.device_info import DeviceInfo, query_device_info
from app.radar.serial_manager import SerialManager, scan_ports, suggest_ports

log = logging.getLogger("garage_pi.radar")

CP2105 = "10C4:EA70"  # IWR6843LEVM USB-UART bridge


def find_ports(cli: str = "auto", data: str = "auto") -> tuple[str, str]:
    """Resolve "auto" ports.  Uses the CP2105 descriptor text ("Enhanced" = CLI, "Standard" = data);
    on Linux falls back to the bridge's interface order (interface 0 = CLI, 1 = data)."""
    if cli != "auto" and data != "auto":
        return cli, data
    ports = scan_ports()
    c, d = suggest_ports(ports)
    if not (c and d):
        bridge = sorted((p for p in ports if CP2105 in p.hwid.upper()), key=lambda p: (p.hwid, p.device))
        if len(bridge) >= 2:
            c, d = bridge[0].device, bridge[1].device
    if not (c and d):
        raise RuntimeError("IWR6843LEVM not found (no CP2105 Enhanced/Standard ports). Found: "
                           + ", ".join(f"{p.device} [{p.description}]" for p in ports))
    return (c if cli == "auto" else cli), (d if data == "auto" else data)


class SerialRadarLink:
    """Opens the EVM, sends the configuration, streams the data port."""

    kind = "serial"

    def __init__(self, settings: dict, config: RadarConfig):
        self.settings = settings
        self.config = config
        self.serial = SerialManager()
        self.device = DeviceInfo()
        self.ports: tuple[str, str] | None = None
        self.last_error = ""

    def start(self, on_bytes: Callable[[bytes, float], None], on_error: Callable[[str], None]) -> None:
        cli_port, data_port = find_ports(self.settings["cli_port"], self.settings["data_port"])
        self.serial.open(cli_port, data_port, int(self.settings["cli_baud"]), int(self.settings["data_baud"]))
        self.ports = (cli_port, data_port)
        cli = CliInterface(self.serial.cli)
        self.device = query_device_info(cli)
        log.info("radar on %s / %s: %s", cli_port, data_port, self.device.summary)
        results = cli.send_config(self.config)
        bad = [r for r in results if not r.ok]
        if bad:
            self.close()
            raise RuntimeError(f"radar rejected '{bad[0].command}': {bad[0].text}")
        self.serial.start_reader(on_bytes, on_error)

    def close(self) -> None:
        try:
            if self.serial.cli is not None:
                CliInterface(self.serial.cli).stop()
        except Exception:
            pass
        self.serial.close()

    @property
    def bytes_received(self) -> int:
        return self.serial.bytes_received

    def describe(self) -> str:
        return f"{self.ports[0]} / {self.ports[1]} - {self.device.summary}" if self.ports else "not connected"


class ReplayRadarLink:
    """Plays a recording's raw UART bytes (for testing the service without hardware)."""

    kind = "replay"

    def __init__(self, recording_dir, speed: float = 1.0, loop: bool = False):
        from app.recording.session import read_raw_file

        self.path = recording_dir
        self.chunks = read_raw_file(f"{recording_dir}/raw_uart.bin")
        self.speed, self.loop = speed, loop
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.bytes_received = 0
        self.finished = threading.Event()
        self.device = DeviceInfo(True, {"Platform": "REPLAY", "mmWave SDK Version": "-"}, str(recording_dir))

    def start(self, on_bytes, on_error) -> None:
        self._stop.clear()
        self.finished.clear()

        def run():
            while not self._stop.is_set():
                t_rec0, t_wall0 = self.chunks[0][0], time.time()
                for ts, data in self.chunks:
                    if self._stop.is_set():
                        return
                    due = t_wall0 + (ts - t_rec0) / self.speed
                    delay = due - time.time()
                    if delay > 0:
                        time.sleep(delay)
                    self.bytes_received += len(data)
                    on_bytes(data, t_wall0 + (ts - t_rec0))  # recorded timing, even when played faster
                if not self.loop:
                    self.finished.set()
                    return

        self._thread = threading.Thread(target=run, name="replay-link", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def describe(self) -> str:
        return f"replay of {self.path} at {self.speed:g}x"
