"""Serial-port discovery and the data-port reader thread.

COM numbers are never hard-coded.  Ports are enumerated and their role (CLI vs
data) is guessed from the USB descriptor text:

* IWR6843LEVM / AOP EVM (Silicon Labs CP2105 dual UART):
    "Enhanced COM Port" -> CLI,   "Standard COM Port" -> data
* XDS110-based EVMs (e.g. IWR6843ISK):
    "Application/User UART" -> CLI,   "Auxiliary Data Port" -> data

The guess is only a default; the user can pick any port in the UI.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import serial
from serial.tools import list_ports

DEFAULT_CLI_BAUD = 115200
DEFAULT_DATA_BAUD = 921600

_CLI_HINTS = ("enhanced com port", "application/user", "user uart")
_DATA_HINTS = ("standard com port", "auxiliary data port")
# never guess these as radar ports (e.g. "Standard Serial over Bluetooth link")
_EXCLUDE = ("bluetooth", "com0com")


@dataclass
class PortInfo:
    device: str
    description: str
    hwid: str
    role: str = ""  # "cli", "data" or ""

    @property
    def label(self) -> str:
        role = f" [{self.role.upper()}?]" if self.role else ""
        return f"{self.device} - {self.description}{role}"


def guess_role(description: str) -> str:
    d = description.lower()
    if any(x in d for x in _EXCLUDE):
        return ""
    if any(h in d for h in _CLI_HINTS):
        return "cli"
    if any(h in d for h in _DATA_HINTS):
        return "data"
    return ""


def scan_ports() -> list[PortInfo]:
    ports = []
    for p in sorted(list_ports.comports(), key=lambda p: p.device):
        desc = p.description or ""
        ports.append(PortInfo(p.device, desc, p.hwid or "", guess_role(desc)))
    return ports


def suggest_ports(ports: list[PortInfo]) -> tuple[str | None, str | None]:
    cli = next((p.device for p in ports if p.role == "cli"), None)
    data = next((p.device for p in ports if p.role == "data"), None)
    return cli, data


class SerialManager:
    """Owns the CLI and data serial ports and the data reader thread."""

    def __init__(self):
        self.cli: serial.Serial | None = None
        self.data: serial.Serial | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.bytes_received = 0
        self.settings: dict = {}

    @property
    def is_open(self) -> bool:
        return bool(self.cli and self.cli.is_open and self.data and self.data.is_open)

    def open(self, cli_port: str, data_port: str, cli_baud: int = DEFAULT_CLI_BAUD, data_baud: int = DEFAULT_DATA_BAUD):
        self.close()
        if cli_port == data_port:
            raise ValueError("CLI and data ports must be different for the TI demos.")
        self.cli = serial.Serial(cli_port, cli_baud, timeout=0.1)
        try:
            self.data = serial.Serial(data_port, data_baud, timeout=0.02)
            try:  # Windows: enlarge the driver buffer so bursts (heatmap TLVs) cannot overrun it
                self.data.set_buffer_size(rx_size=1 << 18)
            except (AttributeError, serial.SerialException):
                pass
        except Exception:
            self.cli.close()
            self.cli = None
            raise
        self.settings = {"cli_port": cli_port, "cli_baud": cli_baud, "data_port": data_port, "data_baud": data_baud}

    def start_reader(self, on_bytes: Callable[[bytes, float], None], on_error: Callable[[str], None] | None = None):
        if not self.data:
            raise RuntimeError("Data port is not open")
        self.stop_reader()
        self._stop.clear()
        self.data.reset_input_buffer()

        def run():
            while not self._stop.is_set():
                try:
                    # block for up to `timeout` collecting up to 16 KB: few large reads, not many 1-byte ones
                    chunk = self.data.read(max(16384, self.data.in_waiting))
                except (serial.SerialException, OSError, TypeError, AttributeError) as e:
                    if on_error and not self._stop.is_set():
                        on_error(f"Data port read failed: {e}")
                    break
                if chunk:
                    self.bytes_received += len(chunk)
                    on_bytes(chunk, time.time())

        self._thread = threading.Thread(target=run, name="radar-data-reader", daemon=True)
        self._thread.start()

    def stop_reader(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def close(self):
        self.stop_reader()
        for s in (self.cli, self.data):
            try:
                if s and s.is_open:
                    s.close()
            except Exception:
                pass
        self.cli = self.data = None
