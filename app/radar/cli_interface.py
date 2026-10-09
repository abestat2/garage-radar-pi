"""Command-line interface to the TI mmWave demo running on the radar."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import serial

from .config_parser import RadarConfig


@dataclass
class CliResponse:
    command: str
    ok: bool
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class CliInterface:
    def __init__(self, port: serial.Serial):
        self.port = port

    def send(self, command: str, timeout: float = 1.0) -> CliResponse:
        self.port.reset_input_buffer()
        self.port.write((command.strip() + "\n").encode("ascii"))
        lines: list[str] = []
        buf = b""
        deadline = time.time() + timeout
        while time.time() < deadline:
            chunk = self.port.read(max(1, self.port.in_waiting))
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("ascii", errors="replace").strip()
                if line:
                    lines.append(line)
                low = line.lower()
                if low.endswith("done"):
                    return CliResponse(command, True, lines)
                if "error" in low or "not recognized" in low:
                    return CliResponse(command, False, lines)
        # sensorStart/version have no reliable terminator on every firmware
        tail = buf.decode("ascii", errors="replace").strip()
        if tail:
            lines.append(tail)
        return CliResponse(command, bool(lines) and not any("error" in l.lower() for l in lines), lines)

    def send_config(
        self,
        config: RadarConfig,
        progress: Callable[[int, int, CliResponse], None] | None = None,
        stop_on_error: bool = True,
    ) -> list[CliResponse]:
        cmds = config.commands()
        results = []
        for i, cmd in enumerate(cmds):
            timeout = 3.0 if cmd.startswith(("sensorStart", "sensorStop", "flushCfg")) else 1.0
            r = self.send(cmd, timeout=timeout)
            results.append(r)
            if progress:
                progress(i + 1, len(cmds), r)
            if not r.ok and stop_on_error:
                break
            time.sleep(0.01)
        return results

    def start(self) -> CliResponse:
        return self.send("sensorStart", timeout=3.0)

    def stop(self) -> CliResponse:
        return self.send("sensorStop", timeout=3.0)
