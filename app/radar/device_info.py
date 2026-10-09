"""Firmware/demo identification via the CLI `version` command."""
from __future__ import annotations

from dataclasses import dataclass, field

from .cli_interface import CliInterface

PLATFORMS = {0xA6843: "xWR6843", 0xA6443: "xWR6443", 0xA1843: "xWR1843", 0xA1642: "xWR1642"}


@dataclass
class DeviceInfo:
    responded: bool = False
    fields: dict[str, str] = field(default_factory=dict)
    raw: str = ""

    @property
    def summary(self) -> str:
        if not self.responded:
            return "No CLI response"
        sdk = self.fields.get("mmWave SDK Version") or self.fields.get("SDK Version", "?")
        plat = self.fields.get("Platform", "?")
        return f"{plat}, SDK {sdk}"

    def to_dict(self) -> dict:
        return {"responded": self.responded, "fields": self.fields, "raw": self.raw}


def query_device_info(cli: CliInterface) -> DeviceInfo:
    r = cli.send("version", timeout=1.5)
    info = DeviceInfo(responded=bool(r.lines), raw=r.text)
    for line in r.lines:
        if ":" in line:
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if k and v and not k.lower().startswith("mmwdemo"):
                info.fields[k] = v
    return info


def platform_name(code: int) -> str:
    return PLATFORMS.get(code, f"0x{code:X}" if code else "unknown")
