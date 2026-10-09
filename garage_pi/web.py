"""Status / configuration / calibration web page and JSON API (standard library only).

    GET  /                          the page (static/index.html)
    GET  /api/status                live snapshot
    GET  /api/config                settings (passwords masked)
    POST /api/config                {"section": {...}}  -> saved; service restarts its outputs
    POST /api/command               {"cmd": "record"|"stop_record"|"fit"|"revert_reference"|"reconnect"|
                                             "reload"|"led_test"|"remove_label", ...}
    GET  /api/recordings            list
    GET  /api/recordings/<name>.zip download one recording (open it in the laptop app)
    POST /api/recordings/<name>/delete
    GET  /api/radar_config          radar.cfg text (the laptop app needs it for the raw stream)
    POST /api/radar_config          replace radar.cfg (text body)
    POST /api/garage                replace garage.json (body = file saved by the laptop app)
    GET  /api/garage                current garage.json
    GET  /api/events.csv            event history
Optional HTTP basic auth (user "admin") when web.password is set.  LAN use only.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import shutil
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from app.geometry.garage import Garage
from app.radar.config_parser import RadarConfig

log = logging.getLogger("garage_pi.web")
STATIC = Path(__file__).resolve().parent / "static"
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def make_handler(service):
    cfg = service.cfg

    class Handler(BaseHTTPRequestHandler):
        server_version = "GarageRadar"

        def log_message(self, fmt, *args):
            log.debug("%s - %s", self.address_string(), fmt % args)

        # ------------------------------------------------------------ helpers
        def _auth_ok(self) -> bool:
            pw = cfg["web"].get("password")
            if not pw:
                return True
            h = self.headers.get("Authorization", "")
            if h.startswith("Basic "):
                try:
                    user, _, given = base64.b64decode(h[6:]).decode().partition(":")
                    return given == pw
                except Exception:
                    return False
            return False

        def _send(self, code: int, body: bytes, ctype: str = "application/json", extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, default=str).encode())

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length", "0") or 0)
            if n > 5_000_000:
                raise ValueError("request too large")
            return self.rfile.read(n) if n else b""

        def _guard(self) -> bool:
            if self._auth_ok():
                return True
            self._send(401, b'{"error": "login required"}', extra={"WWW-Authenticate": 'Basic realm="garage radar"'})
            return False

        # ------------------------------------------------------------ GET
        def do_GET(self):
            if not self._guard():
                return
            try:
                p = self.path.split("?")[0]
                if p in ("/", "/index.html"):
                    return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if p == "/api/status":
                    return self._json(service.snapshot())
                if p == "/api/config":
                    return self._json(cfg.public())
                if p == "/api/radar_config":
                    return self._send(200, cfg.file("radar.cfg").read_bytes(), "text/plain; charset=utf-8")
                if p == "/api/garage":
                    return self._json(service.garage.to_dict())
                if p == "/api/events.csv":
                    f = cfg.file("events.csv")
                    return self._send(200, f.read_bytes() if f.exists() else b"time,stall,event,detail\n", "text/csv")
                if p == "/api/recordings":
                    out = []
                    for d in sorted(cfg.recordings_dir.glob("*/session.json"), reverse=True):
                        m = json.loads(d.read_text())
                        size = sum(f.stat().st_size for f in d.parent.iterdir())
                        out.append({"name": d.parent.name, "frames": m.get("frames"), "states": m.get("expected_states"),
                                    "size_mb": round(size / 1e6, 2), "created": m.get("created")})
                    return self._json(out)
                m = re.match(r"^/api/recordings/([^/]+)\.zip$", p)
                if m and NAME_RE.match(m.group(1)) and (cfg.recordings_dir / m.group(1)).is_dir():
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                        for f in (cfg.recordings_dir / m.group(1)).iterdir():
                            z.write(f, f"{m.group(1)}/{f.name}")
                    return self._send(200, buf.getvalue(), "application/zip",
                                      {"Content-Disposition": f'attachment; filename="{m.group(1)}.zip"'})
                self._json({"error": "not found"}, 404)
            except Exception as e:
                log.exception("GET %s", self.path)
                self._json({"error": str(e)}, 500)

        # ------------------------------------------------------------ POST
        def do_POST(self):
            if not self._guard():
                return
            try:
                p = self.path.split("?")[0]
                body = self._body()
                if p == "/api/command":
                    c = json.loads(body or b"{}")
                    cmd = c.get("cmd")
                    if cmd == "record":
                        states = [str(s).upper() for s in c.get("states", [])]
                        name = re.sub(r"[^A-Za-z0-9_-]+", "_", c.get("name") or "pi_recording")
                        service.commands.put(("record", name, states, float(c.get("seconds", 40)), bool(c.get("reference"))))
                    elif cmd == "remove_label":
                        service.commands.put(("remove_label", str(c.get("name"))))
                    elif cmd in ("stop_record", "fit", "revert_reference", "reconnect", "reload", "led_test"):
                        service.commands.put((cmd,))
                    else:
                        return self._json({"error": f"unknown command {cmd!r}"}, 400)
                    return self._json({"ok": True})
                if p == "/api/config":
                    changes = json.loads(body or b"{}")
                    for sec in ("mqtt", "web"):  # keep the stored password when the masked value comes back
                        if changes.get(sec, {}).get("password") == "********":
                            changes[sec].pop("password")
                    cfg.update(changes)
                    return self._json({"ok": True, "note": "saved - restart the service to apply GPIO/LED/MQTT/port changes"})
                if p == "/api/radar_config":
                    text = body.decode()
                    RadarConfig(text).derived()  # validate it parses
                    cfg.file("radar.cfg").write_text(text)
                    service.commands.put(("reload",))
                    return self._json({"ok": True, "note": "radar.cfg replaced; a different chirp needs a new reference"})
                if p == "/api/garage":
                    g = Garage.from_dict(json.loads(body))
                    g.save(cfg.file("garage.json"))
                    service.commands.put(("reload",))
                    return self._json({"ok": True, "stalls": len(g.stalls), "zones": len(g.ignore_zones)})
                m = re.match(r"^/api/recordings/([^/]+)/delete$", p)
                if m and NAME_RE.match(m.group(1)) and (cfg.recordings_dir / m.group(1)).is_dir():
                    service.commands.put(("remove_label", m.group(1)))
                    shutil.rmtree(cfg.recordings_dir / m.group(1))
                    return self._json({"ok": True})
                self._json({"error": "not found"}, 404)
            except Exception as e:
                log.exception("POST %s", self.path)
                self._json({"error": str(e)}, 400)

    return Handler


class WebServer:
    def __init__(self, service, port: int, host: str = "0.0.0.0"):
        self.httpd = ThreadingHTTPServer((host, port), make_handler(service))
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="web", daemon=True)
        self._thread.start()
        log.info("web page on port %s", self.port)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
