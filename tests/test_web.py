"""Unit tests: the web page / JSON API against a stub service (real HTTP on a loopback port)."""
import base64
import io
import json
import queue
import urllib.error
import urllib.request
import zipfile

import pytest

from garage_pi.config import init_data_dir
from garage_pi.web import STATIC, WebServer
from app.geometry.garage import Garage


class StubService:
    def __init__(self, cfg):
        self.cfg = cfg
        self.commands = queue.Queue()
        self.garage = Garage(width=6, depth=6)
        self.fail = False

    def snapshot(self):
        if self.fail:
            raise RuntimeError("snapshot exploded")
        return {"health": "ok", "events": []}


@pytest.fixture
def env(tmp_path):
    cfg = init_data_dir(tmp_path / "gr")
    svc = StubService(cfg)
    web = WebServer(svc, 0, host="127.0.0.1")
    yield svc, cfg, f"http://127.0.0.1:{web.port}"
    web.close()


def call(url, data=None, headers=None, method=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def post_json(url, obj):
    return call(url, json.dumps(obj).encode(), {"Content-Type": "application/json"})


def make_recording(cfg, name="rec1"):
    d = cfg.recordings_dir / name
    d.mkdir(parents=True)
    (d / "session.json").write_text(json.dumps({"frames": 5, "expected_states": ["OCCUPIED"], "created": "t"}))
    (d / "raw_uart.bin").write_bytes(b"x" * 1000)
    return d


# ---------------------------------------------------------------- GET
def test_page_and_status(env):
    svc, cfg, url = env
    status, body, hdr = call(url + "/")
    assert status == 200 and body == (STATIC / "index.html").read_bytes() and "text/html" in hdr["Content-Type"]
    assert call(url + "/index.html")[0] == 200 and hdr["Cache-Control"] == "no-store"
    assert json.loads(call(url + "/api/status?x=1")[1]) == {"health": "ok", "events": []}
    svc.fail = True
    status, body, _ = call(url + "/api/status")
    assert status == 500 and "exploded" in json.loads(body)["error"]


def test_config_garage_radar_config_and_events(env):
    svc, cfg, url = env
    cfg.update({"mqtt": {"password": "hunter2"}})
    assert json.loads(call(url + "/api/config")[1])["mqtt"]["password"] == "********"
    assert json.loads(call(url + "/api/garage")[1])["width"] == 6
    assert call(url + "/api/radar_config")[1].decode().startswith("%")
    assert call(url + "/api/events.csv")[1] == b"time,stall,event,detail\n"
    cfg.file("events.csv").write_bytes(b"a,b\n")
    status, body, hdr = call(url + "/api/events.csv")
    assert body == b"a,b\n" and hdr["Content-Type"] == "text/csv"
    assert call(url + "/nope")[0] == 404


def test_recordings_list_and_zip_download(env):
    svc, cfg, url = env
    assert json.loads(call(url + "/api/recordings")[1]) == []
    make_recording(cfg, "rec1")
    listing = json.loads(call(url + "/api/recordings")[1])
    assert listing[0]["name"] == "rec1" and listing[0]["frames"] == 5 and listing[0]["size_mb"] >= 0
    status, body, hdr = call(url + "/api/recordings/rec1.zip")
    assert status == 200 and hdr["Content-Type"] == "application/zip" and "rec1.zip" in hdr["Content-Disposition"]
    assert sorted(zipfile.ZipFile(io.BytesIO(body)).namelist()) == ["rec1/raw_uart.bin", "rec1/session.json"]
    assert call(url + "/api/recordings/missing.zip")[0] == 404
    assert call(url + "/api/recordings/bad%20name.zip")[0] == 404


# ---------------------------------------------------------------- POST commands
@pytest.mark.parametrize("payload,expected", [
    ({"cmd": "record", "name": "my run!", "states": ["occupied", "empty"], "seconds": "12", "reference": 1},
     ("record", "my_run_", ["OCCUPIED", "EMPTY"], 12.0, True)),
    ({"cmd": "record"}, ("record", "pi_recording", [], 40.0, False)),
    ({"cmd": "remove_label", "name": "x"}, ("remove_label", "x")),
    ({"cmd": "simulate_arrival", "stall": "2"}, ("simulate_arrival", 2)),
    ({"cmd": "fit"}, ("fit",)),
    ({"cmd": "stop_simulation"}, ("stop_simulation",)),
])
def test_commands_are_queued(env, payload, expected):
    svc, _, url = env
    status, body, _ = post_json(url + "/api/command", payload)
    assert status == 200 and json.loads(body) == {"ok": True}
    assert svc.commands.get_nowait() == expected


def test_unknown_command_and_bad_json(env):
    svc, _, url = env
    status, body, _ = post_json(url + "/api/command", {"cmd": "format_disk"})
    assert status == 400 and "unknown command" in json.loads(body)["error"] and svc.commands.empty()
    assert call(url + "/api/command", b"{broken")[0] == 400
    assert post_json(url + "/api/unknown", {})[0] == 404


def test_config_update_keeps_masked_password(env):
    svc, cfg, url = env
    cfg.update({"mqtt": {"password": "hunter2"}})
    status, body, _ = post_json(url + "/api/config", {"mqtt": {"password": "********", "host": "broker"},
                                                      "web": {"password": "********"}})
    assert status == 200 and "saved" in json.loads(body)["note"]
    assert cfg["mqtt"]["password"] == "hunter2" and cfg["mqtt"]["host"] == "broker"
    post_json(url + "/api/config", {"mqtt": {"password": "new"}})
    assert cfg["mqtt"]["password"] == "new"


def test_radar_config_upload(env):
    svc, cfg, url = env
    status, _, _ = call(url + "/api/radar_config", b"sensorStop\nsensorStart\n")
    assert status == 200 and cfg.file("radar.cfg").read_text() == "sensorStop\nsensorStart\n"
    assert svc.commands.get_nowait() == ("reload",)
    before = cfg.file("radar.cfg").read_text()
    assert call(url + "/api/radar_config", b"\xff\xfe")[0] == 400  # not decodable as text
    assert cfg.file("radar.cfg").read_text() == before


def test_garage_upload(env):
    svc, cfg, url = env
    g = Garage(width=7, depth=5)
    status, body, _ = post_json(url + "/api/garage", g.to_dict())
    assert status == 200 and json.loads(body)["stalls"] == 3
    assert Garage.load(cfg.file("garage.json")).width == 7 and svc.commands.get_nowait() == ("reload",)
    assert post_json(url + "/api/garage", {"width": 1})[0] == 400  # missing keys


def test_recording_delete(env):
    svc, cfg, url = env
    d = make_recording(cfg)
    assert post_json(url + "/api/recordings/rec1/delete", {})[0] == 200
    assert not d.exists() and svc.commands.get_nowait() == ("remove_label", "rec1")
    assert post_json(url + "/api/recordings/rec1/delete", {})[0] == 404


def test_request_body_limit(env):
    import socket

    _, _, url = env
    port = int(url.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(b"POST /api/command HTTP/1.1" + bytes([13, 10]) + b"Host: x" + bytes([13, 10])
                     + b"Content-Length: 6000000" + bytes([13, 10, 13, 10]))
        reply = b""
        while chunk := sock.recv(4096):
            reply += chunk
    assert reply.startswith(b"HTTP/1.0 400") or reply.startswith(b"HTTP/1.1 400")
    assert b"request too large" in reply


# ---------------------------------------------------------------- auth
def test_basic_auth_when_password_set(env):
    svc, cfg, url = env
    cfg.update({"web": {"password": "pw"}})
    status, body, hdr = call(url + "/api/status")
    assert status == 401 and "Basic" in hdr["WWW-Authenticate"]
    assert call(url + "/api/command", b"{}")[0] == 401

    def auth(user, pw):
        return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}

    assert call(url + "/api/status", headers=auth("admin", "pw"))[0] == 200
    assert call(url + "/api/status", headers=auth("anyone", "pw"))[0] == 200
    assert call(url + "/api/status", headers=auth("admin", "wrong"))[0] == 401
    assert call(url + "/api/status", headers={"Authorization": "Basic !!!notbase64"})[0] == 401
    assert call(url + "/api/status", headers={"Authorization": "Bearer pw"})[0] == 401
