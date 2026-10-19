"""The recording download/delete endpoints must never reach outside the recordings folder."""
import io
import json
import queue
import urllib.error
import urllib.request
import zipfile

import pytest

from garage_pi.config import init_data_dir
from garage_pi.web import WebServer, recording_dir


class StubService:
    def __init__(self, cfg):
        self.cfg = cfg
        self.commands = queue.Queue()


@pytest.fixture
def env(tmp_path):
    cfg = init_data_dir(tmp_path / "gr")
    cfg.update({"mqtt": {"password": "hunter2"}})
    rec = cfg.recordings_dir / "rec1"
    rec.mkdir()
    (rec / "session.json").write_text(json.dumps({"frames": 1}))
    svc = StubService(cfg)
    web = WebServer(svc, 0, host="127.0.0.1")
    yield svc, cfg, f"http://127.0.0.1:{web.port}"
    web.close()


def request(url, data=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


@pytest.mark.parametrize("name", [".", "..", "...", ".hidden", "../x", "a/b", "", "bad name", "x\\y"])
def test_recording_dir_rejects_unsafe_names(tmp_path, name):
    (tmp_path / ".hidden").mkdir()
    assert recording_dir(tmp_path, name) is None


def test_recording_dir_accepts_real_folders_only(tmp_path):
    (tmp_path / "2026-10-18_101500_run.1").mkdir()
    (tmp_path / "a_file").write_text("x")
    assert recording_dir(tmp_path, "2026-10-18_101500_run.1") == tmp_path / "2026-10-18_101500_run.1"
    assert recording_dir(tmp_path, "a_file") is None and recording_dir(tmp_path, "missing") is None


def test_recording_dir_rejects_a_symlink_that_leaves_the_folder(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "recordings"
    root.mkdir()
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks need privileges on this platform")
    assert recording_dir(root, "link") is None


def test_zip_download_cannot_escape_recordings_dir(env):
    _, cfg, url = env
    for path in ("/api/recordings/...zip", "/api/recordings/....zip", "/api/recordings/%2e%2e.zip"):
        status, body = request(url + path)
        assert status == 404, path
        assert b"hunter2" not in body
    status, body = request(url + "/api/recordings/rec1.zip")  # the legitimate download still works
    assert status == 200 and zipfile.ZipFile(io.BytesIO(body)).namelist() == ["rec1/session.json"]


def test_delete_cannot_remove_the_data_directory(env):
    svc, cfg, url = env
    for path in ("/api/recordings/../delete", "/api/recordings/./delete", "/api/recordings/%2e%2e/delete"):
        status, _ = request(url + path, b"{}")
        assert status == 404, path
    assert cfg.file("config.json").exists() and cfg.file("radar.cfg").exists() and cfg.recordings_dir.exists()
    assert (cfg.recordings_dir / "rec1").exists() and svc.commands.empty()


def test_delete_still_removes_a_real_recording(env):
    svc, cfg, url = env
    assert request(url + "/api/recordings/rec1/delete", b"{}")[0] == 200
    assert not (cfg.recordings_dir / "rec1").exists() and svc.commands.get_nowait() == ("remove_label", "rec1")
