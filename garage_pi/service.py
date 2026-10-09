"""The headless garage service: radar link -> parser -> detection pipeline -> outputs.

Threads: the radar link (bytes in), the service loop (all processing and state, single-threaded),
LEDs, MQTT, the web server and the raw-stream server.  Other threads talk to the loop only through
`self.commands` (a queue) and read `self.snapshot()`.
"""
from __future__ import annotations

import csv
import json
import logging
import queue
import threading
import time
from pathlib import Path

import app.detection.pipeline as pipeline_mod
from app.detection.calibrate import CalibrationError, build_reference, calibrate
from app.detection.heatmap_occupancy import EmptyReference, config_fingerprint, heatmap_enabled, load_occupancy
from app.detection.parameters import DetectionParams
from app.detection.pipeline import DetectionPipeline, stall_results_rows
from app.geometry.garage import Garage
from app.geometry.site import default_garage
from app.models.garage_state import StallState
from app.radar.config_parser import RadarConfig
from app.radar.packet_parser import PacketParser
from app.recording.recorder import Recorder
from app.recording.replay import load_recording

from .config import ServiceConfig
from .gpio_out import StallOutputs
from .leds import LedBar
from .mqtt_ha import MqttPublisher
from .raw_stream import RawStreamServer

log = logging.getLogger("garage_pi")
VERSION = "0.1.0"
LIVE_REF = "empty_reference_live.npz"
CAPTURED_REF = "empty_reference.npz"
LABELS = "calibration_labels.json"


class GarageService:
    def __init__(self, cfg: ServiceConfig, link_factory, use_hardware: bool = True):
        self.cfg = cfg
        self.link_factory = link_factory  # callable(radar settings, RadarConfig) -> link
        self.use_hardware = use_hardware
        self.commands: queue.Queue = queue.Queue()
        self._bytes: queue.Queue = queue.Queue(maxsize=2000)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.started = time.time()
        pipeline_mod.MAX_ACCUMULATED = int(cfg["performance"]["max_accumulated_points"])

        self.link = None
        self.link_error = ""
        self.link_since: float | None = None
        self._next_connect = 0.0
        self.last_frame_wall: float | None = None
        self.frames = 0
        self.frame_times: list[float] = []
        self.parser = PacketParser()
        self.recorder = Recorder()
        self.rec_job: dict | None = None
        self.busy = ""  # long-running job description (calibration fit)
        self.last_report: list[str] = []
        self.events: list[dict] = []
        self.last_result = None
        self._last_live_save = time.time()

        self.load_models()
        n = len(self.garage.stalls)
        self.gpio = StallOutputs(cfg["gpio"], use_hardware)
        self.leds = LedBar(cfg["leds"], n, use_hardware)
        self.mqtt = MqttPublisher(cfg["mqtt"], n, VERSION)
        self.raw = RawStreamServer(int(cfg["raw_stream"]["port"])) if cfg["raw_stream"]["enabled"] else None
        self._published: list = [None] * n
        self._last_status_pub = 0.0

    # ================================================================ models / files
    def load_models(self) -> None:
        f = self.cfg.file
        self.radar_cfg = RadarConfig.load(f("radar.cfg")) if f("radar.cfg").exists() else RadarConfig()
        self.garage = Garage.load(f("garage.json")) if f("garage.json").exists() else default_garage()
        self.params = DetectionParams.load(f("detection_params.json")) if f("detection_params.json").exists() \
            else DetectionParams()
        self.pipeline = DetectionPipeline(self.garage, self.params)
        occ = load_occupancy(self.garage, self.radar_cfg, self.params, self.cfg.data_dir)
        live = f(LIVE_REF)
        self.reference_source = "captured" if occ.reference is not None else "none"
        if live.exists() and occ.reference is not None:
            try:
                lr = EmptyReference.load(live)
                if lr.fingerprint == occ.fingerprint and lr.image.shape == occ.reference.image.shape and \
                        live.stat().st_mtime >= f(CAPTURED_REF).stat().st_mtime:
                    occ.reference = lr
                    self.reference_source = f"self-updated ({time.strftime('%Y-%m-%d %H:%M', time.localtime(live.stat().st_mtime))})"
            except Exception as e:
                log.warning("ignoring %s: %s", live, e)
        self.pipeline.set_occupancy(occ)
        self.pipeline.last_data_time = None

    def save_live_reference(self) -> None:
        occ = self.pipeline.occupancy
        if occ is None or occ.live is None or occ.reference is None or occ.adapted_s <= 0:
            return
        ref = EmptyReference(occ.live.copy(), occ.fingerprint, occ.reference.frames,
                             [s for s in occ.reference.sources if s != "self-updated"] + ["self-updated"])
        tmp = self.cfg.file(LIVE_REF + ".tmp.npz")
        ref.save(tmp)
        tmp.replace(self.cfg.file(LIVE_REF))
        self._last_live_save = time.time()
        log.info("self-updated reference saved (%.0f s of adaptation)", occ.adapted_s)

    # ================================================================ radar link
    def _on_bytes(self, data: bytes, ts: float) -> None:  # link thread
        if self.raw:
            self.raw.broadcast(data)
        if self.recorder.active:
            self.recorder.write_raw(data, ts)
        try:
            self._bytes.put_nowait((data, ts))
        except queue.Full:
            pass

    def _on_link_error(self, msg: str) -> None:
        self.link_error = msg
        self.commands.put(("reconnect",))

    def _ensure_link(self) -> None:
        if self.link is not None or time.time() < self._next_connect:
            return
        try:
            link = self.link_factory(self.cfg["radar"], self.radar_cfg)
            self.parser.reset()
            self.pipeline.reset()
            link.start(self._on_bytes, self._on_link_error)
            self.link, self.link_error, self.link_since = link, "", time.time()
            self.last_frame_wall = None
            log.info("radar link up: %s", link.describe())
        except Exception as e:
            self.link_error = str(e)
            self._next_connect = time.time() + float(self.cfg["radar"]["retry_s"])
            log.warning("radar link failed: %s (retry in %.0f s)", e, float(self.cfg["radar"]["retry_s"]))

    def _drop_link(self, why: str) -> None:
        if self.link is not None:
            log.warning("restarting radar link: %s", why)
            try:
                self.link.close()
            except Exception:
                pass
        self.link, self.link_error = None, why
        self._next_connect = time.time() + 2.0

    # ================================================================ main loop
    def run(self) -> None:
        log.info("garage service %s starting, data dir %s", VERSION, self.cfg.data_dir)
        try:
            while not self._stop.is_set():
                self._ensure_link()
                self._handle_commands()
                self._drain()
                self._housekeeping()
        finally:
            self.shutdown()

    def stop(self) -> None:
        self._stop.set()

    def _drain(self) -> None:
        try:
            data, ts = self._bytes.get(timeout=0.1)
        except queue.Empty:
            return
        batch = [(data, ts)]
        while len(batch) < 200:
            try:
                batch.append(self._bytes.get_nowait())
            except queue.Empty:
                break
        for data, ts in batch:
            for frame in self.parser.feed(data, ts):
                self._process(frame)

    def _process(self, frame) -> None:
        res = self.pipeline.process(frame)
        self.last_result = res
        self.frames += 1
        self.last_frame_wall = time.time()
        self.frame_times = (self.frame_times + [self.last_frame_wall])[-20:]
        if self.recorder.active:
            self.recorder.write_frame(frame, res.state, stall_results_rows(res.state))
        for e in res.state.events:
            self._event(e)
        self._outputs(res.state.stalls)

    def _system_ok(self) -> bool:
        occ = self.pipeline.occupancy
        fresh = self.last_frame_wall is not None and time.time() - self.last_frame_wall < float(self.cfg["radar"]["frame_timeout_s"])
        return bool(self.link is not None and fresh and occ is not None and occ.last is not None and occ.last.ready)

    def _outputs(self, stalls) -> None:
        ok = self._system_ok()
        states = [s.state for s in stalls]
        self.gpio.update(states, ok)
        self.leds.set_states(states, "busy" if (self.busy or self.recorder.active) else ("ok" if ok else "error"))
        for i, s in enumerate(stalls):
            key = (s.state, self.gpio.levels[i] if i < len(self.gpio.levels) else None)
            if key != self._published[i]:
                self.mqtt.publish_stall(i, s, key[1])
                self._published[i] = key

    def _event(self, e) -> None:
        log.info("EVENT %s %s (%s)", e.name, e.kind, e.detail)
        d = {"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e.timestamp)), "stall": e.stall_index + 1,
             "event": e.kind, "detail": e.detail}
        self.events = ([d] + self.events)[:100]
        self.leds.event(e.stall_index, e.kind)
        self.mqtt.publish_event(e)
        if self.recorder.active:
            self.recorder.write_event(e)
        path = self.cfg.file("events.csv")
        new = not path.exists()
        try:
            with open(path, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["time", "stall", "event", "detail"])
                w.writerow([d["time"], d["stall"], d["event"], d["detail"]])
        except OSError as ex:
            log.warning("events.csv: %s", ex)

    def _housekeeping(self) -> None:
        now = time.time()
        timeout = float(self.cfg["radar"]["frame_timeout_s"])
        if self.link is not None and now - (self.last_frame_wall or self.link_since or now) > timeout:
            if getattr(self.link, "finished", None) is not None and self.link.finished.is_set():
                pass  # replay ended: keep the last state
            else:
                self._drop_link(f"no radar frames for {timeout:.0f} s")
        if self.link is None or (self.last_frame_wall and now - self.last_frame_wall > timeout):
            if self.last_result is not None:  # sensor error: hold pins, OK pin low, LEDs show error
                stalls = self.pipeline.check_timeout(now) or self.last_result.state.stalls
                self._outputs(stalls)
            else:
                self.gpio.update([StallState.UNKNOWN] * len(self.garage.stalls), False)
                self.leds.set_states([StallState.SENSOR_ERROR] * len(self.garage.stalls), "error")
        if self.rec_job and now >= self.rec_job["until"]:
            self._finish_recording()
        if now - self._last_live_save > float(self.cfg["reference"]["save_interval_s"]):
            self.save_live_reference()
            self._last_live_save = now
        if now - self._last_status_pub > 30:
            self.mqtt.publish_status(self.status_summary())
            self._last_status_pub = now

    # ================================================================ commands (from web / MQTT)
    def _handle_commands(self) -> None:
        while True:
            try:
                cmd = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                getattr(self, "_cmd_" + cmd[0])(*cmd[1:])
            except Exception as e:
                log.exception("command %s failed", cmd[0])
                self.last_report = [f"ERROR: {cmd[0]}: {e}"]

    def _cmd_reconnect(self) -> None:
        self._drop_link(self.link_error or "reconnect requested")

    def _cmd_reload(self) -> None:
        """Re-read radar.cfg / garage.json / params / reference / calibration (after uploads or a fit)."""
        self.save_live_reference()
        self.load_models()
        self._published = [None] * len(self.garage.stalls)
        self._drop_link("configuration reloaded")

    def _cmd_record(self, name: str, states: list[str], seconds: float, reference: bool) -> None:
        if self.recorder.active:
            raise RuntimeError("a recording is already running")
        if self.link is None:
            raise RuntimeError("the radar is not running")
        seconds = min(float(seconds), float(self.cfg["recording"]["max_seconds"]))
        meta = {"source": f"pi:{self.link.kind}", "garage": self.garage.to_dict(),
                "detection_params": self.params.to_dict(), "expected_states": states,
                "radar_config_name": "radar.cfg", "notes": f"recorded on the Pi; reference={reference}"}
        path = self.recorder.start(self.cfg.recordings_dir, name, meta, self.radar_cfg.text(),
                                   f"states {states}; reference {reference}")
        self.rec_job = {"path": path, "until": time.time() + seconds, "states": states, "reference": reference,
                        "labelled": bool(states) and all(s in ("EMPTY", "OCCUPIED") for s in states)}
        self.last_report = [f"recording {path.name} for {seconds:.0f} s ..."]

    def _cmd_stop_record(self) -> None:
        if self.rec_job:
            self._finish_recording()

    def _finish_recording(self) -> None:
        job, self.rec_job = self.rec_job, None
        path = self.recorder.stop()
        if not job or not path:
            return
        msg = [f"saved recording {path.name}"]
        if job["labelled"]:
            labels = self._labels()
            dur = max(json.loads((path / "session.json").read_text()).get("frames", 0), 1)
            labels["segments"].append({"recording": path.name, "t": [3, 10_000], "states": job["states"]})
            if job["reference"]:
                labels["reference"].append(path.name)
            self._save_labels(labels)
            msg.append(f"added to the calibration set ({dur} frames, states {job['states']})")
            if job["reference"]:
                msg += self._rebuild_reference(labels)
        self.last_report = msg

    def _rebuild_reference(self, labels: dict) -> list[str]:
        recs = [load_recording(self.cfg.recordings_dir / n, source="raw") for n in labels["reference"]]
        ref, cfg = build_reference(recs, labels["reference"])
        if config_fingerprint(cfg) != config_fingerprint(self.radar_cfg):
            raise CalibrationError("the empty recordings used a different radar configuration than radar.cfg")
        ref.save(self.cfg.file(CAPTURED_REF))
        self.cfg.file(LIVE_REF).unlink(missing_ok=True)
        self._cmd_reload()
        return [f"empty-garage reference rebuilt from {len(recs)} recording(s); self-updated copy reset",
                "fit the thresholds once you also have recordings with cars parked"]

    def _cmd_fit(self) -> None:
        """Fit thresholds on a background thread (takes a while on a Pi Zero), then reload."""
        if self.busy:
            raise RuntimeError(f"busy: {self.busy}")
        labels = self._labels()
        self.busy = "fitting thresholds"

        def job():
            try:
                ref, cal, report = calibrate(labels, self.cfg.recordings_dir, self.garage, self.params)
                cal.notes = f"fitted on the Pi {time.strftime('%Y-%m-%d %H:%M')} from {len(labels['segments'])} recordings"
                ref.save(self.cfg.file(CAPTURED_REF))
                cal.save(self.cfg.file("occupancy_calibration.json"))
                self.cfg.file(LIVE_REF).unlink(missing_ok=True)
                self.last_report = report + ["calibration saved and loaded"]
                self.commands.put(("reload",))
            except Exception as e:
                self.last_report = [f"calibration FAILED: {e}"]
            finally:
                self.busy = ""

        threading.Thread(target=job, name="calibration", daemon=True).start()

    def _cmd_revert_reference(self) -> None:
        self.cfg.file(LIVE_REF).unlink(missing_ok=True)
        self._cmd_reload()
        self.last_report = ["reverted to the captured empty-garage reference"]

    def _cmd_remove_label(self, name: str) -> None:
        labels = self._labels()
        labels["segments"] = [s for s in labels["segments"] if s["recording"] != name]
        labels["reference"] = [r for r in labels["reference"] if r != name]
        self._save_labels(labels)

    def _cmd_led_test(self) -> None:
        self.leds.test(6.0)

    def _labels(self) -> dict:
        p = self.cfg.file(LABELS)
        return json.loads(p.read_text()) if p.exists() else {"reference": [], "segments": []}

    def _save_labels(self, labels: dict) -> None:
        self.cfg.file(LABELS).write_text(json.dumps(labels, indent=2))

    # ================================================================ status
    def status_summary(self) -> dict:
        occ = self.pipeline.occupancy
        fps = (len(self.frame_times) - 1) / max(self.frame_times[-1] - self.frame_times[0], 1e-6) \
            if len(self.frame_times) > 1 else 0.0
        health = "ok" if self._system_ok() else ("starting" if self.link is not None else "radar offline")
        return {"health": health, "frame_rate": round(fps, 2), "frames": self.frames,
                "damaged_packets": self.parser.bad_packets, "salvaged_frames": self.parser.frames_salvaged,
                "occupancy": (occ.problem() or "ready") if occ else "off", "version": VERSION}

    def snapshot(self) -> dict:
        res = self.last_result
        occ = self.pipeline.occupancy
        stalls = []
        if res is not None:
            for i, s in enumerate(res.state.stalls):
                stalls.append({"name": s.name, "state": s.state.value, "confidence": round(s.confidence, 3),
                               "changed": None if s.changed is None else round(s.changed * 100, 1),
                               "threshold": None if s.threshold is None else round(s.threshold * 100, 1),
                               "persistence": round(s.persistence, 1), "explanation": s.explanation,
                               "gpio": self.gpio.describe()[i] if i < len(self.gpio.describe()) else None})
        cal = occ.calibration if occ else None
        return {
            "status": self.status_summary(),
            "uptime_s": round(time.time() - self.started),
            "link": {"kind": getattr(self.link, "kind", None), "description": self.link.describe() if self.link else "",
                     "error": self.link_error, "bytes": getattr(self.link, "bytes_received", 0)},
            "stalls": stalls,
            "system_ok_pin": {"gpio": self.gpio.ok_pin_number, "ok": self.gpio.ok},
            "events": self.events[:30],
            "reference": {"source": self.reference_source,
                          "frames": occ.reference.frames if occ and occ.reference else 0,
                          "adapted_s": round(occ.adapted_s) if occ else 0,
                          "problem": occ.problem() if occ else "off",
                          "heatmap_in_config": heatmap_enabled(self.radar_cfg)},
            "calibration": {"notes": cal.notes if cal else "none", "fingerprint_ok": bool(cal and occ and cal.fingerprint == occ.fingerprint),
                            "labels": self._labels(), "report": self.last_report, "busy": self.busy},
            "recording": None if not self.rec_job else {"name": self.rec_job["path"].name,
                                                        "remaining_s": round(self.rec_job["until"] - time.time()),
                                                        "states": self.rec_job["states"], "reference": self.rec_job["reference"]},
            "mqtt": {"enabled": self.mqtt.enabled, "connected": self.mqtt.connected, "error": self.mqtt.last_error},
            "raw_stream": {"port": self.raw.port if self.raw else None, "clients": self.raw.clients if self.raw else 0},
            "garage": {"stalls": [s.name for s in self.garage.stalls], "zones": [z.name for z in self.garage.ignore_zones]},
        }

    def shutdown(self) -> None:
        log.info("shutting down")
        try:
            if self.rec_job:
                self._finish_recording()
            self.save_live_reference()
        except Exception as e:
            log.warning("saving on shutdown failed: %s", e)
        for closer in (lambda: self.link and self.link.close(), self.gpio.close, self.leds.close, self.mqtt.close,
                       lambda: self.raw and self.raw.close()):
            try:
                closer()
            except Exception:
                pass
