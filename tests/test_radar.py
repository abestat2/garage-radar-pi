"""Unit tests: radar packet parser, CLI config, CLI interface, device info and serial manager (no hardware)."""
import struct
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import serial

from app.models.radar_point import empty_points
from app.radar import packet_parser as pp
from app.radar import serial_manager as sm
from app.radar.cli_interface import CliInterface, CliResponse
from app.radar.config_parser import DerivedParams, RadarConfig, _fmt
from app.radar.device_info import DeviceInfo, platform_name, query_device_info

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "profiles" / "iwr6843_garage_v2.cfg"


# ---------------------------------------------------------------- packet parser
def sample_points(n=5):
    p = empty_points(n)
    p["x"], p["y"], p["z"] = np.linspace(-1, 1, n), np.linspace(1, 5, n), np.linspace(0, 1, n)
    p["doppler"] = np.linspace(-1, 1, n)
    p["snr"], p["noise"] = 12.5, 3.0
    return p


def packet_with_tlvs(tlvs, frame_number=1, num_obj=0):
    body = b"".join(struct.pack(pp.TLV_HDR_FMT, t, len(b)) + b for t, b in tlvs)
    total = pp.HEADER_LEN + len(body)
    return struct.pack(pp.HEADER_FMT, pp.MAGIC, 0x03060000, total, 0xA6843, frame_number, 0, num_obj, len(tlvs), 0) + body


def test_build_and_parse_round_trip():
    pts = sample_points()
    frame = pp.parse_packet(pp.build_packet(7, pts), timestamp=3.5)
    assert frame.frame_number == 7 and frame.timestamp == 3.5 and frame.platform == 0xA6843
    assert frame.tlv_types == [pp.TLV_DETECTED_POINTS, pp.TLV_SIDE_INFO] and not frame.damaged
    assert frame.points["x"] == pytest.approx(pts["x"]) and frame.points["snr"] == pytest.approx(pts["snr"])
    assert len(pp.build_packet(1, pts)) % 32 == 0


def test_build_without_side_info_leaves_nan_snr():
    frame = pp.parse_packet(pp.build_packet(1, sample_points(), include_side_info=False))
    assert frame.tlv_types == [pp.TLV_DETECTED_POINTS] and np.isnan(frame.points["snr"]).all()


def test_incremental_feed_handles_split_chunks_and_garbage():
    parser = pp.PacketParser()
    stream = b"\xff\xfe junk " + pp.build_packet(1, sample_points()) + pp.build_packet(2, sample_points(3))
    frames = []
    for i in range(0, len(stream), 7):
        frames += parser.feed(stream[i:i + 7], timestamp=1.0)
    assert [f.frame_number for f in frames] == [1, 2] and parser.frames_parsed == 2
    assert parser.bytes_discarded >= 8 and parser.bad_packets == 0
    parser.reset()
    assert parser.feed(b"") == []


def test_feed_default_timestamp_is_wall_clock():
    frames = pp.PacketParser().feed(pp.build_packet(1, sample_points()))
    assert abs(frames[0].timestamp - time.time()) < 5


def test_feed_rejects_absurd_length_and_resyncs():
    bad = struct.pack(pp.HEADER_FMT, pp.MAGIC, 0, 0xFFFFFFF, 0, 1, 0, 0, 0, 0)
    parser = pp.PacketParser()
    frames = parser.feed(bad + pp.build_packet(5, sample_points()))
    assert parser.bad_packets == 1 and [f.frame_number for f in frames] == [5]


def test_feed_salvages_damaged_packet_before_next_magic():
    good = pp.build_packet(1, sample_points())
    damaged = good[:-20]  # lost the end of the side-info payload in transit
    next_pkt = pp.build_packet(2, sample_points(2))
    parser = pp.PacketParser()
    frames = parser.feed(damaged + next_pkt)
    assert parser.bad_packets == 1
    assert frames[0].damaged and pp.TLV_DETECTED_POINTS in frames[0].tlv_types
    assert frames[-1].frame_number == 2 and not frames[-1].damaged
    assert parser.frames_salvaged == 1


def test_feed_discards_unsalvageable_packet():
    junk_header = struct.pack(pp.HEADER_FMT, pp.MAGIC, 0, 200, 0, 1, 0, 0, 0, 0)  # claims 0 TLVs
    parser = pp.PacketParser()
    frames = parser.feed(junk_header + b"\x00" * 20 + pp.build_packet(9, sample_points(2)))
    assert [f.frame_number for f in frames] == [9] and parser.bytes_discarded > 0


def test_feed_waits_for_complete_packet():
    pkt = pp.build_packet(1, sample_points())
    parser = pp.PacketParser()
    assert parser.feed(pkt[:30]) == [] and parser.feed(pkt[30:-1]) == []
    assert len(parser.feed(pkt[-1:])) == 1


def test_parse_packet_truncated_raises_unless_partial():
    pkt = pp.build_packet(1, sample_points())
    cut = pkt[:pp.HEADER_LEN + 4]
    with pytest.raises(ValueError):
        pp.parse_packet(cut)
    assert pp.parse_packet(cut, partial=True).tlv_types == []
    with pytest.raises(ValueError):
        pp.parse_packet(pkt[:pp.HEADER_LEN + 8 + 10])
    assert pp.parse_packet(pkt[:pp.HEADER_LEN + 8 + 10], partial=True).tlv_types == []


def test_tlv_plausible():
    assert pp._tlv_plausible(pp.TLV_DETECTED_POINTS, 32, 2, 100)
    assert not pp._tlv_plausible(pp.TLV_DETECTED_POINTS, 30, 2, 100)
    assert not pp._tlv_plausible(pp.TLV_SIDE_INFO, 7, 2, 100) and pp._tlv_plausible(pp.TLV_SIDE_INFO, 8, 2, 100)
    assert not pp._tlv_plausible(999, 4, 2, 100) and not pp._tlv_plausible(pp.TLV_STATS, 500, 0, 100)
    assert pp._tlv_plausible(pp.TLV_STATS, 24, 0, 100)


def test_heatmap_tlv_decoded_as_complex():
    iq = np.array([[2, 1], [4, 3]], dtype="<i2")  # (imag, real) pairs
    f = pp.parse_packet(packet_with_tlvs([(pp.TLV_AZIMUTH_STATIC_HEATMAP, iq.tobytes())]))
    assert f.az_heatmap.dtype == np.complex64 and f.az_heatmap.tolist() == [1 + 2j, 3 + 4j]


def test_compressed_points_tlv():
    units = struct.pack("<5f", 0.01, 0.02, 0.5, 0.1, 2.0)
    body = struct.pack("<bbhHH", 0, 10, 4, 20, 5)  # el, az, doppler, range, snr
    f = pp.parse_packet(packet_with_tlvs([(pp.TLV_COMPRESSED_POINTS, units + body)]))
    assert len(f.points) == 1
    r, az = 20 * 0.1, 10 * 0.02
    assert f.points["x"][0] == pytest.approx(r * np.sin(az), abs=1e-4)
    assert f.points["y"][0] == pytest.approx(r * np.cos(az), abs=1e-4)
    assert f.points["doppler"][0] == pytest.approx(2.0) and f.points["snr"][0] == pytest.approx(10.0)
    assert len(pp._parse_compressed_points(b"short")) == 0


def test_target_list_tlv():
    vals = [0.0] * 27
    vals[0:6] = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    vals[26] = 0.9
    payload = struct.pack(pp._TARGET_FMT, 42, *vals)
    f = pp.parse_packet(packet_with_tlvs([(pp.TLV_TARGET_LIST, payload)]))
    t = f.tracks[0]
    assert (t.track_id, t.x, t.y, t.z) == (42, 1.0, 2.0, 3.0) and t.confidence == pytest.approx(0.9)
    assert t.vx == pytest.approx(0.1) and t.vz == pytest.approx(0.3)


def test_unknown_tlvs_listed_not_decoded():
    f = pp.parse_packet(packet_with_tlvs([(pp.TLV_STATS, b"\x00" * 24)]))
    assert f.tlv_types == [6] and f.num_points == 0


# ---------------------------------------------------------------- config parser
CFG = """% comment
sensorStop
channelCfg 15 7 0
profileCfg 0 60 7 7 57.14 0 0 40 1 256 5209 0 0 158 % trailing
frameCfg 0 2 16 0 250 1 0
cfarCfg -1 0 2 32 16 5 0 6 1
cfarCfg -1 1 0 4 2 3 1 10 1
cfarFovCfg -1 0 0.25 9.0
aoaFovCfg -1 -60 60 -40 40
clutterRemoval -1 0
sensorStart
"""


def test_config_commands_and_find():
    c = RadarConfig(CFG, name="x.cfg")
    cmds = c.commands()
    assert cmds[0] == "sensorStop" and "% comment" not in " ".join(cmds) and cmds[-1] == "sensorStart"
    assert c.find("frameCfg")[4] == "250"
    assert c.find("cfarCfg", "-1", "1")[7] == "10"
    assert c.find("nothing") is None and c.find("cfarCfg", "9") is None
    assert len(c.find_all("cfarCfg")) == 2
    assert c.text().endswith("sensorStart\n")


def test_config_save_load_round_trip(tmp_path):
    c = RadarConfig(CFG)
    c.save(tmp_path / "a.cfg")
    again = RadarConfig.load(tmp_path / "a.cfg")
    assert again.name == "a.cfg" and again.lines == c.lines


def test_radar_params_read_and_apply():
    c = RadarConfig(CFG)
    assert c.get_radar_params() == {
        "frame_period_ms": 250.0, "clutter_removal": False, "cfar_range_threshold_db": 6.0,
        "cfar_doppler_threshold_db": 10.0, "cfar_max_range_m": 9.0, "aoa_az_limit_deg": 60.0,
        "aoa_el_limit_deg": 40.0}
    c.apply_radar_params({"frame_period_ms": 100.0, "clutter_removal": True, "cfar_range_threshold_db": 12,
                          "cfar_doppler_threshold_db": 8.5, "cfar_max_range_m": 7.5,
                          "aoa_az_limit_deg": 45, "aoa_el_limit_deg": 30})
    p = c.get_radar_params()
    assert p["frame_period_ms"] == 100.0 and p["clutter_removal"] is True and p["cfar_range_threshold_db"] == 12.0
    assert p["cfar_doppler_threshold_db"] == 8.5 and p["cfar_max_range_m"] == 7.5
    assert p["aoa_az_limit_deg"] == 45.0 and p["aoa_el_limit_deg"] == 30.0
    assert "% trailing" in c.text()  # comments survive edits


def test_set_arg_edge_cases():
    c = RadarConfig("profileCfg 0 60 % note\nmissing 1\n")
    assert c.set_arg("profileCfg", (), 1, 61.5) and c.lines[0] == "profileCfg 0 61.5 % note"
    assert not c.set_arg("profileCfg", (), 9, 1)  # index beyond the line
    assert not c.set_arg("nope", (), 0, 1)
    assert RadarConfig().get_radar_params() == {}


def test_ensure_line_replaces_or_inserts_before_sensorstart():
    c = RadarConfig("a 1\nsensorStart\n")
    c.ensure_line("b", [2, 0.5, True])
    assert c.lines == ["a 1", "b 2 0.5 1", "sensorStart"]
    c.ensure_line("b", [3])
    assert c.lines[1] == "b 3"
    d = RadarConfig("a 1")
    d.ensure_line("z", [1])
    assert d.lines[-1] == "z 1"


def test_fmt():
    assert _fmt(True) == "1" and _fmt(2.50) == "2.5" and _fmt(7) == "7" and _fmt("x") == "x"


def test_derived_params_for_real_profile():
    d = RadarConfig.load(PROFILE).derived()
    assert (d.num_tx, d.num_rx) == (3, 4) and d.adc_samples == 256 and d.frame_period_ms == 250.0
    assert d.frame_rate_hz == pytest.approx(4.0) and d.chirps_per_frame == 48
    assert d.range_resolution_m == pytest.approx(0.076, abs=0.002) and 16 < d.max_range_m < 19
    assert d.max_velocity_mps == pytest.approx(6.4, abs=0.2)
    labels = [k for k, _ in d.summary()]
    assert "Range resolution" in labels and len(labels) == 10


def test_derived_params_without_profile_or_frame():
    assert RadarConfig("").derived() == DerivedParams()
    d = RadarConfig("profileCfg 0 60 7 7 57.14 0 0 40 1 256 5209 0 0 158").derived()
    assert d.range_resolution_m > 0 and d.frame_period_ms == 0.0


# ---------------------------------------------------------------- CLI interface
class FakePort:
    """Minimal pyserial stand-in: replies to each written line from a script."""

    def __init__(self, replies=None, default=b"Done\n"):
        self.replies, self.default = replies or {}, default
        self.written, self.rx = [], b""
        self.is_open = True

    @property
    def in_waiting(self):
        return len(self.rx)

    def reset_input_buffer(self):
        self.rx = b""

    def write(self, data):
        self.written.append(data.decode())
        self.rx += self.replies.get(data.decode().strip(), self.default)

    def read(self, n=1):
        out, self.rx = self.rx[:n], self.rx[n:]
        if not out:
            time.sleep(0.005)
        return out

    def close(self):
        self.is_open = False


def test_cli_send_done_and_error():
    cli = CliInterface(FakePort({"bad": b"Error: nope\n", "plain": b"hello\n"}))
    ok = cli.send("good")
    assert ok.ok and ok.lines == ["Done"] and ok.text == "Done" and cli.port.written == ["good\n"]
    assert not cli.send("bad").ok
    assert not CliInterface(FakePort({"x": b"mmwDemo:/>x: not recognized as a CLI command\n"})).send("x").ok
    tail = cli.send("plain", timeout=0.05)  # no terminator: judged on what arrived
    assert tail.ok and tail.lines == ["hello"]


def test_cli_send_silent_port_times_out_not_ok():
    cli = CliInterface(FakePort(default=b""))
    r = cli.send("version", timeout=0.05)
    assert r == CliResponse("version", False, [])


def test_cli_send_unterminated_tail_is_kept():
    r = CliInterface(FakePort({"v": b"Platform: xWR6843"})).send("v", timeout=0.05)
    assert r.ok and r.lines == ["Platform: xWR6843"]


def test_send_config_progress_and_stop_on_error():
    cfg = RadarConfig("sensorStop\nflushCfg\nbadCmd\nsensorStart\n")
    port = FakePort({"badCmd": b"Error\n"})
    seen = []
    results = CliInterface(port).send_config(cfg, progress=lambda i, n, r: seen.append((i, n, r.ok)))
    assert len(results) == 3 and seen[-1] == (3, 4, False)  # stopped before sensorStart
    port2 = FakePort({"badCmd": b"Error\n"})
    assert len(CliInterface(port2).send_config(cfg, stop_on_error=False)) == 4


def test_start_stop_commands():
    port = FakePort()
    cli = CliInterface(port)
    assert cli.start().ok and cli.stop().ok
    assert port.written == ["sensorStart\n", "sensorStop\n"]


# ---------------------------------------------------------------- device info
def test_device_info_parsing():
    reply = b"mmwDemo:/>version\nPlatform : xWR6843\nmmWave SDK Version : 03.05.00.04\nmmwDemo: x\nDone\n"
    info = query_device_info(CliInterface(FakePort({"version": reply})))
    assert info.responded and info.fields["Platform"] == "xWR6843"
    assert info.summary == "xWR6843, SDK 03.05.00.04" and info.to_dict()["responded"] is True
    assert "mmwDemo" not in info.fields


def test_device_info_no_response_and_fallbacks():
    info = query_device_info(CliInterface(FakePort(default=b"")))
    assert not info.responded and info.summary == "No CLI response"
    assert DeviceInfo(True, {"SDK Version": "1"}).summary == "?, SDK 1"
    assert platform_name(0xA6843) == "xWR6843" and platform_name(0x1234) == "0x1234" and platform_name(0) == "unknown"


# ---------------------------------------------------------------- serial manager
def test_guess_role_and_port_suggestions():
    assert sm.guess_role("Silicon Labs Dual CP210x USB to UART Bridge: Enhanced COM Port") == "cli"
    assert sm.guess_role("CP210x: Standard COM Port") == "data"
    assert sm.guess_role("XDS110 Class Application/User UART") == "cli"
    assert sm.guess_role("XDS110 Class Auxiliary Data Port") == "data"
    assert sm.guess_role("Standard Serial over Bluetooth link") == "" and sm.guess_role("USB Serial") == ""
    ports = [sm.PortInfo("COM3", "x", "h", "cli"), sm.PortInfo("COM4", "y", "h", "data")]
    assert sm.suggest_ports(ports) == ("COM3", "COM4") and sm.suggest_ports([]) == (None, None)
    assert sm.PortInfo("COM3", "desc", "", "cli").label == "COM3 - desc [CLI?]"
    assert sm.PortInfo("COM9", "desc", "").label == "COM9 - desc"


def test_scan_ports_sorts_and_classifies(monkeypatch):
    class P:
        def __init__(self, device, description, hwid=None):
            self.device, self.description, self.hwid = device, description, hwid

    monkeypatch.setattr(sm.list_ports, "comports",
                        lambda: [P("COM5", "Standard COM Port"), P("COM2", None), P("COM3", "Enhanced COM Port", "H")])
    got = sm.scan_ports()
    assert [p.device for p in got] == ["COM2", "COM3", "COM5"]
    assert [p.role for p in got] == ["", "cli", "data"] and got[0].hwid == ""


class OpenablePort(FakePort):
    instances = []

    def __init__(self, name, baud, timeout=None):
        super().__init__()
        self.name, self.baud = name, baud
        self.chunks = []
        self.fail_after = None
        OpenablePort.instances.append(self)

    def set_buffer_size(self, rx_size):
        self.buffer = rx_size  # pylint: disable=attribute-defined-outside-init

    def read(self, n=1):
        if self.fail_after is not None:
            raise serial.SerialException("unplugged")
        if self.chunks:
            return self.chunks.pop(0)
        time.sleep(0.005)
        return b""


@pytest.fixture
def fake_serial(monkeypatch):
    OpenablePort.instances = []
    monkeypatch.setattr(sm.serial, "Serial", OpenablePort)
    return OpenablePort


def test_open_close_and_validation(fake_serial):
    m = sm.SerialManager()
    assert not m.is_open
    with pytest.raises(ValueError):
        m.open("COM3", "COM3")
    m.open("COM3", "COM4")
    assert m.is_open and m.settings == {"cli_port": "COM3", "cli_baud": 115200, "data_port": "COM4",
                                        "data_baud": 921600}
    assert m.data.buffer == 1 << 18
    cli, data = m.cli, m.data
    m.close()
    assert not cli.is_open and not data.is_open and m.cli is None and not m.is_open


def test_open_failure_closes_cli(fake_serial, monkeypatch):
    created = []

    def factory(name, baud, timeout=None):
        if name == "BAD":
            raise serial.SerialException("busy")
        created.append(OpenablePort(name, baud, timeout))
        return created[-1]

    monkeypatch.setattr(sm.serial, "Serial", factory)
    m = sm.SerialManager()
    with pytest.raises(serial.SerialException):
        m.open("COM3", "BAD")
    assert m.cli is None and not created[0].is_open


def test_open_tolerates_missing_buffer_size(fake_serial):
    class NoBuf(OpenablePort):
        def set_buffer_size(self, rx_size):
            raise AttributeError

    sm.serial.Serial = NoBuf
    m = sm.SerialManager()
    m.open("A", "B")
    assert m.is_open
    m.close()


def test_reader_thread_delivers_bytes_and_reports_errors(fake_serial):
    m = sm.SerialManager()
    with pytest.raises(RuntimeError):
        m.start_reader(lambda b, t: None)
    m.open("COM3", "COM4")
    got, errors = [], []
    done = threading.Event()

    def on_bytes(b, t):
        got.append(b)
        if len(got) == 2:
            done.set()

    m.data.chunks = [b"abc", b"de"]
    m.start_reader(on_bytes, errors.append)
    assert done.wait(2) and b"".join(got) == b"abcde" and m.bytes_received == 5
    m.data.fail_after = 0
    m._thread.join(2)
    assert errors and "unplugged" in errors[0]
    m.stop_reader()
    m.close()


def test_stop_reader_suppresses_error_callback(fake_serial):
    m = sm.SerialManager()
    m.open("COM3", "COM4")
    errors = []
    m.start_reader(lambda b, t: None, errors.append)
    m.stop_reader()
    assert errors == [] and m._thread is None
    m.close()
