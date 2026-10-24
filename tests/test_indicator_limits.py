"""The indicators' stop and full-bar distances: set on the web page, held by the detector, reported back."""
import json
import urllib.error
import urllib.request

import pytest

from garage_pi.indicators import IndicatorLink, IndicatorStatus, check_limits, parse_line


@pytest.fixture
def data_dir(tmp_path):
    from garage_pi.config import init_data_dir

    cfg = init_data_dir(tmp_path / "gr")
    cfg.update({"web": {"port": 0}, "raw_stream": {"port": 0}})
    return cfg


def test_limit_lines_parse():
    assert parse_line("DIST stall=2 min_mm=300 max_mm=2000\r") == {"kind": "DIST", "stall": 2, "min_mm": 300, "max_mm": 2000}
    assert parse_line("DIST stall=1 min_mm=0 max_mm=0") == {"kind": "DIST", "stall": 1, "min_mm": 0, "max_mm": 0}
    # An indicator report carries the limits the indicator is using
    assert parse_line("IND stall=1 state=Present battery_mv=3850 min_mm=300 max_mm=2000") == {
        "stall": 1, "state": "Present", "battery_mv": 3850, "min_mm": 300, "max_mm": 2000}
    for junk in ("DIST stall=1 min_mm=300", "DIST stall=1 max_mm=2000", "DIST stall=1 min_mm=lots max_mm=2",
                 "DIST min_mm=300 max_mm=2000", "IND stall=1 contact=ok min_mm=x"):
        assert parse_line(junk) is None


def test_check_limits():
    assert check_limits(300, 2000) == (300, 2000)
    assert check_limits("300", "2000") == (300, 2000)  # from a web form
    assert check_limits(30, 4000) == (30, 4000)
    assert check_limits(None, None) == (0, 0) and check_limits(0, 0) == (0, 0)  # clear
    for lo, hi in ((29, 2000), (300, 4001), (2000, 300), (300, 300), (300, 0), (0, 2000), (-5, 100)):
        with pytest.raises(ValueError):
            check_limits(lo, hi)
    for lo, hi in (("a", 2000), (300, [1]), (True, 2000), (300.5, "x")):
        with pytest.raises(ValueError):
            check_limits(lo, hi)


def test_status_tracks_wanted_and_active_limits():
    st = IndicatorStatus(3)
    s = st.snapshot()[0]
    assert s["limits_wanted"] is None and s["limits_active"] is None  # nothing reported yet
    assert not st.apply(parse_line("DIST stall=1 min_mm=350 max_mm=1800"))  # not an MQTT-published change
    assert not st.apply(parse_line("DIST stall=2 min_mm=0 max_mm=0"))
    snap = st.snapshot()
    assert snap[0]["limits_wanted"] == {"min_mm": 350, "max_mm": 1800} and snap[0]["limits_active"] is None
    assert snap[1]["limits_wanted"] == {"min_mm": 0, "max_mm": 0}  # reported, but none saved
    assert snap[2]["limits_wanted"] is None

    # What the indicator says it is using arrives with its exchange, and a zero report does not erase it
    st.apply(parse_line("IND stall=1 state=Absent battery_mv=3900 min_mm=300 max_mm=2000"))
    assert st.snapshot()[0]["limits_active"] == {"min_mm": 300, "max_mm": 2000}
    st.apply(parse_line("IND stall=1 contact=ok battery_mv=0 age_s=-1 min_mm=0 max_mm=0"))
    assert st.snapshot()[0]["limits_active"] == {"min_mm": 300, "max_mm": 2000}
    st.apply(parse_line("IND stall=1 state=Absent battery_mv=3900 min_mm=350 max_mm=1800"))
    assert st.snapshot()[0]["limits_active"] == {"min_mm": 350, "max_mm": 1800}
    assert not st.apply(parse_line("DIST stall=9 min_mm=300 max_mm=2000"))  # no such stall


class Port:
    def __init__(self):
        self.written = []

    def write(self, data):
        self.written.append(data)


def test_set_limits_sends_a_command_to_the_detector():
    link = IndicatorLink({"port": "/dev/null"}, IndicatorStatus(3))
    link.close()
    with pytest.raises(RuntimeError):
        link.set_limits(1, 300, 2000)  # no open port
    link._serial = port = Port()
    assert link.set_limits(1, "300", 2000) == (300, 2000)
    assert link.set_limits(3, 0, 0) == (0, 0)
    assert port.written == [b"SETDIST stall=1 min_mm=300 max_mm=2000\n", b"SETDIST stall=3 min_mm=0 max_mm=0\n"]
    for stall, lo, hi in ((0, 300, 2000), (4, 300, 2000), (1, 2000, 300), (1, 300, 9000), (1, "x", 2000)):
        with pytest.raises(ValueError):
            link.set_limits(stall, lo, hi)
    assert len(port.written) == 2  # nothing invalid reached the detector


def test_web_sets_indicator_limits(data_dir):
    from garage_pi.service import GarageService
    from garage_pi.web import WebServer

    svc = GarageService(data_dir, lambda s, c: None, use_hardware=False)
    web = WebServer(svc, 0, host="127.0.0.1")

    def post(body):
        req = urllib.request.Request(f"http://127.0.0.1:{web.port}/api/indicator_limits", json.dumps(body).encode())
        try:
            r = urllib.request.urlopen(req, timeout=5)
            return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    class FakeLink:
        port = "fake"
        error = ""
        sent = []
        connected = True

        def set_limits(self, stall, min_mm, max_mm):
            if not self.connected:
                raise RuntimeError("not connected to the detector on fake")
            lo, hi = check_limits(min_mm, max_mm)
            self.sent.append((stall, lo, hi))
            return lo, hi

    try:
        assert post({"stall": 1, "min_mm": 300, "max_mm": 2000})[0] == 409  # no link to the detector
        svc.indicator_link = link = FakeLink()
        code, body = post({"stall": 2, "min_mm": 350, "max_mm": 1800})
        assert code == 200 and (body["min_mm"], body["max_mm"]) == (350, 1800)
        assert post({"stall": 3})[0] == 200  # blank: back to the built-in values
        assert post({"stall": 1, "min_mm": 2000, "max_mm": 300})[0] == 400
        assert FakeLink.sent[:2] == [(2, 350, 1800), (3, 0, 0)]
        link.connected = False
        assert post({"stall": 1, "min_mm": 300, "max_mm": 2000})[0] == 409
        snap = svc.snapshot()["indicators"]["stalls"][0]
        assert snap["limits_wanted"] is None and snap["limits_active"] is None
    finally:
        svc.indicator_link = None
        web.close()
        svc.shutdown()


def test_indicator_distance_is_a_number_or_nan():
    assert parse_line("IND stall=1 state=Present battery_mv=3850 min_mm=300 max_mm=2000 distance_mm=1450")["distance_mm"] == 1450
    assert parse_line("IND stall=1 state=Absent battery_mv=3850 min_mm=300 max_mm=2000 distance_mm=nan")["distance_mm"] is None
    assert parse_line("IND stall=1 state=Absent battery_mv=3850 distance_mm=far") is None
    assert "distance_mm" not in parse_line("IND stall=1 state=Absent battery_mv=3850")  # older detector firmware

    st = IndicatorStatus(2)
    assert st.snapshot()[0]["distance_mm"] is None
    st.apply(parse_line("IND stall=1 state=Present battery_mv=3850 min_mm=300 max_mm=2000 distance_mm=1450"))
    assert st.snapshot()[0]["distance_mm"] == 1450 and st.snapshot()[1]["distance_mm"] is None
    # The periodic summary is not an exchange, so it neither sets nor clears the distance
    st.apply(parse_line("IND stall=1 contact=ok battery_mv=3850 age_s=3 min_mm=300 max_mm=2000"))
    assert st.snapshot()[0]["distance_mm"] == 1450
    st.apply(parse_line("IND stall=1 state=Present battery_mv=3850 min_mm=300 max_mm=2000 distance_mm=nan"))
    assert st.snapshot()[0]["distance_mm"] is None
