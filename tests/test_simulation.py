"""Simulated car arrivals on the ESP pins (web page test button)."""
import pytest

from app.models.garage_state import StallState as S
from garage_pi.gpio_out import StallOutputs


def outputs():
    g = StallOutputs({"stall_pins": [1, 2, 3], "ok_pin": 4}, use_hardware=False)
    g.update([S.OCCUPIED, S.EMPTY, S.EMPTY], True, now=0)
    return g


def pins(g):
    return [p.value for p in g.pins], g.ok_pin.value


def test_arrival_is_empty_then_a_car_then_the_real_level_again():
    g = outputs()
    g.simulate_arrival(0, empty_s=5, present_s=60, now=100)  # stall 1 really has a car
    assert pins(g) == ([0, 0, 0], 1)  # empty first, even over a parked car
    assert g.simulation(0, now=101) == {"phase": "empty", "remaining_s": 4}
    g.refresh(now=105)
    assert pins(g) == ([1, 0, 0], 1)
    assert g.simulation(0, now=110) == {"phase": "car", "remaining_s": 55}
    g.refresh(now=165)
    assert pins(g) == ([1, 0, 0], 1) and g.simulation(0, now=165) is None  # back to the real car


def test_simulation_does_not_change_real_levels_or_edges():
    g = outputs()
    g.simulate_arrival(1, empty_s=5, present_s=60, now=100)
    g.refresh(now=106)
    assert pins(g)[0] == [1, 1, 0]
    # The radar's states keep updating underneath: no edge for the simulated car
    assert g.update([S.OCCUPIED, S.EMPTY, S.EMPTY], True, now=107) == []
    assert g.levels == [True, False, False] and g.describe()[1]["car_present"] is False
    assert g.simulation(1, now=107)["phase"] == "car"
    g.refresh(now=170)
    assert pins(g)[0] == [1, 0, 0]  # the pin follows the real (empty) stall again


def test_simulation_holds_ok_high_while_the_radar_is_not_ready():
    g = outputs()
    g.update([S.UNKNOWN] * 3, False, now=50)
    assert pins(g)[1] == 0
    g.simulate_arrival(2, empty_s=5, present_s=60, now=100)
    g.refresh(now=106)
    assert pins(g) == ([1, 0, 1], 1)
    g.stop_simulations(now=107)
    assert pins(g) == ([1, 0, 0], 0)  # stopped early: real levels and the real (low) OK pin


def test_unknown_stall_is_refused():
    with pytest.raises(ValueError):
        outputs().simulate_arrival(5, 5, 60)


@pytest.fixture
def data_dir(tmp_path):
    from garage_pi.config import init_data_dir

    cfg = init_data_dir(tmp_path / "gr")
    cfg.update({"web": {"port": 0}, "raw_stream": {"port": 0}})
    return cfg


def test_web_command_starts_and_stops_a_simulation(data_dir):
    import json
    import urllib.request

    from garage_pi.service import GarageService
    from garage_pi.web import WebServer

    svc = GarageService(data_dir, lambda s, c: None, use_hardware=False)
    web = WebServer(svc, 0, host="127.0.0.1")
    try:
        def post(body):
            req = urllib.request.Request(f"http://127.0.0.1:{web.port}/api/command", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=5).read())

        assert post({"cmd": "simulate_arrival", "stall": 2})["ok"]
        svc._handle_commands()
        assert svc.gpio.simulation(1)["phase"] == "empty"
        assert svc.snapshot()["system_ok_pin"]["gpio"] == 23
        assert post({"cmd": "stop_simulation"})["ok"]
        svc._handle_commands()
        assert svc.gpio.simulation(1) is None
    finally:
        web.close()
        svc.shutdown()
