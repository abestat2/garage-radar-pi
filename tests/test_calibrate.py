"""Unit tests: heatmap reference/calibration files, the neighbour leak model and the calibration fit."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.detection import calibrate as cal_mod
from app.detection.calibrate import CalibrationError, build_reference, calibrate, fit_neighbour_model
from app.detection.heatmap_occupancy import (DEFAULT_STALL_CAL, EmptyReference, HeatmapLayout,
                                             OccupancyCalibration, StallCalibration, config_fingerprint,
                                             heatmap_enabled, load_occupancy, mean_image)
from app.detection.parameters import DetectionParams
from app.geometry.garage import Garage
from app.models.radar_frame import RadarFrame
from app.radar.config_parser import RadarConfig

ROOT = Path(__file__).resolve().parents[1]
CFG_TEXT = (ROOT / "configs" / "profiles" / "iwr6843_garage_v2.cfg").read_text()


def segment(states, centre, n=40, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.column_stack([rng.normal(c, 0.005, n) for c in centre])
    return tuple(states), vals


def three_stall_samples():
    """EMPTY ~0.05, OCCUPIED ~0.40, +0.03 leak into an empty stall per occupied neighbour."""
    segs, seed = [], 0
    for states in ([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]):
        centre = [0.40 if s else 0.05 + 0.03 * (sum(states) - s) for s in states]
        segs.append(segment([bool(s) for s in states], centre, seed=seed))
        seed += 1
    return segs


# ---------------------------------------------------------------- neighbour model
def test_fit_neighbour_model_recovers_leaks():
    nb = fit_neighbour_model(three_stall_samples(), 3)
    assert nb.base == pytest.approx([0.05] * 3, abs=0.01)
    for i in range(3):
        for j in range(3):
            if i != j:
                assert nb.leak[i][j] == pytest.approx(0.03, abs=0.01) and not nb.leak_assumed[i][j]
        assert nb.occ_med[i] == pytest.approx(0.40, abs=0.01) and nb.occ_low_any[i] < 0.40
    assert any(k.startswith("0:") for k in nb.occ_low)


def test_fit_neighbour_model_copies_unmeasured_leaks_from_mirror_stall():
    # stall 3 is only ever occupied together with stall 1, so stall 1's reading is never seen
    # with stall 3 occupied while stall 1 is empty: that leak is assumed from the mirror (3 <- 1)
    segs = [segment([False, False, False], [0.05] * 3, seed=1),
            segment([True, False, False], [0.40, 0.05, 0.09], seed=2),
            segment([False, True, False], [0.07, 0.40, 0.07], seed=3),
            segment([True, False, True], [0.40, 0.05, 0.40], seed=4)]
    nb = fit_neighbour_model(segs, 3)
    assert bool(nb.leak_assumed[0][2]) and not nb.leak_assumed[2][0]
    assert nb.leak[0][2] == pytest.approx(nb.leak[2][0]) and nb.leak[2][0] > 0.02


def test_fit_neighbour_model_needs_both_states():
    only_empty = [segment([False, False, False], [0.05] * 3)]
    with pytest.raises(CalibrationError, match="stall 1"):
        fit_neighbour_model(only_empty, 3)
    with pytest.raises(CalibrationError):
        fit_neighbour_model([(tuple([False] * 3), np.zeros((0, 3)))], 3)


def test_neighbour_model_conditional_thresholds():
    nb = fit_neighbour_model(three_stall_samples(), 3)
    alone = nb.calibration_for(1, [False, False, False])
    crowded = nb.calibration_for(1, [True, False, True])
    assert crowded.threshold > alone.threshold and crowded.empty_level > alone.empty_level
    assert alone.empty_level < alone.threshold < alone.occupied_level
    assert nb.key(1, [True, False, True]) == "OO" and nb.to_dict()["base"] == nb.base


# ---------------------------------------------------------------- files
def test_occupancy_calibration_round_trip(tmp_path):
    nb = fit_neighbour_model(three_stall_samples(), 3)
    cal = OccupancyCalibration("fp", [StallCalibration(0.2, 0.05, 0.4)] * 3, "notes", nb)
    cal.save(tmp_path / "c.json")
    again = OccupancyCalibration.load(tmp_path / "c.json")
    assert again.fingerprint == "fp" and again.notes == "notes" and again.stalls[0].threshold == 0.2
    assert again.neighbours.base == nb.base
    bare = OccupancyCalibration("fp2", [DEFAULT_STALL_CAL])
    bare.save(tmp_path / "b.json")
    assert OccupancyCalibration.load(tmp_path / "b.json").neighbours is None


def test_empty_reference_save_load_combine(tmp_path):
    a, b = np.array([[1.0, 5.0]], np.float32), np.array([[3.0, 2.0]], np.float32)
    ref = EmptyReference.combine([a, b], "fp", 20, ["one", "two"])
    assert ref.image.tolist() == [[3.0, 5.0]]
    ref.save(tmp_path / "ref.npz")
    again = EmptyReference.load(tmp_path / "ref.npz")
    assert again.fingerprint == "fp" and again.frames == 20 and again.sources == ["one", "two"]
    assert again.image.tolist() == ref.image.tolist()


def test_config_fingerprint_and_heatmap_enabled():
    cfg = RadarConfig(CFG_TEXT)
    fp = config_fingerprint(cfg)
    assert len(fp) == 12 and fp == config_fingerprint(RadarConfig(CFG_TEXT))
    changed = RadarConfig(CFG_TEXT)
    changed.set_arg("frameCfg", (), 4, 100)
    assert config_fingerprint(changed) != fp
    cfar_only = RadarConfig(CFG_TEXT)
    cfar_only.set_arg("cfarCfg", ("-1", "0"), 7, 20)
    assert config_fingerprint(cfar_only) == fp  # CFAR does not change the heatmap's meaning
    assert heatmap_enabled(cfg) and not heatmap_enabled(RadarConfig("guiMonitor -1 1 1 0 0 0 1"))
    assert not heatmap_enabled(RadarConfig(""))


def test_heatmap_layout_image_and_mean_image():
    layout = HeatmapLayout.from_config(RadarConfig(CFG_TEXT))
    assert layout.num_range_bins == 256 and layout.num_azimuth_antennas == 8
    assert layout.beam_half_width_sin == pytest.approx(0.18)
    good = np.ones(layout.num_range_bins * 8, np.complex64)
    img = layout.image(good)
    assert img.shape[0] == layout.num_range_bins and img.dtype == np.float32
    assert layout.image(None) is None and layout.image(good[:-1]) is None
    f = RadarFrame(1, 0.0)
    f.az_heatmap = good
    none_frame = RadarFrame(2, 0.1)
    mean, n = mean_image([f, none_frame, f], layout)
    assert n == 2 and mean.shape == img.shape
    assert mean_image([none_frame], layout) == (None, 0)


def test_load_occupancy_reads_available_files(tmp_path):
    cfg, params, garage = RadarConfig(CFG_TEXT), DetectionParams(), Garage()
    bare = load_occupancy(garage, cfg, params, tmp_path)
    assert bare.reference is None and bare.calibration is None and bare.problem()
    layout = HeatmapLayout.from_config(cfg)
    ref = EmptyReference(np.zeros((layout.num_range_bins, 64), np.float32), config_fingerprint(cfg), 5, ["x"])
    ref.save(tmp_path / "empty_reference.npz")
    OccupancyCalibration(ref.fingerprint, [StallCalibration(0.2, 0.05, 0.4)] * 3).save(tmp_path / "occupancy_calibration.json")
    full = load_occupancy(garage, cfg, params, tmp_path)
    assert full.reference.frames == 5 and len(full.calibration.stalls) == 3


# ---------------------------------------------------------------- build_reference / calibrate
def fake_recording(cfg_text=CFG_TEXT, frames=()):
    return SimpleNamespace(config_text=cfg_text, frames=list(frames))


def heat_frame(t, level=1.0):
    layout = HeatmapLayout.from_config(RadarConfig(CFG_TEXT))
    f = RadarFrame(1, t)
    f.az_heatmap = np.full(layout.num_range_bins * layout.num_azimuth_antennas, level, np.complex64)
    return f


def test_build_reference_success_and_errors():
    recs = [fake_recording(frames=[heat_frame(0), heat_frame(1)]), fake_recording(frames=[heat_frame(0, 2.0)])]
    ref, cfg = build_reference(recs, ["closed", "open"])
    assert ref.frames == 3 and ref.sources == ["closed", "open"] and cfg.derived().num_tx == 3
    with pytest.raises(CalibrationError, match="no empty-garage"):
        build_reference([], [])
    with pytest.raises(CalibrationError, match="different chirp"):
        build_reference([recs[0], fake_recording(cfg_text=CFG_TEXT.replace("frameCfg 0 2 16 0 250", "frameCfg 0 2 16 0 100"),
                                                 frames=[heat_frame(0)])], ["a", "b"])
    with pytest.raises(CalibrationError, match="no heatmap"):
        build_reference([fake_recording(frames=[RadarFrame(1, 0.0)])], ["blank"])


class FakeOccupancy:
    """Stands in for HeatmapOccupancy: reports the frame's heatmap level as every stall's change."""

    def __init__(self, *a, **k):
        self.cfg = a[1]

    def reset(self):
        pass

    def update(self, az, *rest):
        level = float(az[0].real)
        return SimpleNamespace(ready=True, changed=[level, level, level])

    def thresholds(self, changed=None):
        return [StallCalibration(0.2, 0.05, 0.4)] * 3


@pytest.fixture
def patched(monkeypatch, tmp_path):
    recs = {}

    def fake_load(path, source="raw"):
        return recs[Path(path).name]

    monkeypatch.setattr(cal_mod, "load_recording", fake_load)
    monkeypatch.setattr(cal_mod, "HeatmapOccupancy", FakeOccupancy)
    return recs


def timed_recording(level):
    return fake_recording(frames=[heat_frame(t * 0.25, level) for t in range(40)])


def test_calibrate_end_to_end_with_labelled_segments(patched):
    # level 0.05 => every stall empty, 0.40 => every stall occupied (independent of the neighbours)
    patched["empty"], patched["full"] = timed_recording(0.05), timed_recording(0.40)
    labels = {"reference": ["empty"], "segments": [
        {"recording": "empty", "t": [0, 9], "states": ["EMPTY", "EMPTY", "EMPTY"]},
        {"recording": "full", "t": [0, 9], "states": ["OCCUPIED", "OCCUPIED", "OCCUPIED"]}]}
    ref, cal, report = calibrate(labels, "ignored", Garage(), DetectionParams())
    assert ref.frames == 40 and len(cal.stalls) == 3 and cal.neighbours is not None
    assert all(0.05 < s.threshold < 0.40 for s in cal.stalls)
    assert any("joint decoding" in line for line in report) and any("neighbour model" in line for line in report)


def test_calibrate_with_supplied_reference_and_error_paths(patched):
    patched["empty"], patched["full"] = timed_recording(0.05), timed_recording(0.40)
    ref, _ = build_reference([patched["empty"]], ["empty"])
    seg = lambda rec, states, t=(0, 9): {"recording": rec, "t": list(t), "states": states}  # noqa: E731
    ok = {"segments": [seg("empty", ["EMPTY"] * 3), seg("full", ["OCCUPIED"] * 3)]}
    _, cal, _ = calibrate(ok, "x", Garage(), reference=ref)
    assert len(cal.stalls) == 3
    with pytest.raises(CalibrationError, match="no labelled"):
        calibrate({"segments": []}, "x", Garage(), reference=ref)
    ref.fingerprint = "other"
    with pytest.raises(CalibrationError, match="different radar configuration"):
        calibrate(ok, "x", Garage(), reference=ref)
    ref.fingerprint = config_fingerprint(RadarConfig(CFG_TEXT))
    with pytest.raises(CalibrationError, match="states for 3 stalls"):
        calibrate({"segments": [seg("empty", ["EMPTY"])]}, "x", Garage(), reference=ref)
    with pytest.raises(CalibrationError, match="no heatmap frames"):
        calibrate({"segments": [seg("empty", ["EMPTY"] * 3, t=(500, 600))]}, "x", Garage(), reference=ref)
