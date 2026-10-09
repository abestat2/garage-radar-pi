"""Fit the parked-car occupancy model (empty-garage reference + thresholds) from labelled recordings.

Shared by the laptop tool (tools/build_occupancy_model.py) and the Pi's calibration web page, so
both always produce identical models.

    labels = {"reference": [recording names, garage EMPTY],
              "segments": [{"recording": name, "t": [t0, t1], "states": ["OCCUPIED", "EMPTY", ...]}]}

The numbers are in-sample (thresholds are fitted to the same recordings): check them on new
recordings before trusting them.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from app.geometry.garage import Garage
from app.radar.config_parser import RadarConfig
from app.recording.replay import load_recording

from .heatmap_occupancy import (
    EmptyReference, HeatmapLayout, HeatmapOccupancy, NeighbourModel, OccupancyCalibration, StallCalibration,
    config_fingerprint, mean_image,
)
from .parameters import DetectionParams


class CalibrationError(Exception):
    pass


def fit_neighbour_model(seg_samples, nst: int) -> NeighbourModel:
    """Leak model from labelled segments: an empty stall's reading = base + sum of leaks from occupied
    neighbours (least squares).  Leaks that no segment measures are copied from the mirror stall."""
    base, spread, leak, assumed = [], [], [], []
    occ_low, occ_low_any, occ_med = {}, [], []
    for i in range(nst):
        empty_rows, empty_vals, occ_by_key, occ_all = [], [], {}, []
        for states, vals in seg_samples:
            if not len(vals):
                continue
            key = "".join("O" if states[j] else "E" for j in range(nst) if j != i)
            if states[i]:
                occ_by_key.setdefault(key, []).extend(vals[:, i])
                occ_all.extend(vals[:, i])
            else:
                empty_rows += [[1.0] + [float(states[j]) for j in range(nst) if j != i]] * len(vals)
                empty_vals += list(vals[:, i])
        if not empty_vals or not occ_all:
            raise CalibrationError(f"stall {i + 1} needs at least one EMPTY and one OCCUPIED labelled recording")
        A, y = np.array(empty_rows), np.array(empty_vals)
        others = [j for j in range(nst) if j != i]
        measured = [bool(A[:, 1 + k].any()) for k in range(len(others))]
        cols = [0] + [1 + k for k in range(len(others)) if measured[k]]
        coef, *_ = np.linalg.lstsq(A[:, cols], y, rcond=None)
        row, arow, c = [0.0] * nst, [False] * nst, 1
        for k, j in enumerate(others):
            if measured[k]:
                row[j] = float(max(coef[c], 0.0))
                c += 1
            else:
                arow[j] = True
        base.append(float(max(coef[0], 0.0)))
        e0 = y[(A[:, 1:] == 0).all(axis=1)] if len(y) else y
        spread.append(float(max(np.percentile(e0, 99) - np.median(e0), 0.015)) if len(e0) else 0.03)
        leak.append(row)
        assumed.append(arow)
        for key, v in occ_by_key.items():
            if len(v) >= 20:
                occ_low[f"{i}:{key}"] = float(np.percentile(v, 1))
        occ_low_any.append(float(np.percentile(occ_all, 1)))
        occ_med.append(float(np.median(occ_all)))
    for i in range(nst):  # unmeasured leaks: take the mirror stall's (i <-> nst-1-i) measured value
        for j in range(nst):
            if assumed[i][j]:
                mi, mj = nst - 1 - i, nst - 1 - j
                leak[i][j] = leak[mi][mj] if not assumed[mi][mj] else max(max(r) for r in leak)
    return NeighbourModel(base, spread, leak, assumed, occ_low, occ_low_any, occ_med)


def build_reference(recordings: list, names: list[str]) -> tuple[EmptyReference, RadarConfig]:
    if not recordings:
        raise CalibrationError("no empty-garage recordings")
    cfg = RadarConfig(recordings[0].config_text)
    layout, fp = HeatmapLayout.from_config(cfg), config_fingerprint(cfg)
    imgs, n = [], 0
    for name, r in zip(names, recordings):
        if config_fingerprint(RadarConfig(r.config_text)) != fp:
            raise CalibrationError(f"{name} uses a different chirp configuration")
        img, k = mean_image(r.frames, layout)
        if img is None:
            raise CalibrationError(f"{name} has no heatmap frames (is the heatmap enabled in the radar config?)")
        imgs.append(img)
        n += k
    return EmptyReference.combine(imgs, fp, n, list(names)), cfg


def _segment(rec, t):
    t0 = rec.frames[0].timestamp
    return [f for f in rec.frames if t[0] <= f.timestamp - t0 <= t[1]]


def calibrate(labels: dict, rec_dir: str | Path, garage: Garage, params: DetectionParams | None = None,
              reference: EmptyReference | None = None) -> tuple[EmptyReference, OccupancyCalibration, list[str]]:
    """Returns (reference, calibration, report lines).  `reference` overrides building one from
    labels["reference"] (e.g. a reference captured live on the Pi)."""
    prm = params or DetectionParams()
    rec_dir = Path(rec_dir)
    cache: dict = {}

    def rec(name):
        if name not in cache:
            cache[name] = load_recording(rec_dir / name, source="raw")
        return cache[name]

    report: list[str] = []
    if reference is None:
        names = labels.get("reference", [])
        reference, cfg = build_reference([rec(n) for n in names], names)
    else:
        if not labels.get("segments"):
            raise CalibrationError("no labelled recordings")
        cfg = RadarConfig(rec(labels["segments"][0]["recording"]).config_text)
        if config_fingerprint(cfg) != reference.fingerprint:
            raise CalibrationError("the reference was captured with a different radar configuration")
    fp = reference.fingerprint
    report.append(f"reference: {reference.frames} heatmaps from {len(reference.sources)} empty scene(s), fingerprint {fp}")

    occ = HeatmapOccupancy(garage, cfg, reference, None, prm.occ_change_db, prm.occ_window_frames,
                           prm.occ_assumed_height, prm.entry_approach_m, prm.occ_edge_margin, adapt_tau_s=0)
    nst = len(garage.stalls)
    samples = {i: {"EMPTY": [], "OCCUPIED": []} for i in range(nst)}
    seg_samples = []
    for seg in labels["segments"]:
        r = rec(seg["recording"])
        if len(seg["states"]) != nst:
            raise CalibrationError(f"{seg['recording']}: {len(seg['states'])} states for {nst} stalls")
        occ.reset()
        vals = np.array([o.changed for f in _segment(r, seg["t"]) if (o := occ.update(f.az_heatmap)).ready])
        if not len(vals):
            raise CalibrationError(f"{seg['recording']} t{seg['t']}: no heatmap frames")
        for i, st in enumerate(seg["states"]):
            samples[i][st].extend(vals[:, i].tolist())
        seg_samples.append((tuple(s == "OCCUPIED" for s in seg["states"]), vals))
        report.append(f"  {seg['recording']} t{seg['t']}: " +
                      "  ".join(f"S{i + 1} {st[:3]} {np.median(vals[:, i]) * 100:4.0f}%" for i, st in enumerate(seg["states"])))
    nb = fit_neighbour_model(seg_samples, nst)

    stalls = []
    for i in range(nst):
        e, o = np.array(samples[i]["EMPTY"]), np.array(samples[i]["OCCUPIED"])
        e_hi, o_lo = np.percentile(e, 99), np.percentile(o, 1)
        thr = (e_hi + o_lo) / 2 if o_lo > e_hi else (np.median(e) + np.median(o)) / 2
        stalls.append(StallCalibration(float(thr), float(np.median(e)), float(np.median(o))))
        report.append(f"  {garage.stalls[i].name}: empty median {np.median(e) * 100:4.1f}% (99th {e_hi * 100:4.1f}%), "
                      f"occupied median {np.median(o) * 100:4.1f}% (1st {o_lo * 100:4.1f}%), "
                      f"unconditional margin {(o_lo - e_hi) * 100:+5.1f} pts")
    cal = OccupancyCalibration(fp, stalls, notes="in-sample fit", neighbours=nb)

    report.append("neighbour model (rise of an EMPTY stall's reading when a neighbour is occupied):")
    for i in range(nst):
        leaks = ", ".join(f"S{j + 1} {nb.leak[i][j] * 100:+.1f}{' (assumed)' if nb.leak_assumed[i][j] else ''}"
                          for j in range(nst) if j != i)
        report.append(f"  S{i + 1}: base {nb.base[i] * 100:.1f}%, leaks {leaks}")
    for i in range(nst):
        row = []
        for combo in itertools.product([False, True], repeat=nst - 1):
            occ_list = list(combo[:i]) + [False] + list(combo[i:])
            row.append(f"{nb.key(i, occ_list)}: {nb.calibration_for(i, occ_list).threshold * 100:4.1f}%")
        report.append(f"  S{i + 1} thresholds by neighbours: " + "  ".join(row))
    est = HeatmapOccupancy(garage, cfg, reference, cal, prm.occ_change_db, prm.occ_window_frames,
                           prm.occ_assumed_height, prm.entry_approach_m, prm.occ_edge_margin, adapt_tau_s=0)
    errors = total = 0
    for (states, vals), seg in zip(seg_samples, labels["segments"]):
        wrong = sum(tuple(v[i] >= t.threshold for i, t in enumerate(est.thresholds(list(v)))) != states for v in vals)
        errors += wrong
        total += len(vals)
        if wrong:
            report.append(f"  decode errors: {seg['recording']} t{seg['t']}: {wrong}/{len(vals)} frames")
    report.append(f"joint decoding, in-sample: {errors}/{total} frames wrong")
    return reference, cal, report
