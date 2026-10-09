"""Per-stall state machine with confidence, explanation and ENTERED/EXITED events.

Evidence, strongest first:
  1. Heatmap occupancy (heatmap_occupancy.py): the stall's "changed fraction" of the zero-Doppler
     range-azimuth image vs the empty-garage reference, against a calibrated per-stall threshold.
     This is what sees PARKED cars (the on-chip point cloud mostly does not).
  2. Motion: moving points (Doppler) inside the stall or in the approach zone in front of its door,
     over a short window.  Approaching the radar (negative Doppler) = toward the house wall = entering.
  3. Fallback when the heatmap is unavailable: point-cloud vehicle tracks.  Without the heatmap the
     classifier never reports a confident EMPTY - "no points" is not evidence of an empty stall.

States: ENTERING/EXITING are reported as soon as motion is seen; OCCUPIED needs
`occupancy_persistence_s` of consistent evidence and EMPTY `empty_persistence_s`.
Events: ENTERED when a stall becomes OCCUPIED from EMPTY/ENTERING; EXITED when it becomes EMPTY
from OCCUPIED/EXITING.  Settling from the initial UNKNOWN does not create an event.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.geometry.parking_zones import ParkingStall
from app.models.garage_state import StallEvent, StallResult, StallState, Track

from .heatmap_occupancy import OccupancyReading, StallCalibration
from .parameters import DetectionParams
from .vehicle_detector import is_moving

OCCUPIED_LIKE = (StallState.OCCUPIED, StallState.ENTERING, StallState.EXITING)


@dataclass
class MotionEvidence:
    approaching: int = 0  # moving points with negative Doppler (toward the radar / house wall)
    receding: int = 0


@dataclass
class _StallMemory:
    state: StallState = StallState.UNKNOWN
    since: float | None = None
    pending: StallState | None = None
    pending_since: float | None = None
    settled: bool = False  # left the initial UNKNOWN at least once
    motion_seen: float | None = None  # last time ENTERING/EXITING motion was seen
    history: list[StallState] = field(default_factory=list)


class StallClassifier:
    def __init__(self, n_stalls: int = 3):
        self.mem = [_StallMemory() for _ in range(n_stalls)]

    def reset(self, n_stalls: int | None = None) -> None:
        self.mem = [_StallMemory() for _ in range(n_stalls or len(self.mem))]

    def classify(
        self,
        stalls: list[ParkingStall],
        coverage: list[float],
        xy: np.ndarray,
        tracks: list[Track],
        now: float,
        last_data_time: float | None,
        p: DetectionParams,
        sensor_error: str | None = None,
        occupancy: OccupancyReading | None = None,
        calibration: list[StallCalibration] | None = None,
        motion: list[MotionEvidence] | None = None,
    ) -> tuple[list[StallResult], list[StallEvent]]:
        if len(self.mem) != len(stalls):
            self.reset(len(stalls))
        results, events = [], []
        for i, (stall, cov) in enumerate(zip(stalls, coverage)):
            r, ev = self._one(i, stall, cov, xy, tracks, now, last_data_time, p, sensor_error, occupancy,
                              calibration[i] if calibration else None, motion[i] if motion else MotionEvidence())
            results.append(r)
            if ev:
                events.append(ev)
        return results, events

    # ------------------------------------------------------------------
    def _one(self, i, stall, cov, xy, tracks, now, last_data_time, p, sensor_error, occ, cal, mot):
        m = self.mem[i]
        res = StallResult(stall_index=i, name=stall.name, state=m.state, confidence=0.0,
                          motion_in=mot.approaching, motion_out=mot.receding)
        prev = m.state

        if sensor_error or last_data_time is None or now - last_data_time > p.sensor_timeout_s:
            why = sensor_error or ("no radar data received" if last_data_time is None
                                   else f"no radar data for {now - last_data_time:.1f} s")
            self._enter(m, StallState.SENSOR_ERROR, now)
            res.state, res.explanation = StallState.SENSOR_ERROR, [why]
            return res, None
        if cov < p.min_fov_coverage:
            self._enter(m, StallState.UNKNOWN, now)
            res.state = StallState.UNKNOWN
            res.explanation = [f"only {cov * 100:.0f}% of the stall is inside the radar FOV "
                               f"(need {p.min_fov_coverage * 100:.0f}%)"]
            return res, None

        inside = stall.contains(xy) if len(xy) else np.zeros(0, dtype=bool)
        res.points_inside = int(inside.sum())
        heat_ok = occ is not None and occ.ready and cal is not None

        # ------------------------------------------------ static evidence
        best = self._best_track(tracks, inside, now, p)
        if best is not None:
            res.track_id = best.track_id
            res.width, res.depth, res.height = best.cluster.width, best.cluster.depth, best.cluster.height
        if heat_ok:
            ch, thr = occ.changed[i], cal.threshold
            res.changed, res.threshold = ch, thr
            # hysteresis, but never reaching into the expected EMPTY readings (narrow gaps next to a parked
            # neighbour, where the threshold is conditional)
            band = min(thr * p.occ_hysteresis, 0.5 * max(thr - cal.empty_level, 0.0))
            static = ch >= (thr - band if m.state in OCCUPIED_LIKE else thr + band)
        else:
            static = best is not None and not is_moving(best, p)

        # ------------------------------------------------ motion evidence
        entering = mot.approaching >= p.motion_min_points and mot.approaching >= 2 * mot.receding
        exiting = mot.receding >= p.motion_min_points and mot.receding >= 2 * mot.approaching
        if not heat_ok and best is not None and is_moving(best, p):
            entering = entering or best.velocity[1] > 0
            exiting = exiting or best.velocity[1] < 0
        if entering or exiting:
            m.motion_seen = now

        # ------------------------------------------------ target state
        if entering and m.state not in (StallState.OCCUPIED,):
            target = StallState.ENTERING
        elif exiting and m.state in (StallState.OCCUPIED, StallState.EXITING):
            target = StallState.EXITING
        elif static:
            target = StallState.OCCUPIED
        elif m.state == StallState.ENTERING and m.motion_seen and now - m.motion_seen < p.entering_timeout_s:
            target = StallState.ENTERING  # car still arriving: give the heatmap time to fill in
        elif heat_ok:
            target = StallState.EMPTY
        else:
            target = StallState.UNKNOWN  # no heatmap and no vehicle points: not evidence of EMPTY

        if target in (StallState.ENTERING, StallState.EXITING):
            self._enter(m, target, now)
            m.pending = None
        elif target == m.state:
            m.pending = None
        elif target == StallState.UNKNOWN and m.state != StallState.UNKNOWN and not heat_ok:
            self._enter(m, target, now)
            m.pending = None
        else:
            if m.pending != target:
                m.pending, m.pending_since = target, now
            need = p.occupancy_persistence_s if target == StallState.OCCUPIED else p.empty_persistence_s
            if now - m.pending_since >= need:
                self._enter(m, target, now)
                m.pending = None

        res.state = m.state
        res.persistence = now - m.since if m.since is not None else 0.0
        res.confidence, res.explanation = self._explain(res, best, cov, m, now, p, heat_ok, occ, cal)

        event = None
        if m.state != prev:
            if m.state == StallState.OCCUPIED and prev in (StallState.EMPTY, StallState.ENTERING) and m.settled:
                event = StallEvent(now, i, stall.name, "ENTERED", self._detail(res))
            elif m.state == StallState.EMPTY and prev in (StallState.OCCUPIED, StallState.EXITING) and m.settled:
                event = StallEvent(now, i, stall.name, "EXITED", self._detail(res))
            if m.state in (StallState.EMPTY, StallState.OCCUPIED):
                m.settled = True
        return res, event

    @staticmethod
    def _detail(r: StallResult) -> str:
        if r.changed is not None:
            return f"changed area {r.changed * 100:.0f}% (threshold {r.threshold * 100:.0f}%)"
        return f"{r.points_inside} points"

    @staticmethod
    def _best_track(tracks, inside, now, p):
        best, best_key = None, (0.0, 0)
        for t in tracks:
            if not t.is_vehicle or t.last_seen < now - 1e-9:
                continue
            idx = np.asarray(t.cluster.point_indices, dtype=int)
            if len(idx) == 0 or len(inside) == 0:
                continue
            n_in = int(inside[idx].sum())
            frac = n_in / len(idx)
            if frac >= p.stall_min_fraction and n_in >= p.stall_min_points and (frac, n_in) > best_key:
                best, best_key = t, (frac, n_in)
        return best

    @staticmethod
    def _enter(m: _StallMemory, state: StallState, now: float) -> None:
        if m.state != state:
            m.state, m.since = state, now
        elif m.since is None:
            m.since = now

    @staticmethod
    def _explain(res, best, cov, m, now, p, heat_ok, occ, cal) -> tuple[float, list[str]]:
        exp = []
        if heat_ok:
            exp.append(f"Changed car-zone area {res.changed * 100:.0f}% vs threshold {res.threshold * 100:.0f}% "
                       f"(empty ~{cal.empty_level * 100:.0f}%, parked ~{cal.occupied_level * 100:.0f}%)")
            i = res.stall_index
            if occ.health and i < len(occ.health) and occ.health[i]:
                exp.append("\u26a0 " + occ.health[i])
        else:
            exp.append("Heatmap occupancy unavailable: " + (occ.reason if occ is not None else "not configured"))
        exp.append(f"Moving points in/approaching: {res.motion_in} in, {res.motion_out} out")
        exp.append(f"Points in stall: {res.points_inside}   FOV coverage: {cov * 100:.0f}%")
        if m.pending is not None:
            need = p.occupancy_persistence_s if m.pending == StallState.OCCUPIED else p.empty_persistence_s
            exp.append(f"pending → {m.pending.value} ({now - m.pending_since:.1f}/{need:.1f} s)")
        cov_f = 0.6 + 0.4 * min(1.0, cov)
        s = res.state
        if s in (StallState.UNKNOWN, StallState.SENSOR_ERROR):
            return 0.0, exp
        if s in (StallState.ENTERING, StallState.EXITING):
            n = res.motion_in if s == StallState.ENTERING else res.motion_out
            return round(min(1.0, 0.4 + 0.6 * n / (3.0 * p.motion_min_points)) * cov_f, 3), exp
        if heat_ok:
            thr = res.threshold
            if s == StallState.OCCUPIED:
                span = max(cal.occupied_level - thr, 1e-3)
                c = 0.5 + 0.5 * np.clip((res.changed - thr) / span, 0, 1)
            else:
                span = max(thr - cal.empty_level, 1e-3)
                c = 0.5 + 0.5 * np.clip((thr - res.changed) / span, 0, 1)
            return round(float(c) * cov_f, 3), exp
        # fallback (point-cloud tracks only): modest confidence at best
        if s == StallState.OCCUPIED and best is not None:
            return round(0.5 * cov_f, 3), exp
        return 0.2, exp
