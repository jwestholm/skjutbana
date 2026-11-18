"""Opt-in winner-selection hypotheses (all DEFAULT OFF, env read at call time).

One hypothesis per flag. With every flag unset the live path is unchanged:
no candidate/track fields are added and the installed selector key, readiness
and emission are the original ones.

H8a SKJUTBANA_EVIDENCE_OVER_TIMING=1
    Onset timing is only the existing association gate; among eligible tracks
    the winner is the highest ranking value (best_score, or rank_value when a
    rank flag H6/H7a/H7b is on); ties -> more hits, earlier first_seen_ts,
    lower track_id.
H8b SKJUTBANA_BEST_READY_TOPK=<k>
    Among the top-k eligible tracks (current selector ordering) the first READY
    one (same readiness rule) is returned; when none is ready the top track is
    returned (unchanged waiting). Eligibility (producer shot, emitted, window)
    is applied before ranking, so it is never bypassed.
H6  SKJUTBANA_UNCAPPED_RANK=1
    Ranking-only V2 value without the 35 clip (candidate["v26_rank_raw"],
    candidate["v26_rank_excess"] = the part removed by the clip, scaled by the
    same known-hole penalty). Rank of any candidate = score + excess, so the
    same additive confirmation/agreement/bank bonuses apply; legacy candidates
    have excess 0 (rank = score). candidate["score"] is never changed.
H7a SKJUTBANA_HOLE_SHAPE_PRIOR=1 (+ SKJUTBANA_HOLE_RADIUS_PX="0.7,2.5")
    Ranking factor in [0.3, 1] from equivalent radius sqrt(area/pi) and
    circularity (V2 _candidate_features). Missing features -> 1.
H7b SKJUTBANA_SIGNED_DARKENING_PRIOR=1
    Ranking factor in [0.3, 1] from v2_darkening / v2_absdiff at the V2 peak
    pixel (darkening = max(ref-cur, 0), absdiff = |ref-cur| of the same
    images): 0 => pure brightening, 1 => pure darkening. Ratio >= 0.8 -> 1,
    <= 0.2 -> 0.3, linear between. Missing features -> 1.
H9a SKJUTBANA_NO_OLD_HOLE_SNAP=1
    No snap to an old hole's XY when the track centroid is >= 4 px from it
    (SKJUTBANA_NO_OLD_HOLE_SNAP_MIN_DIST_PX) and its own change is strong:
    max(pre_shot_change, v2225_confirm_center_abs, v2225_confirm_darkening)
    of track.last_candidate >= 5 grey (SKJUTBANA_NO_OLD_HOLE_SNAP_MIN_CHANGE).
H11 SKJUTBANA_HOLE_LIKENESS_RANK=1
    Hole-likeness ranking: each V2 candidate gets candidate["h11_hole_snr"] = matched-filter
    response of the signed PRE->POST darkening (2*darkening - absdiff) with the mean radial
    profile of the real airgun + AR15 hole stamps, divided by the local RMS of that signal
    (texture/motion noise).  Tracks rank by the max value over their observations; among
    the eligible tracks whose onset is within SKJUTBANA_H11_ONSET_SLACK_S (0.06) of the
    earliest eligible onset, the highest rank wins (timing stays primary across frames).
    Offline evidence 2026-11-18 (automation/bench/hole_likeness.py, AUDIT_FINDINGS sec. 14).
H9b SKJUTBANA_SOFT_KNOWN_HOLE_PENALTY=1
    Known-hole penalty x0.6/x0.8/x0.9 instead of x0.15/x0.4/x0.7, in both the
    legacy and V2 candidate paths.
"""
from __future__ import annotations

import math
import os
from typing import Any

DEFAULT_KNOWN_HOLE_FACTORS = (0.15, 0.4, 0.7)
SOFT_KNOWN_HOLE_FACTORS = (0.6, 0.8, 0.9)
V2_SCORE_MIN = 3.6
V2_SCORE_MAX = 35.0


def _flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "") or default)
    except ValueError:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _f(value: Any, default: float | None = None) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def hole_likeness_rank() -> bool:
    return _flag("SKJUTBANA_HOLE_LIKENESS_RANK")


def h11_onset_slack_s():
    """H11 onset window (s) around the earliest eligible onset; None when H11 is off.
    SKJUTBANA_H11_ONSET_SLACK_S (default 0.06 s ~ 1.5 camera frames)."""
    if not hole_likeness_rank():
        return None
    return max(0.0, _float_env("SKJUTBANA_H11_ONSET_SLACK_S", 0.06))


def evidence_over_timing() -> bool:
    """H8a, implied by H11 (hole-likeness must be able to change the winner)."""
    return _flag("SKJUTBANA_EVIDENCE_OVER_TIMING") or hole_likeness_rank()


def best_ready_topk() -> int:
    try:
        k = int(float(os.environ.get("SKJUTBANA_BEST_READY_TOPK", "0") or 0))
    except ValueError:
        return 0
    return k if k > 1 else 0


def uncapped_rank() -> bool:
    return _flag("SKJUTBANA_UNCAPPED_RANK")


def hole_shape_prior() -> bool:
    return _flag("SKJUTBANA_HOLE_SHAPE_PRIOR")


def signed_darkening_prior() -> bool:
    return _flag("SKJUTBANA_SIGNED_DARKENING_PRIOR")


def no_old_hole_snap() -> bool:
    return _flag("SKJUTBANA_NO_OLD_HOLE_SNAP")


def soft_known_hole_penalty() -> bool:
    return _flag("SKJUTBANA_SOFT_KNOWN_HOLE_PENALTY")


def rank_flags_on() -> bool:
    return uncapped_rank() or hole_shape_prior() or signed_darkening_prior() or hole_likeness_rank()


def active_flags() -> list[str]:
    names = []
    if evidence_over_timing():
        names.append("H8a")
    if best_ready_topk():
        names.append("H8b")
    if uncapped_rank():
        names.append("H6")
    if hole_shape_prior():
        names.append("H7a")
    if signed_darkening_prior():
        names.append("H7b")
    if no_old_hole_snap():
        names.append("H9a")
    if soft_known_hole_penalty():
        names.append("H9b")
    if hole_likeness_rank():
        names.append("H11")
    if hole_mf_proposals():
        names.append("H12")
    return names


def known_hole_factors() -> tuple[float, float, float]:
    return SOFT_KNOWN_HOLE_FACTORS if soft_known_hole_penalty() else DEFAULT_KNOWN_HOLE_FACTORS


def known_hole_factor(dist: float, duplicate_radius_px: float) -> float:
    near, mid, far = known_hole_factors()
    if dist <= duplicate_radius_px * 0.5:
        return near
    if dist <= duplicate_radius_px:
        return mid
    if dist <= duplicate_radius_px * 1.5:
        return far
    return 1.0


# ---------------------------------------------------------------------------
# Ranking value (H6/H7a/H7b)
# ---------------------------------------------------------------------------


def v2_rank_raw(raw_score: float) -> float:
    """H6: the V2 formula without the upper 35 clip (lower 3.6 clip kept)."""
    return float(max(V2_SCORE_MIN, float(raw_score)))


def hole_radius_range() -> tuple[float, float]:
    text = os.environ.get("SKJUTBANA_HOLE_RADIUS_PX", "0.7,2.5") or "0.7,2.5"
    try:
        lo, hi = (float(p) for p in text.split(",", 1))
    except ValueError:
        lo, hi = 0.7, 2.5
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0.0 or hi < lo:
        lo, hi = 0.7, 2.5
    return lo, hi


def shape_factor(candidate: dict) -> float:
    area = _f(candidate.get("area"))
    circ = _f(candidate.get("circularity"))
    factor = 1.0
    if area is not None and area > 0.0:
        lo, hi = hole_radius_range()
        r_eq = math.sqrt(area / math.pi)
        if r_eq > hi:
            factor *= hi / r_eq
        elif r_eq < lo:
            factor *= r_eq / lo
    if circ is not None:
        factor *= 0.6 + 0.4 * min(1.0, max(0.0, circ / 0.6))
    return float(min(1.0, max(0.3, factor)))


def darkening_factor(candidate: dict) -> float:
    dark = _f(candidate.get("v2_darkening"))
    absdiff = _f(candidate.get("v2_absdiff"))
    if dark is None or absdiff is None or absdiff <= 1e-6:
        return 1.0
    ratio = min(1.0, max(0.0, dark / absdiff))
    if ratio >= 0.8:
        return 1.0
    if ratio <= 0.2:
        return 0.3
    return float(0.3 + 0.7 * (ratio - 0.2) / 0.6)


# Mean radial profile (r = 0..14 camera px) of the real hole stamps (59 airgun + 8 AR15,
# /data/skjutbana/hole_library, normalised to peak 1), computed 2026-11-18.
H11_STAMP_PROFILE = (0.9237, 0.8437, 0.6733, 0.462, 0.2442, 0.1045, 0.046, 0.017, 0.0068,
                     0.0031, 0.0015, 0.0011, 0.0009, 0.0009, 0.0006)
_H11_KERNEL = None


def h11_kernel():
    """Zero-mean, unit-norm 29x29 matched filter from H11_STAMP_PROFILE."""
    global _H11_KERNEL
    if _H11_KERNEL is None:
        import numpy as np
        yy, xx = np.mgrid[-14:15, -14:15]
        rr = np.hypot(xx, yy)
        k = np.interp(rr, np.arange(len(H11_STAMP_PROFILE)), H11_STAMP_PROFILE, right=0.0)
        inside = rr <= 14
        k[inside] -= k[inside].mean()
        k[~inside] = 0.0
        _H11_KERNEL = (k / np.sqrt((k ** 2).sum())).astype(np.float32)
    return _H11_KERNEL


def h11_snr_map(darkening, absdiff):
    """Matched-filter SNR map of the signed change (positive = darker after the shot)."""
    import cv2
    import numpy as np
    signed = (2.0 * np.asarray(darkening, dtype=np.float32) - np.asarray(absdiff, dtype=np.float32))
    signed = signed - cv2.blur(signed, (41, 41))
    mf = cv2.filter2D(signed, cv2.CV_32F, h11_kernel())
    # Noise from an ANNULUS (61x61 window minus the central 21x21): the blob itself must not
    # inflate its own noise, otherwise every isolated blob saturates at the same SNR
    # (seen 2026-11-18: noise candidates ~13-14 regardless of amplitude).  Floor 0.5 grey.
    sq = signed * signed
    big, small = 61, 21
    ring = (cv2.blur(sq, (big, big)) * (big * big) - cv2.blur(sq, (small, small)) * (small * small))
    ring = np.maximum(ring / float(big * big - small * small), 0.0)
    noise = np.maximum(np.sqrt(ring), 0.5)
    return mf / noise


def hole_mf_proposals() -> int:
    """H12 SKJUTBANA_HOLE_MF_PROPOSALS=<n>: add the n strongest matched-filter peaks
    (h11_snr_map local maxima, SNR >= SKJUTBANA_HOLE_MF_MIN_SNR, default 8) as extra FAST
    proposals.  Recall hypothesis; meant to be evaluated together with H11 ranking."""
    try:
        return max(0, int(float(os.environ.get("SKJUTBANA_HOLE_MF_PROPOSALS", "0") or 0)))
    except ValueError:
        return 0


def h12_peaks(snr, existing_xy, n: int, valid=None) -> list:
    """Top-n local maxima of the SNR map not within 4 px of an existing proposal."""
    import cv2
    import numpy as np
    min_snr = _float_env("SKJUTBANA_HOLE_MF_MIN_SNR", 8.0)
    dil = cv2.dilate(snr, np.ones((9, 9), np.uint8))
    mask = (snr >= dil) & (snr >= min_snr)
    if valid is not None:
        mask &= valid.astype(bool)
    ys, xs = np.nonzero(mask)
    order = np.argsort(-snr[ys, xs])
    taken = [tuple(p) for p in existing_xy]
    out = []
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        if any((x - a) ** 2 + (y - b) ** 2 <= 16 for a, b in taken):
            continue
        out.append((float(snr[y, x]), x, y))
        taken.append((x, y))
        if len(out) >= n:
            break
    return out


import threading as _threading

_H11_TLS = _threading.local()


def _cached_snr(darkening, absdiff):
    """One SNR map per (darkening, absdiff) array pair per thread (extractor + merge reuse)."""
    key = (id(darkening), id(absdiff))
    if getattr(_H11_TLS, "key", None) == key and getattr(_H11_TLS, "ref", None) is darkening:
        return _H11_TLS.snr
    snr = h11_snr_map(darkening, absdiff)
    _H11_TLS.key, _H11_TLS.ref, _H11_TLS.snr = key, darkening, snr
    return snr


H11_MAP_VERSION = "hole_morph_v2map_v1"


def attach_h11(candidates: list, darkening, absdiff, x0: int, y0: int, snr=None) -> None:
    """Hole-morphology evidence, V2-map source (the variant H11 consumes).

    Matched filter on the V2 extractor's registered + photometrically normalised change maps
    (2*darkening - absdiff = ref - cur), annulus-normalised.  Contract fields per candidate:
    ev_hole_morph_snr / _valid / _version.  Outside the map = left UNMEASURED.  Runs when
    H11 is on or SKJUTBANA_H11_FULL_ROI_MAP=1 (shadow).  The per-candidate PATCH source
    (hole_morphology_evidence.py) lost most of the gain in replay (single noisy PRE frame,
    2026-11-18) and stays a separate shadow feature."""
    if not candidates or not (hole_likeness_rank() or os.environ.get("SKJUTBANA_H11_FULL_ROI_MAP") == "1"):
        return
    from src.engine.candidate_evidence import CandidateEvidence
    from src.engine.hole_morphology_evidence import EVIDENCE_FAMILY

    if snr is None:
        snr = _cached_snr(darkening, absdiff)
    h, w = snr.shape
    for candidate in candidates:
        try:
            px = int(round(float(candidate["camera_x"]) - x0))
            py = int(round(float(candidate["camera_y"]) - y0))
        except (KeyError, TypeError, ValueError):
            continue
        ya, yb, xa, xb = max(0, py - 2), min(h, py + 3), max(0, px - 2), min(w, px + 3)
        if yb > ya and xb > xa:
            observation = CandidateEvidence(EVIDENCE_FAMILY, H11_MAP_VERSION, float(snr[ya:yb, xa:xb].max()))
            observation.write_fields(candidate, "ev_hole_morph", "snr")


def candidate_rank_value(candidate: dict) -> float:
    """Ranking-only value of one candidate under the active rank flags."""
    if hole_likeness_rank():
        # Hole-morphology evidence (src/engine/hole_morphology_evidence.py), measured for every
        # candidate source in the worker. None = UNMEASURED: never treated as a bad hole.
        return _f(candidate.get("ev_hole_morph_snr"))
    value = _f(candidate.get("score"), 0.0)
    if uncapped_rank():
        value += max(0.0, _f(candidate.get("v26_rank_excess"), 0.0))
    if hole_shape_prior():
        value *= shape_factor(candidate)
    if signed_darkening_prior():
        value *= darkening_factor(candidate)
    return float(value)


def observe_track_rank(track: Any, candidate: dict) -> None:
    """Tracking seam: track.rank_value = max rank over owned observations.

    Only called while a rank flag is on; best_score is never touched.
    """
    value = candidate_rank_value(candidate)
    if value is None:
        return
    previous = _f(getattr(track, "rank_value", None))
    track.rank_value = float(value if previous is None else max(previous, value))


def track_rank_value(track: Any) -> float:
    if rank_flags_on():
        value = _f(getattr(track, "rank_value", None))
        if value is not None:
            return value
    return float(_f(getattr(track, "best_score", 0.0), 0.0))


# ---------------------------------------------------------------------------
# H9a
# ---------------------------------------------------------------------------


def track_own_change(track: Any) -> float | None:
    cand = getattr(track, "last_candidate", None) or {}
    values = [_f(cand.get(k)) for k in ("pre_shot_change", "v2225_confirm_center_abs", "v2225_confirm_darkening")]
    values = [v for v in values if v is not None]
    return max(values) if values else None


def suppress_old_hole_snap(track: Any, dist: float) -> bool:
    if not no_old_hole_snap():
        return False
    if float(dist) < _float_env("SKJUTBANA_NO_OLD_HOLE_SNAP_MIN_DIST_PX", 4.0):
        return False
    change = track_own_change(track)
    return change is not None and change >= _float_env("SKJUTBANA_NO_OLD_HOLE_SNAP_MIN_CHANGE", 5.0)
