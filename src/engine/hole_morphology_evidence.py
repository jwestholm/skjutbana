"""Hole-morphology evidence family (v1) — matched filter with the real-hole stamp profile.

One evidence family in the evidence chain (ROADMAP "NEXT — broaden physical evidence"):
for every candidate, from every source (V2 FAST/FULL, v24 tile probe, legacy), measure on a
small causal patch pair — the producer shot's PRE frame and the candidate's current frame —
how much the PRE->POST change looks like a real pellet hole:

    PRE' = gain*PRE + offset, fitted on the annulus 10..30 px, optional best of 3x3 integer shifts (SKJUTBANA_HOLE_EVIDENCE_SHIFT=1, ~9x cost)
           (local photometric + registration compensation; v1 without it lost most of the
           full-map variant's gain, 2026-11-18)
    signed  = PRE' - CUR                                  positive = darker after the shot
    mf      = <signed[29x29 centre], K>                   K = zero-mean, unit-norm radial kernel
                                                         from the mean profile of the real
                                                         airgun + AR15 stamps (hole library)
    noise   = robust std of `signed` in the annulus 10..30 px (texture / motion / compression)
    snr     = mf / max(noise, 0.5)

PATCH source of the family (shadow/diagnostic + nightly-training feature; the V2-map source in
winner_selection_flags.attach_h11 is the one H11 ranks by). Contract per candidate (never
changes `score`, never vetoes):
    ev_hole_morph_patch_snr      float, present only when measured
    ev_hole_morph_patch_valid    1.0 measured / 0.0 unavailable
    ev_hole_morph_patch_missing  reason when unavailable (no_pre, border, shape, no_xy)
    ev_hole_morph_patch_version  EVIDENCE_VERSION
Replay 2026-11-18 (H11 ranking by this source): AI-test 22-23/78 vs 29 with the V2-map source.
Per job: cost in ms in the worker's stages (key "hole_morph_evidence").

Runs inside the async detector worker (shot_async_v2224._run_job) after all candidate sources
are merged, so the main thread pays nothing.  Enabled by SKJUTBANA_HOLE_EVIDENCE=1 (shadow).
Missing evidence is UNKNOWN, not negative evidence.
"""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

EVIDENCE_FAMILY = "hole_morphology"
EVIDENCE_VERSION = "hole_morph_mf_v2"
# Mean radial profile r = 0..14 camera px of the real stamps (59 airgun + 8 AR15), peak 1.
STAMP_PROFILE = (0.9237, 0.8437, 0.6733, 0.462, 0.2442, 0.1045, 0.046, 0.017, 0.0068,
                 0.0031, 0.0015, 0.0011, 0.0009, 0.0009, 0.0006)
HALF = 30          # patch 61x61
KHALF = 14         # kernel 29x29
_KERNEL = None
_ANNULUS = None


def enabled() -> bool:
    """PATCH source, shadow only (H11 consumes the V2-map source in winner_selection_flags)."""
    return os.environ.get("SKJUTBANA_HOLE_EVIDENCE") == "1"


def _kernel() -> np.ndarray:
    global _KERNEL, _ANNULUS
    if _KERNEL is None:
        yy, xx = np.mgrid[-KHALF:KHALF + 1, -KHALF:KHALF + 1]
        rr = np.hypot(xx, yy)
        k = np.interp(rr, np.arange(len(STAMP_PROFILE)), STAMP_PROFILE, right=0.0)
        inside = rr <= KHALF
        k[inside] -= k[inside].mean()
        k[~inside] = 0.0
        _KERNEL = (k / np.sqrt((k ** 2).sum())).astype(np.float32)
        py, px = np.mgrid[-HALF:HALF + 1, -HALF:HALF + 1]
        pr = np.hypot(px, py)
        _ANNULUS = (pr >= 10) & (pr <= HALF)
    return _KERNEL


def measure(pre: np.ndarray | None, cur: np.ndarray | None, candidates: list[dict[str, Any]]) -> dict[str, float]:
    """Attach the evidence fields to every candidate in place; returns {"ms", "measured"}."""
    started = time.perf_counter()
    kernel = _kernel()
    measured = 0
    ok_frames = (isinstance(pre, np.ndarray) and isinstance(cur, np.ndarray)
                 and pre.ndim == 2 and pre.shape == getattr(cur, "shape", None))
    h, w = (pre.shape if ok_frames else (0, 0))
    for cand in candidates or []:
        cand["ev_hole_morph_patch_version"] = EVIDENCE_VERSION
        if not ok_frames:
            cand["ev_hole_morph_patch_valid"] = 0.0
            cand["ev_hole_morph_patch_missing"] = "no_pre" if not isinstance(pre, np.ndarray) else "shape"
            continue
        try:
            x = int(round(float(cand["camera_x"])))
            y = int(round(float(cand["camera_y"])))
        except (KeyError, TypeError, ValueError):
            cand["ev_hole_morph_patch_valid"] = 0.0
            cand["ev_hole_morph_patch_missing"] = "no_xy"
            continue
        if not (HALF <= x < w - HALF and HALF <= y < h - HALF):
            cand["ev_hole_morph_patch_valid"] = 0.0
            cand["ev_hole_morph_patch_missing"] = "border"
            continue
        if not (HALF + 1 <= x < w - HALF - 1 and HALF + 1 <= y < h - HALF - 1):
            cand["ev_hole_morph_patch_valid"] = 0.0
            cand["ev_hole_morph_patch_missing"] = "border"
            continue
        b = cur[y - HALF:y + HALF + 1, x - HALF:x + HALF + 1].astype(np.float32)
        bring = b[_ANNULUS]
        best = None
        shifts = (-1, 0, 1) if os.environ.get("SKJUTBANA_HOLE_EVIDENCE_SHIFT") == "1" else (0,)
        for dy in shifts:
            for dx in shifts:
                a = pre[y + dy - HALF:y + dy + HALF + 1, x + dx - HALF:x + dx + HALF + 1].astype(np.float32)
                aring = a[_ANNULUS]
                va = float(aring.var())
                gain = float(((aring - aring.mean()) * (bring - bring.mean())).mean() / va) if va > 1e-3 else 1.0
                gain = min(2.0, max(0.5, gain))
                offset = float(bring.mean() - gain * aring.mean())
                resid = float(((gain * aring + offset - bring) ** 2).mean())
                if best is None or resid < best[0]:
                    best = (resid, a, gain, offset)
        _resid, a, gain, offset = best
        signed = gain * a + offset - b
        centre = signed[HALF - KHALF:HALF + KHALF + 1, HALF - KHALF:HALF + KHALF + 1]
        mf = float((centre * kernel).sum())
        ring = signed[_ANNULUS]
        noise = 1.4826 * float(np.median(np.abs(ring - np.median(ring))))
        cand["ev_hole_morph_patch_snr"] = mf / max(noise, 0.5)
        cand["ev_hole_morph_patch_valid"] = 1.0
        cand.pop("ev_hole_morph_patch_missing", None)
        measured += 1
    return {"ms": (time.perf_counter() - started) * 1000.0, "measured": float(measured)}


def pre_for_shot(scanner: Any, shot_id: int) -> np.ndarray | None:
    """The producer shot's immutable PRE (V2.22.3/V2.22.5 shot-local snapshot)."""
    pre = (getattr(scanner, "_v2225_pre_by_shot", None) or {}).get(int(shot_id))
    if isinstance(pre, np.ndarray):
        return pre if pre.ndim == 2 else None
    snap = getattr(scanner, "pre_shot_snapshot", None)
    return snap if isinstance(snap, np.ndarray) and snap.ndim == 2 else None
