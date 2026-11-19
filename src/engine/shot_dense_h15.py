"""H15 (opt-in, default OFF): late dense second opinion.

SKJUTBANA_H15_DENSE=1
    Keep a short ring of viewport-cropped detector frames.  At peak + SKJUTBANA_H15_DELAY_S
    (default 0.6) submit ONE job per shot to the existing detector worker: matched filter
    (winner_selection_flags.h11_kernel, the H11 stamp profile) on D = 1 - median(post)/median(pre)
    over the whole viewport, normalised by local RMS (exactly automation/bench/hole_likeness.dense_peaks).
    PRE  = median of frames in [max(peak-0.4, previous peak+0.35), peak-0.02]
    POST = median of frames in [peak+0.15, min(submit frame, next peak+0.10)]
    The top peak, if it passes the acceptance rule, becomes the shot's decision through an explicit
    H15 authority track (span/hits = the real POST window).  The second audio peak of a CO2 pair
    (<= 0.12 s after another peak) gets no H15 job; it keeps the ordinary path.  While the H15 result is pending (until SKJUTBANA_H15_DEADLINE_S,
    default 1.3 s after the peak) the shot is NOT resolved by the ordinary selector; if H15
    yields nothing by then, the ordinary selector decides (fail-open).
SKJUTBANA_H15_MIN_S   minimum dense score to accept (default 16; below -> ordinary selector).  Chosen on
                      AI-test dev sessions; physical CO2 echo/no-impact events score ~13-15, real holes ~21-25.
SKJUTBANA_H15_MIN_RATIO / _MAX_CORE / _MAX_BIPOLAR  acceptance rule (1.15 / 0.4 / 1.0), see _choose()
SKJUTBANA_H15_MODE    'authority' (default) | 'shadow' (compute + log only, never wait or decide)

Evidence (2026-11-19, AUDIT §30/§31): offline on 4 live AI-test sessions the dense top-1 was
correct for 157/335 shots vs 121 for the live-path chain (holdout 80 vs 65).  Measurement first;
this module only makes it testable in replay/live.  Ownership rules: frames after the next shot's
peak (+0.10 s) never enter POST; one job per shot; emitted xy of other shots are excluded.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any

import numpy as np

_LOCK = threading.Lock()


def enabled() -> bool:
    return os.environ.get("SKJUTBANA_H15_DENSE") == "1"


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def delay_s() -> float:
    return _f("SKJUTBANA_H15_DELAY_S", 0.6)


def deadline_s() -> float:
    return _f("SKJUTBANA_H15_DEADLINE_S", 1.3)


def mode() -> str:
    return os.environ.get("SKJUTBANA_H15_MODE", "authority").strip().lower()


CO2_PAIR_S = 0.12   # a CO2 shot gives two audio peaks 66-94 ms apart (physical 2026-11-18)


class _State:
    def __init__(self) -> None:
        self.owner: int | None = None              # id(scanner) generation owner
        self.roi: tuple[int, int, int, int] | None = None
        self.shape: tuple[int, int] | None = None
        self.results: dict[tuple[int, float], dict[str, Any]] = {}
        self.submitted: set[tuple[int, float]] = set()
        self.futures: list[Any] = []
        self.installed = False

    def reset(self, owner: int | None) -> None:
        with _LOCK:
            for fut in self.futures:
                try:
                    fut.cancel()
                except Exception:
                    pass
            self.futures = []
            self.results = {}
            self.submitted = set()
            self.roi = None
            self.shape = None
            self.owner = owner


_S = _State()


def _key(event: Any) -> tuple[int, float]:
    """Shot ids can be reused after a scanner restart; the peak time disambiguates."""
    return int(getattr(event, "shot_id", 0) or 0), round(float(getattr(event, "peak_ts", 0.0) or 0.0), 4)


def _viewport_roi(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    try:
        from src.engine.input.hit_input import hit_input
        from src.engine.settings import load_viewport_rect
        vp = load_viewport_rect()
        pts = [hit_input.screen_to_camera(x, y) for x, y in
               ((vp.x, vp.y), (vp.right, vp.y), (vp.right, vp.bottom), (vp.x, vp.bottom))]
        pts = [p for p in pts if p is not None]
        if len(pts) == 4:
            x0 = max(0, int(min(p[0] for p in pts)) - 20)
            y0 = max(0, int(min(p[1] for p in pts)) - 20)
            x1 = min(w, int(max(p[0] for p in pts)) + 20)
            y1 = min(h, int(max(p[1] for p in pts)) + 20)
            if x1 - x0 > 64 and y1 - y0 > 64:
                return x0, y0, x1, y1
    except Exception as exc:
        print(f"[H15] viewport roi unavailable ({type(exc).__name__}: {exc}); using full frame")
    return 0, 0, w, h


def _median(frames: list[np.ndarray]) -> np.ndarray | None:
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0].astype(np.float32)
    return np.median(np.stack(frames), axis=0).astype(np.float32)


def dense_top(pre: np.ndarray, post: np.ndarray, top_n: int = 6) -> list[tuple[float, float, float]]:
    """(x, y, dense_s) of the strongest local maxima, ROI coordinates.  Same math as
    automation/bench/hole_likeness.dense_peaks."""
    import cv2
    from src.engine.winner_selection_flags import h11_kernel
    p = np.maximum(pre, 1.0)
    d = 1.0 - post / p
    d = d - cv2.blur(d, (41, 41))
    mf = cv2.filter2D(d, cv2.CV_32F, h11_kernel())
    noise = np.sqrt(np.maximum(cv2.blur(d * d, (31, 31)), 1e-8)) + 1e-3
    s = mf / noise
    # ignore the 20 px border (filter support)
    s[:20, :] = 0
    s[-20:, :] = 0
    s[:, :20] = 0
    s[:, -20:] = 0
    dil = cv2.dilate(s, np.ones((9, 9), np.uint8))
    ys, xs = np.nonzero((s >= dil) & (s > 0))
    order = np.argsort(-s[ys, xs])[:top_n]
    return [(float(xs[i]), float(ys[i]), float(s[ys[i], xs[i]])) for i in order]


def _core_bipolar(pre: np.ndarray, post: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """core (mean darkening r<=2) and bipolar (bright/dark ratio r<=10) of D = 1 - post/pre,
    annulus-median removed - the same definitions as hole_likeness.features()."""
    r = 20
    xi, yi = int(round(x)), int(round(y))
    h, w = pre.shape
    if not (r <= xi < w - r and r <= yi < h - r):
        return 9.0, 9.0
    sl = (slice(yi - r, yi + r + 1), slice(xi - r, xi + r + 1))
    d = 1.0 - post[sl] / np.maximum(pre[sl], 1.0)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    rr = np.hypot(xx, yy)
    d = d - np.median(d[(rr >= 9) & (rr <= 20)])
    core = float(d[rr <= 2].mean())
    bright = float(max(0.0, -d[rr <= 10].min()))
    bipolar = bright / max(core, 1e-3) if core > 0 else 9.0
    return core, bipolar


def _peaks(scanner: Any) -> list[tuple[int, float]]:
    return sorted(((int(getattr(ev, "shot_id", 0) or 0), float(getattr(ev, "peak_ts", 0.0) or 0.0))
                   for ev in list(getattr(scanner, "audio_events", []) or [])), key=lambda t: t[1])


def on_frame(scanner: Any, gray: np.ndarray, frame_ts: float) -> None:
    """Main-thread hook (called once per detector frame)."""
    if not enabled() or getattr(scanner, "_v2224_worker_clone", False):
        return
    _install_selector_patch()
    ts = float(frame_ts)
    g = np.asarray(gray)
    if g.ndim != 2:
        return
    if _S.owner != id(scanner) or _S.shape != g.shape:
        _S.reset(id(scanner))          # new scanner / camera format: drop all state and jobs
        _S.shape = g.shape
        _S.roi = _viewport_roi(g.shape)
        print(f"[H15] dense second opinion on, roi={_S.roi} delay={delay_s():.2f}s mode={mode()}")
    x0, y0, x1, y1 = _S.roi
    # The detector only runs while a shot is open, so PRE frames come from the scanner's own
    # frame_history (every new camera frame, ~360 deep); crops are copied only for a submission.
    history = list(getattr(scanner, "frame_history", []) or [])
    ring = [(float(fr.timestamp), fr.gray) for fr in history if getattr(fr, "gray", None) is not None]
    peaks = _peaks(scanner)
    for ev in list(getattr(scanner, "audio_events", []) or []):
        key = _key(ev)
        sid, _pk = key
        if sid <= 0 or key in _S.submitted or str(getattr(ev, "state", "")) != "pending":
            continue
        peak = float(getattr(ev, "peak_ts", 0.0) or 0.0)
        if ts < peak + delay_s():
            continue
        _S.submitted.add(key)
        if any(peak - CO2_PAIR_S <= p < peak - 0.01 for _s, p in peaks):
            # second peak of a CO2 pair: the first peak's H15 owns the hole; this event keeps
            # the ordinary path (no second dense pick of the same hole, no spatial veto needed)
            _S.results[key] = dict(status="co2_pair_follower")
            continue
        prev_items = [(p, s_) for s_, p in peaks if p < peak - 0.01]
        prev, prev_sid = max(prev_items) if prev_items else (peak - 99.0, 0)
        nxt = min([p for _s, p in peaks if p > peak + CO2_PAIR_S] or [peak + 99.0])   # skip own echo
        crop = lambda f: np.asarray(f)[y0:y1, x0:x1].copy()
        # PRE must already contain the previous shot's hole (visible ~0.15 s after its peak).
        pre_t0 = max(peak - 0.4, prev + _f("SKJUTBANA_H15_PRE_AFTER_PREV_S", 0.20))
        pre_items = [(t, f) for t, f in ring if pre_t0 <= t <= peak - 0.02]
        exclude = None
        if not pre_items and prev > peak - 0.4:
            # Burst (doubles 0.2-0.3 s apart): no frame after the previous hole appeared.  Use the
            # ordinary PRE window and explicitly exclude ONLY the previous shot's own decided XY
            # (event ownership, not a general spatial veto).
            exclude = _decided_xy(scanner, prev_sid, (prev_sid, round(prev, 4)))
            if exclude is not None:
                pre_items = [(t, f) for t, f in ring if peak - 0.4 <= t <= peak - 0.02]
        post_items = [(t, f) for t, f in ring if peak + 0.15 <= t <= min(ts, nxt + 0.10)]
        if not pre_items or not post_items:
            # no previous-shot-safe PRE (or no POST): evidence unavailable
            _S.results[key] = dict(status="no_frames", n_pre=len(pre_items), n_post=len(post_items))
            print(f"[H15] shot={sid} no_frames pre={len(pre_items)} post={len(post_items)}")
            continue
        windows = dict(pre_t=(pre_items[0][0] - peak, pre_items[-1][0] - peak),
                       post_t=(post_items[0][0] - peak, post_items[-1][0] - peak),
                       post_first_ts=post_items[0][0], post_last_ts=post_items[-1][0],
                       burst_exclude=exclude)
        _submit(key, peak, ts, [crop(f) for _t, f in pre_items], [crop(f) for _t, f in post_items], windows)


def _decided_xy(scanner: Any, prev_sid: int, prev_key: tuple[int, float]) -> tuple[float, float] | None:
    """XY decided for the previous shot: its accepted H15 pick, else its matched track."""
    res = _S.results.get(prev_key) or {}
    rule = res.get("rule") or {}
    if rule.get("accepted"):
        return float(rule["x"]), float(rule["y"])
    for ev in list(getattr(scanner, "audio_events", []) or []):
        if int(getattr(ev, "shot_id", 0) or 0) == int(prev_sid):
            tid = getattr(ev, "matched_track_id", None)
            tr = (getattr(scanner, "_active_tracks", {}) or {}).get(tid) if tid is not None else None
            if tr is not None:
                return float(tr.camera_x), float(tr.camera_y)
    return None


def _submit(key: tuple[int, float], peak: float, ts: float, pre: list, post: list, windows: dict) -> None:
    from src.engine.shot_async_v2224 import async_detector_v2224 as det
    x0, y0, _x1, _y1 = _S.roi
    owner = _S.owner
    t_submit = time.monotonic()

    def job():
        t0 = time.monotonic()
        try:
            pre_m, post_m = _median(pre), _median(post)
            pk = dense_top(pre_m, post_m, top_n=8)
            ex = windows.get("burst_exclude")
            if ex is not None:
                pk = [p for p in pk if (p[0] + x0 - ex[0]) ** 2 + (p[1] + y0 - ex[1]) ** 2 > 144.0]
            feats = [_core_bipolar(pre_m, post_m, x, y) for x, y, _s in pk]
            res = dict(status="ok", peaks=[(x + x0, y + y0, s, c, b) for (x, y, s), (c, b) in zip(pk, feats)],
                       n_pre=len(pre), n_post=len(post), frame_ts=ts, **windows,
                       compute_ms=(time.monotonic() - t0) * 1000.0,
                       queue_ms=(t0 - t_submit) * 1000.0)
        except Exception as exc:
            res = dict(status=f"error {type(exc).__name__}: {exc}")
        with _LOCK:
            if _S.owner == owner:          # obsolete generation -> dropped
                _S.results[key] = res
        return None

    executor = getattr(det, "_executor", None)
    if executor is None:
        job()
        return
    try:
        fut = executor.submit(job)
        with _LOCK:
            _S.futures = [f for f in _S.futures if not f.done()] + [fut]
    except Exception as exc:
        _S.results[key] = dict(status=f"submit_failed {exc}")


def _choose(key: tuple[int, float], age: float) -> tuple[float, float, float, dict] | None:
    """Accept the strongest peak only when it is clearly a hole:
    score >= MIN_S, >= MIN_RATIO x the next peak, core <= MAX_CORE, bipolar <= MAX_BIPOLAR.
    Rule fitted on AI-test dev sessions 1223/1227, holdout 1336/1343: 84 correct vs 80 (always
    dense) vs 65 (chain), wrong H15 decisions 88 -> 12 (AUDIT §31).  Else -> ordinary selector.
    A result first seen after the deadline is rejected (bounded authority)."""
    with _LOCK:
        res = _S.results.get(key)
    if not res or res.get("status") != "ok":
        return None
    if "seen_age_s" not in res:
        res["seen_age_s"] = age
    if "rule" not in res:
        peaks = res["peaks"]
        x, y, s, core, bipolar = peaks[0]
        s2 = peaks[1][2] if len(peaks) > 1 else 0.0
        late = res["seen_age_s"] > deadline_s()
        ok = (not late and s >= _f("SKJUTBANA_H15_MIN_S", 16.0)
              and (s2 <= 0.0 or s / s2 >= _f("SKJUTBANA_H15_MIN_RATIO", 1.15))
              and core <= _f("SKJUTBANA_H15_MAX_CORE", 0.4)
              and bipolar <= _f("SKJUTBANA_H15_MAX_BIPOLAR", 1.0))
        res["rule"] = dict(x=x, y=y, s=s, ratio=(s / s2) if s2 > 0 else None, core=core, bipolar=bipolar,
                           late=late, accepted=ok)
        if not ok:
            print(f"[H15] shot={key[0]} rejected s={s:.1f} ratio={(s / s2) if s2 > 0 else 0:.2f} "
                  f"core={core:.2f} bipolar={bipolar:.2f} late={late} -> ordinary selector")
    rule = res["rule"]
    if not rule["accepted"]:
        return None
    return rule["x"], rule["y"], rule["s"], res


def _pair_partner_reported(scanner: Any, event: Any, x: float, y: float) -> bool:
    """True if an audio event within CO2_PAIR_S of this one already emitted a hit within 12 px."""
    peak = float(getattr(event, "peak_ts", 0.0) or 0.0)
    sid = int(getattr(event, "shot_id", 0) or 0)
    tracks = getattr(scanner, "_active_tracks", {}) or {}
    for ev in list(getattr(scanner, "audio_events", []) or []):
        if int(getattr(ev, "shot_id", 0) or 0) == sid or str(getattr(ev, "state", "")) == "pending":
            continue
        if abs(float(getattr(ev, "peak_ts", 0.0) or 0.0) - peak) > CO2_PAIR_S:
            continue
        tr = tracks.get(getattr(ev, "matched_track_id", None))
        if tr is not None and (tr.camera_x - x) ** 2 + (tr.camera_y - y) ** 2 <= 144.0:
            return True
    return False


def _install_selector_patch() -> None:
    """Installed on the first H15 frame.  By then every other selector wrapper (v251-v253) is
    installed (they install at engine start); the wrapper is a no-op when H15 is disabled."""
    if _S.installed:
        return
    _S.installed = True
    from src.engine.camera.hit_scanner import HitScanner, HoleTrack
    previous_best = HitScanner._best_track_for_event
    previous_ready = HitScanner._track_is_ready

    def best_h15(self, event):
        key = _key(event)
        if not enabled() or mode() == "shadow" or key[0] <= 0 or _S.owner != id(self):
            return previous_best(self, event)
        if getattr(event, "_h15_pair_done", False):
            return None   # the CO2 pair partner already reported this hole; the pair is one shot
        tracks = getattr(self, "_active_tracks", {})
        tid = getattr(event, "_h15_track_id", None)
        if tid is not None and tid in tracks:
            return tracks[tid]
        peak = float(getattr(event, "peak_ts", 0.0) or 0.0)
        age = time.time() - peak
        pick = _choose(key, age)
        if pick is not None:
            x, y, s, res = pick
            if _pair_partner_reported(self, event, x, y):
                event._h15_pair_done = True
                print(f"[H15] shot={key[0]} hole already reported by its CO2 pair partner -> no second emission")
                return None
            # Explicit H15 authority track: its time span/hits are the real POST window
            # (first/last POST frame, number of POST frames), not an ordinary confirmation.
            track = HoleTrack(track_id=self._next_track_id, camera_x=x, camera_y=y, created_at=time.time(),
                              first_seen_ts=float(res["post_first_ts"]), last_seen_ts=float(res["post_last_ts"]),
                              hits=int(res["n_post"]), best_score=float(s), state="h15_dense_authority",
                              last_candidate=dict(camera_x=x, camera_y=y, score=float(s), h15_dense=1.0,
                                                  h15_dense_s=float(s), v2224_producer_shot_id=key[0],
                                                  h15_n_pre=float(res["n_pre"]), h15_n_post=float(res["n_post"])))
            self._next_track_id += 1
            # Identity merge: ordinary tracks at this same hole (<= 12 px) ARE this decision; mark
            # them consumed so a CO2 echo event cannot re-emit the hole (34 vs 23 emissions on the
            # physical session before this).
            for other in tracks.values():
                if not other.emitted and (other.camera_x - x) ** 2 + (other.camera_y - y) ** 2 <= 144.0:
                    other.emitted = True
                    try:
                        other.last_candidate["h15_consumed_by_shot"] = float(key[0])
                    except Exception:
                        pass
            tracks[track.track_id] = track
            event._h15_track_id = track.track_id
            try:
                event.audio_trigger["h15"] = {k: v for k, v in res.items() if k != "peaks"}
            except Exception:
                pass
            print(f"[H15] shot={key[0]} dense xy=({x:.0f},{y:.0f}) s={s:.1f} age={age * 1000:.0f}ms "
                  f"compute={res.get('compute_ms', 0):.0f}ms queue={res.get('queue_ms', 0):.0f}ms")
            return track
        with _LOCK:
            known = key in _S.results
        if age < deadline_s() and not known:
            return None   # await H15 (explicit, bounded)
        return previous_best(self, event)   # fail-open

    def ready_h15(self, track, now_ts, event):
        if enabled() and (getattr(track, "last_candidate", {}) or {}).get("h15_dense"):
            return True   # explicit H15 authority (see best_h15), never an ordinary track
        return previous_ready(self, track, now_ts, event)

    HitScanner._best_track_for_event = best_h15
    HitScanner._track_is_ready = ready_h15
