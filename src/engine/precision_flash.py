"""Opt-in precision flash: one short white flash per shot burst for exact hole XY.

Why: a hole is only as visible as the projected light behind it.  In dark or
textured game scenes a 2-3 mm hole darkens 8-17 grey levels, on bright light
30-50.  A short uniform flash makes every hole a dark dot on a bright field;
comparing it with the previous flash image (which already contains all older
holes) gives the new holes with ~1 px accuracy, independent of game content.

Policy (engine mechanics; the game only declares its needs):

* ``hit_precision`` on the game/scene, one of the engine levels (ROADMAP.md):
  ``SHOT`` | ``HIT_MISS`` | ``OBJECT`` | ``ZONE`` | ``EXACT`` (case-insensitive;
  legacy aliases ``shot_only``->SHOT, ``exact``->EXACT, ``region``->OBJECT).
* ``hit_aggregation``: ``"per_burst"`` (default) | ``"series"``.  Defaults: EXACT
  per_burst flashes once per shot burst; ZONE/EXACT with ``series`` flash ONCE when
  the game calls ``precision_flash.flush_series()`` (e.g. at ceasefire) and get
  ``on_precision_series(result)``; ordinary hits are delivered live meanwhile.
  All other levels never flash unless ``precision_flash="on"``.
* ``precision_flash`` on the game/scene: ``"auto"`` (default) | ``"off"`` | ``"on"``.
  ``"off"`` always wins and may be changed by the game at any time (e.g. during
  a cut-scene).  ``"on"`` enables it for a lower precision level.
* Env ``SKJUTBANA_PRECISION_FLASH=off`` disables it globally; ``=on`` forces it
  on (testing only).

Behaviour when enabled:

* One reference flash when the mode becomes active and the range is quiet.
* After a shot burst: wait until no new audio peak for ``settle_s`` (doubles and
  strings share ONE flash), then flash once for ``duration_s``.  Never more often
  than ``min_interval_s``.
* The flash frames are selected by brightness (full-frame plateau, no rolling-
  shutter transition), averaged, and divided by the reference flash; new dark
  blobs are the new holes.  That flash becomes the next reference.
* While a reference exists, ordinary camera hits are held: the flash result is
  the XY authority.  If a flash measurement is invalid (no clean plateau) the
  held ordinary hits are released unchanged, so the game never loses a shot.

Everything is a no-op unless the policy enables it; games without
``hit_precision`` behave exactly as before.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PRECISION_LEVELS = ("SHOT", "HIT_MISS", "OBJECT", "ZONE", "EXACT")
LEVEL_ALIASES = {"shot_only": "SHOT", "exact": "EXACT", "region": "OBJECT"}
AGGREGATIONS = ("per_burst", "series")
FLASH_PREFERENCES = ("auto", "off", "on", "object_glow")


def canonical_level(value: Any, shot_input_mode: Any = None) -> str:
    text = str(value or "").strip()
    level = LEVEL_ALIASES.get(text.lower(), text.upper())
    if level in PRECISION_LEVELS:
        return level
    return "SHOT" if shot_input_mode == "SHOT_ONLY" else "OBJECT"


@dataclass
class FlashConfig:
    # Budget: the game needs the hit within ~1 s of the shot.
    # single shot: settle 0.30 + flash 0.12 + frames ~0.22 + analysis ~0.05 = ~0.7-0.8 s
    settle_s: float = 0.35          # quiet time after the last peak (doubles 0.15-0.3 s share one flash)
    max_wait_s: float = 0.90        # flash at the latest this long after the burst's first peak
    duration_s: float = 0.18        # ~4 camera frames at 25 fps -> 2-3 full plateau frames to average
    min_interval_s: float = 0.50    # never flash more often than this
    reference_quiet_s: float = 0.5  # range quiet this long before the reference flash
    capture_min_after_s: float = 0.22  # measure once a frame received this long after the flash exists (projector + receive latency)
    capture_wait_s: float = 0.35    # ... or at the latest after this
    # 2026-11-22 hole lab: flat pale yellow beats white (weakest real hole SNR 41.5 vs 29.6) and glares less.
    color: tuple = (230, 220, 150)
    plateau_fraction: float = 0.93  # frame/band brightness vs the brightest flash frame
    min_flash_delta: float = 4.0    # grey levels: flash frames must be this much brighter than before
    reference_retry_s: float = 1.0  # failed reference flash: wait before retrying (max 3 attempts)
    reference_max_attempts: int = 3
    blur_sigma_px: float = 15.0     # illumination normalisation scale
    min_darkening: float = 0.06     # relative darkening floor for a new hole
    noise_k: float = 6.0            # threshold = max(min_darkening, k * robust noise)
    min_area_px: int = 2
    max_area_px: int = 400
    lit_fraction: float = 0.6       # evidence only where the flash lights the screen ...
    edge_margin_px: int = 15        # ... and this far inside the lit edge (edge sway artefacts)
    hold_release_grace_s: float = 1.5  # ordinary hits for an already measured burst are dropped this long
    match_radius_px: float = 30.0   # held ordinary hit within this of a flash hole = same shot
    exclude_tail_s: float = 0.25    # frames received until flash end + this are hidden from ordinary detection
    receive_latency_s: float = 0.06 # frame receive time - exposure end (measured ~0.066 s); conservative

    @classmethod
    def from_settings(cls) -> "FlashConfig":
        cfg = cls()
        try:
            from src.engine.settings import _load_settings_dict
            data = (_load_settings_dict() or {}).get("precision_flash") or {}
        except Exception:
            data = {}
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, tuple(value) if key == "color" else type(getattr(cfg, key))(value))
        return cfg


def resolve_aggregation(scene: Any) -> str:
    value = str(getattr(scene, "hit_aggregation", None) or "per_burst").lower()
    return value if value in AGGREGATIONS else "per_burst"


def resolve_policy(scene: Any, env: dict | None = None) -> tuple[bool, str, str]:
    """Return (enabled, level, preference). Unknown values fall back safely."""
    env = os.environ if env is None else env
    precision = canonical_level(getattr(scene, "hit_precision", None), getattr(scene, "shot_input_mode", None))
    preference = str(getattr(scene, "precision_flash", None) or "auto").lower()
    if preference not in FLASH_PREFERENCES:
        preference = "auto"
    forced = str(env.get("SKJUTBANA_PRECISION_FLASH", "") or "").lower()
    if preference == "off" or forced == "off":
        return False, precision, preference
    # Per-game opt-in; explicit scene/environment off above always wins.
    if getattr(scene, "object_highlight", False) is True:
        preference = "object_glow"
    if preference in ("on", "object_glow") or forced == "on":
        return True, precision, preference
    if precision == "EXACT":
        return True, precision, preference
    return (precision == "ZONE" and resolve_aggregation(scene) == "series"), precision, preference


# --------------------------------------------------------------------------- measurement

def _gray(frame: np.ndarray, rect) -> np.ndarray:
    import cv2
    x, y, w, h = (int(v) for v in rect)
    crop = frame[y:y + h, x:x + w]
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return crop.astype(np.float32)


def select_plateau(frames: list, rect, *, baseline: float | None, cfg: FlashConfig):
    """Pick frames fully inside the flash: every horizontal band near the flash maximum.

    Returns (indices, info).  Rolling shutter makes transition frames bright in
    only part of the image, so each of four row bands must be on the plateau.
    """
    bands = []
    for frame in frames:
        g = _gray(frame, rect)
        h = g.shape[0]
        bands.append([float(g[i * h // 4:(i + 1) * h // 4].mean()) for i in range(4)])
    if not bands:
        return [], dict(reason="NO_FRAMES")
    arr = np.array(bands)
    peak = arr.max(axis=0)
    ok = np.all(arr >= cfg.plateau_fraction * peak, axis=1)
    if baseline is not None and float(peak.mean()) < baseline + cfg.min_flash_delta:
        return [], dict(reason="FLASH_NOT_SEEN", peak=float(peak.mean()), baseline=baseline)
    idx = [i for i, good in enumerate(ok) if good]
    return idx, dict(reason="OK" if idx else "NO_PLATEAU", peak=float(peak.mean()),
                     baseline=baseline, band_means=arr.round(1).tolist())


def average_gray(frames: list, rect) -> np.ndarray:
    return np.mean([_gray(f, rect) for f in frames], axis=0).astype(np.float32)


def find_new_holes(reference: np.ndarray, post: np.ndarray, cfg: FlashConfig, *, max_holes: int = 8, valid_mask=None):
    """New dark blobs in ``post`` relative to ``reference`` (both flash-lit, same crop).

    Relative darkening ``1 - post/ref`` cancels the projector's brightness
    profile; dividing by a large blur of the ratio cancels slow exposure or
    projector drift between the two flashes.  Old holes are dark in both and
    vanish from the ratio.
    """
    import cv2
    ref = np.maximum(reference, 8.0)
    ratio = post / ref
    if valid_mask is None:
        trend = cv2.GaussianBlur(ratio, (0, 0), cfg.blur_sigma_px)
    else:
        weights=np.asarray(valid_mask,dtype=np.float32)
        trend=cv2.GaussianBlur(ratio*weights,(0,0),cfg.blur_sigma_px)/np.maximum(
            cv2.GaussianBlur(weights,(0,0),cfg.blur_sigma_px),1e-6)
    rel = ratio / np.maximum(trend, 1e-3)
    dark = cv2.GaussianBlur(1.0 - rel, (0, 0), 0.8)
    valid = reference > 25.0                                   # outside the lit area: no evidence
    lit = reference >= cfg.lit_fraction * float(np.percentile(reference, 90))
    if cfg.edge_margin_px > 0:
        k = 2 * int(cfg.edge_margin_px) + 1
        lit = cv2.erode(lit.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    valid &= lit
    if valid_mask is not None:
        scoped=np.asarray(valid_mask,dtype=np.uint8)
        if cfg.edge_margin_px>0:scoped=cv2.erode(scoped,np.ones((2*int(cfg.edge_margin_px)+1,)*2,np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        valid &= scoped>0
    dark[~valid] = 0.0
    sample = dark[valid]
    med = float(np.median(sample)) if sample.size else 0.0
    mad = float(np.median(np.abs(sample - med))) * 1.4826 if sample.size else 0.0
    threshold = max(cfg.min_darkening, cfg.noise_k * mad)
    mask = (dark >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    holes = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < cfg.min_area_px or area > cfg.max_area_px:
            continue
        ys, xs = np.nonzero(labels == label)
        weights = dark[ys, xs]
        total = float(weights.sum())
        if total <= 0:
            continue
        holes.append(dict(x=float((xs * weights).sum() / total), y=float((ys * weights).sum() / total),
                          peak=float(weights.max()), mass=total, area=area))
        if valid_mask is not None:
            radius=max(4,int(stats[label,cv2.CC_STAT_WIDTH]),int(stats[label,cv2.CC_STAT_HEIGHT]))
            patch=dark[max(0,int(ys.min())-radius):int(ys.max())+radius+1,
                       max(0,int(xs.min())-radius):int(xs.max())+radius+1]
            holes[-1]['bipolar_change']=bool(float(-patch.min())>=threshold)
    holes.sort(key=lambda h: h["mass"], reverse=True)
    return holes[:max_holes], dict(threshold=threshold, noise_mad=mad)


# --------------------------------------------------------------------------- controller

@dataclass
class _Burst:
    first_peak: float
    last_peak: float
    peaks: list = field(default_factory=list)
    held: list = field(default_factory=list)


class PrecisionFlash:
    """Main-thread state machine; image work runs on a short worker thread."""

    IDLE, ARMED, FLASHING, CAPTURING, MEASURING = "IDLE", "ARMED", "FLASHING", "CAPTURING", "MEASURING"

    def __init__(self, cfg: FlashConfig | None = None, *, clock=time.time, camera=None,
                 emit=None, scanport=None, log=print, monotonic=time.monotonic):
        self.cfg = cfg or FlashConfig()
        self.clock = clock
        self.monotonic = monotonic
        self.camera = camera
        # capture the unfiltered ring now (live wiring later hides flash frames from detection)
        self.ring_source = getattr(camera, "get_ring_snapshot", None)
        self.emit = emit                  # callable(camera_x, camera_y) -> ordinary hit delivery
        self.scanport = scanport          # (x, y, w, h) camera crop; None = load from settings
        self.log = log
        self.enabled = False
        self.precision = "OBJECT"
        self.aggregation = "per_burst"
        self.scene: Any = None
        self.series_peaks: list = []
        self.pending_series = False
        self.series_results: list = []
        self.state = self.IDLE
        self.reference: np.ndarray | None = None
        self.reference_ts: float | None = None
        self.burst: _Burst | None = None
        self.next_burst: _Burst | None = None   # peaks during a flash/measurement
        self._seen_peaks: list = []
        self.flash_windows: list = []          # (on, off|None) for ordinary-detection exclusion
        self.pending_reference = False
        self.flash_on_ts: float | None = None
        self.flash_off_ts: float | None = None
        self.last_flash_ts = -1e9
        self.last_peak_ts = -1e9
        self.enabled_since: float | None = None
        self._worker: threading.Thread | None = None
        self._result: dict | None = None
        self._lock = threading.Lock()
        self.resolved_bursts: list = []   # (first_peak, last_peak, resolved_ts)
        self.results: list = []           # every measurement (also for the bench)
        self.flash_count = 0
        self.reference_attempts = 0
        self.last_reference_attempt = -1e9
        self.mode=None;self._glow=None;self._generation=0;self._last_glow_error=None

    # ----------------------------------------------------------------- policy
    def sync_policy(self, scene: Any) -> None:
        enabled, precision, preference = resolve_policy(scene)
        mode='object_glow' if preference=='object_glow' else 'flash'
        changed=enabled!=self.enabled or mode!=self.mode or scene is not self.scene
        if changed:
            self._generation+=1
            if self._glow is not None:self._glow.disable('policy_or_scene_changed')
            retired=(self.scene is not None and scene is not self.scene) or (self.mode is not None and mode!=self.mode)
            # A delayed hit belongs to its old scene/method, never a new game.
            if not retired:
                self._release_held('POLICY_CHANGED')
                if self.next_burst is not None:
                    for x,y,_ in self.next_burst.held:
                        if self.emit is not None:self.emit(x,y)
            self.state,self.burst,self.next_burst=self.IDLE,None,None
            self.reference=None;self.pending_reference=False;self.pending_series=False
            with self._lock:self._result=None
            if self.flash_windows and self.flash_windows[-1][1] is None:
                self.flash_windows[-1]=(self.flash_windows[-1][0],self.clock())
            self.reference_attempts=0;self.enabled_since=self.clock() if enabled else None
            self.log(f'[PRECISION-FLASH] mode={mode} enabled={enabled} precision={precision}')
        self.enabled,self.precision,self.mode,self.scene=enabled,precision,mode,scene
        self.aggregation=resolve_aggregation(scene)
        if enabled and mode=='object_glow':
            if self._glow is None:
                from src.engine.precision_object_glow import ObjectGlow
                self._glow=ObjectGlow(self)
            try:self._glow.sync(scene)
            except Exception as exc:
                self._glow.disable('invalid_glow_context:'+type(exc).__name__)
                if str(exc)!=self._last_glow_error:self.log('[OBJECT-GLOW] context unavailable: '+str(exc))
                self._last_glow_error=str(exc)

    def holds_ordinary_hits(self) -> bool:
        # series mode: ordinary hits stay live; the series flash is a separate result
        return self.enabled and (self.mode=="object_glow" or (self.reference is not None and self.aggregation != "series"))

    def flush_series(self) -> bool:
        """Series end (e.g. ceasefire): one flash that measures every new hole since the reference."""
        if not self.enabled or self.aggregation != "series" or self.reference is None:
            return False
        if self.state not in (self.IDLE, self.ARMED):
            return False
        self.pending_series = True
        self._start_flash(self.clock())
        return True

    # ----------------------------------------------------------------- inputs
    def on_peak(self, peak_ts: float) -> None:
        if not self.enabled:
            return
        if self.mode=='object_glow':
            self.last_peak_ts=max(self.last_peak_ts,float(peak_ts))
            if self._glow is not None:self._glow.on_peak(float(peak_ts))
            return
        peak_ts = float(peak_ts)
        if any(abs(peak_ts - p) < 1e-6 for p in self._seen_peaks):
            return                                   # registered before the shot step already
        self._seen_peaks = (self._seen_peaks + [peak_ts])[-32:]
        self.last_peak_ts = max(self.last_peak_ts, peak_ts)
        if self.aggregation == "series":
            self.series_peaks.append(peak_ts)       # no per-burst flash; counted for the series result
            return
        in_flash = self.state in (self.FLASHING, self.CAPTURING, self.MEASURING)
        if in_flash:
            # its hole may be missing from (or only partly in) this flash: own burst, own flash
            if self.next_burst is None:
                self.next_burst = _Burst(first_peak=peak_ts, last_peak=peak_ts)
            target = self.next_burst
        else:
            if self.burst is None:
                self.burst = _Burst(first_peak=peak_ts, last_peak=peak_ts)
            target = self.burst
        target.last_peak = max(target.last_peak, peak_ts)
        target.peaks.append(peak_ts)
        if self.state == self.IDLE:
            self.state = self.ARMED

    def hold_ordinary_hit(self, x: float, y: float) -> bool:
        """Return True when the ordinary hit is withheld (flash is the XY authority)."""
        if not self.holds_ordinary_hits():
            return False
        if self.mode=="object_glow":return True  # local OBJECT authority, never release CURRENT guesses
        now = self.clock()
        for first, last, resolved in self.resolved_bursts[-8:]:
            if resolved <= now <= resolved + self.cfg.hold_release_grace_s:
                self.log(f"[PRECISION-FLASH] dropped late ordinary hit ({x:.1f},{y:.1f}) after flash result")
                return True
        target = self.burst if self.burst is not None else self.next_burst
        if target is None:
            # ordinary hit without an audio burst of ours: keep live behaviour
            return False
        target.held.append((float(x), float(y), now))
        return True

    def excludes_frame(self, ts: float) -> bool:
        """True for camera frames that may show the flash (hidden from ordinary detection)."""
        if self._glow is not None and self._glow.excludes_frame(ts):return True
        if not self.flash_windows:
            return False
        for on, off in self.flash_windows[-4:]:
            if ts >= on and (off is None or ts <= off + self.cfg.exclude_tail_s):
                return True
        return False

    # ----------------------------------------------------------------- main thread tick
    def tick(self) -> None:
        if not self.enabled:
            return
        if self.mode=="object_glow":
            if self._glow is not None:self._glow.tick()
            return
        now = self.clock()
        if self.reference is None and self.state == self.IDLE and self.burst is None and self.next_burst is None:
            since = self.enabled_since or now
            quiet = (now - since >= self.cfg.reference_quiet_s
                     and now - self.last_peak_ts >= self.cfg.reference_quiet_s)
            retry_ok = (self.reference_attempts < self.cfg.reference_max_attempts
                        and now - self.last_reference_attempt >= self.cfg.reference_retry_s)
            if quiet and retry_ok:
                self.pending_reference = True
                self.reference_attempts += 1
                self.last_reference_attempt = now
                self._start_flash(now)
            return
        if self.state == self.ARMED and self.burst is not None:
            quiet = now - self.burst.last_peak >= self.cfg.settle_s
            overdue = now - self.burst.first_peak >= self.cfg.max_wait_s
            spaced = now - self.last_flash_ts >= self.cfg.min_interval_s
            if (quiet or overdue) and spaced:
                if self.reference is None:
                    # no reference yet: cannot measure -> ordinary hits stand
                    self._release_held("NO_REFERENCE")
                    self.burst, self.state = None, self.IDLE
                    return
                self._start_flash(now)
        elif self.state == self.FLASHING and now - self.flash_on_ts >= self.cfg.duration_s:
            self.flash_off_ts = now
            self.flash_windows[-1] = (self.flash_on_ts, now)
            self.state = self.CAPTURING
        elif self.state == self.CAPTURING and (now - self.flash_off_ts >= self.cfg.capture_wait_s
                                               or self._latest_frame_ts() >= self.flash_off_ts
                                               + self.cfg.capture_min_after_s):
            self._start_measurement()
        elif self.state == self.MEASURING:
            with self._lock:
                result, self._result = self._result, None
            if result is not None:
                self._finish(result)

    def _latest_frame_ts(self) -> float:
        getter = getattr(self.camera, "get_latest_timestamp", None)
        try:
            ts = getter() if callable(getter) else None
        except Exception:
            ts = None
        return float(ts) if ts is not None else -1e18

    def _start_flash(self, now: float) -> None:
        self.state = self.FLASHING
        self.flash_on_ts = now
        self.last_flash_ts = now
        self.flash_count += 1
        self.flash_windows = (self.flash_windows + [(now, None)])[-8:]

    def is_flashing(self) -> bool:
        if self.mode=="object_glow":return self.enabled and self._glow is not None and self._glow.enabled and self._glow.state=="SHOWING"
        return self.enabled and self.state == self.FLASHING

    def render(self, screen, area=None) -> None:
        """Paint the flash over the whole projected window (call right before flip).

        The whole window, not only the game viewport: content outside the
        viewport is projected onto the board too, and moving content there
        would otherwise differ between reference and post flash.
        """
        if self.mode=="object_glow":
            if self._glow is not None:self._glow.render(screen)
            return
        if self.is_flashing():
            screen.fill(self.cfg.color, area)

    # ----------------------------------------------------------------- measurement
    def _rect(self):
        if self.scanport is not None:
            return self.scanport
        from src.engine.settings import load_scanport_rect
        r = load_scanport_rect()
        return (r.x, r.y, r.w, r.h) if r is not None else None

    def _start_measurement(self) -> None:
        self.state = self.MEASURING
        on, off = self.flash_on_ts, self.flash_off_ts
        ring = self.ring_source() if self.ring_source else []
        frames = [cf for cf in ring if on - 0.6 <= cf.timestamp <= off + self.cfg.capture_wait_s]
        is_reference = self.pending_reference and not self.pending_series
        # Snapshot everything the worker needs: on_peak keeps mutating bursts.
        # A shot during the flash must not leak into this image (it would
        # become part of the next reference and never be found).
        late = list(self.next_burst.peaks) if self.next_burst is not None else []
        if self.pending_series:
            late = [p for p in self.series_peaks if p >= on]
        cutoff = min(late) if late else None
        expected = max(1, len(self.burst.peaks)) if self.burst is not None else 1
        if self.pending_series:
            expected = min(30, max(1, sum(1 for p in self.series_peaks if p < on)))
        rect = self._rect()
        reference = self.reference
        generation=self._generation

        def work():
            try:
                result = self._measure(frames, on, rect, reference, is_reference, expected, cutoff)
            except Exception as exc:  # never break the game loop
                result = dict(valid=False, reason=f"ERROR:{type(exc).__name__}:{exc}")
            result.update(flash_on_ts=on, flash_off_ts=off, reference_flash=is_reference)
            with self._lock:
                if generation==self._generation:self._result = result

        self._worker = threading.Thread(target=work, name="precision-flash", daemon=True)
        self._worker.start()

    def _measure(self, frames, on, rect, reference, is_reference, expected, cutoff=None) -> dict:
        if rect is None:
            return dict(valid=False, reason="NO_SCANPORT")
        before = [cf.frame_bgr for cf in frames if cf.timestamp < on]
        # a frame received at t was exposed until ~t - receive_latency: keep frames
        # whose exposure ended before the cutoff peak (that hole cannot be in them)
        limit = None if cutoff is None else cutoff + self.cfg.receive_latency_s
        during = [cf for cf in frames if cf.timestamp >= on and (limit is None or cf.timestamp < limit)]
        baseline = float(_gray(before[-1], rect).mean()) if before else None
        idx, info = select_plateau([cf.frame_bgr for cf in during], rect, baseline=baseline, cfg=self.cfg)
        if not idx:
            return dict(valid=False, reason=info.get("reason"), plateau=info, frames=len(during))
        image = average_gray([during[i].frame_bgr for i in idx], rect)
        out = dict(valid=True, image=image, plateau_frames=len(idx), frames=len(during),
                   plateau_ts=[during[i].timestamp for i in idx], flash_mean=float(image.mean()),
                   saturated_fraction=float((image >= 250).mean()))
        if is_reference or reference is None:
            out["holes"] = []
            return out
        n = int(expected)
        holes, stats = find_new_holes(reference, image, self.cfg, max_holes=8)
        for hole in holes:
            hole["camera_x"] = hole["x"] + rect[0]
            hole["camera_y"] = hole["y"] + rect[1]
        out.update(holes=holes[:n], extra_holes=holes[n:], expected=n, **stats)
        return out

    def _finish(self, result: dict) -> None:
        now = self.clock()
        burst, self.burst = self.burst, None
        image = result.pop("image", None)
        if result.get("valid") and image is not None:
            self.reference, self.reference_ts = image, now
        if self.pending_series:
            self.pending_series = False
            on = result.get("flash_on_ts") or now
            result.update(series=True, peaks=[p for p in self.series_peaks if p < on])
            self.series_peaks = [p for p in self.series_peaks if p >= on]
            self.series_results.append(result)
            self.results.append(result)
            self.log(f"[PRECISION-FLASH] series valid={result.get('valid')} shots={len(result['peaks'])} "
                     f"holes={len(result.get('holes', []))}")
            callback = getattr(self.scene, "on_precision_series", None)
            if callable(callback):
                try:
                    callback(result)
                except Exception as exc:
                    self.log(f"[PRECISION-FLASH] on_precision_series failed: {exc}")
            self.state = self.ARMED if self.burst is not None else self.IDLE
            return
        if result.get("reference_flash"):
            self.pending_reference = False
            detail = result.get("plateau") or {}
            self.log(f"[PRECISION-FLASH] reference valid={result.get('valid')} reason={result.get('reason','OK')} "
                     f"frames={result.get('frames')} peak={detail.get('peak')} baseline={detail.get('baseline')}")
        elif burst is not None:
            if result.get("valid"):
                holes = result.get("holes", [])
                on = result.get("flash_on_ts") or now
                peaks = sorted(p for p in burst.peaks if p < on) or sorted(burst.peaks)
                for i, hole in enumerate(holes):
                    # hole->peak order is unknown; peak_ts gives the game the burst's shot times
                    peak = peaks[min(i, len(peaks) - 1)] if peaks else None
                    self._emit_with_context(hole["camera_x"], hole["camera_y"], peak)
                released = self._reconcile(burst, holes, result)
                self.resolved_bursts.append((burst.first_peak, burst.last_peak, now))
                self.log(f"[PRECISION-FLASH] burst peaks={len(burst.peaks)} holes={len(holes)} "
                         f"held_ordinary={len(burst.held)} released={released} plateau={result.get('plateau_frames')}")
            elif self.next_burst is not None:
                # a new shot during the flash cut the image away: measure both
                # bursts with the next flash against the same reference
                nb = self.next_burst
                merged = _Burst(first_peak=burst.first_peak, last_peak=max(burst.last_peak, nb.last_peak),
                                peaks=list(burst.peaks) + list(nb.peaks), held=list(burst.held) + list(nb.held))
                self.next_burst = merged
                self.log(f"[PRECISION-FLASH] flash cut by a new shot ({result.get('reason')}); "
                         f"merged into next flash (peaks={len(merged.peaks)})")
            else:
                self.log(f"[PRECISION-FLASH] invalid flash ({result.get('reason')}); releasing ordinary hits")
                for x, y, _ in burst.held:
                    if self.emit is not None:
                        self.emit(x, y)
            result["peaks"] = list(burst.peaks)
        self.results.append(result)
        # peaks that arrived during this flash form the next burst with its own flash
        self.burst, self.next_burst = self.next_burst, None
        self.state = self.ARMED if self.burst is not None else self.IDLE

    _next_shot_id = 1_000_000_000      # own id range: never collides with HitScanner shot ids

    def _emit_with_context(self, x: float, y: float, peak_ts: float | None) -> None:
        """Deliver like an ordinary camera hit, with a unique shot_id and the shot time.

        Games de-duplicate hits by shot_id (or debounce hits without one), so two
        holes of a double shot delivered together must carry distinct ids.
        """
        if self.emit is None:
            return
        PrecisionFlash._next_shot_id += 1
        try:
            from src.engine import shot_context_v250 as sc
            ctx = sc.ShotEmissionContextV250(shot_id=PrecisionFlash._next_shot_id,
                                             peak_ts=float(peak_ts if peak_ts is not None else self.clock()),
                                             scanner_state="precision_flash")
            old = getattr(sc._tls, "shot_context_v250", None)
            sc._tls.shot_context_v250 = ctx
            try:
                self.emit(x, y)
            finally:
                sc._tls.shot_context_v250 = old
        except ImportError:
            self.emit(x, y)

    def _reconcile(self, burst: _Burst, holes: list, result: dict) -> int:
        """Flash XY replaces matching ordinary hits; never lose a shot the flash missed.

        Held ordinary hits near a flash hole are the same shot (dropped).  If
        the flash found fewer holes than shots, that many unmatched ordinary
        hits are released unchanged (earliest first).
        """
        unmatched = []
        for x, y, ts in burst.held:
            near = any(((x - h["camera_x"]) ** 2 + (y - h["camera_y"]) ** 2) ** 0.5 <= self.cfg.match_radius_px
                       for h in holes)
            if not near:
                unmatched.append((x, y, ts))
        missing = max(0, int(result.get("expected", len(holes))) - len(holes))
        release = sorted(unmatched, key=lambda h: h[2])[:missing]
        for x, y, _ in release:
            if self.emit is not None:
                self.emit(x, y)
        result["held_ordinary"] = [list(h[:2]) for h in burst.held]
        result["released_ordinary"] = [list(h[:2]) for h in release]
        return len(release)

    def _release_held(self, reason) -> None:
        if self.burst is None:
            return
        held, self.burst.held = self.burst.held, []
        for x, y, _ in held:
            if self.emit is not None:
                self.emit(x, y)


# --------------------------------------------------------------------------- live wiring

_CONTROLLER: PrecisionFlash | None = None


def get_controller() -> PrecisionFlash | None:
    return _CONTROLLER


def flush_series() -> bool:
    """Game API: the shooting series ended (e.g. ceasefire); measure it with one flash."""
    return _CONTROLLER.flush_series() if _CONTROLLER is not None else False


def install(app: Any) -> PrecisionFlash | None:
    """Create the controller once.  Nothing is wrapped until a scene enables the flash."""
    global _CONTROLLER
    if _CONTROLLER is not None:
        return _CONTROLLER
    if str(os.environ.get("SKJUTBANA_PRECISION_FLASH", "")).lower() == "off":
        return None
    from src.engine.camera.camera_manager import camera_manager
    _CONTROLLER = PrecisionFlash(FlashConfig.from_settings(), camera=camera_manager, emit=None)
    return _CONTROLLER


_WIRED = False


def _wire(controller: PrecisionFlash) -> None:
    """Lazily wrap the ordinary hit delivery and hide flash frames from detection.

    Only happens the first time a scene enables the flash; all wrappers are
    pass-through while the flash is disabled or no flash window exists.
    """
    global _WIRED
    if _WIRED:
        return
    _WIRED = True
    from src.engine.camera.camera_manager import camera_manager
    from src.engine.input.hit_input import hit_input

    original_push = getattr(hit_input, "push_camera_hit", None)
    if original_push is not None:
        controller.emit = original_push

        def gated_push(camera_x, camera_y):
            if controller.hold_ordinary_hit(camera_x, camera_y):
                return None
            return original_push(camera_x, camera_y)

        hit_input.push_camera_hit = gated_push

    for name in ("get_new_frames_since_last_pickup", "get_ring_snapshot"):
        original = getattr(camera_manager, name, None)
        if original is None:
            continue

        def filtered(_original=original):
            frames = _original()
            if not controller.flash_windows and controller._glow is None:
                return frames
            return [cf for cf in frames if not controller.excludes_frame(cf.timestamp)]

        setattr(camera_manager, name, filtered)


def _register_peaks(controller: PrecisionFlash) -> None:
    """Register audio peaks before the shot step can emit an ordinary hit for them."""
    try:
        from src.engine.audio.audio_peak_detector import audio_peak_detector
    except Exception:
        return
    pending = ()
    snapshot = getattr(audio_peak_detector, "pending_events_snapshot", None)
    if callable(snapshot):
        try:
            pending = snapshot() or ()
        except Exception:
            pending = ()
    for event in pending:
        ts = getattr(event, "timestamp", None)
        if ts is not None and float(ts) > controller.last_peak_ts:
            controller.on_peak(float(ts))


def pre_step(app: Any) -> None:
    """Called once per live frame BEFORE the shot-path step."""
    controller = _CONTROLLER
    if controller is None:
        return
    controller.sync_policy(getattr(app, "scene", None))
    if controller.enabled:
        _wire(controller)
        _register_peaks(controller)


def frame_hook(app: Any, new_peak: Any) -> None:
    """Called once per live frame after the shot-path step, before scene render."""
    controller = _CONTROLLER
    if controller is None or not controller.enabled:
        return
    if new_peak is not None:
        ts = new_peak if isinstance(new_peak, (int, float)) else getattr(new_peak, "peak_ts", None)
        if ts is not None:
            controller.on_peak(float(ts))
    controller.tick()


def render_hook(screen, scene=None) -> None:
    controller = _CONTROLLER
    if controller is None:return
    current=controller.scene if scene is None else scene
    if current is not controller.scene or not resolve_policy(current)[0]:controller.sync_policy(current)
    if not controller.is_flashing():return
    controller.render(screen)


def presented_hook(timestamp=None):
    """After the real display flip: timestamp when the local mask became visible/absent."""
    controller=_CONTROLLER
    if controller is not None and controller._glow is not None:
        controller._glow.presented(controller.clock() if timestamp is None else float(timestamp))


def attach_object_glow(query):
    controller=_CONTROLLER
    if controller is None or not controller.enabled or controller.mode!='object_glow' or controller._glow is None:
        query.finish(reason='object_glow_disabled')
    else:controller._glow.attach(query)
