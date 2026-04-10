from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.engine.audio.audio_peak_detector import AudioPeakEvent, audio_peak_detector
from src.engine.camera.camera_manager import camera_manager
from src.engine.input.hit_input import hit_input
from src.engine.settings import (
    load_camera_calibration,
    load_content_rect,
    load_scanport_rect,
    load_viewport_rect,
)


@dataclass
class ScanportFrame:
    timestamp: float
    gray: np.ndarray


@dataclass
class AudioShotEvent:
    shot_id: int
    peak_ts: float
    created_at: float
    state: str = "pending"
    emitted: bool = False
    matched_track_id: int | None = None
    matched_hole_id: int | None = None
    confidence: float = 0.0
    note: str = ""


@dataclass
class HoleTrack:
    track_id: int
    camera_x: float
    camera_y: float
    created_at: float
    first_seen_ts: float
    last_seen_ts: float
    hits: int = 1
    best_score: float = 0.0
    emitted: bool = False
    missed_frames: int = 0
    state: str = "tentative"
    last_candidate: dict[str, float] = field(default_factory=dict)


@dataclass
class HoleEntry:
    hole_id: int
    camera_x: float
    camera_y: float
    score: float
    first_seen_ts: float
    last_seen_ts: float
    hit_count: int = 1


class HitScanner:
    STATE_OFF = "OFF"
    STATE_ARMING = "ARMING"
    STATE_ACTIVE = "ACTIVE"

    def __init__(self) -> None:
        self.enabled = False
        self.state = self.STATE_OFF
        self.arm_duration_s = 1.0
        self.arm_until_ts = 0.0
        self.last_emit_ts = 0.0
        self.global_emit_cooldown_s = 0.0

        self.last_audio_event_ts = 0.0
        self.audio_event_count = 0
        self._audio_subscribed = False
        self._last_frame_ts: float | None = None

        self.frame_history: deque[ScanportFrame] = deque(maxlen=360)
        self.audio_events: deque[AudioShotEvent] = deque(maxlen=128)
        self._next_shot_id = 1
        self._next_track_id = 1
        self._next_hole_id = 1

        self.association_lead_s = 0.08
        self.association_lag_s = 0.22
        self.event_timeout_s = 0.95
        self.track_confirm_frames = 3
        self.track_confirm_span_s = 0.09
        self.track_drop_after_missed_frames = 5
        self.track_merge_radius_px = 12.0

        self.recent_bg_min_age_s = 0.18
        self.recent_bg_max_age_s = 1.20
        self.max_background_frames = 12

        self.min_area = 2.0
        self.max_area = 180.0
        self.min_radius = 0.8
        self.max_radius = 12.0
        self.min_circularity = 0.02
        self.border_margin = 3

        self.min_change_threshold = 5.0
        self.min_blackhat_threshold = 5.0
        self.min_score_threshold = 3.5
        self.patch_radius = 10
        self.inner_radius = 2
        self.outer_radius = 7
        self.min_center_darkening = 2.0
        self.min_local_contrast_gain = 0.6
        self.min_persistent_post_frames = 3

        self.duplicate_radius_px = 18.0
        self.rehit_radius_px = 12.0
        self.rehit_gain_required = 4.0
        self.max_known_holes = 512
        self.known_holes: list[dict[str, float]] = []

        self._active_tracks: dict[int, HoleTrack] = {}

        self.scene_reference_gray: np.ndarray | None = None
        self.surface_reference_gray: np.ndarray | None = None
        self.reference_capture_frames_needed = 6
        self.reference_capture_kind: str | None = None
        self.reference_capture_buffer: list[np.ndarray] = []

        self.last_status = "off"
        self.debug_frames: dict[str, np.ndarray] = {}
        self.last_candidates: list[dict[str, float]] = []
        self.last_stable_tracks: list[HoleTrack] = []
        self.last_threshold_value: float = 0.0
        self.last_change_threshold_value: float = 0.0
        self.last_vote_threshold_value: float = 0.0
        self.last_window_debug: dict[str, float] = {}
        self.last_best_candidate: dict[str, float] | None = None
        self.last_event_debug: dict[str, float | str] = {}

    # ------------------------------------------------------------------
    # Public control API
    # ------------------------------------------------------------------

    def enable(self) -> None:
        self.enabled = True
        self.state = self.STATE_ARMING
        self.arm_until_ts = time.time() + self.arm_duration_s
        self.last_emit_ts = 0.0
        self.last_audio_event_ts = 0.0
        self.audio_event_count = 0
        self._last_frame_ts = None
        if not self._audio_subscribed:
            audio_peak_detector.subscribe(self._on_audio_peak)
            self._audio_subscribed = True
        self.frame_history.clear()
        self.audio_events.clear()
        self._active_tracks.clear()
        self.reference_capture_buffer.clear()
        self.reference_capture_kind = None
        self.debug_frames.clear()
        self.last_candidates = []
        self.last_stable_tracks = []
        self.last_threshold_value = 0.0
        self.last_change_threshold_value = 0.0
        self.last_vote_threshold_value = 0.0
        self.last_window_debug = {}
        self.last_best_candidate = None
        self.last_event_debug = {}
        self.last_status = "arming"

    def disable(self) -> None:
        self.enabled = False
        self.state = self.STATE_OFF
        if self._audio_subscribed:
            audio_peak_detector.unsubscribe(self._on_audio_peak)
            self._audio_subscribed = False
        self.frame_history.clear()
        self.audio_events.clear()
        self._active_tracks.clear()
        self.reference_capture_buffer.clear()
        self.reference_capture_kind = None
        self.debug_frames.clear()
        self.last_candidates = []
        self.last_stable_tracks = []
        self.last_threshold_value = 0.0
        self.last_change_threshold_value = 0.0
        self.last_vote_threshold_value = 0.0
        self.last_window_debug = {}
        self.last_best_candidate = None
        self.last_event_debug = {}
        self.audio_event_count = 0
        self.last_status = "off"

    def request_scene_reference(self) -> None:
        self.reference_capture_kind = "scene"
        self.reference_capture_buffer.clear()

    def capture_scene_reference(self) -> None:
        self.request_scene_reference()

    def capture_surface_refresh(self, reset_holes: bool = False) -> None:
        self.reference_capture_kind = "surface"
        self.reference_capture_buffer.clear()
        if reset_holes:
            self.reset_hole_map()

    def reset_hole_map(self) -> None:
        self.known_holes.clear()
        self._active_tracks.clear()

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        del dt
        if not self.enabled or self.state == self.STATE_OFF:
            self.last_candidates = []
            self.last_stable_tracks = []
            return

        frame_bgr = camera_manager.get_latest_frame()
        if frame_bgr is None:
            self.last_status = "no_camera_frame"
            return

        frame_ts = camera_manager.get_latest_timestamp() or time.time()
        now = time.time()

        is_new_frame = self._last_frame_ts is None or abs(frame_ts - self._last_frame_ts) > 1e-6
        if is_new_frame:
            self._last_frame_ts = frame_ts
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            self.frame_history.append(ScanportFrame(timestamp=frame_ts, gray=gray))
            self.debug_frames["camera_gray"] = gray
        else:
            gray = self.frame_history[-1].gray if self.frame_history else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.state == self.STATE_ARMING:
            if now >= self.arm_until_ts:
                self.state = self.STATE_ACTIVE
                self.last_status = "active"
                if self.scene_reference_gray is None:
                    self.request_scene_reference()
            else:
                self.last_status = "arming"
                return

        if is_new_frame:
            self._maybe_capture_reference(gray)
            if self._has_open_events():
                candidates = self._detect_frame_candidates(gray=gray, frame_ts=frame_ts)
                self._update_tracks(candidates, frame_ts)
            else:
                self._age_tracks_without_candidates(frame_ts)
                self.last_candidates = []

        emitted_now = self._resolve_audio_events(now)
        self._prune_finished_events(now)
        self._prune_old_holes(now)
        self._refresh_debug_views()

        self.last_status = (
            f"active peaks={self.audio_event_count} events={len(self.audio_events)} "
            f"tracks={len(self._active_tracks)} holes={len(self.known_holes)} emit={emitted_now}"
        )

    # ------------------------------------------------------------------
    # Debug/status
    # ------------------------------------------------------------------

    def get_debug_snapshot(self) -> dict:
        return {
            "enabled": self.enabled,
            "state": self.state,
            "last_status": self.last_status,
            "audio_event_count": self.audio_event_count,
            "pending_trigger_windows": len(self.audio_events),
            "known_holes_count": len(self.known_holes),
            "candidates_count": len(self.last_candidates),
            "stable_tracks_count": len(self.last_stable_tracks),
            "debug_frames": dict(self.debug_frames),
            "candidates": [dict(c) for c in self.last_candidates],
            "stable_tracks": [
                {
                    "track_id": t.track_id,
                    "camera_x": t.camera_x,
                    "camera_y": t.camera_y,
                    "hits": t.hits,
                    "score": t.best_score,
                    "emitted": t.emitted,
                    "state": t.state,
                }
                for t in self.last_stable_tracks
            ],
            "known_holes": list(self.known_holes),
            "threshold_value": self.last_threshold_value,
            "change_threshold_value": self.last_change_threshold_value,
            "vote_threshold_value": self.last_vote_threshold_value,
            "window_debug": dict(self.last_window_debug),
            "best_candidate": None if self.last_best_candidate is None else dict(self.last_best_candidate),
            "last_event_debug": dict(self.last_event_debug),
        }

    def get_status_lines(self) -> list[str]:
        ref_state = "scene" if self.scene_reference_gray is not None else "none"
        if self.surface_reference_gray is not None:
            ref_state += "+surface"
        return [
            f"HitScanner state: {self.state}",
            f"Audio peaks heard: {self.audio_event_count}",
            f"Pending events: {len(self.audio_events)}",
            f"Tracks: {len(self._active_tracks)} | Known holes: {len(self.known_holes)}",
            f"Reference: {ref_state}",
            f"Thr combined: {self.last_threshold_value:.1f}",
            f"Thr change: {self.last_change_threshold_value:.1f}",
            f"Thr vote: {self.last_vote_threshold_value:.1f}",
            f"Status: {self.last_status}",
        ]

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _on_audio_peak(self, ev: AudioPeakEvent) -> None:
        self.last_audio_event_ts = max(self.last_audio_event_ts, ev.timestamp)
        self.audio_event_count += 1
        if not self.enabled or self.state != self.STATE_ACTIVE:
            return
        self.audio_events.append(
            AudioShotEvent(
                shot_id=self._next_shot_id,
                peak_ts=ev.timestamp,
                created_at=time.time(),
            )
        )
        self._next_shot_id += 1

    def _has_open_events(self) -> bool:
        return any(ev.state == "pending" for ev in self.audio_events)

    def _resolve_audio_events(self, now_ts: float) -> int:
        emitted_now = 0
        for event in list(self.audio_events):
            if event.state != "pending":
                continue

            track = self._best_track_for_event(event)
            if track is not None:
                event.matched_track_id = track.track_id
                event.confidence = max(event.confidence, float(track.best_score))
                self.last_event_debug = {
                    "shot_id": float(event.shot_id),
                    "track_id": float(track.track_id),
                    "score": float(track.best_score),
                    "peak_ts": float(event.peak_ts),
                }

                if self._track_is_ready(track, now_ts, event):
                    if self._emit_track_result(track, event):
                        emitted_now += 1
                    continue

            if now_ts - event.peak_ts >= self.event_timeout_s:
                event.state = "missed"
                event.note = "timeout"

        return emitted_now

    def _prune_finished_events(self, now_ts: float) -> None:
        keep: list[AudioShotEvent] = []
        for ev in self.audio_events:
            age = now_ts - ev.peak_ts
            if ev.state == "pending" or age <= max(1.5, self.event_timeout_s + 0.5):
                keep.append(ev)
        self.audio_events = deque(keep, maxlen=self.audio_events.maxlen)

    def _best_track_for_event(self, event: AudioShotEvent) -> HoleTrack | None:
        best: HoleTrack | None = None
        best_metric: tuple[float, float] | None = None
        for track in self._active_tracks.values():
            onset_dt = track.first_seen_ts - event.peak_ts
            if onset_dt < -self.association_lead_s or onset_dt > self.association_lag_s:
                continue
            if track.emitted and event.matched_track_id != track.track_id:
                continue
            metric = (abs(onset_dt), -track.best_score)
            if best is None or metric < best_metric:
                best = track
                best_metric = metric
        return best

    def _track_is_ready(self, track: HoleTrack, now_ts: float, event: AudioShotEvent) -> bool:
        age = track.last_seen_ts - track.first_seen_ts
        if track.hits >= self.track_confirm_frames and age >= self.track_confirm_span_s:
            return True
        if now_ts - event.peak_ts >= 0.30 and track.hits >= max(2, self.track_confirm_frames - 1):
            return True
        return False

    def _emit_track_result(self, track: HoleTrack, event: AudioShotEvent) -> bool:
        near = self._is_near_known_hole(track.camera_x, track.camera_y)
        if near is not None:
            hole, dist = near
            required = float(hole.get("score", 0.0)) + self.rehit_gain_required
            if dist <= self.rehit_radius_px or track.best_score < required:
                hole["hit_count"] = float(hole.get("hit_count", 1.0) + 1.0)
                hole["timestamp"] = time.time()
                hole["last_score"] = float(track.best_score)
                hit_input.push_camera_hit(hole["camera_x"], hole["camera_y"])
                event.state = "rehit"
                event.emitted = True
                event.matched_hole_id = int(hole.get("hole_id", 0))
                track.emitted = True
                track.state = "rehit"
                self.last_emit_ts = time.time()
                self.last_best_candidate = {
                    "camera_x": float(hole["camera_x"]),
                    "camera_y": float(hole["camera_y"]),
                    "score": float(track.best_score),
                    "event": float(event.shot_id),
                }
                return True

        hole = self._remember_known_hole(track)
        hit_input.push_camera_hit(track.camera_x, track.camera_y)
        event.state = "matched"
        event.emitted = True
        event.matched_hole_id = int(hole.get("hole_id", 0))
        track.emitted = True
        track.state = "confirmed"
        self.last_emit_ts = time.time()
        self.last_best_candidate = {
            "camera_x": float(track.camera_x),
            "camera_y": float(track.camera_y),
            "score": float(track.best_score),
            "event": float(event.shot_id),
        }
        return True

    # ------------------------------------------------------------------
    # Reference capture / background models
    # ------------------------------------------------------------------

    def _maybe_capture_reference(self, gray: np.ndarray) -> None:
        if self.reference_capture_kind is None and self.scene_reference_gray is None and len(self.frame_history) >= 6:
            self.reference_capture_kind = "scene"
            self.reference_capture_buffer.clear()

        if self.reference_capture_kind is None:
            return

        if self._has_open_events():
            return

        self.reference_capture_buffer.append(gray.copy())
        if len(self.reference_capture_buffer) < self.reference_capture_frames_needed:
            return

        ref = np.median(np.stack(self.reference_capture_buffer, axis=0), axis=0).astype(np.uint8)
        if self.reference_capture_kind == "scene":
            self.scene_reference_gray = ref
            self.debug_frames["scene_reference"] = ref
        elif self.reference_capture_kind == "surface":
            self.surface_reference_gray = ref
            self.debug_frames["surface_reference"] = ref
        self.reference_capture_buffer.clear()
        self.reference_capture_kind = None

    def _build_recent_background(self, frame_ts: float) -> np.ndarray | None:
        candidates: list[np.ndarray] = []
        for fr in reversed(self.frame_history):
            age = frame_ts - fr.timestamp
            if age < self.recent_bg_min_age_s:
                continue
            if age > self.recent_bg_max_age_s:
                break
            candidates.append(fr.gray)
            if len(candidates) >= self.max_background_frames:
                break

        if len(candidates) < 3:
            return None
        return np.median(np.stack(candidates, axis=0), axis=0).astype(np.uint8)

    def _build_pre_shot_background(self) -> np.ndarray | None:
        """
        Bygg en bakgrundsbild från frames INNAN kulan träffade tavlan.

        Mikrofonen sitter i kameran, ~50cm från tavlan.
        Tidslinje vid 6m avstånd:
        - t=0: Avfyrning
        - t=30-60ms: Kulan träffar tavlan → hål uppstår
        - t=~1.5ms efter träff: Mikrofon hör smällen (50cm / 340m/s)

        Audio peak ≈ kulan har redan träffat. Hålet finns redan i bilden.
        Vi behöver frames från INNAN peak_ts för att ha en ren bakgrund.

        Vid 30fps (33ms/frame) och snabbskytte (5 skott/s = 200ms mellanrum)
        tar vi frames som är minst 40ms äldre än peak_ts (säkerhetsmarginal
        för att kulan kan ha träffat 1-2 frames innan ljudet nådde mikrofonen).
        """
        earliest_peak_ts = None
        for ev in self.audio_events:
            if ev.state == "pending":
                if earliest_peak_ts is None or ev.peak_ts < earliest_peak_ts:
                    earliest_peak_ts = ev.peak_ts

        if earliest_peak_ts is None:
            return None

        # Kulan träffade ~1-2 frames innan audio peak (30-66ms vid 30fps).
        # Vi vill ha frames från innan kulan träffade, alltså minst 40ms
        # före peak_ts, och max 200ms tillbaka (snabbskytte-gräns).
        pre_shot_frames: list[np.ndarray] = []
        cutoff_latest = earliest_peak_ts - 0.04   # 40ms före peak
        cutoff_earliest = earliest_peak_ts - 0.20  # 200ms före peak

        for fr in reversed(self.frame_history):
            if fr.timestamp > cutoff_latest:
                continue
            if fr.timestamp < cutoff_earliest:
                break
            pre_shot_frames.append(fr.gray)
            if len(pre_shot_frames) >= 5:
                break

        if len(pre_shot_frames) < 1:
            return None
        if len(pre_shot_frames) == 1:
            return pre_shot_frames[0]
        return np.median(np.stack(pre_shot_frames, axis=0), axis=0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Detection / tracking
    # ------------------------------------------------------------------

    def _detect_frame_candidates(self, gray: np.ndarray, frame_ts: float) -> list[dict[str, float]]:
        roi_mask = self._frame_roi_mask(gray.shape)
        roi_pixels = int(np.count_nonzero(roi_mask))
        self.last_window_debug = {
            "roi_pixels": float(roi_pixels),
            "frame_ts": float(frame_ts),
            "pending_events": float(sum(1 for ev in self.audio_events if ev.state == "pending")),
        }
        if roi_pixels <= 0:
            self.last_threshold_value = 0.0
            self.last_change_threshold_value = 0.0
            self.last_vote_threshold_value = 0.0
            return []

        pre_shot_bg = self._build_pre_shot_background()
        if pre_shot_bg is None or pre_shot_bg.shape != gray.shape:
            # Ingen pre-shot-referens ännu — fall tillbaka på legacy-detektion
            return self._detect_frame_candidates_legacy(gray, frame_ts, roi_mask, roi_pixels)

        pre_shot_blur = cv2.GaussianBlur(pre_shot_bg, (5, 5), 0)

        # ---- Samla post-shot frames och diffa var och en mot pre-shot ----
        # Hitta den senaste pending audio-peaken
        earliest_peak_ts = None
        for ev in self.audio_events:
            if ev.state == "pending":
                if earliest_peak_ts is None or ev.peak_ts < earliest_peak_ts:
                    earliest_peak_ts = ev.peak_ts

        # Samla frames efter skottet (peak_ts + 20ms och framåt)
        post_shot_frames: list[np.ndarray] = []
        if earliest_peak_ts is not None:
            for fr in self.frame_history:
                if fr.timestamp < earliest_peak_ts + 0.02:
                    continue
                post_shot_frames.append(fr.gray)

        # Inkludera nuvarande frame
        post_shot_frames.append(gray)

        if len(post_shot_frames) < 1:
            return []

        self.last_window_debug["post_shot_frames"] = float(len(post_shot_frames))

        # ---- Bygg vote-map: för varje pixel, i hur många frames har den ändrats? ----
        # En förändring som syns i de flesta frames = persistent (hål/märke).
        # En förändring som bara syns i 1-2 frames = skakning/blänk/brus.
        diff_threshold = 8  # Minsta pixelskillnad för att räknas som "ändrad"
        vote_map = np.zeros(gray.shape, dtype=np.float32)

        for post_frame in post_shot_frames:
            post_blur = cv2.GaussianBlur(post_frame, (5, 5), 0)
            frame_diff = cv2.absdiff(pre_shot_blur, post_blur)

            # Subtrahera global shift (projektorflicker, autoexponering)
            roi_vals = frame_diff[roi_mask > 0]
            if roi_vals.size > 0:
                global_shift = float(np.median(roi_vals))
            else:
                global_shift = 0.0
            shift_val = int(min(255, max(0, round(global_shift))))
            compensated = cv2.subtract(frame_diff, np.full_like(frame_diff, shift_val))

            # Räkna vote: pixlar med förändring över tröskeln
            vote_map += (compensated > diff_threshold).astype(np.float32)

        # Normalisera till 0-255 baserat på antal frames
        n_frames = float(len(post_shot_frames))
        # vote_ratio: andel frames där pixeln ändrats (0.0 - 1.0)
        vote_ratio = vote_map / max(1.0, n_frames)

        # Kräv att förändringen syns i minst 50% av frames
        min_vote_ratio = 0.5
        persistent_mask = (vote_ratio >= min_vote_ratio).astype(np.uint8) * 255
        persistent_mask = cv2.bitwise_and(persistent_mask, roi_mask)

        # Kolla om för stor del av bilden är persistent (video/animation)
        persistent_pixels = int(np.count_nonzero(persistent_mask))
        change_ratio = float(persistent_pixels) / max(1, roi_pixels)
        self.last_window_debug["change_ratio"] = change_ratio
        self.last_window_debug["global_shift"] = global_shift

        if change_ratio > 0.04:
            self.last_window_debug["pre_shot_rejected"] = 1.0
            return self._detect_frame_candidates_legacy(gray, frame_ts, roi_mask, roi_pixels)

        # ---- Bygg combined map från vote-data ----
        # Skala vote_ratio till 0-255 för tröskling och konturanalys
        combined = np.clip(vote_ratio * 255.0, 0, 255).astype(np.uint8)
        combined = cv2.bitwise_and(combined, roi_mask)

        # Använd senaste frame-diff som delta-map (för patch-verifiering)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        current_diff = cv2.absdiff(pre_shot_blur, gray_blur)
        roi_vals = current_diff[roi_mask > 0]
        if roi_vals.size > 0:
            gs = float(np.median(roi_vals))
        else:
            gs = 0.0
        gs_val = int(min(255, max(0, round(gs))))
        combined_delta = cv2.subtract(current_diff, np.full_like(current_diff, gs_val))
        combined_delta = cv2.bitwise_and(combined_delta, roi_mask)

        self.debug_frames["pre_shot_delta"] = combined_delta
        self.debug_frames["vote_map"] = np.clip(vote_map * (255.0 / max(1.0, n_frames)), 0, 255).astype(np.uint8)

        # Tröskling
        roi_values = combined[roi_mask > 0]
        if roi_values.size == 0:
            return []

        adaptive_thr = float(np.percentile(roi_values, 98.5))
        self.last_threshold_value = max(3.0, adaptive_thr)
        self.last_change_threshold_value = 0.0
        self.last_vote_threshold_value = 0.0

        merged = persistent_mask.copy()
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        self.debug_frames["roi_polygon"] = roi_mask
        self.debug_frames["combined_map"] = combined
        self.debug_frames["change_map"] = combined_delta
        self.debug_frames["candidate_mask"] = merged

        # ---- Hitta konturer i persistent mask ----
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[dict[str, float]] = []
        blackhat = np.zeros_like(gray_blur)  # Dummy för _verify_patch-signaturen

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > self.max_area:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            circularity = 0.0
            if perimeter > 1e-6:
                circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))
            if circularity < self.min_circularity:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            radius = float(radius)
            if radius < self.min_radius or radius > self.max_radius:
                continue

            if (
                cx < self.border_margin
                or cy < self.border_margin
                or cx >= gray.shape[1] - self.border_margin
                or cy >= gray.shape[0] - self.border_margin
            ):
                continue

            # Vote-score: genomsnittlig vote_ratio i centrum
            ix, iy = int(round(cx)), int(round(cy))
            r_check = max(2, int(radius))
            y0c = max(0, iy - r_check)
            y1c = min(gray.shape[0], iy + r_check + 1)
            x0c = max(0, ix - r_check)
            x1c = min(gray.shape[1], ix + r_check + 1)
            if y1c > y0c and x1c > x0c:
                center_vote = float(np.mean(vote_ratio[y0c:y1c, x0c:x1c]))
            else:
                center_vote = 0.0

            patch = self._verify_patch(
                gray=gray_blur,
                combined=combined,
                combined_delta=combined_delta,
                blackhat=blackhat,
                pre_shot_delta=combined_delta,
                camera_x=cx,
                camera_y=cy,
                radius=radius,
            )
            if patch is None:
                continue

            # Boost score med vote-persistens
            base_score = patch["score"]
            score = base_score + 5.0 * center_vote

            candidate = {
                "camera_x": float(cx),
                "camera_y": float(cy),
                "area": float(area),
                "radius": float(radius),
                "circularity": float(circularity),
                "score": float(score),
                "center_darkening": float(patch["center_darkening"]),
                "local_contrast_gain": float(patch["local_contrast_gain"]),
                "blackhat_value": float(patch["blackhat_value"]),
                "change_value": float(patch["change_value"]),
                "pre_shot_change": float(center_vote),
                "timestamp": float(frame_ts),
            }
            candidates.append(candidate)

        candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        self.last_candidates = candidates[:24]
        return self.last_candidates

    def _detect_frame_candidates_legacy(
        self,
        gray: np.ndarray,
        frame_ts: float,
        roi_mask: np.ndarray,
        roi_pixels: int,
    ) -> list[dict[str, float]]:
        """Fallback-detektion utan pre-shot-referens (scene_ref + blackhat)."""
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        delta_maps: list[np.ndarray] = []
        if self.scene_reference_gray is not None and self.scene_reference_gray.shape == gray.shape:
            delta_maps.append(cv2.absdiff(
                cv2.GaussianBlur(self.scene_reference_gray, (5, 5), 0), gray_blur
            ))
        if self.surface_reference_gray is not None and self.surface_reference_gray.shape == gray.shape:
            delta_maps.append(cv2.absdiff(
                cv2.GaussianBlur(self.surface_reference_gray, (5, 5), 0), gray_blur
            ))
        recent_bg = self._build_recent_background(frame_ts)
        if recent_bg is not None:
            delta_maps.append(cv2.absdiff(
                cv2.GaussianBlur(recent_bg, (5, 5), 0), gray_blur
            ))

        blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        blackhat = cv2.morphologyEx(gray_blur, cv2.MORPH_BLACKHAT, blackhat_kernel)

        if delta_maps:
            fallback_delta = delta_maps[0]
            for dmap in delta_maps[1:]:
                fallback_delta = np.maximum(fallback_delta, dmap)
        else:
            fallback_delta = np.zeros_like(gray_blur)

        combined_delta = fallback_delta
        combined = np.maximum(fallback_delta, blackhat)
        combined = cv2.bitwise_and(combined, roi_mask)
        combined_delta = cv2.bitwise_and(combined_delta, roi_mask)

        roi_values = combined[roi_mask > 0]
        if roi_values.size == 0:
            return []

        adaptive_thr = float(np.percentile(roi_values, 98.8))
        change_thr = float(np.percentile(combined_delta[roi_mask > 0], 98.5))

        self.last_threshold_value = max(self.min_score_threshold, adaptive_thr)
        self.last_change_threshold_value = max(self.min_change_threshold, change_thr)
        self.last_vote_threshold_value = 0.0

        combined_bin = (combined >= self.last_threshold_value).astype(np.uint8) * 255
        change_bin = (combined_delta >= self.last_change_threshold_value).astype(np.uint8) * 255
        merged = cv2.bitwise_or(combined_bin, change_bin)
        merged = cv2.bitwise_and(merged, roi_mask)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        self.debug_frames["roi_polygon"] = roi_mask
        self.debug_frames["combined_map"] = combined
        self.debug_frames["change_map"] = combined_delta
        self.debug_frames["blackhat_map"] = blackhat
        self.debug_frames["candidate_mask"] = merged

        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[dict[str, float]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > self.max_area:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            circularity = 0.0
            if perimeter > 1e-6:
                circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))
            if circularity < self.min_circularity:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            radius = float(radius)
            if radius < self.min_radius or radius > self.max_radius:
                continue

            if (
                cx < self.border_margin
                or cy < self.border_margin
                or cx >= gray.shape[1] - self.border_margin
                or cy >= gray.shape[0] - self.border_margin
            ):
                continue

            patch = self._verify_patch(
                gray=gray_blur,
                combined=combined,
                combined_delta=combined_delta,
                blackhat=blackhat,
                pre_shot_delta=None,
                camera_x=cx,
                camera_y=cy,
                radius=radius,
            )
            if patch is None:
                continue

            candidate = {
                "camera_x": float(cx),
                "camera_y": float(cy),
                "area": float(area),
                "radius": float(radius),
                "circularity": float(circularity),
                "score": float(patch["score"]),
                "center_darkening": float(patch["center_darkening"]),
                "local_contrast_gain": float(patch["local_contrast_gain"]),
                "blackhat_value": float(patch["blackhat_value"]),
                "change_value": float(patch["change_value"]),
                "pre_shot_change": float(patch.get("pre_shot_change", 0.0)),
                "timestamp": float(frame_ts),
            }
            candidates.append(candidate)

        candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        self.last_candidates = candidates[:24]
        return self.last_candidates

    def _verify_patch(
        self,
        *,
        gray: np.ndarray,
        combined: np.ndarray,
        combined_delta: np.ndarray,
        blackhat: np.ndarray,
        pre_shot_delta: np.ndarray | None,
        camera_x: float,
        camera_y: float,
        radius: float,
    ) -> dict[str, float] | None:
        x = int(round(camera_x))
        y = int(round(camera_y))
        r = max(self.patch_radius, int(np.ceil(radius * 3.0)))
        x0 = max(0, x - r)
        y0 = max(0, y - r)
        x1 = min(gray.shape[1], x + r + 1)
        y1 = min(gray.shape[0], y + r + 1)
        if x1 <= x0 or y1 <= y0:
            return None

        patch_combined = combined[y0:y1, x0:x1]
        patch_delta = combined_delta[y0:y1, x0:x1]

        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist_sq = (xx - camera_x) ** 2 + (yy - camera_y) ** 2
        center_mask = dist_sq <= float(self.inner_radius * self.inner_radius)
        ring_mask = (dist_sq >= float(self.outer_radius * self.outer_radius)) & (
            dist_sq <= float((self.outer_radius + 3) * (self.outer_radius + 3))
        )

        if not np.any(center_mask) or not np.any(ring_mask):
            return None

        # center_change: hur mycket har centrum förändrats (ljusare ELLER mörkare).
        # Ersätter gamla center_darkening som bara fångade mörkare.
        center_change = float(np.mean(patch_combined[center_mask]))
        change_value = float(np.mean(patch_delta[center_mask]))
        ring_value = float(np.mean(patch_combined[ring_mask]))

        # local_contrast: absolut skillnad mellan centrum och ring.
        # Hål/märken har hög kontrast mot omgivningen oavsett riktning.
        local_contrast = abs(center_change - ring_value)

        # Pre-shot change: hur mycket har denna punkt förändrats sedan
        # precis innan skottet? Hög = nytt hål/märke.
        pre_shot_change = 0.0
        if pre_shot_delta is not None:
            patch_pre = pre_shot_delta[y0:y1, x0:x1]
            pre_shot_change = float(np.mean(patch_pre[center_mask]))

        if center_change < self.min_center_darkening:
            return None
        if local_contrast < self.min_local_contrast_gain:
            return None

        # Score: pre-shot-förändring dominerar om tillgänglig.
        if pre_shot_change > 0.5:
            score = (
                0.25 * center_change
                + 0.15 * change_value
                + 0.15 * local_contrast
                + 0.80 * pre_shot_change
            )
        else:
            score = (
                0.45 * center_change
                + 0.35 * change_value
                + 0.35 * local_contrast
            )
        if score < self.min_score_threshold:
            return None

        return {
            "score": float(score),
            "center_darkening": float(center_change),
            "change_value": float(change_value),
            "blackhat_value": 0.0,
            "local_contrast_gain": float(local_contrast),
            "pre_shot_change": float(pre_shot_change),
        }

    def _update_tracks(self, candidates: list[dict[str, float]], frame_ts: float) -> None:
        for track in self._active_tracks.values():
            track.missed_frames += 1

        for candidate in candidates:
            best_track: HoleTrack | None = None
            best_dist = float("inf")
            for track in self._active_tracks.values():
                dist = float(np.hypot(track.camera_x - candidate["camera_x"], track.camera_y - candidate["camera_y"]))
                if dist <= self.track_merge_radius_px and dist < best_dist:
                    best_track = track
                    best_dist = dist

            if best_track is None:
                track = HoleTrack(
                    track_id=self._next_track_id,
                    camera_x=float(candidate["camera_x"]),
                    camera_y=float(candidate["camera_y"]),
                    created_at=float(frame_ts),
                    first_seen_ts=float(frame_ts),
                    last_seen_ts=float(frame_ts),
                    hits=1,
                    best_score=float(candidate.get("score", 0.0)),
                    emitted=False,
                    missed_frames=0,
                    state="tentative",
                    last_candidate=dict(candidate),
                )
                self._active_tracks[track.track_id] = track
                self._next_track_id += 1
                continue

            alpha = 0.35
            best_track.camera_x = float((1.0 - alpha) * best_track.camera_x + alpha * candidate["camera_x"])
            best_track.camera_y = float((1.0 - alpha) * best_track.camera_y + alpha * candidate["camera_y"])
            best_track.last_seen_ts = float(frame_ts)
            best_track.hits += 1
            best_track.best_score = float(max(best_track.best_score, candidate.get("score", 0.0)))
            best_track.last_candidate = dict(candidate)
            best_track.missed_frames = 0
            if best_track.hits >= self.track_confirm_frames:
                best_track.state = "stable"

        self._drop_dead_tracks(frame_ts)

    def _age_tracks_without_candidates(self, frame_ts: float) -> None:
        for track in self._active_tracks.values():
            track.missed_frames += 1
        self._drop_dead_tracks(frame_ts)

    def _drop_dead_tracks(self, frame_ts: float) -> None:
        to_delete: list[int] = []
        for track_id, track in self._active_tracks.items():
            if track.missed_frames > self.track_drop_after_missed_frames:
                if track.hits < 2 and (frame_ts - track.first_seen_ts) < 0.12:
                    to_delete.append(track_id)
                elif track.emitted:
                    to_delete.append(track_id)
                elif (frame_ts - track.last_seen_ts) > 0.60:
                    to_delete.append(track_id)
        for track_id in to_delete:
            self._active_tracks.pop(track_id, None)

    def _refresh_debug_views(self) -> None:
        stable = list(self._active_tracks.values())
        stable.sort(key=lambda t: t.best_score, reverse=True)
        self.last_stable_tracks = stable[:8]

    # ------------------------------------------------------------------
    # ROI / hole map
    # ------------------------------------------------------------------

    def _frame_roi_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """
        Build the search mask in full camera-frame coordinates.

        Important:
        content_rect is viewport-local in this project, not absolute screen coords.
        So for homography/inverse_homography we must first convert content_rect
        to absolute screen coords by offsetting with viewport_rect.x/y.
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)

        calibration = load_camera_calibration() or {}
        raw_inv = calibration.get("inverse_homography")
        content = load_content_rect()
        viewport = load_viewport_rect()

        if raw_inv and content is not None and viewport is not None:
            try:
                H_inv = np.array(raw_inv, dtype=np.float32)
                if H_inv.shape == (3, 3):
                    screen_pts = np.array(
                        [
                            [viewport.x + content.x, viewport.y + content.y],
                            [viewport.x + content.x + content.w, viewport.y + content.y],
                            [viewport.x + content.x + content.w, viewport.y + content.y + content.h],
                            [viewport.x + content.x, viewport.y + content.y + content.h],
                        ],
                        dtype=np.float32,
                    ).reshape(-1, 1, 2)
                    cam_pts = cv2.perspectiveTransform(screen_pts, H_inv).reshape(-1, 2)
                    cam_pts[:, 0] = np.clip(cam_pts[:, 0], 0, w - 1)
                    cam_pts[:, 1] = np.clip(cam_pts[:, 1], 0, h - 1)
                    polygon = np.round(cam_pts).astype(np.int32)
                    if cv2.contourArea(polygon) >= 10.0:
                        cv2.fillConvexPoly(mask, polygon, 255)
                        roi_debug = np.zeros((h, w), dtype=np.uint8)
                        cv2.polylines(roi_debug, [polygon], True, 255, 2)
                        self.debug_frames["roi_polygon"] = roi_debug
                        return mask
            except Exception:
                pass

        # Fallback: older ratio-based content mapping inside scanport.
        scanport = load_scanport_rect()
        if scanport is None:
            mask[:, :] = 255
            self.debug_frames["roi_polygon"] = np.zeros((h, w), dtype=np.uint8)
            return mask

        x0 = int(max(0, min(w, round(scanport.x))))
        y0 = int(max(0, min(h, round(scanport.y))))
        x1 = int(max(0, min(w, round(scanport.x + scanport.w))))
        y1 = int(max(0, min(h, round(scanport.y + scanport.h))))
        if x1 <= x0 or y1 <= y0:
            mask[:, :] = 255
            self.debug_frames["roi_polygon"] = np.zeros((h, w), dtype=np.uint8)
            return mask

        mask[y0:y1, x0:x1] = 255
        roi_debug = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(roi_debug, (x0, y0), (x1 - 1, y1 - 1), 255, 2)
        self.debug_frames["roi_polygon"] = roi_debug
        return mask

    def _is_near_known_hole(self, camera_x: float, camera_y: float) -> tuple[dict[str, float], float] | None:
        best: tuple[dict[str, float], float] | None = None
        for hole in self.known_holes:
            dist = float(np.hypot(float(hole["camera_x"]) - camera_x, float(hole["camera_y"]) - camera_y))
            if dist > self.duplicate_radius_px:
                continue
            if best is None or dist < best[1]:
                best = (hole, dist)
        return best

    def _remember_known_hole(self, track: HoleTrack) -> dict[str, float]:
        hole = {
            "hole_id": float(self._next_hole_id),
            "camera_x": float(track.camera_x),
            "camera_y": float(track.camera_y),
            "score": float(track.best_score),
            "timestamp": time.time(),
            "hit_count": 1.0,
        }
        self._next_hole_id += 1
        self.known_holes.append(hole)
        if len(self.known_holes) > self.max_known_holes:
            self.known_holes = self.known_holes[-self.max_known_holes :]
        return hole

    def _prune_old_holes(self, now_ts: float) -> None:
        del now_ts
        if len(self.known_holes) > self.max_known_holes:
            self.known_holes = self.known_holes[-self.max_known_holes :]


hit_scanner = HitScanner()
