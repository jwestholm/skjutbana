from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from src.engine.audio.audio_peak_detector import AudioPeakEvent, audio_peak_detector
from src.engine.camera.camera_manager import camera_manager
from src.engine.input.hit_input import hit_input
from src.engine.settings import (
    load_scanport_rect,
    load_scanner_debug_overlay_enabled,
    load_viewport_rect,
)


@dataclass
class ScanportFrame:
    timestamp: float
    gray: np.ndarray


@dataclass
class TriggerWindow:
    peak_ts: float
    processed: bool = False
    created_at: float = 0.0


@dataclass
class HoleTrack:
    camera_x: float
    camera_y: float
    created_at: float
    last_seen: float
    hits: int = 1
    best_score: float = 0.0
    emitted: bool = False


class HitScanner:
    """
    Robust audio-assisterad träffscanner.

    Mål:
    - Ett skott -> max en vinnare
    - Gamla hål ignoreras normalt
    - Hål-i-hål tillåts om förändringen är tydligt starkare än tidigare
    - Tejp/projektor/brus ska få svårt att vinna

    Viktigt:
    - Detektion sker i scanporten
    - Punkten som skickas vidare är full-kamera-koordinat
    - Mapping till viewport/content sker i hit_input
    """

    STATE_OFF = "OFF"
    STATE_ARMING = "ARMING"
    STATE_ACTIVE = "ACTIVE"

    def __init__(self) -> None:
        self.enabled = False
        self.state = self.STATE_OFF

        self.arm_duration_s = 1.0
        self.arm_until_ts = 0.0

        self.last_emit_ts = 0.0
        self.global_emit_cooldown_s = 0.18

        self.last_audio_event_ts = 0.0
        self.audio_event_count = 0
        self._audio_subscribed = False

        # Historik
        self.frame_history: deque[ScanportFrame] = deque(maxlen=240)
        self.trigger_windows: deque[TriggerWindow] = deque(maxlen=48)

        # Tidsfönster runt peak
        self.pre_start_s = 0.28
        self.pre_end_s = 0.05
        self.post_start_s = 0.03
        self.post_end_s = 0.40
        self.analysis_lag_s = 0.42

        # Kandidatfilter
        self.min_area = 3.0
        self.max_area = 220.0
        self.min_radius = 1.0
        self.max_radius = 12.0
        self.min_circularity = 0.015
        self.border_margin = 3

        # Tröskelgolv
        self.min_change_threshold = 4.0
        self.min_combined_threshold = 8.0
        self.min_vote_threshold = 1.0

        # Patch-verifiering
        self.patch_radius = 8
        self.inner_radius = 2
        self.outer_radius = 6

        self.min_center_darkening = 4.0
        self.min_onset_darkening = 3.5
        self.min_late_darkening = 3.5
        self.min_local_contrast_gain = 0.8
        self.min_persistent_post_frames = 2

        # Kända hål / re-hit
        self.duplicate_radius_px = 22.0
        self.rehit_gain_required = 4.0
        self.max_known_holes = 256

        # Debug
        self.last_status = "off"
        self.debug_frames: dict[str, np.ndarray] = {}
        self.last_candidates: list[dict[str, float]] = []
        self.last_stable_tracks: list[HoleTrack] = []
        self.last_threshold_value: float = 0.0
        self.last_change_threshold_value: float = 0.0
        self.last_vote_threshold_value: float = 0.0
        self.last_window_debug: dict[str, float] = {}
        self.last_best_candidate: dict[str, float] | None = None

        # Kända hål lagras som dictar för att kunna jämföra styrka vid re-hit
        self.known_holes: list[dict[str, float]] = []

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def enable(self) -> None:
        self.enabled = True
        self.state = self.STATE_ARMING
        self.arm_until_ts = time.time() + self.arm_duration_s
        self.last_emit_ts = 0.0
        self.last_audio_event_ts = 0.0
        self.audio_event_count = 0

        if not self._audio_subscribed:
            audio_peak_detector.subscribe(self._on_audio_peak)
            self._audio_subscribed = True

        self.frame_history.clear()
        self.trigger_windows.clear()
        self.debug_frames.clear()
        self.last_candidates = []
        self.last_stable_tracks = []
        self.last_threshold_value = 0.0
        self.last_change_threshold_value = 0.0
        self.last_vote_threshold_value = 0.0
        self.last_window_debug = {}
        self.last_best_candidate = None
        self.last_status = "arming"

    def disable(self) -> None:
        self.enabled = False
        self.state = self.STATE_OFF

        if self._audio_subscribed:
            audio_peak_detector.unsubscribe(self._on_audio_peak)
            self._audio_subscribed = False

        self.frame_history.clear()
        self.trigger_windows.clear()
        self.debug_frames.clear()
        self.last_candidates = []
        self.last_stable_tracks = []
        self.last_threshold_value = 0.0
        self.last_change_threshold_value = 0.0
        self.last_vote_threshold_value = 0.0
        self.last_window_debug = {}
        self.last_best_candidate = None
        self.audio_event_count = 0
        self.last_status = "off"

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

        scanport_gray = self._extract_scanport_gray(frame_bgr)
        if scanport_gray is None:
            self.last_status = "scanport_not_ready"
            return

        now = time.time()
        self.frame_history.append(ScanportFrame(timestamp=now, gray=scanport_gray))
        self.debug_frames["scanport_gray"] = scanport_gray

        if self.state == self.STATE_ARMING:
            if now >= self.arm_until_ts:
                self.state = self.STATE_ACTIVE
                self.last_status = "active"
            else:
                self.last_status = "arming"
            return

        emitted_now = 0
        candidates_for_debug: list[dict[str, float]] = []
        stable_for_debug: list[HoleTrack] = []

        for tw in list(self.trigger_windows):
            if tw.processed:
                continue

            if now < tw.peak_ts + self.analysis_lag_s:
                continue

            result = self._process_trigger_window(tw.peak_ts)
            tw.processed = True

            if result is None:
                continue

            best_candidate, candidates, stable_tracks = result

            candidates_for_debug.extend(candidates)
            stable_for_debug.extend(stable_tracks)
            self.last_best_candidate = best_candidate

            if best_candidate is None:
                continue

            if now - self.last_emit_ts < self.global_emit_cooldown_s:
                self.last_status = "emit_cooldown"
                continue

            hit_input.push_camera_hit(
                best_candidate["camera_x"],
                best_candidate["camera_y"],
            )
            self.last_emit_ts = now
            self._remember_known_hole(best_candidate)
            emitted_now += 1

        candidates_for_debug.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        stable_for_debug.sort(key=lambda t: t.best_score, reverse=True)

        self.last_candidates = candidates_for_debug[:16]
        self.last_stable_tracks = stable_for_debug[:8]
        self.last_status = (
            f"active peaks={self.audio_event_count} windows={len(self.trigger_windows)} "
            f"cand={len(candidates_for_debug)} stable={len(stable_for_debug)} emit={emitted_now}"
        )

    def get_debug_snapshot(self) -> dict:
        return {
            "enabled": self.enabled,
            "state": self.state,
            "last_status": self.last_status,
            "audio_event_count": self.audio_event_count,
            "pending_trigger_windows": len(self.trigger_windows),
            "known_holes_count": len(self.known_holes),
            "candidates_count": len(self.last_candidates),
            "stable_tracks_count": len(self.last_stable_tracks),
            "debug_frames": dict(self.debug_frames),
            "candidates": [dict(c) for c in self.last_candidates],
            "stable_tracks": [
                {
                    "camera_x": t.camera_x,
                    "camera_y": t.camera_y,
                    "hits": t.hits,
                    "score": t.best_score,
                    "emitted": t.emitted,
                }
                for t in self.last_stable_tracks
            ],
            "known_holes": list(self.known_holes),
            "threshold_value": self.last_threshold_value,
            "change_threshold_value": self.last_change_threshold_value,
            "vote_threshold_value": self.last_vote_threshold_value,
            "window_debug": dict(self.last_window_debug),
            "best_candidate": None if self.last_best_candidate is None else dict(self.last_best_candidate),
        }

    def get_status_lines(self) -> list[str]:
        return [
            f"HitScanner state: {self.state}",
            f"Audio peaks heard: {self.audio_event_count}",
            f"Trigger windows: {len(self.trigger_windows)}",
            f"Known holes: {len(self.known_holes)}",
            f"Thr combined: {self.last_threshold_value:.1f}",
            f"Thr change: {self.last_change_threshold_value:.1f}",
            f"Thr vote: {self.last_vote_threshold_value:.1f}",
            f"Status: {self.last_status}",
        ]

    # ------------------------------------------------------------
    # Audio event listener
    # ------------------------------------------------------------

    def _on_audio_peak(self, ev: AudioPeakEvent) -> None:
        self.last_audio_event_ts = max(self.last_audio_event_ts, ev.timestamp)
        self.audio_event_count += 1

        if not self.enabled:
            return

        if self.state != self.STATE_ACTIVE:
            return

        self.trigger_windows.append(
            TriggerWindow(
                peak_ts=ev.timestamp,
                processed=False,
                created_at=time.time(),
            )
        )

    # ------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------

    def _extract_scanport_gray(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        scanport = load_scanport_rect()
        if scanport is None:
            return None

        h, w = frame_bgr.shape[:2]
        x = max(0, int(scanport.x))
        y = max(0, int(scanport.y))
        sw = max(1, int(scanport.w))
        sh = max(1, int(scanport.h))

        if x >= w or y >= h:
            return None

        sw = min(sw, w - x)
        sh = min(sh, h - y)

        if sw <= 1 or sh <= 1:
            return None

        crop = frame_bgr[y : y + sh, x : x + sw]
        if crop.size == 0:
            return None

        return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    def _process_trigger_window(
        self,
        peak_ts: float,
    ) -> tuple[dict[str, float] | None, list[dict[str, float]], list[HoleTrack]] | None:
        pre_frames: list[np.ndarray] = []
        post_frames: list[np.ndarray] = []

        for fr in self.frame_history:
            dt = fr.timestamp - peak_ts
            if -self.pre_start_s <= dt <= -self.pre_end_s:
                pre_frames.append(fr.gray)
            if self.post_start_s <= dt <= self.post_end_s:
                post_frames.append(fr.gray)

        self.last_window_debug = {
            "pre_count": float(len(pre_frames)),
            "post_count": float(len(post_frames)),
            "peak_ts": float(peak_ts),
        }

        if len(pre_frames) < 3 or len(post_frames) < 4:
            self.last_status = f"window_not_ready pre={len(pre_frames)} post={len(post_frames)}"
            return None

        pre_stack = np.stack(pre_frames, axis=0)
        post_stack = np.stack(post_frames, axis=0)

        pre_ref = np.median(pre_stack, axis=0).astype(np.uint8)
        post_ref = np.median(post_stack, axis=0).astype(np.uint8)

        # Tidig och sen post för onset/persistence
        early_count = max(2, min(4, len(post_frames) // 2))
        late_count = max(2, min(4, len(post_frames) // 2))

        early_post = np.median(np.stack(post_frames[:early_count], axis=0), axis=0).astype(np.uint8)
        late_post = np.median(np.stack(post_frames[-late_count:], axis=0), axis=0).astype(np.uint8)

        self.debug_frames["pre_ref"] = pre_ref
        self.debug_frames["post_ref"] = post_ref
        self.debug_frames["early_post"] = early_post
        self.debug_frames["late_post"] = late_post

        candidates = self._detect_candidates(pre_ref, post_ref, early_post, late_post, post_frames)
        best_candidate = self._pick_best_candidate(candidates)
        stable_tracks = self._build_tracks_from_candidates(candidates, peak_ts)

        return best_candidate, candidates, stable_tracks

    def _make_ignore_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """
        Ignore-mask i scanport-lokal koordinat.
        Just nu maskar vi bort vänster debugpanel om scanner debug overlay är på.
        """
        h, w = shape
        mask = np.ones((h, w), dtype=np.uint8) * 255

        if not load_scanner_debug_overlay_enabled():
            return mask

        viewport = load_viewport_rect()
        scanport = load_scanport_rect()
        if scanport is None or viewport.w <= 0 or viewport.h <= 0:
            return mask

        # ScannerStatusOverlay panel approx:
        # x=10, y=10, w≈760, h dynamisk.
        # Vi maskar konservativt en vänsterpanelzon i viewport-koordinater.
        panel_rects_viewport = [
            pygame_rect_like(10, 10, min(760, viewport.w), min(520, viewport.h))
        ]

        for rect in panel_rects_viewport:
            x0 = int(round((rect.x / float(viewport.w)) * w))
            y0 = int(round((rect.y / float(viewport.h)) * h))
            x1 = int(round(((rect.x + rect.w) / float(viewport.w)) * w))
            y1 = int(round(((rect.y + rect.h) / float(viewport.h)) * h))

            x0 = max(0, min(w, x0))
            x1 = max(0, min(w, x1))
            y0 = max(0, min(h, y0))
            y1 = max(0, min(h, y1))

            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 0

        return mask

    def _detect_candidates(
        self,
        pre_ref: np.ndarray,
        post_ref: np.ndarray,
        early_post: np.ndarray,
        late_post: np.ndarray,
        post_frames: list[np.ndarray],
    ) -> list[dict[str, float]]:
        pre_f = pre_ref.astype(np.float32)
        post_f = post_ref.astype(np.float32)
        early_f = early_post.astype(np.float32)
        late_f = late_post.astype(np.float32)

        pre_blur = cv2.GaussianBlur(pre_f, (0, 0), 0.9)
        post_blur = cv2.GaussianBlur(post_f, (0, 0), 0.9)
        early_blur = cv2.GaussianBlur(early_f, (0, 0), 0.9)
        late_blur = cv2.GaussianBlur(late_f, (0, 0), 0.9)

        delta_dark = np.clip(pre_blur - post_blur, 0.0, 255.0)
        onset_dark = np.clip(pre_blur - early_blur, 0.0, 255.0)
        late_dark = np.clip(pre_blur - late_blur, 0.0, 255.0)

        local_mean = cv2.GaussianBlur(post_blur, (0, 0), 6.0)
        local_dark = np.clip(local_mean - post_blur, 0.0, 255.0)

        post_u8 = np.clip(post_blur, 0, 255).astype(np.uint8)
        blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        blackhat = cv2.morphologyEx(post_u8, cv2.MORPH_BLACKHAT, blackhat_kernel).astype(np.float32)

        # Temporal vote: hur ofta mörkningen syns i post-fönstret
        base_threshold = max(2.0, float(np.mean(delta_dark) + 0.45 * np.std(delta_dark)))
        vote_count = np.zeros_like(pre_f, dtype=np.float32)
        for fr in post_frames:
            fr_f = cv2.GaussianBlur(fr.astype(np.float32), (0, 0), 0.9)
            frame_delta = np.clip(pre_blur - fr_f, 0.0, 255.0)
            vote_count += (frame_delta >= base_threshold).astype(np.float32)

        vote_norm = (vote_count / float(len(post_frames))) * 255.0 if post_frames else vote_count

        combined = (
            0.34 * delta_dark
            + 0.18 * onset_dark
            + 0.18 * late_dark
            + 0.12 * local_dark
            + 0.10 * blackhat
            + 0.08 * vote_norm
        )

        change_mu = float(np.mean(delta_dark))
        change_sigma = float(np.std(delta_dark))
        change_threshold = max(self.min_change_threshold, change_mu + 1.35 * change_sigma)
        self.last_change_threshold_value = change_threshold

        comb_mu = float(np.mean(combined))
        comb_sigma = float(np.std(combined))
        combined_threshold = max(self.min_combined_threshold, comb_mu + 1.45 * comb_sigma)
        self.last_threshold_value = combined_threshold

        vote_mu = float(np.mean(vote_norm))
        vote_sigma = float(np.std(vote_norm))
        vote_threshold = max(self.min_vote_threshold, vote_mu + 0.80 * vote_sigma)
        self.last_vote_threshold_value = vote_threshold

        change_mask = (delta_dark >= change_threshold).astype(np.uint8) * 255
        combined_mask = (combined >= combined_threshold).astype(np.uint8) * 255
        vote_mask = (vote_norm >= vote_threshold).astype(np.uint8) * 255
        onset_mask = (onset_dark >= self.min_onset_darkening).astype(np.uint8) * 255
        late_mask = (late_dark >= self.min_late_darkening).astype(np.uint8) * 255

        mask = cv2.bitwise_and(change_mask, combined_mask)
        mask = cv2.bitwise_and(mask, onset_mask)
        mask = cv2.bitwise_and(mask, late_mask)
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(change_mask, vote_mask))

        # Ignore UI-zoner om debug overlay projiceras
        ignore_mask = self._make_ignore_mask(mask.shape)
        mask = cv2.bitwise_and(mask, ignore_mask)

        kernel_open = np.ones((2, 2), dtype=np.uint8)
        kernel_close = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = post_ref.shape[:2]
        out: list[dict[str, float]] = []
        scanport = load_scanport_rect()
        if scanport is None:
            return out

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > self.max_area:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < self.min_radius or radius > self.max_radius:
                continue

            if (
                cx < self.border_margin
                or cy < self.border_margin
                or cx > w - self.border_margin
                or cy > h - self.border_margin
            ):
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue

            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < self.min_circularity:
                continue

            ix = max(0, min(int(round(cx)), w - 1))
            iy = max(0, min(int(round(cy)), h - 1))

            patch_ok, patch = self._verify_patch(
                x=ix,
                y=iy,
                pre_ref=pre_ref,
                post_ref=post_ref,
                early_post=early_post,
                late_post=late_post,
                post_frames=post_frames,
            )
            if not patch_ok:
                continue

            camera_x = float(ix + scanport.x)
            camera_y = float(iy + scanport.y)

            # Hantera gamla hål / hål-i-hål
            known = self._find_nearest_known_hole(camera_x, camera_y)
            known_gain = 0.0
            is_rehit = False
            if known is not None:
                known_gain = patch["center_darkening"] - known.get("center_darkening", 0.0)
                if known_gain < self.rehit_gain_required:
                    continue
                is_rehit = True

            score = (
                0.32 * float(combined[iy, ix])
                + 0.18 * patch["center_darkening"]
                + 0.14 * patch["onset_darkening"]
                + 0.12 * patch["late_darkening"]
                + 0.10 * patch["persistent_count"]
                + 0.08 * patch["contrast_gain"]
                + 0.06 * float(vote_norm[iy, ix])
            )

            if is_rehit:
                score += 3.0

            out.append(
                {
                    "camera_x": camera_x,
                    "camera_y": camera_y,
                    "score": float(score),
                    "delta_score": float(delta_dark[iy, ix]),
                    "combined_value": float(combined[iy, ix]),
                    "vote_value": float(vote_norm[iy, ix]),
                    "local_dark_value": float(local_dark[iy, ix]),
                    "center_darkening": float(patch["center_darkening"]),
                    "onset_darkening": float(patch["onset_darkening"]),
                    "late_darkening": float(patch["late_darkening"]),
                    "contrast_gain": float(patch["contrast_gain"]),
                    "persistent_count": float(patch["persistent_count"]),
                    "area": area,
                    "radius": float(radius),
                    "circularity": circularity,
                    "is_rehit": 1.0 if is_rehit else 0.0,
                    "known_gain": float(known_gain),
                }
            )

        out.sort(key=lambda c: c["score"], reverse=True)

        self.debug_frames["delta_dark"] = np.clip(delta_dark, 0, 255).astype(np.uint8)
        self.debug_frames["onset_dark"] = np.clip(onset_dark, 0, 255).astype(np.uint8)
        self.debug_frames["late_dark"] = np.clip(late_dark, 0, 255).astype(np.uint8)
        self.debug_frames["local_dark"] = np.clip(local_dark, 0, 255).astype(np.uint8)
        self.debug_frames["blackhat"] = np.clip(blackhat, 0, 255).astype(np.uint8)
        self.debug_frames["vote_norm"] = np.clip(vote_norm, 0, 255).astype(np.uint8)
        self.debug_frames["score"] = np.clip(combined, 0, 255).astype(np.uint8)
        self.debug_frames["mask"] = mask
        self.debug_frames["ignore_mask"] = ignore_mask

        return out

    def _verify_patch(
        self,
        x: int,
        y: int,
        pre_ref: np.ndarray,
        post_ref: np.ndarray,
        early_post: np.ndarray,
        late_post: np.ndarray,
        post_frames: list[np.ndarray],
    ) -> tuple[bool, dict[str, float]]:
        r = self.patch_radius
        h, w = pre_ref.shape[:2]

        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)

        pre_patch = pre_ref[y0:y1, x0:x1].astype(np.float32)
        post_patch = post_ref[y0:y1, x0:x1].astype(np.float32)
        early_patch = early_post[y0:y1, x0:x1].astype(np.float32)
        late_patch = late_post[y0:y1, x0:x1].astype(np.float32)

        ph, pw = pre_patch.shape[:2]
        if ph < 5 or pw < 5:
            return False, {}

        yy, xx = np.mgrid[0:ph, 0:pw]
        cx = x - x0
        cy = y - y0
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2

        inner_mask = dist2 <= (self.inner_radius ** 2)
        outer_mask = (dist2 > (self.inner_radius ** 2)) & (dist2 <= (self.outer_radius ** 2))

        if not np.any(inner_mask) or not np.any(outer_mask):
            return False, {}

        pre_center = float(np.mean(pre_patch[inner_mask]))
        post_center = float(np.mean(post_patch[inner_mask]))
        early_center = float(np.mean(early_patch[inner_mask]))
        late_center = float(np.mean(late_patch[inner_mask]))

        pre_ring = float(np.mean(pre_patch[outer_mask]))
        post_ring = float(np.mean(post_patch[outer_mask]))

        center_darkening = pre_center - post_center
        onset_darkening = pre_center - early_center
        late_darkening = pre_center - late_center

        pre_local_contrast = pre_ring - pre_center
        post_local_contrast = post_ring - post_center
        contrast_gain = post_local_contrast - pre_local_contrast

        if center_darkening < self.min_center_darkening:
            return False, {}
        if onset_darkening < self.min_onset_darkening:
            return False, {}
        if late_darkening < self.min_late_darkening:
            return False, {}

        persistent_count = 0
        best_frame_darkening = 0.0

        for fr in post_frames:
            patch = fr[y0:y1, x0:x1].astype(np.float32)
            if patch.shape != post_patch.shape:
                continue

            fr_center = float(np.mean(patch[inner_mask]))
            fr_darkening = pre_center - fr_center
            best_frame_darkening = max(best_frame_darkening, fr_darkening)

            if fr_darkening >= (self.min_center_darkening * 0.75):
                persistent_count += 1

        if persistent_count < self.min_persistent_post_frames and best_frame_darkening < (self.min_center_darkening + 2.0):
            return False, {}

        if contrast_gain < self.min_local_contrast_gain and center_darkening < (self.min_center_darkening + 2.5):
            return False, {}

        return True, {
            "center_darkening": center_darkening,
            "onset_darkening": onset_darkening,
            "late_darkening": late_darkening,
            "contrast_gain": contrast_gain,
            "persistent_count": float(persistent_count),
        }

    def _pick_best_candidate(self, candidates: list[dict[str, float]]) -> dict[str, float] | None:
        if not candidates:
            return None

        # Kandidaterna är redan sorterade på score fallande.
        return candidates[0]

    def _build_tracks_from_candidates(
        self,
        candidates: list[dict[str, float]],
        peak_ts: float,
    ) -> list[HoleTrack]:
        stable: list[HoleTrack] = []

        for c in candidates[:4]:
            stable.append(
                HoleTrack(
                    camera_x=c["camera_x"],
                    camera_y=c["camera_y"],
                    created_at=peak_ts,
                    last_seen=peak_ts,
                    hits=1,
                    best_score=float(c["score"]),
                    emitted=False,
                )
            )

        return stable

    # ------------------------------------------------------------
    # Known holes / duplicate / re-hit
    # ------------------------------------------------------------

    def _find_nearest_known_hole(self, camera_x: float, camera_y: float) -> dict[str, float] | None:
        best = None
        best_dist = None

        for hole in self.known_holes:
            dx = camera_x - hole["camera_x"]
            dy = camera_y - hole["camera_y"]
            dist = float(np.hypot(dx, dy))
            if dist <= self.duplicate_radius_px:
                if best is None or dist < best_dist:
                    best = hole
                    best_dist = dist

        return best

    def _remember_known_hole(self, candidate: dict[str, float]) -> None:
        camera_x = float(candidate["camera_x"])
        camera_y = float(candidate["camera_y"])

        existing = self._find_nearest_known_hole(camera_x, camera_y)
        now = time.time()

        if existing is not None:
            # Hål-i-hål eller uppdaterad styrka
            existing["camera_x"] = camera_x
            existing["camera_y"] = camera_y
            existing["score"] = max(existing.get("score", 0.0), float(candidate["score"]))
            existing["center_darkening"] = max(
                existing.get("center_darkening", 0.0),
                float(candidate.get("center_darkening", 0.0)),
            )
            existing["last_seen"] = now
            existing["hits"] = float(existing.get("hits", 1.0) + 1.0)
            return

        self.known_holes.append(
            {
                "camera_x": camera_x,
                "camera_y": camera_y,
                "score": float(candidate["score"]),
                "center_darkening": float(candidate.get("center_darkening", 0.0)),
                "last_seen": now,
                "hits": 1.0,
            }
        )

        if len(self.known_holes) > self.max_known_holes:
            self.known_holes = self.known_holes[-self.max_known_holes:]


def pygame_rect_like(x: int, y: int, w: int, h: int):
    class _R:
        def __init__(self, x, y, w, h):
            self.x = x
            self.y = y
            self.w = w
            self.h = h

    return _R(x, y, w, h)


hit_scanner = HitScanner()