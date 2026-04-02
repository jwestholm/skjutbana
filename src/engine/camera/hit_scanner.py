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
    STATE_OFF = "OFF"
    STATE_ARMING = "ARMING"
    STATE_ACTIVE = "ACTIVE"

    def __init__(self) -> None:
        self.enabled = False
        self.state = self.STATE_OFF

        self.arm_duration_s = 1.0
        self.arm_until_ts = 0.0
        self.last_emit_ts = 0.0
        self.global_emit_cooldown_s = 0.16
        self.last_audio_event_ts = 0.0
        self.audio_event_count = 0
        self._audio_subscribed = False

        self.frame_history: deque[ScanportFrame] = deque(maxlen=240)
        self.trigger_windows: deque[TriggerWindow] = deque(maxlen=48)

        self.pre_start_s = 0.28
        self.pre_end_s = 0.05
        self.post_start_s = 0.03
        self.post_end_s = 0.42
        self.analysis_lag_s = 0.44

        self.min_area = 3.0
        self.max_area = 240.0
        self.min_radius = 1.0
        self.max_radius = 14.0
        self.min_circularity = 0.01
        self.border_margin = 3

        self.min_change_threshold = 4.0
        self.min_combined_threshold = 8.0
        self.min_vote_threshold = 1.0

        self.patch_radius = 10
        self.inner_radius = 2
        self.outer_radius = 7
        self.min_center_darkening = 3.0
        self.min_onset_darkening = 2.5
        self.min_late_darkening = 2.5
        self.min_local_contrast_gain = 0.5
        self.min_persistent_post_frames = 2

        self.duplicate_radius_px = 20.0
        self.rehit_gain_required = 4.0
        self.max_known_holes = 256
        self.known_holes: list[dict[str, float]] = []

        self.last_status = "off"
        self.debug_frames: dict[str, np.ndarray] = {}
        self.last_candidates: list[dict[str, float]] = []
        self.last_stable_tracks: list[HoleTrack] = []
        self.last_threshold_value: float = 0.0
        self.last_change_threshold_value: float = 0.0
        self.last_vote_threshold_value: float = 0.0
        self.last_window_debug: dict[str, float] = {}
        self.last_best_candidate: dict[str, float] | None = None

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

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        now = time.time()
        self.frame_history.append(ScanportFrame(timestamp=now, gray=gray))
        self.debug_frames["camera_gray"] = gray

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

            hit_input.push_camera_hit(best_candidate["camera_x"], best_candidate["camera_y"])
            self.last_emit_ts = now
            self._remember_known_hole(best_candidate)
            emitted_now += 1

        candidates_for_debug.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        stable_for_debug.sort(key=lambda t: t.best_score, reverse=True)
        self.last_candidates = candidates_for_debug[:24]
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

    def _on_audio_peak(self, ev: AudioPeakEvent) -> None:
        self.last_audio_event_ts = max(self.last_audio_event_ts, ev.timestamp)
        self.audio_event_count += 1

        if not self.enabled or self.state != self.STATE_ACTIVE:
            return

        self.trigger_windows.append(
            TriggerWindow(
                peak_ts=ev.timestamp,
                processed=False,
                created_at=time.time(),
            )
        )

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

        pre_ref = np.median(np.stack(pre_frames, axis=0), axis=0).astype(np.uint8)
        post_ref = np.median(np.stack(post_frames, axis=0), axis=0).astype(np.uint8)

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

    def _frame_roi_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """
        Build the search mask in full camera-frame coordinates.

        Important:
        content_rect is viewport-local in this project, not absolute screen coords.
        So for homography/inverse_homography we must first convert content_rect to
        absolute screen coords by offsetting with viewport_rect.x/y.
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)

        calibration = load_camera_calibration() or {}
        raw_inv = calibration.get("inverse_homography")
        content = load_content_rect()
        viewport = load_viewport_rect()

        if raw_inv:
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

        if viewport.w <= 0 or viewport.h <= 0 or scanport.w <= 0 or scanport.h <= 0:
            mask[:, :] = 255
            self.debug_frames["roi_polygon"] = np.zeros((h, w), dtype=np.uint8)
            return mask

        rx0 = content.x / float(viewport.w)
        ry0 = content.y / float(viewport.h)
        rx1 = (content.x + content.w) / float(viewport.w)
        ry1 = (content.y + content.h) / float(viewport.h)

        x0 = int(round(scanport.x + rx0 * scanport.w))
        y0 = int(round(scanport.y + ry0 * scanport.h))
        x1 = int(round(scanport.x + rx1 * scanport.w))
        y1 = int(round(scanport.y + ry1 * scanport.h))

        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))

        if x1 <= x0 or y1 <= y0:
            mask[:, :] = 255
            self.debug_frames["roi_polygon"] = np.zeros((h, w), dtype=np.uint8)
            return mask

        mask[y0:y1, x0:x1] = 255
        roi_debug = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(roi_debug, (x0, y0), (x1 - 1, y1 - 1), 255, 2)
        self.debug_frames["roi_polygon"] = roi_debug
        return mask

    def _detect_candidates(
        self,
        pre_ref: np.ndarray,
        post_ref: np.ndarray,
        early_post: np.ndarray,
        late_post: np.ndarray,
        post_frames: list[np.ndarray],
    ) -> list[dict[str, float]]:
        roi_mask = self._frame_roi_mask(pre_ref.shape)
        self.debug_frames["mask"] = roi_mask

        diff_early = cv2.subtract(pre_ref, early_post)
        diff_late = cv2.subtract(pre_ref, late_post)
        combined = cv2.max(diff_early, diff_late)

        diff_early = cv2.bitwise_and(diff_early, diff_early, mask=roi_mask)
        diff_late = cv2.bitwise_and(diff_late, diff_late, mask=roi_mask)
        combined = cv2.bitwise_and(combined, combined, mask=roi_mask)

        self.debug_frames["diff_early"] = diff_early
        self.debug_frames["diff_late"] = diff_late
        self.debug_frames["combined"] = combined

        nonzero = combined[roi_mask > 0]
        if nonzero.size == 0:
            self.last_threshold_value = self.min_combined_threshold
            self.last_change_threshold_value = self.min_change_threshold
            self.last_vote_threshold_value = self.min_vote_threshold
            return []

        adaptive_thr = max(
            self.min_combined_threshold,
            float(np.percentile(nonzero, 92)) * 0.72,
        )
        change_thr = max(
            self.min_change_threshold,
            float(np.percentile(nonzero, 85)) * 0.46,
        )

        early_bin = np.uint8(diff_early >= change_thr) * 255
        late_bin = np.uint8(diff_late >= change_thr) * 255
        combined_bin = np.uint8(combined >= adaptive_thr) * 255

        votes = np.zeros_like(combined, dtype=np.float32)
        for fr in post_frames:
            delta = cv2.subtract(pre_ref, fr)
            delta = cv2.bitwise_and(delta, delta, mask=roi_mask)
            votes += (delta >= change_thr).astype(np.float32)

        vote_thr = max(self.min_vote_threshold, float(self.min_persistent_post_frames))
        vote_bin = np.uint8(votes >= vote_thr) * 255

        merged = cv2.bitwise_or(combined_bin, early_bin)
        merged = cv2.bitwise_or(merged, late_bin)
        merged = cv2.bitwise_and(merged, vote_bin)

        kernel = np.ones((3, 3), np.uint8)
        merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, kernel, iterations=1)
        merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel, iterations=1)

        self.debug_frames["score_map"] = np.uint8(np.clip(votes * (255.0 / max(1.0, votes.max())), 0, 255))
        self.debug_frames["thresholded"] = merged
        self.last_threshold_value = float(adaptive_thr)
        self.last_change_threshold_value = float(change_thr)
        self.last_vote_threshold_value = float(vote_thr)

        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[dict[str, float]] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > self.max_area:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            radius = float(radius)
            if radius < self.min_radius or radius > self.max_radius:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if (
                x <= self.border_margin
                or y <= self.border_margin
                or (x + w) >= (merged.shape[1] - self.border_margin)
                or (y + h) >= (merged.shape[0] - self.border_margin)
            ):
                continue

            perimeter = float(cv2.arcLength(contour, True))
            circularity = 0.0 if perimeter <= 1e-6 else (4.0 * np.pi * area) / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            patch_info = self._verify_patch(
                pre_ref,
                early_post,
                late_post,
                post_frames,
                int(round(cx)),
                int(round(cy)),
            )
            if patch_info is None:
                continue

            candidate = {
                "camera_x": float(cx),
                "camera_y": float(cy),
                "area": area,
                "radius": radius,
                "circularity": float(circularity),
                "center_darkening": patch_info["center_darkening"],
                "onset_darkening": patch_info["onset_darkening"],
                "late_darkening": patch_info["late_darkening"],
                "local_contrast_gain": patch_info["local_contrast_gain"],
                "persistent_post_frames": patch_info["persistent_post_frames"],
            }

            candidate["score"] = self._score_candidate(candidate)
            candidates.append(candidate)

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

    def _verify_patch(
        self,
        pre_ref: np.ndarray,
        early_post: np.ndarray,
        late_post: np.ndarray,
        post_frames: list[np.ndarray],
        cx: int,
        cy: int,
    ) -> dict[str, float] | None:
        r = self.patch_radius
        x0 = cx - r
        y0 = cy - r
        x1 = cx + r + 1
        y1 = cy + r + 1

        if x0 < 0 or y0 < 0 or x1 > pre_ref.shape[1] or y1 > pre_ref.shape[0]:
            return None

        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        dist2 = xx * xx + yy * yy

        center_mask = dist2 <= (self.inner_radius * self.inner_radius)
        ring_mask = (dist2 >= (self.outer_radius * self.outer_radius)) & (dist2 <= (r * r))
        if not np.any(center_mask) or not np.any(ring_mask):
            return None

        pre_patch = pre_ref[y0:y1, x0:x1].astype(np.float32)
        early_patch = early_post[y0:y1, x0:x1].astype(np.float32)
        late_patch = late_post[y0:y1, x0:x1].astype(np.float32)

        pre_center = float(np.mean(pre_patch[center_mask]))
        early_center = float(np.mean(early_patch[center_mask]))
        late_center = float(np.mean(late_patch[center_mask]))
        pre_ring = float(np.mean(pre_patch[ring_mask]))
        late_ring = float(np.mean(late_patch[ring_mask]))

        center_darkening = pre_center - late_center
        onset_darkening = pre_center - early_center
        late_darkening = pre_center - late_center
        local_contrast_gain = (pre_ring - late_ring) + (pre_center - late_center)

        if center_darkening < self.min_center_darkening:
            return None
        if onset_darkening < self.min_onset_darkening:
            return None
        if late_darkening < self.min_late_darkening:
            return None
        if local_contrast_gain < self.min_local_contrast_gain:
            return None

        persistent_post_frames = 0
        for fr in post_frames:
            patch = fr[y0:y1, x0:x1].astype(np.float32)
            if (pre_center - float(np.mean(patch[center_mask]))) >= self.min_late_darkening:
                persistent_post_frames += 1

        if persistent_post_frames < self.min_persistent_post_frames:
            return None

        return {
            "center_darkening": float(center_darkening),
            "onset_darkening": float(onset_darkening),
            "late_darkening": float(late_darkening),
            "local_contrast_gain": float(local_contrast_gain),
            "persistent_post_frames": float(persistent_post_frames),
        }

    def _score_candidate(self, candidate: dict[str, float]) -> float:
        score = 0.0
        score += candidate.get("center_darkening", 0.0) * 2.2
        score += candidate.get("onset_darkening", 0.0) * 1.0
        score += candidate.get("late_darkening", 0.0) * 1.5
        score += candidate.get("local_contrast_gain", 0.0) * 0.8
        score += candidate.get("persistent_post_frames", 0.0) * 1.25
        score += candidate.get("circularity", 0.0) * 4.0
        radius = candidate.get("radius", 0.0)
        if radius > 0.0:
            score += max(0.0, 6.0 - abs(radius - 3.0))
        return float(score)

    def _is_near_known_hole(self, camera_x: float, camera_y: float):
        best = None
        best_dist = 1e9
        for hole in self.known_holes:
            dx = float(camera_x) - float(hole["camera_x"])
            dy = float(camera_y) - float(hole["camera_y"])
            d = float(np.hypot(dx, dy))
            if d < best_dist:
                best_dist = d
                best = hole
        if best is None or best_dist > self.duplicate_radius_px:
            return None
        return best, best_dist

    def _pick_best_candidate(self, candidates: list[dict[str, float]]):
        for candidate in candidates:
            known = self._is_near_known_hole(candidate["camera_x"], candidate["camera_y"])
            if known is None:
                return candidate

            known_hole, _ = known
            required = float(known_hole.get("score", 0.0)) + self.rehit_gain_required
            if candidate.get("score", 0.0) >= required:
                return candidate

        return None

    def _build_tracks_from_candidates(self, candidates: list[dict[str, float]], peak_ts: float) -> list[HoleTrack]:
        tracks: list[HoleTrack] = []
        for candidate in candidates[:8]:
            tracks.append(
                HoleTrack(
                    camera_x=float(candidate["camera_x"]),
                    camera_y=float(candidate["camera_y"]),
                    created_at=float(peak_ts),
                    last_seen=float(peak_ts),
                    hits=1,
                    best_score=float(candidate.get("score", 0.0)),
                    emitted=False,
                )
            )
        return tracks

    def _remember_known_hole(self, candidate: dict[str, float]) -> None:
        self.known_holes.append(
            {
                "camera_x": float(candidate["camera_x"]),
                "camera_y": float(candidate["camera_y"]),
                "score": float(candidate.get("score", 0.0)),
                "timestamp": time.time(),
            }
        )
        if len(self.known_holes) > self.max_known_holes:
            self.known_holes = self.known_holes[-self.max_known_holes :]


hit_scanner = HitScanner()
