from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from src.engine.audio.audio_peak_detector import audio_peak_detector
from src.engine.camera.camera_manager import camera_manager
from src.engine.input.hit_input import hit_input
from src.engine.settings import load_camera_calibration, load_scanport_rect


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
    Audio-assisterad träffscanner.

    Viktiga designval:
    - analyserar ENDAST rå scanport (ingen warp i detektionen)
    - ljudpeak öppnar extra noggrant triggerfönster
    - bygger pre/post-referenser runt peak
    - letar efter ny mörk, lokal och persistent förändring
    - transformerar först EFTER bekräftelse via hit_input.push_camera_hit(camera_x, camera_y)

    Detta är avsiktligt mer konservativt än tidigare blob-sökning.
    """

    STATE_OFF = "OFF"
    STATE_ARMING = "ARMING"
    STATE_ACTIVE = "ACTIVE"

    def __init__(self) -> None:
        self.enabled = False
        self.state = self.STATE_OFF

        # övergripande state
        self.arm_duration_s = 1.0
        self.arm_until_ts = 0.0
        self.last_emit_ts = 0.0
        self.global_emit_cooldown_s = 0.25
        self.last_audio_event_ts = 0.0

        # scanport frame-buffer
        self.frame_history: deque[ScanportFrame] = deque(maxlen=90)  # ca 3 sek vid 30 fps
        self.trigger_windows: deque[TriggerWindow] = deque(maxlen=20)

        # tidsfönster runt ljudpeak
        self.pre_start_s = 0.18
        self.pre_end_s = 0.03
        self.post_start_s = 0.04
        self.post_end_s = 0.22
        self.analysis_lag_s = 0.24  # vänta tills post-window finns inspelat

        # kandidatfilter
        self.min_area = 6.0
        self.max_area = 90.0
        self.min_radius = 1.4
        self.max_radius = 8.0
        self.min_circularity = 0.10
        self.border_margin = 4

        self.min_change_threshold = 10.0
        self.min_combined_threshold = 18.0

        # patch-verifiering
        self.patch_radius = 7           # 15x15 patch
        self.inner_radius = 2
        self.outer_radius = 6
        self.min_center_darkening = 10.0
        self.min_local_contrast_gain = 8.0
        self.min_persistent_post_frames = 2

        # tracking / duplicate
        self.match_radius_px = 12.0
        self.duplicate_radius_px = 20.0
        self.max_track_idle_s = 0.40
        self.min_confirm_frames = 1  # varje trigger-window är redan strikt; 1 räcker där

        self.tracks: list[HoleTrack] = []
        self.known_holes: list[tuple[float, float, float]] = []

        # debug
        self.last_status = "off"
        self.debug_frames: dict[str, np.ndarray] = {}
        self.last_candidates: list[dict[str, float]] = []
        self.last_stable_tracks: list[HoleTrack] = []
        self.last_threshold_value: float = 0.0
        self.last_change_threshold_value: float = 0.0

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def enable(self) -> None:
        self.enabled = True
        self.state = self.STATE_ARMING
        self.arm_until_ts = time.time() + self.arm_duration_s
        self.last_emit_ts = 0.0
        self.last_audio_event_ts = 0.0

        self.frame_history.clear()
        self.trigger_windows.clear()
        self.tracks.clear()
        self.known_holes.clear()

        self.debug_frames.clear()
        self.last_candidates = []
        self.last_stable_tracks = []
        self.last_threshold_value = 0.0
        self.last_change_threshold_value = 0.0
        self.last_status = "arming"

    def disable(self) -> None:
        self.enabled = False
        self.state = self.STATE_OFF

        self.frame_history.clear()
        self.trigger_windows.clear()
        self.tracks.clear()
        self.known_holes.clear()

        self.debug_frames.clear()
        self.last_candidates = []
        self.last_stable_tracks = []
        self.last_threshold_value = 0.0
        self.last_change_threshold_value = 0.0
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

        # håll några debug-bilder levande
        self.debug_frames["scanport_gray"] = scanport_gray

        if self.state == self.STATE_ARMING:
            if now >= self.arm_until_ts:
                self.state = self.STATE_ACTIVE
                self.last_status = "active"
            else:
                self.last_status = "arming"
            return

        if self.state != self.STATE_ACTIVE:
            return

        # plocka in nya audio peaks
        for ev in audio_peak_detector.get_events_since(self.last_audio_event_ts):
            self.last_audio_event_ts = max(self.last_audio_event_ts, ev.timestamp)
            self.trigger_windows.append(
                TriggerWindow(
                    peak_ts=ev.timestamp,
                    processed=False,
                    created_at=now,
                )
            )

        emitted_now = 0
        candidates_for_debug: list[dict[str, float]] = []
        stable_for_debug: list[HoleTrack] = []

        # processa trigger-fönster som nu har tillräckligt med efter-data
        for tw in list(self.trigger_windows):
            if tw.processed:
                continue

            if now < tw.peak_ts + self.analysis_lag_s:
                continue

            result = self._process_trigger_window(tw.peak_ts)
            tw.processed = True

            if result is None:
                continue

            candidates, stable_tracks = result
            candidates_for_debug.extend(candidates)
            stable_for_debug.extend(stable_tracks)

            for track in stable_tracks:
                if track.emitted:
                    continue

                if self._is_duplicate(track.camera_x, track.camera_y):
                    track.emitted = True
                    continue

                if now - self.last_emit_ts < self.global_emit_cooldown_s:
                    continue

                hit_input.push_camera_hit(track.camera_x, track.camera_y)
                self.last_emit_ts = now
                self._remember_known_hole(track.camera_x, track.camera_y, track.best_score)
                track.emitted = True
                emitted_now += 1

        self.last_candidates = candidates_for_debug
        self.last_stable_tracks = stable_for_debug

        self.last_status = (
            f"active audio={len(self.trigger_windows)} cand={len(candidates_for_debug)} "
            f"stable={len(stable_for_debug)} emit={emitted_now}"
        )

    def get_debug_snapshot(self) -> dict:
        return {
            "enabled": self.enabled,
            "state": self.state,
            "last_status": self.last_status,
            "known_holes_count": len(self.known_holes),
            "tracks_count": len(self.tracks),
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
        }

    def get_status_lines(self) -> list[str]:
        return [
            f"HitScanner state: {self.state}",
            f"Known holes: {len(self.known_holes)}",
            f"Thr combined: {self.last_threshold_value:.1f}",
            f"Thr change: {self.last_change_threshold_value:.1f}",
            f"Status: {self.last_status}",
        ]

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

        crop = frame_bgr[y:y + sh, x:x + sw]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return gray

    def _process_trigger_window(
        self,
        peak_ts: float,
    ) -> tuple[list[dict[str, float]], list[HoleTrack]] | None:
        pre_frames = []
        post_frames = []

        for fr in self.frame_history:
            dt = fr.timestamp - peak_ts

            if -self.pre_start_s <= dt <= -self.pre_end_s:
                pre_frames.append(fr.gray)

            if self.post_start_s <= dt <= self.post_end_s:
                post_frames.append(fr.gray)

        if len(pre_frames) < 2 or len(post_frames) < 3:
            return None

        pre_ref = np.median(np.stack(pre_frames, axis=0), axis=0).astype(np.uint8)
        post_ref = np.median(np.stack(post_frames, axis=0), axis=0).astype(np.uint8)

        candidates = self._detect_candidates(pre_ref, post_ref, post_frames)
        stable_tracks = self._build_tracks_from_candidates(candidates, peak_ts)

        self.debug_frames["pre_ref"] = pre_ref
        self.debug_frames["post_ref"] = post_ref

        return candidates, stable_tracks

    def _detect_candidates(
        self,
        pre_ref: np.ndarray,
        post_ref: np.ndarray,
        post_frames: list[np.ndarray],
    ) -> list[dict[str, float]]:
        pre_blur = cv2.GaussianBlur(pre_ref, (0, 0), 1.0)
        post_blur = cv2.GaussianBlur(post_ref, (0, 0), 1.0)

        # ny mörkare förändring
        delta_dark = cv2.subtract(pre_blur, post_blur)

        # lokal mörkerförstärkning i efter-bilden
        local_mean = cv2.GaussianBlur(post_blur, (0, 0), 7.0)
        darkness = cv2.subtract(local_mean, post_blur)

        blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        blackhat = cv2.morphologyEx(post_blur, cv2.MORPH_BLACKHAT, blackhat_kernel)

        dark_score = cv2.addWeighted(darkness, 0.60, blackhat, 0.40, 0.0)

        change_mu, change_sigma = cv2.meanStdDev(delta_dark)
        change_threshold = max(
            self.min_change_threshold,
            float(change_mu[0][0] + 2.0 * change_sigma[0][0]),
        )
        self.last_change_threshold_value = change_threshold

        _, change_mask = cv2.threshold(
            delta_dark,
            change_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        combined = cv2.addWeighted(
            delta_dark,
            0.72,
            dark_score,
            0.28,
            0.0,
        )

        comb_mu, comb_sigma = cv2.meanStdDev(combined)
        combined_threshold = max(
            self.min_combined_threshold,
            float(comb_mu[0][0] + 2.1 * comb_sigma[0][0]),
        )
        self.last_threshold_value = combined_threshold

        _, combined_mask = cv2.threshold(
            combined,
            combined_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        mask = cv2.bitwise_and(change_mask, combined_mask)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        h, w = post_ref.shape[:2]
        out: list[dict[str, float]] = []

        for contour in contours:
            area = cv2.contourArea(contour)
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

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            ix = max(0, min(int(round(cx)), w - 1))
            iy = max(0, min(int(round(cy)), h - 1))

            delta_value = float(delta_dark[iy, ix])
            combined_value = float(combined[iy, ix])

            if delta_value < self.last_change_threshold_value:
                continue

            patch_ok, center_darkening, contrast_gain, persistent_count = self._verify_patch(
                x=ix,
                y=iy,
                pre_ref=pre_ref,
                post_ref=post_ref,
                post_frames=post_frames,
            )
            if not patch_ok:
                continue

            scanport = load_scanport_rect()
            camera_x = float(ix + scanport.x)
            camera_y = float(iy + scanport.y)

            out.append(
                {
                    "camera_x": camera_x,
                    "camera_y": camera_y,
                    "score": combined_value,
                    "delta_score": delta_value,
                    "center_darkening": center_darkening,
                    "contrast_gain": contrast_gain,
                    "persistent_count": persistent_count,
                    "area": float(area),
                    "radius": float(radius),
                    "circularity": float(circularity),
                }
            )

        self.debug_frames["delta_dark"] = delta_dark
        self.debug_frames["dark_score"] = dark_score
        self.debug_frames["score"] = combined
        self.debug_frames["mask"] = mask

        return out

    def _verify_patch(
        self,
        x: int,
        y: int,
        pre_ref: np.ndarray,
        post_ref: np.ndarray,
        post_frames: list[np.ndarray],
    ) -> tuple[bool, float, float, int]:
        r = self.patch_radius
        h, w = pre_ref.shape[:2]

        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)

        pre_patch = pre_ref[y0:y1, x0:x1]
        post_patch = post_ref[y0:y1, x0:x1]

        ph, pw = pre_patch.shape[:2]
        if ph < 5 or pw < 5:
            return False, 0.0, 0.0, 0

        yy, xx = np.mgrid[0:ph, 0:pw]
        cx = x - x0
        cy = y - y0
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2

        inner_mask = dist2 <= (self.inner_radius ** 2)
        outer_mask = (dist2 > (self.inner_radius ** 2)) & (dist2 <= (self.outer_radius ** 2))

        if not np.any(inner_mask) or not np.any(outer_mask):
            return False, 0.0, 0.0, 0

        pre_center = float(np.mean(pre_patch[inner_mask]))
        post_center = float(np.mean(post_patch[inner_mask]))
        pre_ring = float(np.mean(pre_patch[outer_mask]))
        post_ring = float(np.mean(post_patch[outer_mask]))

        center_darkening = pre_center - post_center
        pre_local_contrast = pre_ring - pre_center
        post_local_contrast = post_ring - post_center
        contrast_gain = post_local_contrast - pre_local_contrast

        if center_darkening < self.min_center_darkening:
            return False, center_darkening, contrast_gain, 0

        if contrast_gain < self.min_local_contrast_gain:
            return False, center_darkening, contrast_gain, 0

        persistent_count = 0
        for fr in post_frames:
            patch = fr[y0:y1, x0:x1]
            if patch.shape != post_patch.shape:
                continue
            fr_center = float(np.mean(patch[inner_mask]))
            if (pre_center - fr_center) >= (self.min_center_darkening * 0.7):
                persistent_count += 1

        if persistent_count < self.min_persistent_post_frames:
            return False, center_darkening, contrast_gain, persistent_count

        return True, center_darkening, contrast_gain, persistent_count

    def _build_tracks_from_candidates(
        self,
        candidates: list[dict[str, float]],
        peak_ts: float,
    ) -> list[HoleTrack]:
        stable: list[HoleTrack] = []

        for c in candidates:
            stable.append(
                HoleTrack(
                    camera_x=c["camera_x"],
                    camera_y=c["camera_y"],
                    created_at=peak_ts,
                    last_seen=peak_ts,
                    hits=1,
                    best_score=c["score"],
                    emitted=False,
                )
            )

        return stable

    # ------------------------------------------------------------
    # Memory / duplicate suppression
    # ------------------------------------------------------------

    def _remember_known_hole(self, camera_x: float, camera_y: float, score: float) -> None:
        if self._is_duplicate(camera_x, camera_y):
            return
        self.known_holes.append((camera_x, camera_y, score))

    def _is_duplicate(self, camera_x: float, camera_y: float) -> bool:
        for hx, hy, _score in self.known_holes:
            if np.hypot(camera_x - hx, camera_y - hy) <= self.duplicate_radius_px:
                return True
        return False


hit_scanner = HitScanner()