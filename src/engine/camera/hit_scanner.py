from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from src.engine.audio.audio_peak_detector import AudioPeakEvent, audio_peak_detector
from src.engine.camera.camera_manager import camera_manager
from src.engine.input.hit_input import hit_input
from src.engine.settings import load_scanport_rect


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

    Designmål:
    - analysera ENDAST när audio trigger säger att ett skott sannolikt gått av
    - jämför stabil referens före skott med stabil referens efter skott
    - hitta små, mörka, permanenta förändringar i rå scanport
    - fungera först på vit/stillastående yta innan vi optimerar för video/spel

    Den här versionen är medvetet mer tolerant än den tidigare:
    - lägre trösklar
    - temporal vote-map från flera post-frames
    - lokal bakgrundsnormalisering
    - mindre aggressiv patch-verifiering
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
        self.global_emit_cooldown_s = 0.20

        self.last_audio_event_ts = 0.0
        self.audio_event_count = 0
        self._audio_subscribed = False

        # scanport frame-buffer
        self.frame_history: deque[ScanportFrame] = deque(maxlen=180)  # ~6 sek vid 30 fps
        self.trigger_windows: deque[TriggerWindow] = deque(maxlen=32)

        # tidsfönster runt ljudpeak
        self.pre_start_s = 0.22
        self.pre_end_s = 0.03
        self.post_start_s = 0.05
        self.post_end_s = 0.30
        self.analysis_lag_s = 0.34

        # kandidatfilter - mer tolerant för vit yta
        self.min_area = 3.0
        self.max_area = 160.0
        self.min_radius = 1.0
        self.max_radius = 10.0
        self.min_circularity = 0.03
        self.border_margin = 3

        # adaptiva thresholdgolv
        self.min_change_threshold = 4.0
        self.min_combined_threshold = 8.0
        self.min_vote_threshold = 1.0

        # patch-verifiering - medvetet mildare nu
        self.patch_radius = 7
        self.inner_radius = 2
        self.outer_radius = 6
        self.min_center_darkening = 4.0
        self.min_local_contrast_gain = 1.5
        self.min_persistent_post_frames = 1

        # duplicate suppression
        self.duplicate_radius_px = 18.0

        self.known_holes: list[tuple[float, float, float]] = []

        # debug
        self.last_status = "off"
        self.debug_frames: dict[str, np.ndarray] = {}
        self.last_candidates: list[dict[str, float]] = []
        self.last_stable_tracks: list[HoleTrack] = []
        self.last_threshold_value: float = 0.0
        self.last_change_threshold_value: float = 0.0
        self.last_vote_threshold_value: float = 0.0
        self.last_window_debug: dict[str, float] = {}

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
        self.known_holes.clear()
        self.debug_frames.clear()
        self.last_candidates = []
        self.last_stable_tracks = []
        self.last_threshold_value = 0.0
        self.last_change_threshold_value = 0.0
        self.last_vote_threshold_value = 0.0
        self.last_window_debug = {}
        self.last_status = "arming"

    def disable(self) -> None:
        self.enabled = False
        self.state = self.STATE_OFF

        if self._audio_subscribed:
            audio_peak_detector.unsubscribe(self._on_audio_peak)
            self._audio_subscribed = False

        self.frame_history.clear()
        self.trigger_windows.clear()
        self.known_holes.clear()
        self.debug_frames.clear()
        self.last_candidates = []
        self.last_stable_tracks = []
        self.last_threshold_value = 0.0
        self.last_change_threshold_value = 0.0
        self.last_vote_threshold_value = 0.0
        self.last_window_debug = {}
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

        if self.state != self.STATE_ACTIVE:
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

        # håll bara de starkaste kandidaterna i debug, så overlay blir läsbar
        candidates_for_debug.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        stable_for_debug.sort(key=lambda t: t.best_score, reverse=True)

        self.last_candidates = candidates_for_debug[:12]
        self.last_stable_tracks = stable_for_debug[:6]
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
    ) -> tuple[list[dict[str, float]], list[HoleTrack]] | None:
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

        if len(pre_frames) < 3 or len(post_frames) < 3:
            self.last_status = (
                f"window_not_ready pre={len(pre_frames)} post={len(post_frames)}"
            )
            return None

        pre_stack = np.stack(pre_frames, axis=0)
        post_stack = np.stack(post_frames, axis=0)

        pre_ref = np.median(pre_stack, axis=0).astype(np.uint8)
        post_ref = np.median(post_stack, axis=0).astype(np.uint8)

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
        pre_f = pre_ref.astype(np.float32)
        post_f = post_ref.astype(np.float32)

        # lätt blur för att minska sensorbrus
        pre_blur = cv2.GaussianBlur(pre_f, (0, 0), 0.9)
        post_blur = cv2.GaussianBlur(post_f, (0, 0), 0.9)

        # ny mörk förändring mellan före och efter
        delta_dark = np.clip(pre_blur - post_blur, 0.0, 255.0)

        # lokal bakgrundsnormalisering: punkt som är mörkare än sin närmiljö
        local_mean = cv2.GaussianBlur(post_blur, (0, 0), 6.0)
        local_dark = np.clip(local_mean - post_blur, 0.0, 255.0)

        # top-hat/blackhat-liknande förstärkning av små mörka detaljer
        post_u8 = np.clip(post_blur, 0, 255).astype(np.uint8)
        blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        blackhat = cv2.morphologyEx(post_u8, cv2.MORPH_BLACKHAT, blackhat_kernel).astype(
            np.float32
        )

        # temporal vote: hur många post-frames visar mörkning på samma pixel?
        per_frame_threshold = max(
            2.0,
            float(np.mean(delta_dark) + 0.50 * np.std(delta_dark)),
        )

        vote_count = np.zeros_like(pre_f, dtype=np.float32)
        for fr in post_frames:
            fr_f = cv2.GaussianBlur(fr.astype(np.float32), (0, 0), 0.9)
            frame_delta = np.clip(pre_blur - fr_f, 0.0, 255.0)
            vote_count += (frame_delta >= per_frame_threshold).astype(np.float32)

        if len(post_frames) > 0:
            vote_norm = (vote_count / float(len(post_frames))) * 255.0
        else:
            vote_norm = vote_count

        # kombinera flera signaler
        combined = (
            0.50 * delta_dark
            + 0.22 * local_dark
            + 0.13 * blackhat
            + 0.15 * vote_norm
        )

        # adaptiva thresholds med lägre golv
        delta_mu = float(np.mean(delta_dark))
        delta_sigma = float(np.std(delta_dark))
        change_threshold = max(
            self.min_change_threshold,
            delta_mu + 1.35 * delta_sigma,
        )
        self.last_change_threshold_value = change_threshold

        comb_mu = float(np.mean(combined))
        comb_sigma = float(np.std(combined))
        combined_threshold = max(
            self.min_combined_threshold,
            comb_mu + 1.45 * comb_sigma,
        )
        self.last_threshold_value = combined_threshold

        vote_mu = float(np.mean(vote_norm))
        vote_sigma = float(np.std(vote_norm))
        vote_threshold = max(
            self.min_vote_threshold,
            vote_mu + 0.80 * vote_sigma,
        )
        self.last_vote_threshold_value = vote_threshold

        change_mask = (delta_dark >= change_threshold).astype(np.uint8) * 255
        combined_mask = (combined >= combined_threshold).astype(np.uint8) * 255
        vote_mask = (vote_norm >= vote_threshold).astype(np.uint8) * 255

        # kräv tydlig ändring + tillräcklig total score, men vote-mask får stärka små hål
        mask = cv2.bitwise_and(change_mask, combined_mask)
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(change_mask, vote_mask))

        # städa brus men låt små hål överleva
        kernel_open = np.ones((2, 2), dtype=np.uint8)
        kernel_close = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

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

            delta_value = float(delta_dark[iy, ix])
            combined_value = float(combined[iy, ix])
            vote_value = float(vote_norm[iy, ix])
            local_dark_value = float(local_dark[iy, ix])

            patch_ok, center_darkening, contrast_gain, persistent_count = self._verify_patch(
                x=ix,
                y=iy,
                pre_ref=pre_ref,
                post_ref=post_ref,
                post_frames=post_frames,
            )

            if not patch_ok:
                continue

            camera_x = float(ix + scanport.x)
            camera_y = float(iy + scanport.y)

            score = (
                combined_value
                + 0.30 * vote_value
                + 0.20 * center_darkening
                + 0.10 * local_dark_value
            )

            out.append(
                {
                    "camera_x": camera_x,
                    "camera_y": camera_y,
                    "score": float(score),
                    "delta_score": delta_value,
                    "combined_value": combined_value,
                    "vote_value": vote_value,
                    "local_dark_value": local_dark_value,
                    "center_darkening": center_darkening,
                    "contrast_gain": contrast_gain,
                    "persistent_count": float(persistent_count),
                    "area": area,
                    "radius": float(radius),
                    "circularity": circularity,
                }
            )

        out.sort(key=lambda c: c["score"], reverse=True)

        self.debug_frames["delta_dark"] = np.clip(delta_dark, 0, 255).astype(np.uint8)
        self.debug_frames["local_dark"] = np.clip(local_dark, 0, 255).astype(np.uint8)
        self.debug_frames["blackhat"] = np.clip(blackhat, 0, 255).astype(np.uint8)
        self.debug_frames["vote_norm"] = np.clip(vote_norm, 0, 255).astype(np.uint8)
        self.debug_frames["score"] = np.clip(combined, 0, 255).astype(np.uint8)
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

        pre_patch = pre_ref[y0:y1, x0:x1].astype(np.float32)
        post_patch = post_ref[y0:y1, x0:x1].astype(np.float32)

        ph, pw = pre_patch.shape[:2]
        if ph < 5 or pw < 5:
            return False, 0.0, 0.0, 0

        yy, xx = np.mgrid[0:ph, 0:pw]
        cx = x - x0
        cy = y - y0
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2

        inner_mask = dist2 <= (self.inner_radius**2)
        outer_mask = (dist2 > (self.inner_radius**2)) & (dist2 <= (self.outer_radius**2))

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

        # mildare grundkrav
        if center_darkening < self.min_center_darkening:
            return False, center_darkening, contrast_gain, 0

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

        # släpp igenom om antingen flera frames bekräftar eller ett riktigt tydligt post-frame finns
        persistent_ok = (
            persistent_count >= self.min_persistent_post_frames
            or best_frame_darkening >= (self.min_center_darkening + 2.0)
        )

        if not persistent_ok:
            return False, center_darkening, contrast_gain, persistent_count

        # lokal kontrast ska helst öka, men behöver inte vara jättestark på vit yta
        if contrast_gain < self.min_local_contrast_gain and center_darkening < (
            self.min_center_darkening + 2.5
        ):
            return False, center_darkening, contrast_gain, persistent_count

        return True, center_darkening, contrast_gain, persistent_count

    def _build_tracks_from_candidates(
        self,
        candidates: list[dict[str, float]],
        peak_ts: float,
    ) -> list[HoleTrack]:
        stable: list[HoleTrack] = []

        for c in candidates[:3]:
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