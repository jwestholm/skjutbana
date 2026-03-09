from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from src.engine.camera.camera_manager import camera_manager
from src.engine.input.hit_input import hit_input
from src.engine.settings import (
    load_camera_calibration,
    load_scanport_rect,
    load_viewport_rect,
)


@dataclass
class HoleTrack:
    board_x: float
    board_y: float
    camera_x: float
    camera_y: float
    created_at: float
    last_seen: float
    hits: int = 1
    best_score: float = 0.0
    emitted: bool = False


class HitScanner:
    """
    Konservativ träffscanner.

    Mål:
    - analysera endast scanport
    - jobba i warpad board/viewport-bild
    - hitta NYA mörkare förändringar
    - kräva persistence
    - skicka riktiga camera_x/camera_y till hit_input.push_camera_hit(...)
    """

    STATE_OFF = "OFF"
    STATE_ARMING = "ARMING"
    STATE_ACTIVE = "ACTIVE"

    def __init__(self) -> None:
        self.enabled = False
        self.state = self.STATE_OFF

        # Tider / state
        self.arm_duration_s = 1.50
        self.arm_until_ts = 0.0
        self.last_emit_ts = 0.0
        self.global_emit_cooldown_s = 0.25

        # Historik / referens
        self.history_size = 14
        self.reference_stride = 3
        self.board_history: deque[np.ndarray] = deque(maxlen=self.history_size)

        # Tracking
        self.min_confirm_frames = 3
        self.max_track_idle_s = 0.45
        self.match_radius_px = 16.0
        self.duplicate_radius_px = 24.0

        # Blobfilter - ganska konservativt
        self.min_area = 6.0
        self.max_area = 120.0
        self.min_radius = 1.5
        self.max_radius = 8.0
        self.min_circularity = 0.10
        self.border_margin = 5

        # Thresholds
        self.min_change_threshold = 14.0
        self.min_combined_threshold = 24.0
        self.min_center_darkness = 12.0

        # Vikter
        self.change_weight = 0.70
        self.dark_weight = 0.30

        # Runtime
        self.tracks: list[HoleTrack] = []
        self.known_holes: list[tuple[float, float, float]] = []

        self.last_status: str = "off"
        self.debug_frames: dict[str, np.ndarray] = {}
        self.last_candidates: list[dict[str, float]] = []
        self.last_stable_tracks: list[HoleTrack] = []
        self.last_board_size: tuple[int, int] = (1, 1)
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

        self.board_history.clear()
        self.tracks.clear()
        self.known_holes.clear()

        self.debug_frames.clear()
        self.last_candidates = []
        self.last_stable_tracks = []
        self.last_board_size = (1, 1)
        self.last_threshold_value = 0.0
        self.last_change_threshold_value = 0.0
        self.last_status = "arming"

    def disable(self) -> None:
        self.enabled = False
        self.state = self.STATE_OFF

        self.board_history.clear()
        self.tracks.clear()
        self.known_holes.clear()

        self.debug_frames.clear()
        self.last_candidates = []
        self.last_stable_tracks = []
        self.last_board_size = (1, 1)
        self.last_threshold_value = 0.0
        self.last_change_threshold_value = 0.0
        self.last_status = "off"

    def scene_rearmed(self) -> None:
        if self.enabled:
            self.enable()

    def update(self, dt: float) -> None:
        del dt

        if not self.enabled or self.state == self.STATE_OFF:
            self.last_candidates = []
            self.last_stable_tracks = []
            return

        frame_bgr = camera_manager.get_latest_frame()
        if frame_bgr is None:
            self.last_status = "no_frame"
            self.last_candidates = []
            self.last_stable_tracks = []
            return

        prepared = self._prepare_board_view(frame_bgr)
        if prepared is None:
            self.last_status = "not_ready"
            self.last_candidates = []
            self.last_stable_tracks = []
            return

        gray, board_to_crop_matrix, scanport = prepared
        self.last_board_size = (gray.shape[1], gray.shape[0])

        self.board_history.append(gray.copy())

        if len(self.board_history) < max(6, self.reference_stride + 3):
            self.last_status = f"arming hist={len(self.board_history)}"
            self.last_candidates = []
            self.last_stable_tracks = []
            return

        reference_gray = self._build_reference_image()
        candidates = self._detect_candidates(
            gray=gray,
            reference_gray=reference_gray,
            board_to_crop_matrix=board_to_crop_matrix,
            scanport=scanport,
        )

        now = time.time()

        self._drop_stale_tracks(now)
        self._ingest_candidates(candidates, now)

        stable_tracks = self._get_stable_tracks(now)

        self.last_candidates = [dict(c) for c in candidates]
        self.last_stable_tracks = [
            HoleTrack(
                board_x=t.board_x,
                board_y=t.board_y,
                camera_x=t.camera_x,
                camera_y=t.camera_y,
                created_at=t.created_at,
                last_seen=t.last_seen,
                hits=t.hits,
                best_score=t.best_score,
                emitted=t.emitted,
            )
            for t in stable_tracks
        ]

        if self.state == self.STATE_ARMING:
            for track in stable_tracks:
                self._remember_known_hole(track.board_x, track.board_y, track.best_score)
                track.emitted = True

            if now >= self.arm_until_ts:
                self.state = self.STATE_ACTIVE
                self.tracks.clear()
                self.last_status = "active"
            else:
                self.last_status = f"arming cand={len(candidates)}"
            return

        if self.state != self.STATE_ACTIVE:
            return

        emitted_now = 0

        for track in stable_tracks:
            if track.emitted:
                continue

            if self._is_duplicate(track.board_x, track.board_y):
                track.emitted = True
                continue

            if now - self.last_emit_ts < self.global_emit_cooldown_s:
                continue

            hit_input.push_camera_hit(track.camera_x, track.camera_y)
            self.last_emit_ts = now
            self._remember_known_hole(track.board_x, track.board_y, track.best_score)
            track.emitted = True
            emitted_now += 1

        self.last_status = (
            f"active cand={len(candidates)} stable={len(stable_tracks)} emit={emitted_now}"
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
                    "board_x": t.board_x,
                    "board_y": t.board_y,
                    "camera_x": t.camera_x,
                    "camera_y": t.camera_y,
                    "hits": t.hits,
                    "score": t.best_score,
                    "emitted": t.emitted,
                }
                for t in self.last_stable_tracks
            ],
            "known_holes": list(self.known_holes),
            "board_size": self.last_board_size,
            "threshold_value": self.last_threshold_value,
            "change_threshold_value": self.last_change_threshold_value,
        }

    def get_status_lines(self) -> list[str]:
        return [
            f"HitScanner state: {self.state}",
            f"Known holes: {len(self.known_holes)}",
            f"Tracks: {len(self.tracks)}",
            f"Thr combined: {self.last_threshold_value:.1f}",
            f"Thr change: {self.last_change_threshold_value:.1f}",
            f"Status: {self.last_status}",
        ]

    # ------------------------------------------------------------
    # Geometry / preparation
    # ------------------------------------------------------------

    def _prepare_board_view(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, object] | None:
        calibration = load_camera_calibration()
        if not calibration:
            return None

        homography = calibration.get("homography")
        if not homography:
            return None

        scanport = load_scanport_rect()
        if scanport is None:
            return None

        viewport = load_viewport_rect()

        H_camera_to_screen = np.array(homography, dtype=np.float32)
        if H_camera_to_screen.shape != (3, 3):
            return None

        frame_h, frame_w = frame_bgr.shape[:2]

        x = max(0, int(scanport.x))
        y = max(0, int(scanport.y))
        w = max(1, int(scanport.w))
        h = max(1, int(scanport.h))

        if x >= frame_w or y >= frame_h:
            return None

        w = min(w, frame_w - x)
        h = min(h, frame_h - y)
        if w <= 1 or h <= 1:
            return None

        crop = frame_bgr[y:y + h, x:x + w]
        if crop.size == 0:
            return None

        # crop -> camera
        T_crop_to_camera = np.array(
            [
                [1.0, 0.0, float(x)],
                [0.0, 1.0, float(y)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # screen -> board-local
        T_screen_to_board = np.array(
            [
                [1.0, 0.0, -float(viewport.x)],
                [0.0, 1.0, -float(viewport.y)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # crop -> board
        H_crop_to_board = T_screen_to_board @ H_camera_to_screen @ T_crop_to_camera

        try:
            H_board_to_crop = np.linalg.inv(H_crop_to_board).astype(np.float32)
        except np.linalg.LinAlgError:
            return None

        warped = cv2.warpPerspective(
            crop,
            H_crop_to_board,
            (int(viewport.w), int(viewport.h)),
            flags=cv2.INTER_LINEAR,
        )

        if warped is None or warped.size == 0:
            return None

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        self.debug_frames["crop_bgr"] = crop
        self.debug_frames["warped_gray"] = gray

        return gray, H_board_to_crop, scanport

    # ------------------------------------------------------------
    # Reference
    # ------------------------------------------------------------

    def _build_reference_image(self) -> np.ndarray:
        hist = list(self.board_history)
        usable = hist[:-self.reference_stride] if len(hist) > self.reference_stride else hist

        if len(usable) == 1:
            return usable[0]

        stack = np.stack(usable, axis=0).astype(np.uint8)
        return np.median(stack, axis=0).astype(np.uint8)

    # ------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------

    def _detect_candidates(
        self,
        gray: np.ndarray,
        reference_gray: np.ndarray,
        board_to_crop_matrix: np.ndarray,
        scanport,
    ) -> list[dict[str, float]]:
        gray_blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        ref_blur = cv2.GaussianBlur(reference_gray, (0, 0), 1.0)

        # Ny mörkare förändring
        dark_change = cv2.subtract(ref_blur, gray_blur)

        # Lokal hål-lik mörkhet
        local_mean = cv2.GaussianBlur(gray_blur, (0, 0), 7.0)
        darkness = cv2.subtract(local_mean, gray_blur)

        blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        blackhat = cv2.morphologyEx(gray_blur, cv2.MORPH_BLACKHAT, blackhat_kernel)

        dark_score = cv2.addWeighted(darkness, 0.60, blackhat, 0.40, 0.0)

        change_mu, change_sigma = cv2.meanStdDev(dark_change)
        change_threshold = max(
            self.min_change_threshold,
            float(change_mu[0][0] + 2.2 * change_sigma[0][0]),
        )
        self.last_change_threshold_value = change_threshold

        _, change_mask = cv2.threshold(
            dark_change,
            change_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        combined = cv2.addWeighted(
            dark_change,
            self.change_weight,
            dark_score,
            self.dark_weight,
            0.0,
        )

        comb_mu, comb_sigma = cv2.meanStdDev(combined)
        combined_threshold = max(
            self.min_combined_threshold,
            float(comb_mu[0][0] + 2.3 * comb_sigma[0][0]),
        )
        self.last_threshold_value = combined_threshold

        _, combined_mask = cv2.threshold(
            combined,
            combined_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        # Kräver både förändring och kombinerad hålscore
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

        height, width = gray.shape[:2]
        candidates: list[dict[str, float]] = []

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
                or cx > width - self.border_margin
                or cy > height - self.border_margin
            ):
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            ix = max(0, min(int(round(cx)), width - 1))
            iy = max(0, min(int(round(cy)), height - 1))

            change_value = float(dark_change[iy, ix])
            dark_value = float(dark_score[iy, ix])
            combined_value = float(combined[iy, ix])

            if change_value < self.last_change_threshold_value:
                continue

            if dark_value < self.min_center_darkness:
                continue

            crop_point = cv2.perspectiveTransform(
                np.array([[[cx, cy]]], dtype=np.float32),
                board_to_crop_matrix,
            )[0, 0]

            crop_x = float(crop_point[0])
            crop_y = float(crop_point[1])

            camera_x = crop_x + float(scanport.x)
            camera_y = crop_y + float(scanport.y)

            candidates.append(
                {
                    "board_x": float(cx),
                    "board_y": float(cy),
                    "camera_x": camera_x,
                    "camera_y": camera_y,
                    "score": combined_value,
                    "change_score": change_value,
                    "dark_score": dark_value,
                    "area": float(area),
                    "radius": float(radius),
                    "circularity": float(circularity),
                }
            )

        self.debug_frames["reference_gray"] = reference_gray
        self.debug_frames["change"] = dark_change
        self.debug_frames["dark_score"] = dark_score
        self.debug_frames["score"] = combined
        self.debug_frames["mask"] = mask

        return candidates

    # ------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------

    def _drop_stale_tracks(self, now: float) -> None:
        self.tracks = [
            track
            for track in self.tracks
            if now - track.last_seen <= self.max_track_idle_s
        ]

    def _ingest_candidates(self, candidates: list[dict[str, float]], now: float) -> None:
        for c in candidates:
            best_track = None
            best_dist = self.match_radius_px

            for track in self.tracks:
                dist = np.hypot(c["board_x"] - track.board_x, c["board_y"] - track.board_y)
                if dist <= best_dist:
                    best_dist = dist
                    best_track = track

            if best_track is None:
                self.tracks.append(
                    HoleTrack(
                        board_x=c["board_x"],
                        board_y=c["board_y"],
                        camera_x=c["camera_x"],
                        camera_y=c["camera_y"],
                        created_at=now,
                        last_seen=now,
                        hits=1,
                        best_score=c["score"],
                    )
                )
                continue

            alpha = 0.60
            best_track.board_x = alpha * best_track.board_x + (1.0 - alpha) * c["board_x"]
            best_track.board_y = alpha * best_track.board_y + (1.0 - alpha) * c["board_y"]
            best_track.camera_x = alpha * best_track.camera_x + (1.0 - alpha) * c["camera_x"]
            best_track.camera_y = alpha * best_track.camera_y + (1.0 - alpha) * c["camera_y"]
            best_track.last_seen = now
            best_track.hits += 1
            best_track.best_score = max(best_track.best_score, c["score"])

    def _get_stable_tracks(self, now: float) -> list[HoleTrack]:
        stable: list[HoleTrack] = []
        for track in self.tracks:
            if track.hits < self.min_confirm_frames:
                continue
            if now - track.last_seen > 0.15:
                continue
            stable.append(track)
        return stable

    # ------------------------------------------------------------
    # Known hole memory
    # ------------------------------------------------------------

    def _remember_known_hole(self, board_x: float, board_y: float, score: float) -> None:
        if self._is_duplicate(board_x, board_y):
            return
        self.known_holes.append((board_x, board_y, score))

    def _is_duplicate(self, board_x: float, board_y: float) -> bool:
        for hx, hy, _score in self.known_holes:
            if np.hypot(board_x - hx, board_y - hy) <= self.duplicate_radius_px:
                return True
        return False


hit_scanner = HitScanner()