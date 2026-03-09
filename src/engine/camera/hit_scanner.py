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
    Visuell träffscanner för projicerad skjutbana.

    Design:
    - analyserar bara scanport
    - warpar scanport -> viewport/board-plan
    - bygger en korttidsreferens i board-plan
    - räknar förändringskarta + mörk/hål-karta
    - emitterar ENDAST via hit_input.push_camera_hit(...)

    Mål:
    - hitta en ny lokal förändring
    - som är mörk/hål-liknande
    - som blir kvar på samma plats
    """

    STATE_OFF = "OFF"
    STATE_ARMING = "ARMING"
    STATE_ACTIVE = "ACTIVE"

    def __init__(self) -> None:
        self.enabled = False
        self.state = self.STATE_OFF

        # State / timing
        self.arm_duration_s = 1.20
        self.min_confirm_frames = 3
        self.max_track_idle_s = 0.45
        self.global_emit_cooldown_s = 0.18
        self.last_emit_ts = 0.0
        self.arm_until_ts = 0.0

        # Tracking / duplicate
        self.match_radius_px = 20.0
        self.duplicate_radius_px = 28.0

        # History / reference
        self.history_size = 12
        self.reference_stride = 2
        self.board_history: deque[np.ndarray] = deque(maxlen=self.history_size)

        # Thresholds / filter
        self.min_combined_threshold = 26.0
        self.min_change_threshold = 8.0
        self.min_area = 10.0
        self.max_area = 420.0
        self.min_radius = 2.0
        self.max_radius = 18.0
        self.min_circularity = 0.18
        self.border_margin = 6

        # Weights
        self.change_weight = 0.58
        self.dark_weight = 0.42

        # Runtime state
        self.tracks: list[HoleTrack] = []
        self.known_holes: list[tuple[float, float, float]] = []

        self.last_status: str = "off"
        self.debug_frames: dict[str, np.ndarray] = {}
        self.last_candidates: list[dict[str, float]] = []
        self.last_stable_tracks: list[HoleTrack] = []
        self.last_board_size: tuple[int, int] = (1, 1)
        self.last_threshold_value: float = 0.0
        self.last_change_threshold_value: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enable(self) -> None:
        self.enabled = True
        self.state = self.STATE_ARMING
        self.arm_until_ts = time.time() + self.arm_duration_s
        self.last_emit_ts = 0.0

        self.tracks.clear()
        self.known_holes.clear()
        self.board_history.clear()

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

        self.tracks.clear()
        self.known_holes.clear()
        self.board_history.clear()

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

        frame = camera_manager.get_latest_frame()
        if frame is None:
            self.last_status = "no_frame"
            self.last_candidates = []
            self.last_stable_tracks = []
            return

        prepared = self._prepare_board_view(frame)
        if prepared is None:
            self.last_status = "not_ready"
            self.last_candidates = []
            self.last_stable_tracks = []
            return

        gray, board_to_crop_matrix, scanport = prepared
        self.last_board_size = (gray.shape[1], gray.shape[0])

        self.board_history.append(gray.copy())

        now = time.time()

        # Vi behöver några frames innan vi kan bygga referens.
        if len(self.board_history) < max(4, self.reference_stride + 2):
            self.last_status = f"arming history={len(self.board_history)}"
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

        self._drop_stale_tracks(now)
        self._ingest_candidates(candidates, now)

        stable_tracks = self._get_stable_tracks(now)

        self.last_candidates = [dict(item) for item in candidates]
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
            # Allt som redan finns under armning betraktas som befintliga hål.
            for track in stable_tracks:
                self._remember_known_hole(track.board_x, track.board_y, track.best_score)
                track.emitted = True

            if now >= self.arm_until_ts:
                self.state = self.STATE_ACTIVE
                self.tracks.clear()
                self.last_status = "active"
            else:
                self.last_status = (
                    f"arming hist={len(self.board_history)} cand={len(candidates)}"
                )
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

    def get_status_lines(self) -> list[str]:
        return [
            f"HitScanner state: {self.state}",
            f"Known holes: {len(self.known_holes)}",
            f"Tracks: {len(self.tracks)}",
            f"Thr combined: {self.last_threshold_value:.1f}",
            f"Thr change: {self.last_change_threshold_value:.1f}",
            f"Status: {self.last_status}",
        ]

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
            "candidates": [dict(item) for item in self.last_candidates],
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

    # ------------------------------------------------------------------
    # Board preparation
    # ------------------------------------------------------------------

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

        translate_crop_to_camera = np.array(
            [
                [1.0, 0.0, float(x)],
                [0.0, 1.0, float(y)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        translate_screen_to_board_local = np.array(
            [
                [1.0, 0.0, -float(viewport.x)],
                [0.0, 1.0, -float(viewport.y)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        H_crop_to_board = (
            translate_screen_to_board_local
            @ H_camera_to_screen
            @ translate_crop_to_camera
        )

        try:
            board_to_crop = np.linalg.inv(H_crop_to_board).astype(np.float32)
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

        return gray, board_to_crop, scanport

    # ------------------------------------------------------------------
    # Reference model
    # ------------------------------------------------------------------

    def _build_reference_image(self) -> np.ndarray:
        """
        Bygger en robust referens från tidigare frames.
        Vi använder äldre frames, inte allra senaste, så att ett nytt hål
        inte direkt skrivs in i referensen.
        """
        hist = list(self.board_history)
        if len(hist) <= self.reference_stride:
            return hist[-1]

        usable = hist[:-self.reference_stride]
        if len(usable) == 1:
            return usable[0]

        stack = np.stack(usable, axis=0).astype(np.uint8)
        reference = np.median(stack, axis=0).astype(np.uint8)
        return reference

    # ------------------------------------------------------------------
    # Candidate detection
    # ------------------------------------------------------------------

    def _detect_candidates(
        self,
        gray: np.ndarray,
        reference_gray: np.ndarray,
        board_to_crop_matrix: np.ndarray,
        scanport,
    ) -> list[dict[str, float]]:
        gray_blur = cv2.GaussianBlur(gray, (0, 0), 1.1)
        ref_blur = cv2.GaussianBlur(reference_gray, (0, 0), 1.1)

        # 1) Förändringskarta - bara mörkare förändringar är intressanta.
        dark_change = cv2.subtract(ref_blur, gray_blur)

        # 2) Hål-liknande mörk lokal struktur i nuvarande frame.
        local_mean = cv2.GaussianBlur(gray_blur, (0, 0), 8.0)
        darkness = cv2.subtract(local_mean, gray_blur)

        blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        blackhat = cv2.morphologyEx(gray_blur, cv2.MORPH_BLACKHAT, blackhat_kernel)

        dark_score = cv2.addWeighted(darkness, 0.62, blackhat, 0.38, 0.0)

        # 3) Ta bort svagt globalt flimmer / mycket små variationer.
        change_mu, change_sigma = cv2.meanStdDev(dark_change)
        change_threshold = max(
            self.min_change_threshold,
            float(change_mu[0][0] + 1.7 * change_sigma[0][0]),
        )
        self.last_change_threshold_value = change_threshold

        _, change_mask = cv2.threshold(
            dark_change,
            change_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        change_mask = cv2.morphologyEx(
            change_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )

        # 4) Kombinera förändring + mörk/hål-score
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
            float(comb_mu[0][0] + 2.0 * comb_sigma[0][0]),
        )
        self.last_threshold_value = combined_threshold

        _, combined_mask = cv2.threshold(
            combined,
            combined_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        # Kräver både förändring och hål-liknande score.
        mask = cv2.bitwise_and(combined_mask, change_mask)

        # Stabilisera blobbar.
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
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

            ix = int(round(cx))
            iy = int(round(cy))
            ix = max(0, min(ix, width - 1))
            iy = max(0, min(iy, height - 1))

            change_value = float(dark_change[iy, ix])
            dark_value = float(dark_score[iy, ix])
            combined_value = float(combined[iy, ix])

            # Extra skydd: kräver både verklig förändring och hål-liknande mörker
            if change_value < self.last_change_threshold_value:
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

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def _drop_stale_tracks(self, now: float) -> None:
        self.tracks = [
            track
            for track in self.tracks
            if now - track.last_seen <= self.max_track_idle_s
        ]

    def _ingest_candidates(self, candidates: list[dict[str, float]], now: float) -> None:
        for candidate in candidates:
            best_track = None
            best_dist = self.match_radius_px

            for track in self.tracks:
                dist = np.hypot(
                    candidate["board_x"] - track.board_x,
                    candidate["board_y"] - track.board_y,
                )
                if dist <= best_dist:
                    best_dist = dist
                    best_track = track

            if best_track is None:
                self.tracks.append(
                    HoleTrack(
                        board_x=candidate["board_x"],
                        board_y=candidate["board_y"],
                        camera_x=candidate["camera_x"],
                        camera_y=candidate["camera_y"],
                        created_at=now,
                        last_seen=now,
                        hits=1,
                        best_score=candidate["score"],
                    )
                )
                continue

            alpha = 0.60
            best_track.board_x = alpha * best_track.board_x + (1.0 - alpha) * candidate["board_x"]
            best_track.board_y = alpha * best_track.board_y + (1.0 - alpha) * candidate["board_y"]
            best_track.camera_x = alpha * best_track.camera_x + (1.0 - alpha) * candidate["camera_x"]
            best_track.camera_y = alpha * best_track.camera_y + (1.0 - alpha) * candidate["camera_y"]
            best_track.last_seen = now
            best_track.hits += 1
            best_track.best_score = max(best_track.best_score, candidate["score"])

    def _get_stable_tracks(self, now: float) -> list[HoleTrack]:
        stable: list[HoleTrack] = []
        for track in self.tracks:
            if track.hits < self.min_confirm_frames:
                continue
            if now - track.last_seen > 0.14:
                continue
            stable.append(track)
        return stable

    # ------------------------------------------------------------------
    # Known hole memory
    # ------------------------------------------------------------------

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