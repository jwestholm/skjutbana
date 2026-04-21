
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pygame

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import numpy as np  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for scenes.ai_training") from exc

from src.engine.ai.runtime import get_ai_runtime


class AITrainingScene:
    """
    Minimal-text AI training scene.

    Flow:
    1. Wait for a shot / shot-like event.
    2. Freeze pre/post camera frames.
    3. Show top 10 candidates.
    4. User clicks roughly where the hit was.
    5. AI learns immediately from that click.
    6. Visual state clears and the user can shoot again.

    Important:
    - AI lives in camera/video space.
    - The user's click is transformed into camera space before training.
    - Training area is a strict central ROI, so the wall/text outside should not dominate.
    """
    wants_hit_scanning = True

    def __init__(self, bg_color=None, *args, **kwargs) -> None:
        self.bg_color = bg_color or (255, 255, 255)
        self.runtime = get_ai_runtime()
        self.app = None
        self.return_menu_state = None

        self.font_small: Optional[pygame.font.Font] = None
        self.font_big: Optional[pygame.font.Font] = None

        self.background_mode = "white"
        self.awaiting_click = False
        self.last_reset_ts = time.time()
        self.session_message = ""

        self.pre_frame_gray: Optional[np.ndarray] = None
        self.post_frame_gray: Optional[np.ndarray] = None
        self.last_camera_frame_bgr: Optional[np.ndarray] = None

        self.current_candidates: List[Dict[str, Any]] = []
        self.clicked_camera_xy: Optional[Tuple[float, float]] = None
        self.last_learning_result: Optional[Dict[str, Any]] = None
        self.last_shot_ts: Optional[float] = None

        self.cached_surface_size: Tuple[int, int] = (1280, 720)

    # ---------- Engine lifecycle ----------

    def enter(self, app=None, return_menu_state=None, *args, **kwargs):
        self.app = app
        self.return_menu_state = return_menu_state
        pygame.font.init()
        self.font_small = pygame.font.SysFont("arial", 20)
        self.font_big = pygame.font.SysFont("arial", 28, bold=True)
        self._hard_reset_visuals()
        return self

    def exit(self):
        return None

    # ---------- Internal helpers ----------

    def _hard_reset_visuals(self) -> None:
        self.awaiting_click = False
        self.current_candidates = []
        self.clicked_camera_xy = None
        self.post_frame_gray = None
        self.last_learning_result = None
        self.session_message = ""
        self.last_reset_ts = time.time()

    def _surface_size(self) -> Tuple[int, int]:
        if self.app is not None and hasattr(self.app, "screen") and self.app.screen is not None:
            try:
                w, h = self.app.screen.get_size()
                self.cached_surface_size = (int(w), int(h))
                return self.cached_surface_size
            except Exception:
                pass
        return self.cached_surface_size

    def _camera_frame_from_app(self) -> Optional[np.ndarray]:
        if self.app is None:
            return None

        candidates = [
            ("camera_manager", "last_frame"),
            ("camera_manager", "frame"),
            ("camera_manager", "latest_frame"),
            ("camera_manager", "current_frame"),
        ]
        for owner_name, attr_name in candidates:
            owner = getattr(self.app, owner_name, None)
            if owner is not None and hasattr(owner, attr_name):
                frame = getattr(owner, attr_name)
                if frame is not None:
                    return frame
            if owner is not None and hasattr(owner, "get_frame"):
                try:
                    frame = owner.get_frame()
                    if frame is not None:
                        return frame
                except Exception:
                    pass
        return None

    def _gray_from_frame(self, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if frame is None:
            return None
        if len(frame.shape) == 2:
            return frame.copy()
        if cv2 is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Fallback: assume BGR/RGB-like final axis
        return np.mean(frame[:, :, :3], axis=2).astype(np.uint8)

    def _capture_pre_if_needed(self) -> None:
        if self.pre_frame_gray is not None:
            return
        frame = self._camera_frame_from_app()
        gray = self._gray_from_frame(frame)
        if gray is not None:
            self.pre_frame_gray = gray
            self.last_camera_frame_bgr = frame

    def _extract_detector_candidates(self) -> List[Dict[str, Any]]:
        if self.app is None:
            return []
        hit_scanner = getattr(self.app, "hit_scanner", None)
        if hit_scanner is None:
            return []
        probe_names = [
            "last_candidates",
            "debug_candidates",
            "recent_candidates",
            "_last_candidates",
        ]
        for name in probe_names:
            value = getattr(hit_scanner, name, None)
            if isinstance(value, list):
                return list(value)
        for method_name in ["get_debug_candidates", "get_recent_candidates"]:
            method = getattr(hit_scanner, method_name, None)
            if callable(method):
                try:
                    value = method()
                    if isinstance(value, list):
                        return list(value)
                except Exception:
                    pass
        return []

    def _detect_new_shot(self) -> bool:
        """
        Training scene needs to work with the existing detector as support.
        We therefore watch a few possible timestamps from hit_scanner/audio detector.
        """
        if self.app is None:
            return False

        timestamps: List[float] = []
        for owner_name in ["hit_scanner", "audio_peak_detector"]:
            owner = getattr(self.app, owner_name, None)
            if owner is None:
                continue
            for attr_name in ["last_shot_ts", "last_peak_ts", "latest_peak_ts", "last_event_ts"]:
                value = getattr(owner, attr_name, None)
                if value is not None:
                    try:
                        timestamps.append(float(value))
                    except Exception:
                        pass

        if not timestamps:
            return False

        shot_ts = max(timestamps)
        if self.last_shot_ts is None or shot_ts > self.last_shot_ts + 1e-6:
            self.last_shot_ts = shot_ts
            return True
        return False

    def _camera_to_screen(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        # For now use direct normalization by camera frame size.
        # This keeps training internally consistent even before perfect game-space projection.
        width, height = self._surface_size()
        if self.post_frame_gray is not None:
            ch, cw = self.post_frame_gray.shape[:2]
        elif self.pre_frame_gray is not None:
            ch, cw = self.pre_frame_gray.shape[:2]
        else:
            ch, cw = height, width
        sx = int(round((xy[0] / max(1, cw - 1)) * width))
        sy = int(round((xy[1] / max(1, ch - 1)) * height))
        return sx, sy

    def _screen_to_camera(self, xy: Tuple[int, int]) -> Tuple[float, float]:
        width, height = self._surface_size()
        if self.post_frame_gray is not None:
            ch, cw = self.post_frame_gray.shape[:2]
        elif self.pre_frame_gray is not None:
            ch, cw = self.pre_frame_gray.shape[:2]
        else:
            ch, cw = height, width
        cx = (xy[0] / max(1, width)) * max(1, cw - 1)
        cy = (xy[1] / max(1, height)) * max(1, ch - 1)
        return (float(cx), float(cy))

    def _roi_screen_rect(self) -> pygame.Rect:
        width, height = self._surface_size()
        if self.post_frame_gray is not None:
            ch, cw = self.post_frame_gray.shape[:2]
        elif self.pre_frame_gray is not None:
            ch, cw = self.pre_frame_gray.shape[:2]
        else:
            cw, ch = width, height
        rx, ry, rw, rh = self.runtime.training_roi_rect(cw, ch)
        x0, y0 = self._camera_to_screen((rx, ry))
        x1, y1 = self._camera_to_screen((rx + rw, ry + rh))
        left = min(x0, x1)
        top = min(y0, y1)
        return pygame.Rect(left, top, abs(x1 - x0), abs(y1 - y0))

    def _finalize_shot_capture(self) -> None:
        frame = self._camera_frame_from_app()
        gray = self._gray_from_frame(frame)
        self.post_frame_gray = gray
        detector_candidates = self._extract_detector_candidates()
        self.current_candidates = self.runtime.rank_candidates(
            gray_pre=self.pre_frame_gray,
            gray_post=self.post_frame_gray,
            detector_candidates=detector_candidates,
            limit=10,
        )
        self.awaiting_click = True
        self.clicked_camera_xy = None

    # ---------- Scene API ----------

    def update(self, dt=0.0):
        self._capture_pre_if_needed()

        # Use existing detector/audio support, but keep training logic bounded to ROI.
        if not self.awaiting_click and self._detect_new_shot():
            self._finalize_shot_capture()

        # After a short quiet period without a real shot, allow manual fallback with SPACE.
        return None

    def draw(self, surface):
        width, height = surface.get_size()
        self.cached_surface_size = (width, height)

        # Very little text. Mostly clean training board.
        bg = (250, 250, 250) if self.background_mode == "white" else (10, 10, 10)
        fg = (15, 15, 15) if self.background_mode == "white" else (245, 245, 245)
        surface.fill(bg)

        roi = self._roi_screen_rect()
        pygame.draw.rect(surface, (205, 205, 205) if self.background_mode == "white" else (80, 80, 80), roi, 2)

        if self.awaiting_click:
            for index, candidate in enumerate(self.current_candidates):
                sx, sy = self._camera_to_screen((candidate["camera_x"], candidate["camera_y"]))
                is_top = index == 0
                color = (220, 30, 30) if is_top else (40, 110, 220)
                radius = 13 if is_top else 9
                pygame.draw.circle(surface, color, (sx, sy), radius, 2)
                if self.font_small is not None:
                    label = self.font_small.render(str(index + 1), True, color)
                    surface.blit(label, (sx + 10, sy - 10))

        if self.clicked_camera_xy is not None:
            sx, sy = self._camera_to_screen(self.clicked_camera_xy)
            pygame.draw.circle(surface, (230, 30, 30), (sx, sy), 18, 3)

        # Minimal mode hints only.
        if self.font_big is not None:
            caption = self.font_big.render("AI", True, fg)
            surface.blit(caption, (18, 12))

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return self.return_menu_state
            if event.key == pygame.K_TAB:
                if self.runtime.settings.get("allow_black_background", True):
                    self.background_mode = "black" if self.background_mode == "white" else "white"
                return None
            if event.key == pygame.K_SPACE and not self.awaiting_click:
                # Manual fallback for cases where audio integration is not firing yet.
                self._finalize_shot_capture()
                return None

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.awaiting_click:
            click_screen = event.pos
            click_camera = self._screen_to_camera(click_screen)
            self.clicked_camera_xy = click_camera
            self.last_learning_result = self.runtime.learn_from_click(
                click_camera_xy=click_camera,
                shown_candidates=self.current_candidates,
                gray_pre=self.pre_frame_gray,
                gray_post=self.post_frame_gray,
            )

            # Prepare next shot immediately. The click DOES train the AI and persists on disk.
            self.pre_frame_gray = self.post_frame_gray
            self._hard_reset_visuals()
            return None

        return None
