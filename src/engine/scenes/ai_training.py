"""
AI Training Scene.

Key flows:
- Real/manual shot: original training flow
- Right click: one synthetic/manual round
- F1: stable auto-training state machine with visible steps and delays

Important behavior:
- During clean-capture/photo, HUD/markers/cursor are hidden
- Synthetic holes remain visible during clean-capture and review
- Auto-training waits between the phases instead of rushing
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

import pygame

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None

from src.engine.ai.runtime import get_ai_runtime
from src.engine.ai.space_mapper import project_screen_point
from src.engine.audio.audio_peak_detector import AudioPeakEvent, audio_peak_detector
from src.engine.camera.camera_manager import camera_manager
from src.engine.camera.hit_scanner import hit_scanner
from src.engine.input.hit_input import hit_input
from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import load_viewport_rect
from src.engine.synthetic.synthetic_hole_overlay import SyntheticHoleOverlay

BG_WHITE = (248, 248, 248)
BG_BLACK = (18, 18, 18)
CYAN = (80, 220, 255)
YELLOW = (255, 210, 70)
ORANGE = (255, 150, 80)
GREEN = (120, 255, 120)
RED = (255, 110, 110)
CLICK_COLOR = RED
HUD_BG = (0, 0, 0, 120)
GRID_LINE = (222, 222, 222)
WHITE = (240, 240, 240)
SOFT_WHITE = (210, 210, 210)


class AITrainingScene(Scene):
    wants_hit_scanning = True
    wants_camera_preview = False

    MODE_NAMES = ["white", "white_grid", "gray", "black", "checker", "checker_anim", "bubbles"]

    def __init__(self, bg_color=None, **kwargs) -> None:
        super().__init__()
        self.runtime = get_ai_runtime()
        self._bubbles: list[dict] = []

        if isinstance(bg_color, (tuple, list)) and len(bg_color) >= 3:
            brightness = sum(int(c) for c in bg_color[:3]) / 3.0
            self.bg_mode_index = 1 if brightness < 96 else 0
        else:
            self.bg_mode_index = 0

        self.font: pygame.font.Font | None = None
        self.small: pygame.font.Font | None = None
        self.tiny: pygame.font.Font | None = None
        self.viewport: pygame.Rect | None = None

        self.awaiting_click = False
        self.ranked_candidates: list[dict[str, Any]] = []
        self.clicked_camera_xy: tuple[float, float] | None = None
        self.last_learning_result: dict[str, Any] | None = None
        self.status_message = ""

        self._pending_click_camera: tuple[float, float] | None = None
        self._pending_click_phase: str | None = None
        self._pending_wait_frames: int = 0

        self._reviewing = False
        self._review_pre_surface: pygame.Surface | None = None
        self._review_post_surface: pygame.Surface | None = None

        self.t = 0.0
        self._animation_frozen = False
        self._last_peak_ts = 0.0

        self.synthetic_overlay: SyntheticHoleOverlay | None = None
        self._overlay_size: tuple[int, int] = (0, 0)

        # Cursor state (avoid blinking by only toggling when the desired state changes).
        self._cursor_visible = True
        self._handled_shot_peak_ts = 0.0

        # Synthetic scheduling / settle timing.
        self.synthetic_trigger_delay_s = 0.20
        self.background_settle_delay_s = 0.25
        self.background_settle_until_ts = 0.0
        self.synthetic_trigger_pending = False
        self.synthetic_trigger_batch_mode = False
        self.synthetic_trigger_screen_xy: tuple[int, int] | None = None
        self.synthetic_trigger_ready_ts = 0.0

        # Single synthetic round (right click).
        self.single_synth_round_active = False
        self.single_target_screen_xy: tuple[int, int] | None = None

        # Auto-training configuration.
        self.auto_training_enabled = False
        self.auto_headless = False  # F2: no visuals, no delays, just train
        self.auto_target_iterations = 100
        self.auto_iteration = 0
        self.auto_target_screen_xy: tuple[int, int] | None = None
        self.auto_active_hole_id: str | None = None
        self.auto_last_trigger_ts = 0.0
        self.auto_min_trigger_gap_s = 0.08

        # Auto-training flow state.
        self.auto_phase = "idle"  # idle / waiting_markers / waiting_review / waiting_next
        self.auto_click_delay_s = 1.0
        self.auto_review_delay_s = 1.0
        self.auto_next_iteration_delay_s = 0.5
        self.auto_click_ready_ts = 0.0
        self.auto_review_ready_ts = 0.0
        self.auto_next_iteration_ts = 0.0

        # Auto stats / report.
        self.auto_report_visible = False
        self.auto_report_lines: list[str] = []
        self._reset_auto_stats()

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _set_cursor_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if visible != self._cursor_visible:
            pygame.mouse.set_visible(visible)
            self._cursor_visible = visible

    def _reset_auto_stats(self) -> None:
        self.auto_stats: dict[str, Any] = {
            "iterations": 0,
            "shots_triggered": 0,
            "candidate_rounds": 0,
            "found": 0,
            "top1": 0,
            "top3": 0,
            "missed": 0,
            "nearest_distance_sum": 0.0,
            "nearest_distance_count": 0,
        }

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 34)
        self.small = pygame.font.Font(None, 24)
        self.tiny = pygame.font.Font(None, 18)
        self.viewport = load_viewport_rect()
        self.runtime = get_ai_runtime()
        self._reset_shot_state(clear_synthetic=True)
        self._last_peak_ts = audio_peak_detector.last_peak_ts
        self._handled_shot_peak_ts = 0.0
        self._set_cursor_visible(True)

    def on_exit(self) -> None:
        self._set_cursor_visible(True)

    def _reset_shot_state(self, *, clear_synthetic: bool = False) -> None:
        self.awaiting_click = False
        self.ranked_candidates = []
        self.clicked_camera_xy = None
        self.last_learning_result = None
        self._animation_frozen = False
        self._reviewing = False
        self._review_pre_surface = None
        self._review_post_surface = None
        self._pending_click_camera = None
        self._pending_click_phase = None
        self._pending_wait_frames = 0
        self._last_peak_ts = audio_peak_detector.last_peak_ts
        self._handled_shot_peak_ts = 0.0

        self.synthetic_trigger_pending = False
        self.synthetic_trigger_batch_mode = False
        self.synthetic_trigger_screen_xy = None
        self.synthetic_trigger_ready_ts = 0.0

        self.single_synth_round_active = False
        self.single_target_screen_xy = None

        if clear_synthetic and self.synthetic_overlay is not None:
            self.synthetic_overlay.clear()
            self.auto_active_hole_id = None
            self.auto_target_screen_xy = None

        self._set_cursor_visible(True)

    def _ensure_overlay(self, screen: pygame.Surface) -> SyntheticHoleOverlay:
        size = (screen.get_width(), screen.get_height())
        if self.synthetic_overlay is None or self._overlay_size != size:
            self.synthetic_overlay = SyntheticHoleOverlay(size[0], size[1], rng_seed=42)
            self._overlay_size = size
        return self.synthetic_overlay

    def _clear_synthetic_holes(self) -> None:
        if self.synthetic_overlay is not None:
            self.synthetic_overlay.clear()
        self.auto_active_hole_id = None
        self.auto_target_screen_xy = None
        self.single_target_screen_xy = None

    def _enter_review(self, click_camera_xy: tuple[float, float], fresh_post_gray=None) -> None:
        self._reviewing = True
        self._review_pre_surface = None
        self._review_post_surface = None

        gray_pre = self.runtime.pre_shot_gray
        gray_post = fresh_post_gray if fresh_post_gray is not None else self.runtime.post_shot_gray
        if gray_post is None:
            return

        ix, iy = int(round(click_camera_xy[0])), int(round(click_camera_xy[1]))
        h, w = gray_post.shape[:2]
        patch_r = 80
        x0, y0 = max(0, ix - patch_r), max(0, iy - patch_r)
        x1, y1 = min(w, ix + patch_r + 1), min(h, iy + patch_r + 1)
        if x1 <= x0 or y1 <= y0:
            return

        post_patch = gray_post[y0:y1, x0:x1]
        self._review_post_surface = self._gray_patch_to_surface(post_patch)

        if gray_pre is not None and gray_pre.shape == gray_post.shape:
            pre_patch = gray_pre[y0:y1, x0:x1]
            self._review_pre_surface = self._gray_patch_to_surface(pre_patch)

    @staticmethod
    def _gray_patch_to_surface(patch) -> pygame.Surface | None:
        if patch is None or patch.size == 0:
            return None
        try:
            if cv2 is not None:
                rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
            else:
                rgb = np.stack([patch, patch, patch], axis=-1)
            h, w = rgb.shape[:2]
            return pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
        except Exception:
            return None

    @staticmethod
    def _save_hole_image(click_camera_xy, post_gray, hole_type: str = "hole") -> None:
        """Save the raw post-shot patch as a grayscale PNG.
        
        Args:
            click_camera_xy: camera coordinates of the hole
            post_gray: grayscale camera frame
            hole_type: 'synt' for synthetic, 'hole' for real shots
        """
        if post_gray is None or np is None:
            return
        try:
            from pathlib import Path

            ix, iy = int(round(click_camera_xy[0])), int(round(click_camera_xy[1]))
            h, w = post_gray.shape[:2]
            patch_r = 64  # 128x128 patch
            x0, y0 = max(0, ix - patch_r), max(0, iy - patch_r)
            x1, y1 = min(w, ix + patch_r), min(h, iy + patch_r)

            if x1 <= x0 or y1 <= y0:
                return

            patch = post_gray[y0:y1, x0:x1]
            if patch.size == 0:
                return

            holes_dir = Path("content/ai/holes")
            holes_dir.mkdir(parents=True, exist_ok=True)

            # Find next number for this type
            prefix = str(hole_type) + "_"
            existing = sorted(holes_dir.glob(f"{prefix}*.png"))
            next_num = 1
            for f in existing:
                try:
                    num = int(f.stem.replace(prefix, ""))
                    next_num = max(next_num, num + 1)
                except ValueError:
                    pass

            path = holes_dir / f"{prefix}{next_num:07d}.png"

            if cv2 is not None:
                cv2.imwrite(str(path), patch)
            else:
                rgb = np.stack([patch, patch, patch], axis=-1)
                surf = pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
                pygame.image.save(surf, str(path))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Auto report / stats
    # ------------------------------------------------------------------
    def _record_detection_stats(self, ranked_candidates: list[dict[str, Any]]) -> None:
        if not self.auto_training_enabled or self.auto_target_screen_xy is None:
            return

        self.auto_stats["candidate_rounds"] += 1

        projected = project_screen_point(float(self.auto_target_screen_xy[0]), float(self.auto_target_screen_xy[1]))
        tx, ty = float(projected.camera_x), float(projected.camera_y)
        match_radius = float(self.runtime.settings.get("click_match_radius_px", 42.0))

        nearest_rank = None
        nearest_distance = None

        for cand in ranked_candidates:
            cx = float(cand.get("camera_x", 0.0))
            cy = float(cand.get("camera_y", 0.0))
            d = math.hypot(cx - tx, cy - ty)
            if nearest_distance is None or d < nearest_distance:
                nearest_distance = d
                nearest_rank = int(cand.get("rank", 999))

        if nearest_distance is not None:
            self.auto_stats["nearest_distance_sum"] += float(nearest_distance)
            self.auto_stats["nearest_distance_count"] += 1

        found = nearest_distance is not None and nearest_distance <= match_radius
        if found:
            self.auto_stats["found"] += 1
            if nearest_rank == 1:
                self.auto_stats["top1"] += 1
            if nearest_rank is not None and nearest_rank <= 3:
                self.auto_stats["top3"] += 1
        else:
            self.auto_stats["missed"] += 1

    def _build_auto_report(self) -> None:
        total = int(self.auto_stats["iterations"])
        found = int(self.auto_stats["found"])
        top1 = int(self.auto_stats["top1"])
        top3 = int(self.auto_stats["top3"])
        missed = int(self.auto_stats["missed"])
        rounds = int(self.auto_stats["candidate_rounds"])
        shots = int(self.auto_stats["shots_triggered"])
        dist_count = int(self.auto_stats["nearest_distance_count"])
        avg_dist = (
            float(self.auto_stats["nearest_distance_sum"]) / float(dist_count)
            if dist_count > 0 else 0.0
        )

        # Funnel diagnostics summary
        funnel_lines = self.runtime.funnel.format_summary_lines()

        self.auto_report_lines = [
            "Autoträning klar",
            "",
            f"Iterationer: {total}",
            f"Skott körda: {shots}",
            f"Rundor med kandidater: {rounds}",
            f"AI hittade hålet: {found} / {total}",
            f"Top-1 rätt: {top1} / {total}",
            f"Top-3 rätt: {top3} / {total}",
            f"Missade: {missed} / {total}",
            f"Medelavstånd bästa kandidat: {avg_dist:.1f} px" if dist_count > 0 else "Medelavstånd bästa kandidat: n/a",
            "",
            "--- Funnel-diagnostik ---",
        ] + funnel_lines + [
            "",
            "Klicka eller tryck en tangent för att stänga rapporten.",
        ]
        self.auto_report_visible = True

        # Save CSV report
        try:
            csv_path = self.runtime.funnel.save_csv("autotrain")
            if csv_path:
                self.auto_report_lines.insert(-1, f"CSV sparad: {csv_path.name}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Synthetic shot setup / firing
    # ------------------------------------------------------------------
    def _toggle_auto_training(self, headless: bool = False) -> None:
        self.auto_training_enabled = not self.auto_training_enabled
        self.auto_headless = headless if self.auto_training_enabled else False
        self.auto_report_visible = False
        self.auto_report_lines = []

        if self.auto_training_enabled:
            self._reset_auto_stats()
            self.runtime.funnel.clear()
            self.auto_iteration = 0
            self.auto_phase = "waiting_next"
            self.auto_next_iteration_ts = time.time() + 0.2
            mode_label = "headless" if headless else "visuell"
            self.status_message = f"Autoträning startad ({mode_label}, F1/F2 stoppar)."
            self._reset_shot_state(clear_synthetic=True)
        else:
            self.auto_phase = "idle"
            self.auto_headless = False
            self.status_message = "Autoträning stoppad."
            self._reset_shot_state(clear_synthetic=True)

    def _choose_auto_screen_point(self, vp: pygame.Rect) -> tuple[int, int]:
        margin = 12
        x = random.randint(vp.left + margin, max(vp.left + margin, vp.right - margin))
        y = random.randint(vp.top + margin, max(vp.top + margin, vp.bottom - margin))
        return x, y

    def _start_auto_iteration(self, screen: pygame.Surface) -> None:
        if self.auto_iteration >= self.auto_target_iterations:
            self.auto_training_enabled = False
            self.auto_phase = "idle"
            self._build_auto_report()
            return

        vp = self.viewport or pygame.Rect(0, 0, screen.get_width(), screen.get_height())
        target_xy = self._choose_auto_screen_point(vp)
        if self._trigger_synthetic_shot_at(screen, target_xy, batch_mode=True):
            self.auto_phase = "waiting_fire"

    def _trigger_synthetic_shot_at(
        self,
        screen: pygame.Surface,
        screen_xy: tuple[int, int],
        *,
        batch_mode: bool,
    ) -> bool:
        now = time.time()
        if now - self.auto_last_trigger_ts < self.auto_min_trigger_gap_s:
            return False

        overlay = self._ensure_overlay(screen)
        if not batch_mode:
            self._clear_synthetic_holes()

        sx = int(round(screen_xy[0]))
        sy = int(round(screen_xy[1]))

        if batch_mode:
            self.auto_target_screen_xy = (sx, sy)
        else:
            self.single_synth_round_active = True
            self.single_target_screen_xy = (sx, sy)

        hole_kind = random.choices(
            ["clean_hole", "torn_hole", "ragged_hole", "dent_ring", "weak_indent"],
            weights=[66, 16, 10, 6, 2],
            k=1,
        )[0]
        radius_px = random.uniform(1.7, 2.9)

        self.auto_active_hole_id = overlay.add_hole(
            sx,
            sy,
            kind=hole_kind,
            radius_px=radius_px,
            strength=random.uniform(0.90, 1.18),
            opacity=random.uniform(0.94, 1.0),
        )

        # Hide cursor before the camera sees the setup.
        self._set_cursor_visible(False)

        self.synthetic_trigger_pending = True
        self.synthetic_trigger_batch_mode = batch_mode
        self.synthetic_trigger_screen_xy = (sx, sy)
        self.synthetic_trigger_ready_ts = max(
            now + self.synthetic_trigger_delay_s,
            self.background_settle_until_ts,
        )

        if batch_mode:
            self.status_message = (
                f"Autoträning {self.auto_iteration + 1}/{self.auto_target_iterations}: "
                "väntar på kamerasettling..."
            )
        else:
            self.status_message = "Syntetisk runda förbereds..."
        return True

    def _fire_pending_synthetic_shot(self) -> bool:
        if not self.synthetic_trigger_pending:
            return False

        batch_mode = self.synthetic_trigger_batch_mode
        event_ts = time.time()

        audio_peak_detector.last_peak_ts = event_ts
        try:
            hit_scanner._on_audio_peak(
                AudioPeakEvent(
                    timestamp=event_ts,
                    peak=max(1.0, float(getattr(audio_peak_detector, "min_abs_peak", 0.2)) * 1.5),
                    rms=0.0,
                )
            )
        except Exception as exc:
            self.status_message = f"Syntetiskt skott misslyckades: {exc}"
            self.auto_training_enabled = False
            self.auto_phase = "idle"
            self.synthetic_trigger_pending = False
            self._clear_synthetic_holes()
            self._set_cursor_visible(True)
            return False

        self.synthetic_trigger_pending = False
        self._animation_frozen = True
        self._last_peak_ts = event_ts
        self.auto_last_trigger_ts = event_ts
        self.auto_stats["shots_triggered"] += 1

        if batch_mode:
            self.auto_phase = "waiting_detection"
            self.status_message = (
                f"Autoträning {self.auto_iteration + 1}/{self.auto_target_iterations}: "
                "syntetiskt skott triggat."
            )
        else:
            self.status_message = "Syntetiskt skott triggat på vald punkt."

        return True

    # ------------------------------------------------------------------
    # Scene events
    # ------------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event):
        if self.auto_report_visible:
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN):
                self.auto_report_visible = False
                self.auto_report_lines = []
                self._reset_shot_state(clear_synthetic=True)
                self.status_message = ""
            return None

        if self._reviewing:
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN):
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    from src.engine.scenes.menu import MenuScene
                    return SceneSwitch(MenuScene())
                self._reset_shot_state(clear_synthetic=True)
                return None
            return None

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                from src.engine.scenes.menu import MenuScene
                return SceneSwitch(MenuScene())

            if event.key == pygame.K_F1:
                self._toggle_auto_training(headless=False)
                return None

            if event.key == pygame.K_F2:
                self._toggle_auto_training(headless=True)
                return None

            if event.key == pygame.K_TAB:
                self.bg_mode_index = (self.bg_mode_index + 1) % len(self.MODE_NAMES)
                self.background_settle_until_ts = time.time() + self.background_settle_delay_s
                self.status_message = "Bakgrund bytt – vänta ett ögonblick innan skott."
                return None

            if event.key == pygame.K_r:
                self.runtime.memory.reset()
                self._reset_auto_stats()
                self._reset_shot_state(clear_synthetic=True)
                self.auto_phase = "idle"
                self.status_message = "AI nollställd."
                return None

            if event.key == pygame.K_SPACE and not self.awaiting_click:
                self._on_shot_detected()
                return None

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.awaiting_click:
                    self._on_training_click(event.pos)
                return None

            if event.button == 3:
                if (not self.awaiting_click and not self._reviewing and not self.synthetic_trigger_pending and
                        not self.auto_training_enabled and not self.auto_report_visible):
                    screen = pygame.display.get_surface()
                    if screen is not None:
                        self._trigger_synthetic_shot_at(screen, event.pos, batch_mode=False)
                return None

        return None

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, dt: float):
        current_peak_ts = audio_peak_detector.last_peak_ts
        if current_peak_ts > self._last_peak_ts and not self._animation_frozen:
            self._animation_frozen = True
            self._last_peak_ts = current_peak_ts

        if not self._animation_frozen:
            self.t += dt

        now = time.time()

        # Stable auto-training state machine.
        if self.auto_training_enabled:
            if self.auto_phase == "waiting_next" and not self.awaiting_click and not self._reviewing and not self.synthetic_trigger_pending:
                if now >= self.auto_next_iteration_ts:
                    screen = pygame.display.get_surface()
                    if screen is not None:
                        self._start_auto_iteration(screen)

            elif self.auto_phase == "waiting_markers" and self.awaiting_click:
                click_ready = now >= self.auto_click_ready_ts if not self.auto_headless else True
                if click_ready and self.auto_target_screen_xy is not None:
                    self._on_training_click(self.auto_target_screen_xy)
                    # _on_training_click starts clean capture. Next visible phase is review.
                    self.auto_phase = "waiting_review"

            elif self.auto_phase == "waiting_review" and self._reviewing:
                if self.auto_headless:
                    # Headless: skip review immediately
                    review_ready = True
                else:
                    review_ready = now >= self.auto_review_ready_ts
                if review_ready:
                    # Auto-dismiss review after it has been visible for 1 second.
                    self._reviewing = False
                    self._review_pre_surface = None
                    self._review_post_surface = None
                    self.clicked_camera_xy = None
                    self.awaiting_click = False
                    self._set_cursor_visible(True)
                    if self.auto_iteration >= self.auto_target_iterations:
                        self.auto_training_enabled = False
                        self.auto_phase = "idle"
                        self._build_auto_report()
                    else:
                        delay = 0.05 if self.auto_headless else self.auto_next_iteration_delay_s
                        self.auto_next_iteration_ts = now + delay
                        self.auto_phase = "waiting_next"

        if self.synthetic_trigger_pending and now >= self.synthetic_trigger_ready_ts:
            self._fire_pending_synthetic_shot()

        if (
            not self.awaiting_click
            and not self._reviewing
            and self.runtime.has_new_shot
            and audio_peak_detector.last_peak_ts > self._handled_shot_peak_ts
        ):
            self._handled_shot_peak_ts = audio_peak_detector.last_peak_ts
            self._on_shot_detected()

        if self._pending_click_phase == "wait_frame":
            self._pending_wait_frames += 1
            needed = 1 if getattr(self, 'auto_headless', False) else 5
            if self._pending_wait_frames >= needed:
                self._pending_click_phase = "capture"
        elif self._pending_click_phase == "capture":
            self._do_clean_capture()

    # ------------------------------------------------------------------
    # Shot / click / learning
    # ------------------------------------------------------------------
    def _do_clean_capture(self) -> None:
        # Important: during clean capture, the cursor must stay hidden.
        self._set_cursor_visible(False)

        click_camera = self._pending_click_camera
        self._pending_click_phase = None
        self._pending_click_camera = None
        self._pending_wait_frames = 0

        if click_camera is None:
            self._set_cursor_visible(True)
            return

        clean_post_gray = None
        try:
            frame_bgr = camera_manager.get_latest_frame()
            if frame_bgr is not None and cv2 is not None:
                clean_post_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            pass

        post_gray = clean_post_gray if clean_post_gray is not None else self.runtime.post_shot_gray
        pre_gray = self.runtime.pre_shot_gray

        result = self.runtime.learn_from_click(
            click_camera_xy=click_camera,
            shown_candidates=self.ranked_candidates,
            gray_pre=pre_gray,
            gray_post=post_gray,
        )
        self.last_learning_result = result

        self.runtime.study_click_area(
            click_camera_xy=click_camera,
            gray_pre=pre_gray,
            gray_post=post_gray,
        )

        # Save hole image to build training image bank
        is_synthetic = self.auto_training_enabled or self.single_synth_round_active
        self._save_hole_image(click_camera, post_gray, hole_type="synt" if is_synthetic else "hole")

        self._enter_review(click_camera, post_gray)

        if self.auto_training_enabled:
            self.auto_iteration += 1
            self.auto_stats["iterations"] = self.auto_iteration
            self.status_message = f"Autoträning: iteration {self.auto_iteration}/{self.auto_target_iterations} klar."
            self.auto_review_ready_ts = time.time() + self.auto_review_delay_s
            self.auto_phase = "waiting_review"
        else:
            # Manual/right-click round: review stays until user dismisses.
            self._set_cursor_visible(True)

    def _on_shot_detected(self) -> None:
        all_candidates = list(hit_scanner.last_candidates)
        if not all_candidates:
            self.status_message = "Skott detekterat men inga kandidater."
            if self.auto_training_enabled:
                self.auto_stats["missed"] += 1
                self.auto_iteration += 1
                self.auto_stats["iterations"] = self.auto_iteration
                if self.auto_iteration >= self.auto_target_iterations:
                    self.auto_training_enabled = False
                    self.auto_phase = "idle"
                    self._build_auto_report()
                else:
                    self.auto_next_iteration_ts = time.time() + self.auto_next_iteration_delay_s
                    self.auto_phase = "waiting_next"
            else:
                self._set_cursor_visible(True)
            return

        # Determine ground truth for funnel diagnostics
        gt_xy = None
        if self.auto_training_enabled and self.auto_target_screen_xy is not None:
            projected = project_screen_point(
                float(self.auto_target_screen_xy[0]),
                float(self.auto_target_screen_xy[1]),
            )
            gt_xy = (projected.camera_x, projected.camera_y)

        # Use full funnel pipeline: reject noise → rank → diagnostics
        # Use same match radius as UI report for consistency
        match_radius = float(self.runtime.settings.get("click_match_radius_px", 42.0))
        self.ranked_candidates, diag = self.runtime.rank_with_funnel(
            all_candidates, gt_xy=gt_xy, limit=150, match_radius_px=match_radius,
        )
        self.awaiting_click = True
        self.clicked_camera_xy = None
        self._record_detection_stats(self.ranked_candidates)

        if self.auto_training_enabled:
            self.auto_click_ready_ts = time.time() + self.auto_click_delay_s
            self.auto_phase = "waiting_markers"
            self.status_message = (
                f"Autoträning {self.auto_iteration + 1}/{self.auto_target_iterations}: "
                f"{len(self.ranked_candidates)} kandidater hittade, auto-klick om {self.auto_click_delay_s:.1f} s."
            )
            self._set_cursor_visible(False)
        else:
            self.status_message = f"Skott! {len(self.ranked_candidates)} kandidater. Klicka var du träffade."
            self._set_cursor_visible(True)

    def _on_training_click(self, screen_pos: tuple[int, int]) -> None:
        # Hide mouse immediately after click so only the projected world remains for photo/capture.
        self._set_cursor_visible(False)

        projected = project_screen_point(float(screen_pos[0]), float(screen_pos[1]))
        click_camera = (projected.camera_x, projected.camera_y)
        self.clicked_camera_xy = click_camera
        self._pending_click_camera = click_camera
        self._pending_click_phase = "clean_render"
        self.awaiting_click = False

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render(self, screen: pygame.Surface) -> None:
        overlay = self._ensure_overlay(screen)
        mode = self.MODE_NAMES[self.bg_mode_index]
        vp = self.viewport or pygame.Rect(0, 0, screen.get_width(), screen.get_height())

        screen.fill((30, 30, 30))
        self._render_background(screen, mode, vp)

        # Synthetic holes must remain visible even in clean capture / review.
        if overlay is not None:
            composed = overlay.composite_on(pygame.surfarray.array3d(screen).swapaxes(0, 1))
            composed = composed.swapaxes(0, 1)
            pygame.surfarray.blit_array(screen, composed)

        # Clean render for photo/capture: show only world + synthetic holes.
        if self._pending_click_phase in ("clean_render", "wait_frame", "capture"):
            self._set_cursor_visible(False)
            if self._pending_click_phase == "clean_render":
                self._pending_click_phase = "wait_frame"
                self._pending_wait_frames = 0
            return

        # Headless mode: only show progress counter, no candidates/review/HUD
        if self.auto_headless and self.auto_training_enabled:
            if self.tiny:
                progress = f"Headless: {self.auto_iteration}/{self.auto_target_iterations}"
                surf = self.tiny.render(progress, True, (180, 180, 180))
                screen.blit(surf, (vp.x + 8, vp.y + 8))
            if self.auto_report_visible:
                self._render_auto_report(screen, vp)
            return

        self._render_candidates(screen, vp)
        self._render_click_feedback(screen)

        if self._reviewing:
            self._render_review(screen, vp)

        self._render_hud(screen, mode, vp)

        if self.auto_report_visible:
            self._render_auto_report(screen, vp)

    def _render_auto_report(self, screen: pygame.Surface, vp: pygame.Rect) -> None:
        lines = self.auto_report_lines or ["Autoträning klar"]
        width = min(720, max(420, vp.w - 80))
        line_h = 28
        height = 80 + len(lines) * line_h
        panel_rect = pygame.Rect(0, 0, width, height)
        panel_rect.center = vp.center

        panel = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        panel.fill((0, 0, 0, 220))
        pygame.draw.rect(panel, (160, 160, 160), panel.get_rect(), 2)
        screen.blit(panel, panel_rect.topleft)

        y = panel_rect.y + 20
        for i, line in enumerate(lines):
            font = self.font if i == 0 and self.font is not None else self.small
            color = WHITE if i != 0 else ORANGE
            if font is None:
                continue
            surf = font.render(line, True, color)
            x = panel_rect.centerx - surf.get_width() // 2
            screen.blit(surf, (x, y))
            y += line_h

    def _render_review(self, screen: pygame.Surface, vp: pygame.Rect) -> None:
        zoom = 3
        gap = 24
        pre = self._review_pre_surface
        post = self._review_post_surface
        if post is None:
            return

        pw, ph = post.get_size()
        scaled_w, scaled_h = pw * zoom, ph * zoom
        label_h = 22
        total_w = scaled_w * 2 + gap if pre is not None else scaled_w
        total_h = scaled_h + label_h + 30

        panel_rect = pygame.Rect(0, 0, total_w + 40, total_h + 20)
        panel_rect.center = (vp.centerx, vp.centery)
        panel = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        panel.fill((0, 0, 0, 200))
        pygame.draw.rect(panel, (120, 120, 120), panel.get_rect(), 2)
        screen.blit(panel, panel_rect.topleft)

        x_start = panel_rect.x + 20
        y_start = panel_rect.y + 10

        if pre is not None:
            scaled_pre = pygame.transform.scale(pre, (scaled_w, scaled_h))
            screen.blit(scaled_pre, (x_start, y_start + label_h))
            pygame.draw.rect(screen, CYAN, (x_start, y_start + label_h, scaled_w, scaled_h), 2)
            cx_pre = x_start + scaled_w // 2
            cy_pre = y_start + label_h + scaled_h // 2
            pygame.draw.line(screen, CYAN, (cx_pre - 10, cy_pre), (cx_pre + 10, cy_pre), 1)
            pygame.draw.line(screen, CYAN, (cx_pre, cy_pre - 10), (cx_pre, cy_pre + 10), 1)
            if self.tiny:
                label = self.tiny.render("FÖRE skott", True, CYAN)
                screen.blit(label, (x_start, y_start))

        post_x = x_start + (scaled_w + gap if pre is not None else 0)
        scaled_post = pygame.transform.scale(post, (scaled_w, scaled_h))
        screen.blit(scaled_post, (post_x, y_start + label_h))
        pygame.draw.rect(screen, ORANGE, (post_x, y_start + label_h, scaled_w, scaled_h), 2)
        cx_post = post_x + scaled_w // 2
        cy_post = y_start + label_h + scaled_h // 2
        pygame.draw.line(screen, ORANGE, (cx_post - 10, cy_post), (cx_post + 10, cy_post), 1)
        pygame.draw.line(screen, ORANGE, (cx_post, cy_post - 10), (cx_post, cy_post + 10), 1)

        if self.tiny:
            label = self.tiny.render("EFTER skott", True, ORANGE)
            screen.blit(label, (post_x, y_start))
            hint = self.tiny.render("Klicka eller tryck för att fortsätta", True, WHITE)
            hint_rect = hint.get_rect(centerx=panel_rect.centerx, top=y_start + label_h + scaled_h + 6)
            screen.blit(hint, hint_rect)

    def _render_background(self, screen: pygame.Surface, mode: str, vp: pygame.Rect) -> None:
        if mode == "black":
            pygame.draw.rect(screen, BG_BLACK, vp)
        elif mode == "gray":
            pygame.draw.rect(screen, (128, 128, 128), vp)
        elif mode == "white_grid":
            pygame.draw.rect(screen, BG_WHITE, vp)
            for x in range(vp.left, vp.right, 48):
                pygame.draw.line(screen, GRID_LINE, (x, vp.top), (x, vp.bottom), 1)
            for y in range(vp.top, vp.bottom, 48):
                pygame.draw.line(screen, GRID_LINE, (vp.left, y), (vp.right, y), 1)
        elif mode == "checker":
            self._draw_checker_static(screen, vp)
        elif mode == "checker_anim":
            self._draw_checker_anim(screen, vp)
        elif mode == "bubbles":
            self._draw_bubbles(screen, vp)
        else:
            pygame.draw.rect(screen, BG_WHITE, vp)

        pygame.draw.rect(screen, (0, 180, 0), vp, 2)

    def _draw_checker_static(self, screen: pygame.Surface, vp: pygame.Rect) -> None:
        cell = 40
        colors = [(220, 220, 220), (60, 60, 60)]
        for row, y in enumerate(range(vp.top, vp.bottom, cell)):
            for col, x in enumerate(range(vp.left, vp.right, cell)):
                color = colors[(row + col) % 2]
                rect = pygame.Rect(x, y, min(cell, vp.right - x), min(cell, vp.bottom - y))
                pygame.draw.rect(screen, color, rect)

    def _draw_checker_anim(self, screen: pygame.Surface, vp: pygame.Rect) -> None:
        cell = 40
        colors = [(220, 220, 220), (60, 60, 60)]
        if not self._animation_frozen:
            offset = int(self.t * 60) % (cell * 2)
            self._checker_frozen_offset = offset
        else:
            offset = getattr(self, "_checker_frozen_offset", 0)

        for y in range(vp.top - cell, vp.bottom + cell, cell):
            for x in range(vp.left - cell, vp.right + cell, cell):
                ax = x + offset
                ay = y + offset
                row = (ay - vp.top) // cell
                col = (ax - vp.left) // cell
                color = colors[(row + col) % 2]
                rect = pygame.Rect(ax, ay, cell, cell).clip(vp)
                if rect.w > 0 and rect.h > 0:
                    pygame.draw.rect(screen, color, rect)

    def _draw_bubbles(self, screen: pygame.Surface, vp: pygame.Rect) -> None:
        pygame.draw.rect(screen, BG_WHITE, vp)

        if not self._bubbles:
            for _ in range(15):
                speed = random.uniform(0.06, 0.25)
                angle = random.uniform(0.0, 2.0 * math.pi)
                self._bubbles.append(
                    {
                        "x": random.uniform(0.1, 0.9),
                        "y": random.uniform(0.1, 0.9),
                        "r": random.uniform(0.03, 0.08),
                        "dx": math.cos(angle) * speed,
                        "dy": math.sin(angle) * speed,
                        "color": (
                            random.randint(40, 220),
                            random.randint(40, 220),
                            random.randint(40, 220),
                        ),
                        "shape": random.choice(["circle", "rect", "triangle"]),
                    }
                )

        if not self._animation_frozen:
            dt = 1.0 / 60.0
            for b in self._bubbles:
                b["x"] += b["dx"] * dt
                b["y"] += b["dy"] * dt

                if b["x"] < 0.05 or b["x"] > 0.95:
                    b["dx"] *= -1.0
                    b["dx"] += random.uniform(-0.02, 0.02)
                    b["dy"] += random.uniform(-0.01, 0.01)
                    b["x"] = max(0.05, min(0.95, b["x"]))

                if b["y"] < 0.05 or b["y"] > 0.95:
                    b["dy"] *= -1.0
                    b["dy"] += random.uniform(-0.02, 0.02)
                    b["dx"] += random.uniform(-0.01, 0.01)
                    b["y"] = max(0.05, min(0.95, b["y"]))

        for b in self._bubbles:
            cx = int(vp.left + b["x"] * vp.w)
            cy = int(vp.top + b["y"] * vp.h)
            r = int(b["r"] * min(vp.w, vp.h))
            color = b["color"]

            if b["shape"] == "circle":
                pygame.draw.circle(screen, color, (cx, cy), r)
                pygame.draw.circle(screen, (30, 30, 30), (cx, cy), r, 2)
            elif b["shape"] == "rect":
                rect = pygame.Rect(cx - r, cy - r, r * 2, r * 2)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (30, 30, 30), rect, 2)
            else:
                points = [(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
                pygame.draw.polygon(screen, color, points)
                pygame.draw.polygon(screen, (30, 30, 30), points, 2)

    def _render_candidates(self, screen: pygame.Surface, vp: pygame.Rect) -> None:
        if not self.awaiting_click or not self.ranked_candidates:
            return

        for cand in self.ranked_candidates:
            rank = int(cand.get("rank", 99))
            cam_x = cand.get("camera_x", 0.0)
            cam_y = cand.get("camera_y", 0.0)

            try:
                sx, sy = hit_input._canonical_camera_to_screen(cam_x, cam_y)
            except Exception:
                continue

            if not (math.isfinite(sx) and math.isfinite(sy)):
                continue

            ix, iy = int(round(sx)), int(round(sy))

            if rank == 1:
                color = ORANGE
                radius = 18
                width = 3
            elif rank <= 3:
                color = YELLOW
                radius = 14
                width = 2
            else:
                color = CYAN
                radius = 10
                width = 2

            pygame.draw.circle(screen, color, (ix, iy), radius, width)
            pygame.draw.line(screen, color, (ix - radius - 4, iy), (ix + radius + 4, iy), 1)
            pygame.draw.line(screen, color, (ix, iy - radius - 4), (ix, iy + radius + 4), 1)

            if self.small is not None:
                label = self.small.render(str(rank), True, color)
                screen.blit(label, (ix + radius + 4, iy - 10))

            if self.tiny is not None:
                ai_score = cand.get("ai_score", 0.0)
                combined = cand.get("combined_score", 0.0)
                score_text = f"ai:{ai_score:.2f} c:{combined:.2f}"
                score_surf = self.tiny.render(score_text, True, color)
                screen.blit(score_surf, (ix + radius + 4, iy + 6))

    def _render_click_feedback(self, screen: pygame.Surface) -> None:
        if self.clicked_camera_xy is None:
            return

        try:
            sx, sy = hit_input._canonical_camera_to_screen(
                self.clicked_camera_xy[0],
                self.clicked_camera_xy[1],
            )
        except Exception:
            return

        ix, iy = int(round(sx)), int(round(sy))
        pygame.draw.circle(screen, CLICK_COLOR, (ix, iy), 16, 3)
        pygame.draw.line(screen, CLICK_COLOR, (ix - 22, iy), (ix + 22, iy), 2)
        pygame.draw.line(screen, CLICK_COLOR, (ix, iy - 22), (ix, iy + 22), 2)

    def _render_hud(self, screen: pygame.Surface, mode: str, vp: pygame.Rect) -> None:
        if self.font is None or self.small is None or self.tiny is None:
            return

        sw, sh = screen.get_size()
        is_dark = mode == "black"

        top_bar = pygame.Surface((sw, 36), pygame.SRCALPHA)
        top_bar.fill(HUD_BG)
        screen.blit(top_bar, (0, 0))

        mode_label = {
            "white": "Vit",
            "white_grid": "Vit + rutnät",
            "gray": "Grå",
            "black": "Svart",
            "checker": "Rutmönster",
            "checker_anim": "Rutmönster (video)",
            "bubbles": "Bubblor",
        }.get(mode, mode)

        summary = self.runtime.memory.summary()
        header = (
            f"AI-träning • {mode_label} • "
            f"{summary['positive_count']} pos / {summary['negative_count']} neg"
        )
        screen.blit(self.small.render(header, True, WHITE), (12, 8))

        if self.awaiting_click:
            status_text = "KLICKA var du träffade"
            status_color = YELLOW
        elif self.auto_training_enabled:
            status_text = f"AUTO {self.auto_iteration}/{self.auto_target_iterations}"
            status_color = ORANGE
        else:
            status_text = "Skjut..."
            status_color = GREEN

        status_surf = self.small.render(status_text, True, status_color)
        screen.blit(status_surf, (sw - status_surf.get_width() - 16, 8))

        if self.status_message:
            bot_bar = pygame.Surface((sw, 28), pygame.SRCALPHA)
            bot_bar.fill(HUD_BG)
            screen.blit(bot_bar, (0, sh - 28))
            screen.blit(self.tiny.render(self.status_message, True, WHITE), (12, sh - 24))

        help_text = "TAB=bakgrund HÖGERKLICK=syntetisk runda SPACE=manuellt skott F1=autoträna R=nollställ ESC=tillbaka"
        help_surf = self.tiny.render(
            help_text,
            True,
            SOFT_WHITE if not is_dark else (140, 140, 140),
        )
        screen.blit(help_surf, (sw - help_surf.get_width() - 12, sh - 48))

        if self.auto_training_enabled:
            stats_text = (
                f"Hittat: {self.auto_stats['found']}  "
                f"Top-1: {self.auto_stats['top1']}  "
                f"Top-3: {self.auto_stats['top3']}  "
                f"Miss: {self.auto_stats['missed']}"
            )
            stats_surf = self.tiny.render(stats_text, True, ORANGE)
            screen.blit(stats_surf, (12, sh - 48))
