"""
AI Training Scene.

Flow:
1. Show a clean training surface (white/black/grid etc.)
2. Wait for a shot (audio peak triggers hit_scanner)
3. AI captures pre/post frames and ranks all candidates
4. User clicks roughly where the hit was
5. AI learns immediately from that click
6. Reset and repeat

Auto mode:
- F1 toggles automatic training on/off
- The scene places a synthetic hole on the projected playfield
- It triggers a fake audio peak directly into the scanner
- The normal chain runs: shot -> candidates -> click -> learning
- Auto mode keeps the normal visual flow, but auto-clicks after a short delay

Right-click mode:
- Right click places one synthetic hole at the clicked point
- It triggers one fake shot and then behaves like a normal manual round
- The user still clicks the shown hit manually after the AI marks candidates
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
TEXT_LIGHT = (245, 245, 245)
TEXT_DARK = (30, 30, 30)
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
    """AI training scene following the standard Scene protocol."""

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
        self.click_flash_timer = 0.0
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

        # Auto-training state
        self.auto_training_enabled = False
        self.auto_target_iterations = 1000
        self.auto_iteration = 0
        self.auto_target_screen_xy: tuple[int, int] | None = None
        self.auto_active_hole_id: str | None = None
        self.auto_waiting_for_shot = False
        self.auto_click_pending = False
        self.auto_last_trigger_ts = 0.0
        self.auto_min_trigger_gap_s = 0.08
        self.auto_status_detail = ""
        self.auto_click_delay_s = 1.0
        self.auto_review_delay_s = 1.0
        self.auto_next_iteration_delay_s = 1.0
        self.auto_click_ready_ts = 0.0
        self.auto_review_ready_ts = 0.0
        self.auto_next_iteration_ts = 0.0

        # Synthetic shot scheduling / settle timing.
        # We wait a short time after drawing the hole or changing background
        # before injecting the fake audio event, so the camera has time to
        # observe the correct "before/after" world state.
        self.synthetic_trigger_delay_s = 0.20
        self.background_settle_delay_s = 0.25
        self.background_settle_until_ts = 0.0
        self.synthetic_trigger_pending = False
        self.synthetic_trigger_batch_mode = False
        self.synthetic_trigger_screen_xy: tuple[int, int] | None = None
        self.synthetic_trigger_ready_ts = 0.0

        # One synthetic manual round triggered by right click.
        # This behaves exactly like a normal shot flow, except the program
        # injects the sound trigger and places the hole for you.
        self.single_synth_round_active = False
        self.single_target_screen_xy: tuple[int, int] | None = None

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 34)
        self.small = pygame.font.Font(None, 24)
        self.tiny = pygame.font.Font(None, 18)
        self.viewport = load_viewport_rect()
        self.runtime = get_ai_runtime()
        self._reset_shot_state()
        self._last_peak_ts = audio_peak_detector.last_peak_ts

    def on_exit(self) -> None:
        pass

    def _reset_shot_state(self) -> None:
        self.awaiting_click = False
        self.ranked_candidates = []
        self.clicked_camera_xy = None
        self.last_learning_result = None
        self.click_flash_timer = 0.0
        self._animation_frozen = False
        self._reviewing = False
        self._review_pre_surface = None
        self._review_post_surface = None
        self._pending_click_camera = None
        self._pending_click_phase = None
        self._pending_wait_frames = 0
        self._last_peak_ts = audio_peak_detector.last_peak_ts
        self.auto_click_pending = False
        self.auto_waiting_for_shot = False
        self.auto_click_ready_ts = 0.0
        self.auto_review_ready_ts = 0.0
        self.single_synth_round_active = False
        self.single_target_screen_xy = None
        self.synthetic_trigger_pending = False
        self.synthetic_trigger_batch_mode = False
        self.synthetic_trigger_screen_xy = None
        self.synthetic_trigger_ready_ts = 0.0
        if self.synthetic_overlay is not None:
            self.synthetic_overlay.clear()
        pygame.mouse.set_visible(True)

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

    # ------------------------------------------------------------------
    # Auto-training
    # ------------------------------------------------------------------
    def _toggle_auto_training(self) -> None:
        self.auto_training_enabled = not self.auto_training_enabled
        if self.auto_training_enabled:
            self.auto_iteration = 0
            self.auto_waiting_for_shot = False
            self.auto_click_pending = False
            self.auto_click_ready_ts = 0.0
            self.auto_review_ready_ts = 0.0
            self.auto_next_iteration_ts = time.time() + 0.2
            self.synthetic_trigger_pending = False
            self.synthetic_trigger_screen_xy = None
            self.synthetic_trigger_ready_ts = 0.0
            self.single_synth_round_active = False
            self.auto_status_detail = "Autoträning startad."
            self.status_message = "Autoträning startad (F1 stoppar)."
            self._reviewing = False
            self._clear_synthetic_holes()
            pygame.mouse.set_visible(False)
        else:
            self.auto_waiting_for_shot = False
            self.auto_click_pending = False
            self.auto_click_ready_ts = 0.0
            self.auto_review_ready_ts = 0.0
            self.synthetic_trigger_pending = False
            self.synthetic_trigger_screen_xy = None
            self.synthetic_trigger_ready_ts = 0.0
            self.auto_status_detail = "Autoträning stoppad."
            self.status_message = "Autoträning stoppad."
            self.single_synth_round_active = False
            self._clear_synthetic_holes()
            pygame.mouse.set_visible(True)

    def _choose_auto_screen_point(self, vp: pygame.Rect) -> tuple[int, int]:
        margin = 48
        x = random.randint(vp.left + margin, max(vp.left + margin, vp.right - margin))
        y = random.randint(vp.top + margin, max(vp.top + margin, vp.bottom - margin))
        return x, y

    def _current_auto_click_target(self) -> tuple[int, int] | None:
        return self.auto_target_screen_xy

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
            weights=[34, 18, 14, 18, 16],
            k=1,
        )[0]

        radius_px = random.uniform(1.9, 3.3)

        self.auto_active_hole_id = overlay.add_hole(
            sx,
            sy,
            kind=hole_kind,
            radius_px=radius_px,
            strength=random.uniform(0.85, 1.20),
            opacity=random.uniform(0.90, 1.0),
        )

        # Hide cursor before the camera sees the synthetic shot setup.
        pygame.mouse.set_visible(False)

        # Give the projector/camera chain a brief moment to catch up so that
        # pre/post frames use the intended background and hole state.
        self.synthetic_trigger_pending = True
        self.synthetic_trigger_batch_mode = batch_mode
        self.synthetic_trigger_screen_xy = (sx, sy)
        self.synthetic_trigger_ready_ts = max(
            now + self.synthetic_trigger_delay_s,
            self.background_settle_until_ts,
        )
        self.status_message = (
            f"Autoträning {self.auto_iteration + 1}/{self.auto_target_iterations}: väntar på kamerasettling..."
            if batch_mode
            else "Syntetisk runda förbereds..."
        )
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
            self.single_synth_round_active = False
            self.synthetic_trigger_pending = False
            self._clear_synthetic_holes()
            pygame.mouse.set_visible(True)
            return False

        self.synthetic_trigger_pending = False
        self._animation_frozen = True
        self._last_peak_ts = event_ts
        self.auto_last_trigger_ts = event_ts
        self.auto_waiting_for_shot = batch_mode
        self.auto_click_pending = False

        if batch_mode:
            self.status_message = (
                f"Autoträning {self.auto_iteration + 1}/{self.auto_target_iterations}: "
                f"syntetiskt skott triggat."
            )
        else:
            self.status_message = "Syntetiskt skott triggat på vald punkt."

        return True

    def _start_auto_iteration(self, screen: pygame.Surface) -> None:
        if self.auto_iteration >= self.auto_target_iterations:
            self.auto_training_enabled = False
            self.auto_status_detail = "Målnivå nådd."
            self.status_message = f"Autoträning klar: {self.auto_iteration} iterationer."
            self._clear_synthetic_holes()
            pygame.mouse.set_visible(True)
            return

        if time.time() < self.auto_next_iteration_ts:
            return

        vp = self.viewport or pygame.Rect(0, 0, screen.get_width(), screen.get_height())
        sx, sy = self._choose_auto_screen_point(vp)
        self._trigger_synthetic_shot_at(screen, (sx, sy), batch_mode=True)

    def _finish_auto_iteration(self) -> None:
        self.auto_iteration += 1
        self.auto_waiting_for_shot = False
        self.auto_click_pending = False
        self.auto_click_ready_ts = 0.0
        self.auto_review_ready_ts = 0.0
        self._clear_synthetic_holes()
        if self.auto_iteration >= self.auto_target_iterations:
            self.auto_training_enabled = False
            self.status_message = f"Autoträning klar: {self.auto_iteration} iterationer."
            pygame.mouse.set_visible(True)
        else:
            self.auto_next_iteration_ts = time.time() + self.auto_next_iteration_delay_s
            self.status_message = (
                f"Autoträning: iteration {self.auto_iteration}/{self.auto_target_iterations} klar."
            )

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event):
        if self._reviewing:
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN):
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    from src.engine.scenes.menu import MenuScene
                    return SceneSwitch(MenuScene())
                self._reset_shot_state()
                return None
            return None

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                from src.engine.scenes.menu import MenuScene
                return SceneSwitch(MenuScene())

            if event.key == pygame.K_F1:
                self._toggle_auto_training()
                return None

            if event.key == pygame.K_TAB:
                self.bg_mode_index = (self.bg_mode_index + 1) % len(self.MODE_NAMES)
                self.background_settle_until_ts = time.time() + self.background_settle_delay_s
                self.status_message = "Bakgrund bytt – vänta ett ögonblick innan skott."
                return None

            if event.key == pygame.K_r:
                self.runtime.memory.reset()
                self._reset_shot_state()
                self._clear_synthetic_holes()
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

            if (
                event.button == 3
                and not self.awaiting_click
                and not self._reviewing
                and self._pending_click_phase is None
                and not self.auto_training_enabled
            ):
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

        if self.synthetic_trigger_pending and time.time() >= self.synthetic_trigger_ready_ts:
            self._fire_pending_synthetic_shot()

        if not self.awaiting_click and not self._reviewing and self.runtime.has_new_shot:
            self._on_shot_detected()

        if self.awaiting_click and self.single_synth_round_active and not self.auto_training_enabled:
            pygame.mouse.set_visible(True)

        if self.awaiting_click and self.auto_training_enabled and self.auto_click_pending:
            if time.time() >= self.auto_click_ready_ts:
                self.auto_click_pending = False
                auto_target = self._current_auto_click_target()
                if auto_target is not None:
                    self._on_training_click(auto_target)

        if self.auto_training_enabled and self._reviewing and self.auto_review_ready_ts > 0.0:
            if time.time() >= self.auto_review_ready_ts:
                self._finish_auto_iteration()
                self._reset_shot_state()
                pygame.mouse.set_visible(False if self.auto_training_enabled else True)
                return

        if self._pending_click_phase == "wait_frame":
            self._pending_wait_frames += 1
            if self._pending_wait_frames >= 5:
                self._pending_click_phase = "capture"
        elif self._pending_click_phase == "capture":
            self._do_clean_capture()

    def _do_clean_capture(self) -> None:
        click_camera = self._pending_click_camera
        self._pending_click_phase = None
        self._pending_click_camera = None
        self._pending_wait_frames = 0

        if click_camera is None:
            pygame.mouse.set_visible(True)
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

        if result.get("positive_added"):
            dist = result.get("nearest_distance", 999.0)
            if dist <= float(self.runtime.settings.get("click_match_radius_px", 42.0)):
                idx = result.get("nearest_index", 0)
                self.status_message = (
                    f"Tränade: kandidat #{idx + 1} (avstånd {dist:.0f}px). "
                    f"+{result.get('negatives_added', 0)} neg. "
                    f"Totalt: {result['total_positives']} pos / {result['total_negatives']} neg"
                )
            else:
                self.status_message = (
                    f"Syntetiskt positivt (ingen kandidat nära). "
                    f"Totalt: {result['total_positives']} pos / {result['total_negatives']} neg"
                )

        self._enter_review(click_camera, post_gray)

        if self.auto_training_enabled:
            self.auto_review_ready_ts = time.time() + self.auto_review_delay_s
            return

        pygame.mouse.set_visible(True)

    def _on_shot_detected(self) -> None:
        all_candidates = list(hit_scanner.last_candidates)
        if not all_candidates:
            self.status_message = "Skott detekterat men inga kandidater."
            if self.auto_training_enabled:
                self._finish_auto_iteration()
                self._reset_shot_state()
                pygame.mouse.set_visible(False if self.auto_training_enabled else True)
            elif self.single_synth_round_active:
                self.single_synth_round_active = False
                self._clear_synthetic_holes()
                pygame.mouse.set_visible(True)
            return

        self.ranked_candidates = self.runtime.rank_candidates(all_candidates, limit=50)
        self.awaiting_click = True
        self.clicked_camera_xy = None

        if self.auto_training_enabled:
            self.auto_waiting_for_shot = False
            self.auto_click_pending = True
            self.auto_click_ready_ts = time.time() + self.auto_click_delay_s
            pygame.mouse.set_visible(False)
            self.status_message = (
                f"Autoträning {self.auto_iteration + 1}/{self.auto_target_iterations}: "
                f"{len(self.ranked_candidates)} kandidater hittade, auto-klick om {self.auto_click_delay_s:.1f} s."
            )
        elif self.single_synth_round_active:
            pygame.mouse.set_visible(True)
            self.status_message = (
                f"Syntetisk runda: {len(self.ranked_candidates)} kandidater hittade. Klicka var du träffade."
            )
        else:
            pygame.mouse.set_visible(True)
            self.status_message = f"Skott! {len(self.ranked_candidates)} kandidater. Klicka var du träffade."

    def _on_training_click(self, screen_pos: tuple[int, int]) -> None:
        projected = project_screen_point(float(screen_pos[0]), float(screen_pos[1]))
        click_camera = (projected.camera_x, projected.camera_y)
        self.clicked_camera_xy = click_camera
        self._pending_click_camera = click_camera
        self._pending_click_phase = "clean_render"
        self.awaiting_click = False
        pygame.mouse.set_visible(False)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render(self, screen: pygame.Surface) -> None:
        overlay = self._ensure_overlay(screen)
        mode = self.MODE_NAMES[self.bg_mode_index]
        vp = self.viewport or pygame.Rect(0, 0, screen.get_width(), screen.get_height())

        screen.fill((30, 30, 30))
        self._render_background(screen, mode, vp)

        # Synthetic holes are part of the projected world and must remain visible
        # during clean capture. Candidate overlays / cursor feedback are hidden later.
        if overlay is not None:
            composed = overlay.composite_on(pygame.surfarray.array3d(screen).swapaxes(0, 1))
            composed = composed.swapaxes(0, 1)
            pygame.surfarray.blit_array(screen, composed)

        if self.auto_training_enabled and not self.awaiting_click and not self._reviewing and self._pending_click_phase is None:
            self._start_auto_iteration(screen)

        if self._pending_click_phase in ("clean_render", "wait_frame", "capture"):
            if self._pending_click_phase == "clean_render":
                self._pending_click_phase = "wait_frame"
                self._pending_wait_frames = 0
                pygame.mouse.set_visible(False)
            return

        self._render_candidates(screen, vp)
        self._render_click_feedback(screen)

        if self._reviewing:
            self._render_review(screen, vp)

        self._render_hud(screen, mode, vp)

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
        elif self.single_synth_round_active:
            status_text = "SYNTH 1x"
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

        help_text = "TAB=bakgrund SPACE=manuellt skott Högerklick=syntetiskt skott F1=autoträna R=nollställ ESC=tillbaka"
        help_surf = self.tiny.render(
            help_text,
            True,
            SOFT_WHITE if not is_dark else (140, 140, 140),
        )
        screen.blit(help_surf, (sw - help_surf.get_width() - 12, sh - 48))
