"""
AI Training Scene.

Flow:
1. Show a clean training surface (white/black/grid etc.)
2. Wait for a shot (audio peak triggers hit_scanner)
3. AI captures pre/post frames and ranks all candidates
4. User clicks roughly where the hit was
5. AI learns immediately from that click
6. Reset and repeat

The click is transformed to camera space. The AI compares it against
shown candidates and learns positive/negative examples.
"""
from __future__ import annotations

import math
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
from src.engine.camera.camera_manager import camera_manager
from src.engine.camera.hit_scanner import hit_scanner
from src.engine.input.hit_input import hit_input
from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import load_viewport_rect


BG_WHITE = (248, 248, 248)
BG_BLACK = (18, 18, 18)
TEXT_LIGHT = (245, 245, 245)
TEXT_DARK = (30, 30, 30)
CYAN = (80, 220, 255)
YELLOW = (255, 210, 70)
ORANGE = (255, 150, 80)
GREEN = (120, 255, 120)
CLICK_COLOR = (255, 110, 110)
HUD_BG = (0, 0, 0, 120)
GRID_LINE = (222, 222, 222)
WHITE = (240, 240, 240)
SOFT_WHITE = (210, 210, 210)


class AITrainingScene(Scene):
    """AI training scene following the standard Scene protocol."""

    wants_hit_scanning = True
    wants_camera_preview = False

    MODE_NAMES = ["white", "black", "grid"]

    def __init__(self, bg_color=None, **kwargs) -> None:
        super().__init__()
        self.runtime = get_ai_runtime()

        # Determine initial background mode from bg_color hint
        if isinstance(bg_color, (tuple, list)) and len(bg_color) >= 3:
            brightness = sum(int(c) for c in bg_color[:3]) / 3.0
            self.bg_mode_index = 1 if brightness < 96 else 0
        else:
            self.bg_mode_index = 0

        self.font: pygame.font.Font | None = None
        self.small: pygame.font.Font | None = None
        self.tiny: pygame.font.Font | None = None

        self.viewport: pygame.Rect | None = None

        # Shot state
        self.awaiting_click = False
        self.ranked_candidates: list[dict[str, Any]] = []
        self.clicked_camera_xy: tuple[float, float] | None = None
        self.last_learning_result: dict[str, Any] | None = None
        self.click_flash_timer = 0.0
        self.status_message = ""

        # Animation
        self.t = 0.0

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 34)
        self.small = pygame.font.Font(None, 24)
        self.tiny = pygame.font.Font(None, 18)
        self.viewport = load_viewport_rect()
        self.runtime = get_ai_runtime()
        self._reset_shot_state()

    def on_exit(self) -> None:
        pass

    def _reset_shot_state(self) -> None:
        self.awaiting_click = False
        self.ranked_candidates = []
        self.clicked_camera_xy = None
        self.last_learning_result = None
        self.click_flash_timer = 0.0

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                from src.engine.scenes.menu import MenuScene
                return SceneSwitch(MenuScene())

            if event.key == pygame.K_TAB:
                self.bg_mode_index = (self.bg_mode_index + 1) % len(self.MODE_NAMES)
                return None

            if event.key == pygame.K_r:
                self.runtime.memory.reset()
                self._reset_shot_state()
                self.status_message = "AI nollställd."
                return None

            if event.key == pygame.K_SPACE and not self.awaiting_click:
                # Manual trigger — capture current state as a shot
                self._on_shot_detected()
                return None

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.awaiting_click:
                self._on_training_click(event.pos)
            return None

        return None

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, dt: float):
        self.t += dt

        # Check if AI runtime detected a new shot
        if not self.awaiting_click and self.runtime.has_new_shot:
            self._on_shot_detected()

        # Clear click flash after a short time
        if self.clicked_camera_xy is not None and not self.awaiting_click:
            self.click_flash_timer += dt
            if self.click_flash_timer > 0.4:
                self.clicked_camera_xy = None
                self.click_flash_timer = 0.0

        return None

    def _on_shot_detected(self) -> None:
        """A shot was detected — rank candidates and wait for click."""
        # Get all candidates from hit_scanner (not just top 10)
        all_candidates = list(hit_scanner.last_candidates)
        if not all_candidates:
            self.status_message = "Skott detekterat men inga kandidater."
            return

        self.ranked_candidates = self.runtime.rank_candidates(all_candidates, limit=50)
        self.awaiting_click = True
        self.clicked_camera_xy = None
        self.status_message = f"Skott! {len(self.ranked_candidates)} kandidater. Klicka var du träffade."

    def _on_training_click(self, screen_pos: tuple[int, int]) -> None:
        """User clicked to indicate where the hit was."""
        # Transform screen click to camera coordinates
        projected = project_screen_point(float(screen_pos[0]), float(screen_pos[1]))
        click_camera = (projected.camera_x, projected.camera_y)
        self.clicked_camera_xy = click_camera

        # Train the AI
        result = self.runtime.learn_from_click(
            click_camera_xy=click_camera,
            shown_candidates=self.ranked_candidates,
            gray_pre=self.runtime.pre_shot_gray,
            gray_post=self.runtime.post_shot_gray,
        )
        self.last_learning_result = result

        if result.get("positive_added"):
            dist = result.get("nearest_distance", 999)
            if dist <= float(self.runtime.settings.get("click_match_radius_px", 42)):
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

        # Reset for next shot
        self.awaiting_click = False
        self.click_flash_timer = 0.0

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, screen: pygame.Surface) -> None:
        mode = self.MODE_NAMES[self.bg_mode_index]
        vp = self.viewport or pygame.Rect(0, 0, screen.get_width(), screen.get_height())

        # Fill area outside viewport with dark border
        screen.fill((30, 30, 30))

        self._render_background(screen, mode, vp)
        self._render_candidates(screen, vp)
        self._render_click_feedback(screen)
        self._render_hud(screen, mode, vp)

    def _render_background(self, screen: pygame.Surface, mode: str, vp: pygame.Rect) -> None:
        if mode == "black":
            pygame.draw.rect(screen, BG_BLACK, vp)
        elif mode == "grid":
            pygame.draw.rect(screen, BG_WHITE, vp)
            for x in range(vp.left, vp.right, 48):
                pygame.draw.line(screen, GRID_LINE, (x, vp.top), (x, vp.bottom), 1)
            for y in range(vp.top, vp.bottom, 48):
                pygame.draw.line(screen, GRID_LINE, (vp.left, y), (vp.right, y), 1)
        else:
            pygame.draw.rect(screen, BG_WHITE, vp)

        # Viewport border
        pygame.draw.rect(screen, (0, 180, 0), vp, 2)

    def _render_candidates(self, screen: pygame.Surface, vp: pygame.Rect) -> None:
        if not self.awaiting_click or not self.ranked_candidates:
            return

        for cand in self.ranked_candidates:
            rank = int(cand.get("rank", 99))
            cam_x = cand.get("camera_x", 0.0)
            cam_y = cand.get("camera_y", 0.0)

            # Transform to screen
            try:
                from src.engine.input.hit_input import hit_input
                sx, sy = hit_input._canonical_camera_to_screen(cam_x, cam_y)
            except Exception:
                continue

            if not (math.isfinite(sx) and math.isfinite(sy)):
                continue

            ix, iy = int(round(sx)), int(round(sy))

            # Color by rank
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
            # Crosshair
            pygame.draw.line(screen, color, (ix - radius - 4, iy), (ix + radius + 4, iy), 1)
            pygame.draw.line(screen, color, (ix, iy - radius - 4), (ix, iy + radius + 4), 1)

            # Rank number
            if self.small is not None:
                label = self.small.render(str(rank), True, color)
                screen.blit(label, (ix + radius + 4, iy - 10))

            # Score
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
            from src.engine.input.hit_input import hit_input
            sx, sy = hit_input._canonical_camera_to_screen(
                self.clicked_camera_xy[0], self.clicked_camera_xy[1]
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
        ink = TEXT_LIGHT if is_dark else TEXT_DARK

        # Top bar
        top_bar = pygame.Surface((sw, 36), pygame.SRCALPHA)
        top_bar.fill(HUD_BG)
        screen.blit(top_bar, (0, 0))

        mode_label = {"white": "Vit", "black": "Svart", "grid": "Rutnät"}.get(mode, mode)
        summary = self.runtime.memory.summary()
        header = (
            f"AI-träning • {mode_label} • "
            f"{summary['positive_count']} pos / {summary['negative_count']} neg"
        )
        screen.blit(self.small.render(header, True, WHITE), (12, 8))

        # Status indicator
        if self.awaiting_click:
            status_text = "KLICKA var du träffade"
            status_color = YELLOW
        else:
            status_text = "Skjut..."
            status_color = GREEN

        status_surf = self.small.render(status_text, True, status_color)
        screen.blit(status_surf, (sw - status_surf.get_width() - 16, 8))

        # Bottom status message
        if self.status_message:
            bot_bar = pygame.Surface((sw, 28), pygame.SRCALPHA)
            bot_bar.fill(HUD_BG)
            screen.blit(bot_bar, (0, sh - 28))
            screen.blit(self.tiny.render(self.status_message, True, WHITE), (12, sh - 24))

        # Help (bottom right)
        help_text = "TAB=bakgrund  SPACE=manuellt skott  R=nollställ  ESC=tillbaka"
        help_surf = self.tiny.render(help_text, True, SOFT_WHITE if not is_dark else (140, 140, 140))
        screen.blit(help_surf, (sw - help_surf.get_width() - 12, sh - 48))
