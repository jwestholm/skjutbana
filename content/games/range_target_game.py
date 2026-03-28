from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygame

from src.engine.range_projection import (
    RangeProjectionGeometry,
    clamp_distance_m,
    projected_ground_anchor_y_px,
    projected_lateral_offset_px,
    projected_target_height_px,
)
from src.engine.settings import (
    load_range_projection_settings,
)


WHITE = (245, 245, 245)
BLACK = (0, 0, 0)
HUD_BG = (0, 0, 0, 150)
ACCENT = (255, 220, 90)
SOFT = (210, 210, 210)


@dataclass
class RangeTargetState:
    distance_m: float = 100.0
    lateral_offset_m: float = 0.0
    hud_enabled: bool = True


class RangeTargetGame:
    """
    Enkel första version av 'helfigur på avstånd'.

    Förväntade filer i game_root:
    - background.png
    - target.png

    Kontroller:
    - PLUS / KP_PLUS / =  -> öka avstånd +5 m
    - MINUS / KP_MINUS    -> minska avstånd -5 m
    - VÄNSTER / HÖGER     -> flytta target i sidled
    - SPACE               -> visa/dölj HUD
    - R                   -> återställ
    - ESC                 -> hanteras av GameScene som vanligt
    """

    def __init__(self, game_root: str, viewport: pygame.Rect) -> None:
        self.game_root = Path(game_root)
        self.viewport = viewport.copy()

        self.state = RangeTargetState()

        self.min_distance_m = 5.0
        self.max_distance_m = 600.0
        self.distance_step_m = 5.0
        self.lateral_step_m = 0.5

        self.target_real_height_cm = 180.0

        self.background_surface: pygame.Surface | None = None
        self.background_scaled: pygame.Surface | None = None
        self.target_surface: pygame.Surface | None = None

        self.font: pygame.font.Font | None = None
        self.small_font: pygame.font.Font | None = None

        self._load_assets()

    def _load_assets(self) -> None:
        bg_path = self.game_root / "background.png"
        target_path = self.game_root / "target.png"

        if not bg_path.exists():
            raise FileNotFoundError(f"Missing background image: {bg_path}")

        if not target_path.exists():
            raise FileNotFoundError(f"Missing target image: {target_path}")

        self.background_surface = pygame.image.load(str(bg_path)).convert()
        self.target_surface = pygame.image.load(str(target_path)).convert_alpha()

        self._rebuild_background_cache()

    def _rebuild_background_cache(self) -> None:
        if self.background_surface is None:
            self.background_scaled = None
            return

        self.background_scaled = pygame.transform.smoothscale(
            self.background_surface,
            (self.viewport.w, self.viewport.h),
        )

    def on_enter(self) -> None:
        if self.font is None:
            self.font = pygame.font.Font(None, 42)
        if self.small_font is None:
            self.small_font = pygame.font.Font(None, 26)

    def on_exit(self) -> None:
        pass

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
            self.state.distance_m = clamp_distance_m(
                self.state.distance_m + self.distance_step_m,
                min_distance_m=self.min_distance_m,
                max_distance_m=self.max_distance_m,
            )
            return None

        if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.state.distance_m = clamp_distance_m(
                self.state.distance_m - self.distance_step_m,
                min_distance_m=self.min_distance_m,
                max_distance_m=self.max_distance_m,
            )
            return None

        if event.key == pygame.K_LEFT:
            self.state.lateral_offset_m -= self.lateral_step_m
            return None

        if event.key == pygame.K_RIGHT:
            self.state.lateral_offset_m += self.lateral_step_m
            return None

        if event.key == pygame.K_SPACE:
            self.state.hud_enabled = not self.state.hud_enabled
            return None

        if event.key == pygame.K_r:
            self.state = RangeTargetState()
            return None

        return None

    def update(self, dt: float):
        del dt
        return None

    def _geometry(self) -> RangeProjectionGeometry:
        settings = load_range_projection_settings()
        return RangeProjectionGeometry(
            wall_distance_m=float(settings.get("wall_distance_m", 6.0)),
            viewport_physical_width_cm=float(settings.get("viewport_physical_width_cm", 100.0)),
            viewport_physical_height_cm=float(settings.get("viewport_physical_height_cm", 50.0)),
        )

    def _render_background(self, screen: pygame.Surface) -> None:
        if self.background_scaled is None:
            return

        screen.blit(self.background_scaled, (self.viewport.x, self.viewport.y))

    def _render_target(self, screen: pygame.Surface) -> None:
        if self.target_surface is None:
            return

        geometry = self._geometry()

        target_h_px = projected_target_height_px(
            real_height_cm=self.target_real_height_cm,
            virtual_distance_m=self.state.distance_m,
            viewport=self.viewport,
            geometry=geometry,
        )

        target_h_px = max(2, int(round(target_h_px)))

        src_w, src_h = self.target_surface.get_size()
        if src_h <= 0:
            return

        aspect = src_w / float(src_h)
        target_w_px = max(1, int(round(target_h_px * aspect)))

        scaled_target = pygame.transform.smoothscale(
            self.target_surface,
            (target_w_px, target_h_px),
        )

        offset_x_px = projected_lateral_offset_px(
            world_lateral_offset_m=self.state.lateral_offset_m,
            virtual_distance_m=self.state.distance_m,
            viewport=self.viewport,
            geometry=geometry,
        )

        foot_x = self.viewport.centerx + offset_x_px
        foot_y = projected_ground_anchor_y_px(
            distance_m=self.state.distance_m,
            viewport=self.viewport,
            min_distance_m=self.min_distance_m,
            max_distance_m=self.max_distance_m,
            near_ground_y_norm=0.92,
            far_ground_y_norm=0.58,
        )

        draw_x = int(round(foot_x - (target_w_px / 2.0)))
        draw_y = int(round(foot_y - target_h_px))

        screen.blit(scaled_target, (draw_x, draw_y))

    def _render_hud(self, screen: pygame.Surface) -> None:
        if not self.state.hud_enabled:
            return

        if self.font is None or self.small_font is None:
            return

        panel = pygame.Surface((210, 74), pygame.SRCALPHA)
        panel.fill(HUD_BG)

        distance_text = self.font.render(f"{int(round(self.state.distance_m))} m", True, ACCENT)
        panel.blit(distance_text, (16, 10))

        hint_text = self.small_font.render("SPACE: HUD  +/-: avstånd", True, SOFT)
        panel.blit(hint_text, (16, 46))

        screen.blit(panel, (self.viewport.x + 16, self.viewport.y + 16))

    def render(self, screen: pygame.Surface) -> None:
        self._render_background(screen)
        self._render_target(screen)
        self._render_hud(screen)


def create_game(game_root: str, viewport: pygame.Rect):
    return RangeTargetGame(game_root=game_root, viewport=viewport)