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
from src.engine.settings import load_range_projection_settings

WHITE = (245, 245, 245)
BLACK = (0, 0, 0)
HUD_BG = (0, 0, 0, 150)
ACCENT = (255, 220, 90)
SOFT = (210, 210, 210)
WARN = (255, 140, 140)


@dataclass
class RangeTargetState:
    distance_m: float = 100.0
    lateral_offset_m: float = 0.0
    hud_enabled: bool = True


class RangeTargetGame:
    """
    Enkel version av "helfigur på avstånd".

    Kontroller:
    - LEFT / RIGHT -> flytta figur i sidled
    - UP           -> närmare
    - DOWN         -> längre bort
    - SPACE        -> visa/dölj HUD
    - R            -> återställ
    - ESC          -> hanteras av GameScene
    """

    def __init__(self, game_root: str, viewport: pygame.Rect) -> None:
        self.game_root = Path(game_root)
        self.viewport = viewport.copy()
        self.state = RangeTargetState()

        self.max_distance_m = 600.0
        self.distance_step_m = 1.0
        self.distance_step_large_m = 5.0
        self.lateral_step_m = 0.25
        self.lateral_step_large_m = 1.0

        # Antagen verklig höjd på figuren
        self.target_real_height_cm = 180.0

        # Horisontell ankarpunkt i target-bilden.
        # 0.5 = mitt i bilden. Justera vid behov om PNG:n inte är perfekt centrerad.
        self.target_anchor_x_norm = 0.5

        self.background_surface: pygame.Surface | None = None
        self.background_scaled: pygame.Surface | None = None
        self.target_surface: pygame.Surface | None = None

        self.font: pygame.font.Font | None = None
        self.small_font: pygame.font.Font | None = None

        self._load_assets()

    # ------------------------------------------------------------
    # Assets
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Scene hooks
    # ------------------------------------------------------------

    def on_enter(self) -> None:
        if self.font is None:
            self.font = pygame.font.Font(None, 40)
        if self.small_font is None:
            self.small_font = pygame.font.Font(None, 24)

        # Se till att startavståndet aldrig ligger innanför väggplanet.
        geometry = self._geometry()
        self.state.distance_m = max(self.state.distance_m, geometry.wall_distance_m)

    def on_exit(self) -> None:
        pass

    # ------------------------------------------------------------
    # Projection / geometry
    # ------------------------------------------------------------

    def _geometry(self) -> RangeProjectionGeometry:
        settings = load_range_projection_settings()
        return RangeProjectionGeometry(
            wall_distance_m=float(settings.get("wall_distance_m", 6.0)),
            viewport_physical_width_cm=float(
                settings.get("viewport_physical_width_cm", 100.0)
            ),
            viewport_physical_height_cm=float(
                settings.get("viewport_physical_height_cm", 50.0)
            ),
        )

    def _min_distance_m(self) -> float:
        # Närmast = står vid väggen/projektionsplanet.
        geometry = self._geometry()
        return max(0.1, float(geometry.wall_distance_m))

    def _target_fits_height_at_distance(self, distance_m: float) -> bool:
        geometry = self._geometry()
        projected_h_cm = self.target_real_height_cm * (
            geometry.wall_distance_m / max(0.01, distance_m)
        )
        return projected_h_cm <= geometry.viewport_physical_height_cm

    # ------------------------------------------------------------
    # Input
    # ------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        mods = pygame.key.get_mods()
        large_step = bool(mods & pygame.KMOD_SHIFT)

        distance_step = self.distance_step_large_m if large_step else self.distance_step_m
        lateral_step = self.lateral_step_large_m if large_step else self.lateral_step_m

        if event.key == pygame.K_UP:
            self.state.distance_m = clamp_distance_m(
                self.state.distance_m - distance_step,
                min_distance_m=self._min_distance_m(),
                max_distance_m=self.max_distance_m,
            )
            return None

        if event.key == pygame.K_DOWN:
            self.state.distance_m = clamp_distance_m(
                self.state.distance_m + distance_step,
                min_distance_m=self._min_distance_m(),
                max_distance_m=self.max_distance_m,
            )
            return None

        if event.key == pygame.K_LEFT:
            self.state.lateral_offset_m -= lateral_step
            return None

        if event.key == pygame.K_RIGHT:
            self.state.lateral_offset_m += lateral_step
            return None

        if event.key == pygame.K_SPACE:
            self.state.hud_enabled = not self.state.hud_enabled
            return None

        if event.key == pygame.K_r:
            self.state = RangeTargetState()
            self.state.distance_m = max(self.state.distance_m, self._min_distance_m())
            return None

        return None

    # ------------------------------------------------------------
    # Update
    # ------------------------------------------------------------

    def update(self, dt: float):
        del dt
        return None

    # ------------------------------------------------------------
    # Render
    # ------------------------------------------------------------

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

        # Världsankare: figurens fötter står på markplanet.
        foot_x = self.viewport.centerx + offset_x_px
        foot_y = projected_ground_anchor_y_px(
            distance_m=self.state.distance_m,
            viewport=self.viewport,
            min_distance_m=self._min_distance_m(),
            max_distance_m=self.max_distance_m,
            near_ground_y_norm=0.92,
            far_ground_y_norm=0.58,
        )

        # Rita från definierad fotankare i bilden, inte bara bildens mitt.
        anchor_x_px = target_w_px * self.target_anchor_x_norm

        draw_x = int(round(foot_x - anchor_x_px))
        draw_y = int(round(foot_y - target_h_px))

        screen.blit(scaled_target, (draw_x, draw_y))

    def _render_hud(self, screen: pygame.Surface) -> None:
        if not self.state.hud_enabled:
            return
        if self.font is None or self.small_font is None:
            return

        geometry = self._geometry()
        fits = self._target_fits_height_at_distance(self.state.distance_m)

        panel = pygame.Surface((370, 128), pygame.SRCALPHA)
        panel.fill(HUD_BG)

        distance_text = self.font.render(
            f"{self.state.distance_m:.1f} m",
            True,
            ACCENT,
        )
        panel.blit(distance_text, (16, 8))

        lateral_text = self.small_font.render(
            f"Sidled: {self.state.lateral_offset_m:+.2f} m",
            True,
            SOFT,
        )
        panel.blit(lateral_text, (18, 48))

        setup_text = self.small_font.render(
            f"Vägg: {geometry.wall_distance_m:.1f} m | Viewport: "
            f"{geometry.viewport_physical_width_cm:.0f} x "
            f"{geometry.viewport_physical_height_cm:.0f} cm",
            True,
            SOFT,
        )
        panel.blit(setup_text, (18, 72))

        fit_msg = (
            "Helfigur får plats"
            if fits
            else "Helfigur får ej plats vid detta avstånd"
        )
        fit_color = SOFT if fits else WARN
        fit_text = self.small_font.render(fit_msg, True, fit_color)
        panel.blit(fit_text, (18, 96))

        screen.blit(panel, (self.viewport.x + 16, self.viewport.y + 16))

    def render(self, screen: pygame.Surface) -> None:
        self._render_background(screen)
        self._render_target(screen)
        self._render_hud(screen)


def create_game(game_root: str, viewport: pygame.Rect):
    return RangeTargetGame(game_root=game_root, viewport=viewport)