from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygame

from src.engine.range_projection import (
    RangeProjectionGeometry,
    clamp_distance_m,
    depth_ratio,
    projected_lateral_offset_px,
    projected_target_height_px,
)
from src.engine.settings import load_range_projection_settings

WHITE = (245, 245, 245)
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
    Helfigur på avstånd, kalibrerad mot just denna bakgrund.

    Kontroller:
    - LEFT / RIGHT -> flytta figur i sidled
    - UP           -> närmare
    - DOWN         -> längre bort
    - SHIFT        -> snabbare rörelse när hålls inne
    - SPACE        -> visa/dölj HUD
    - R            -> återställ
    """

    def __init__(self, game_root: str, viewport: pygame.Rect) -> None:
        self.game_root = Path(game_root)
        self.viewport = viewport.copy()
        self.state = RangeTargetState()

        self.max_distance_m = 600.0

        # Kontinuerlig rörelse när tangenter hålls inne
        self.distance_speed_mps = 18.0
        self.distance_speed_fast_mps = 60.0
        self.lateral_speed_mps = 1.2
        self.lateral_speed_fast_mps = 3.5

        # Verklig figurhöjd
        self.target_real_height_cm = 180.0

        # Om PNG:n inte är helt centrerad kan denna trimmas lite
        self.target_anchor_x_norm = 0.5

        # Visuell tuning:
        # 1.0 = ren fysisk grund
        # lite lägre multiplier och exponent < 1 gör att den känns mindre "sprite-zoomig"
        self.visual_scale_multiplier = 0.88
        self.visual_scale_exponent = 0.90

        # Kalibrerad bana i bakgrunden.
        # Format: (distance_m, x_norm, y_norm)
        # x_norm/y_norm är fotpunktens position i viewporten.
        self.ground_path = [
            (6.3,   0.545, 0.965),
            (12.0,  0.545, 0.930),
            (20.0,  0.546, 0.890),
            (34.0,  0.547, 0.825),
            (50.0,  0.548, 0.765),
            (75.0,  0.549, 0.700),
            (100.0, 0.551, 0.650),
            (125.0, 0.552, 0.610),
            (150.0, 0.553, 0.580),
            (200.0, 0.554, 0.545),
            (300.0, 0.555, 0.500),
            (450.0, 0.556, 0.465),
            (600.0, 0.557, 0.445),
        ]

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

        geometry = self._geometry()
        self.state.distance_m = max(self.state.distance_m, geometry.wall_distance_m)

    def on_exit(self) -> None:
        pass

    # ------------------------------------------------------------
    # Geometry / settings
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
        geometry = self._geometry()
        return max(0.1, float(geometry.wall_distance_m))

    def _target_fits_height_at_distance(self, distance_m: float) -> bool:
        geometry = self._geometry()
        projected_h_cm = self.target_real_height_cm * (
            geometry.wall_distance_m / max(0.01, distance_m)
        )
        return projected_h_cm <= geometry.viewport_physical_height_cm

    # ------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------

    def _sample_ground_path(self, distance_m: float) -> tuple[float, float]:
        """
        Returnerar fotpunktens x_norm, y_norm längs den kalibrerade banan.
        """
        pts = self.ground_path
        d = float(distance_m)

        if d <= pts[0][0]:
            return pts[0][1], pts[0][2]
        if d >= pts[-1][0]:
            return pts[-1][1], pts[-1][2]

        for i in range(len(pts) - 1):
            d0, x0, y0 = pts[i]
            d1, x1, y1 = pts[i + 1]
            if d0 <= d <= d1:
                t = (d - d0) / (d1 - d0)
                x = x0 + (x1 - x0) * t
                y = y0 + (y1 - y0) * t
                return x, y

        return pts[-1][1], pts[-1][2]

    def _foot_position_px(self) -> tuple[float, float]:
        geometry = self._geometry()

        x_norm, y_norm = self._sample_ground_path(self.state.distance_m)

        base_x = self.viewport.x + (x_norm * self.viewport.w)
        base_y = self.viewport.y + (y_norm * self.viewport.h)

        lateral_px = projected_lateral_offset_px(
            world_lateral_offset_m=self.state.lateral_offset_m,
            virtual_distance_m=self.state.distance_m,
            viewport=self.viewport,
            geometry=geometry,
        )

        return base_x + lateral_px, base_y

    def _target_height_px(self) -> int:
        geometry = self._geometry()

        base_h = projected_target_height_px(
            real_height_cm=self.target_real_height_cm,
            virtual_distance_m=self.state.distance_m,
            viewport=self.viewport,
            geometry=geometry,
        )

        ratio = depth_ratio(
            virtual_distance_m=self.state.distance_m,
            wall_distance_m=geometry.wall_distance_m,
        )

        # Samma när-känsla vid ratio=1, men lugnare avtagande längre bort
        tuned_h = base_h * self.visual_scale_multiplier * (ratio ** (self.visual_scale_exponent - 1.0))
        return max(2, int(round(tuned_h)))

    # ------------------------------------------------------------
    # Input
    # ------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
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
        keys = pygame.key.get_pressed()
        fast = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        distance_speed = self.distance_speed_fast_mps if fast else self.distance_speed_mps
        lateral_speed = self.lateral_speed_fast_mps if fast else self.lateral_speed_mps

        if keys[pygame.K_UP]:
            self.state.distance_m -= distance_speed * dt

        if keys[pygame.K_DOWN]:
            self.state.distance_m += distance_speed * dt

        if keys[pygame.K_LEFT]:
            self.state.lateral_offset_m -= lateral_speed * dt

        if keys[pygame.K_RIGHT]:
            self.state.lateral_offset_m += lateral_speed * dt

        self.state.distance_m = clamp_distance_m(
            self.state.distance_m,
            min_distance_m=self._min_distance_m(),
            max_distance_m=self.max_distance_m,
        )

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

        target_h_px = self._target_height_px()

        src_w, src_h = self.target_surface.get_size()
        if src_h <= 0:
            return

        aspect = src_w / float(src_h)
        target_w_px = max(1, int(round(target_h_px * aspect)))

        scaled_target = pygame.transform.smoothscale(
            self.target_surface,
            (target_w_px, target_h_px),
        )

        foot_x, foot_y = self._foot_position_px()

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

        panel = pygame.Surface((380, 128), pygame.SRCALPHA)
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

        fit_msg = "Helfigur får plats" if fits else "Helfigur får ej plats vid detta avstånd"
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