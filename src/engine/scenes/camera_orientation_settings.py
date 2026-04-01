from __future__ import annotations

import pygame

from src.engine.camera.camera_manager import camera_manager
from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import (
    load_camera_mirror_horizontal,
    load_camera_mirror_vertical,
    load_camera_rotation,
    save_camera_mirror_horizontal,
    save_camera_mirror_vertical,
    save_camera_rotation,
)

WHITE = (240, 240, 240)
SOFT_WHITE = (210, 210, 210)
GREEN = (120, 255, 120)
RED = (255, 120, 120)
YELLOW = (255, 220, 100)
PANEL_BG = (0, 0, 0, 170)
HILITE = (255, 255, 255, 28)


class CameraOrientationSettingsScene(Scene):
    """
    Kamerans orientering appliceras centralt i CameraManager innan resten av
    motorn använder framen.

    Kontroller:
    - UP / DOWN: välj rad
    - LEFT / RIGHT / ENTER / SPACE: ändra vald inställning
    - R: återställ till standard
    - ESC: spara och gå tillbaka
    """

    wants_hit_scanning = False
    wants_camera_preview = False

    def __init__(self, bg_color=(0, 0, 0)) -> None:
        self.bg_color = tuple(bg_color)
        self.font = None
        self.small = None
        self.tiny = None

        self.selected_index = 0
        self.rotation = 0
        self.mirror_horizontal = False
        self.mirror_vertical = False

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 42)
        self.small = pygame.font.Font(None, 28)
        self.tiny = pygame.font.Font(None, 22)

        self.rotation = load_camera_rotation()
        self.mirror_horizontal = load_camera_mirror_horizontal()
        self.mirror_vertical = load_camera_mirror_vertical()

    def _go_back(self):
        from src.engine.scenes.menu import MenuScene
        return SceneSwitch(MenuScene())

    def _save(self) -> None:
        save_camera_rotation(self.rotation)
        save_camera_mirror_horizontal(self.mirror_horizontal)
        save_camera_mirror_vertical(self.mirror_vertical)
        camera_manager.reload_transform_settings()

    def _field_count(self) -> int:
        return 3

    def _change_selected(self, delta: int) -> None:
        if self.selected_index == 0:
            choices = [0, 90, 180, 270]
            try:
                idx = choices.index(self.rotation)
            except ValueError:
                idx = 0
            self.rotation = choices[(idx + delta) % len(choices)]
        elif self.selected_index == 1:
            self.mirror_horizontal = not self.mirror_horizontal
        elif self.selected_index == 2:
            self.mirror_vertical = not self.mirror_vertical

        self._save()

    def _reset_defaults(self) -> None:
        self.rotation = 0
        self.mirror_horizontal = False
        self.mirror_vertical = False
        self._save()

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            self._save()
            return self._go_back()

        if event.key == pygame.K_UP:
            self.selected_index = (self.selected_index - 1) % self._field_count()
            return None

        if event.key == pygame.K_DOWN:
            self.selected_index = (self.selected_index + 1) % self._field_count()
            return None

        if event.key == pygame.K_LEFT:
            self._change_selected(-1)
            return None

        if event.key == pygame.K_RIGHT:
            self._change_selected(1)
            return None

        if event.key in (pygame.K_RETURN, pygame.K_SPACE):
            self._change_selected(1)
            return None

        if event.key == pygame.K_r:
            self._reset_defaults()
            return None

        return None

    def update(self, dt: float):
        del dt
        return None

    def render(self, screen: pygame.Surface) -> None:
        screen.fill(self.bg_color)

        panel = pygame.Surface((1040, 520), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        panel_pos = (40, 40)
        screen.blit(panel, panel_pos)

        px, py = panel_pos

        title = self.font.render("Kameraorientering", True, WHITE)
        screen.blit(title, (px + 20, py + 18))

        subtitle = self.tiny.render(
            "Appliceras direkt i CameraManager innan analys, scanport, viewport och koordinater används.",
            True,
            SOFT_WHITE,
        )
        screen.blit(subtitle, (px + 20, py + 58))

        rows = [
            ("Rotation", f"{self.rotation}°"),
            ("Spegel horisontellt", "PÅ" if self.mirror_horizontal else "AV"),
            ("Spegel vertikalt", "PÅ" if self.mirror_vertical else "AV"),
        ]

        y = py + 120
        for idx, (label, value) in enumerate(rows):
            row_rect = pygame.Rect(px + 18, y - 6, 640, 46)
            if idx == self.selected_index:
                hi = pygame.Surface((row_rect.w, row_rect.h), pygame.SRCALPHA)
                hi.fill(HILITE)
                screen.blit(hi, row_rect.topleft)

            label_color = YELLOW if idx == self.selected_index else WHITE
            value_color = GREEN if idx == self.selected_index else SOFT_WHITE

            label_surf = self.small.render(label, True, label_color)
            value_surf = self.small.render(value, True, value_color)

            screen.blit(label_surf, (row_rect.x + 12, row_rect.y + 10))
            screen.blit(value_surf, (row_rect.right - value_surf.get_width() - 12, row_rect.y + 10))
            y += 60

        help_lines = [
            "UP / DOWN = välj rad",
            "LEFT / RIGHT = ändra",
            "ENTER / SPACE = växla vald rad",
            "R = återställ standard",
            "ESC = tillbaka",
        ]
        y = py + 330
        for line in help_lines:
            surf = self.tiny.render(line, True, SOFT_WHITE)
            screen.blit(surf, (px + 20, y))
            y += 24

        status_title = self.small.render("Kamerastatus", True, WHITE)
        screen.blit(status_title, (px + 700, py + 120))

        status_lines = camera_manager.get_status_lines()
        status_y = py + 160
        for line in status_lines[:8]:
            surf = self.tiny.render(line, True, SOFT_WHITE)
            screen.blit(surf, (px + 700, status_y))
            status_y += 24

        warning = self.tiny.render(
            "Tips: börja med 180° om kameran sitter uppochned i taket.",
            True,
            GREEN if self.rotation == 180 else RED,
        )
        screen.blit(warning, (px + 20, py + 455))