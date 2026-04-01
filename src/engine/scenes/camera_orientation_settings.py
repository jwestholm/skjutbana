from __future__ import annotations

import pygame

from src.engine.camera.camera_manager import camera_manager
from src.engine.scene import Scene, SceneSwitch
from src.engine.scenes.menu import MenuScene
from src.engine.settings import (
    load_camera_mirror_horizontal,
    load_camera_mirror_vertical,
    load_camera_rotation,
    save_camera_mirror_horizontal,
    save_camera_mirror_vertical,
    save_camera_rotation,
)

WHITE = (240, 240, 240)
SOFT = (190, 190, 190)
YELLOW = (255, 220, 80)
GREEN = (120, 255, 120)
PANEL_BG = (0, 0, 0, 170)
HILITE_BG = (255, 255, 255, 28)


class CameraOrientationSettingsScene(Scene):
    """
    Kamerans orientering appliceras centralt i CameraManager innan resten av
    motorn använder framen.

    Kontroller:
    - UP / DOWN välj rad
    - LEFT / RIGHT ändra värde
    - R återställ defaults
    - ESC tillbaka till menyn
    """

    wants_hit_scanning = False
    wants_camera_preview = False

    def __init__(self) -> None:
        self.font_title: pygame.font.Font | None = None
        self.font_body: pygame.font.Font | None = None
        self.font_small: pygame.font.Font | None = None

        self.selected_index = 0
        self.status_message = ""
        self.fields: list[dict] = []

    def on_enter(self) -> None:
        self.font_title = pygame.font.Font(None, 44)
        self.font_body = pygame.font.Font(None, 30)
        self.font_small = pygame.font.Font(None, 22)
        self._reload_fields()
        self.status_message = "Ändringarna sparas direkt och används av kameramotorn."

    def on_exit(self) -> None:
        pass

    def _reload_fields(self) -> None:
        self.fields = [
            {
                "label": "Rotation",
                "type": "choice",
                "choices": [0, 90, 180, 270],
                "value": int(load_camera_rotation()),
            },
            {
                "label": "Spegel horisontellt",
                "type": "bool",
                "value": bool(load_camera_mirror_horizontal()),
            },
            {
                "label": "Spegel vertikalt",
                "type": "bool",
                "value": bool(load_camera_mirror_vertical()),
            },
        ]

    def _save_current_state(self) -> None:
        save_camera_rotation(int(self.fields[0]["value"]))
        save_camera_mirror_horizontal(bool(self.fields[1]["value"]))
        save_camera_mirror_vertical(bool(self.fields[2]["value"]))
        camera_manager.reload_transform_settings()

    def _reset_defaults(self) -> None:
        self.fields[0]["value"] = 0
        self.fields[1]["value"] = False
        self.fields[2]["value"] = False
        self._save_current_state()
        self.status_message = "Kamerans orientering återställd till standard."

    def _change_selected(self, delta: int) -> None:
        if not self.fields:
            return

        field = self.fields[self.selected_index]
        field_type = field.get("type")

        if field_type == "choice":
            choices = list(field.get("choices", []))
            if not choices:
                return
            current = field.get("value", choices[0])
            try:
                index = choices.index(current)
            except ValueError:
                index = 0
            index = (index + delta) % len(choices)
            field["value"] = choices[index]

        elif field_type == "bool":
            field["value"] = not bool(field.get("value", False))

        self._save_current_state()
        self.status_message = "Kamerans orientering uppdaterad."

    def handle_event(self, event) -> SceneSwitch | None:
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            return SceneSwitch(next_scene=MenuScene())

        if event.key == pygame.K_UP:
            self.selected_index = (self.selected_index - 1) % len(self.fields)
            return None

        if event.key == pygame.K_DOWN:
            self.selected_index = (self.selected_index + 1) % len(self.fields)
            return None

        if event.key == pygame.K_LEFT:
            self._change_selected(-1)
            return None

        if event.key == pygame.K_RIGHT:
            self._change_selected(1)
            return None

        if event.key == pygame.K_RETURN:
            self._change_selected(1)
            return None

        if event.key == pygame.K_r:
            self._reset_defaults()
            return None

        return None

    def update(self, dt: float) -> SceneSwitch | None:
        del dt
        return None

    def draw(self, surface: pygame.Surface) -> None:
        assert self.font_title is not None
        assert self.font_body is not None
        assert self.font_small is not None

        surface.fill((22, 22, 26))

        panel = pygame.Surface((980, 560), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        panel_rect = panel.get_rect(center=surface.get_rect().center)
        surface.blit(panel, panel_rect)

        title = self.font_title.render("Kameraorientering", True, WHITE)
        surface.blit(title, (panel_rect.x + 30, panel_rect.y + 24))

        help_text = (
            "Rotation/spegling appliceras direkt i CameraManager innan någon annan "
            "del av motorn använder bilden."
        )
        help_surf = self.font_small.render(help_text, True, SOFT)
        surface.blit(help_surf, (panel_rect.x + 32, panel_rect.y + 68))

        y = panel_rect.y + 130
        for index, field in enumerate(self.fields):
            selected = index == self.selected_index

            row_rect = pygame.Rect(panel_rect.x + 24, y - 6, panel_rect.w - 48, 50)
            if selected:
                hilite = pygame.Surface((row_rect.w, row_rect.h), pygame.SRCALPHA)
                hilite.fill(HILITE_BG)
                surface.blit(hilite, row_rect.topleft)

            label_color = YELLOW if selected else WHITE
            value_color = GREEN if selected else SOFT

            label_surf = self.font_body.render(str(field["label"]), True, label_color)
            surface.blit(label_surf, (row_rect.x + 12, row_rect.y + 10))

            value = field["value"]
            if isinstance(value, bool):
                value_text = "På" if value else "Av"
            else:
                value_text = f"{value}°"

            value_surf = self.font_body.render(value_text, True, value_color)
            value_rect = value_surf.get_rect(midright=(row_rect.right - 12, row_rect.y + 24))
            surface.blit(value_surf, value_rect)

            y += 62

        controls = [
            "UP/DOWN: välj rad",
            "LEFT/RIGHT eller ENTER: ändra",
            "R: återställ",
            "ESC: tillbaka",
        ]
        controls_y = panel_rect.bottom - 108
        for i, text in enumerate(controls):
            surf = self.font_small.render(text, True, SOFT)
            surface.blit(surf, (panel_rect.x + 32, controls_y + i * 22))

        status_surf = self.font_small.render(self.status_message, True, GREEN)
        surface.blit(status_surf, (panel_rect.x + 32, panel_rect.bottom - 28))

        status_lines = camera_manager.get_status_lines()
        right_y = panel_rect.y + 130
        for line in status_lines[:6]:
            surf = self.font_small.render(line, True, SOFT)
            rect = surf.get_rect(topright=(panel_rect.right - 24, right_y))
            surface.blit(surf, rect)
            right_y += 22