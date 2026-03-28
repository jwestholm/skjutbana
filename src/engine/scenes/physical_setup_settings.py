from __future__ import annotations

import pygame

from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import (
    load_wall_distance_m,
    save_wall_distance_m,
    load_viewport_physical_width_cm,
    save_viewport_physical_width_cm,
    load_viewport_physical_height_cm,
    save_viewport_physical_height_cm,
    load_viewport_bottom_world_cm,
    save_viewport_bottom_world_cm,
)
from src.engine.scenes.menu import MenuScene

WHITE = (240, 240, 240)
SOFT = (190, 190, 190)
YELLOW = (255, 220, 80)
GREEN = (120, 255, 120)

PANEL_BG = (0, 0, 0, 170)
HILITE_BG = (255, 255, 255, 28)


class PhysicalSetupSettingsScene(Scene):
    """
    Globala motor-/miljöinställningar för fysisk setup.

    Ändrar och sparar:
    - Avstånd till vägg
    - Viewportens verkliga bredd
    - Viewportens verkliga höjd
    - Viewportens nederkant över mark

    Kontroller:
    - UP / DOWN välj rad
    - LEFT / RIGHT minska / öka värde
    - SHIFT större steg
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

    # ------------------------------------------------------------
    # Scene lifecycle
    # ------------------------------------------------------------

    def on_enter(self) -> None:
        self.font_title = pygame.font.Font(None, 44)
        self.font_body = pygame.font.Font(None, 30)
        self.font_small = pygame.font.Font(None, 22)
        self._reload_fields()
        self.status_message = "Ändringarna sparas direkt."

    def on_exit(self) -> None:
        pass

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _reload_fields(self) -> None:
        self.fields = [
            {
                "label": "Avstånd till vägg",
                "unit": "m",
                "value": float(load_wall_distance_m()),
                "default": 6.3,
                "small_step": 0.1,
                "large_step": 0.5,
                "min": 0.5,
                "max": 50.0,
                "save": save_wall_distance_m,
                "format": "{:.1f}",
                "hint": "LEFT/RIGHT ändrar  •  SHIFT = större steg",
            },
            {
                "label": "Viewport bredd (verklig)",
                "unit": "cm",
                "value": float(load_viewport_physical_width_cm()),
                "default": 90.0,
                "small_step": 1.0,
                "large_step": 5.0,
                "min": 10.0,
                "max": 1000.0,
                "save": save_viewport_physical_width_cm,
                "format": "{:.0f}",
                "hint": "Projicerad verklig bredd på tavlan/viewporten",
            },
            {
                "label": "Viewport höjd (verklig)",
                "unit": "cm",
                "value": float(load_viewport_physical_height_cm()),
                "default": 70.0,
                "small_step": 1.0,
                "large_step": 5.0,
                "min": 10.0,
                "max": 1000.0,
                "save": save_viewport_physical_height_cm,
                "format": "{:.0f}",
                "hint": "Projicerad verklig höjd på tavlan/viewporten",
            },
            {
                "label": "Viewport underkant över mark",
                "unit": "cm",
                "value": float(load_viewport_bottom_world_cm()),
                "default": 105.0,
                "small_step": 1.0,
                "large_step": 5.0,
                "min": -100.0,
                "max": 400.0,
                "save": save_viewport_bottom_world_cm,
                "format": "{:.0f}",
                "hint": "Fysisk höjd från mark till tavlans nederkant",
            },
        ]

    def _selected_field(self) -> dict:
        return self.fields[self.selected_index]

    def _adjust_selected(self, direction: int, large_step: bool) -> None:
        field = self._selected_field()
        step = field["large_step"] if large_step else field["small_step"]

        value = float(field["value"]) + (float(direction) * float(step))
        value = max(float(field["min"]), min(float(field["max"]), value))

        field["value"] = value
        field["save"](value)

        self.status_message = (
            f"Sparade: {field['label']} = "
            f"{field['format'].format(value)} {field['unit']}"
        )

    def _reset_defaults(self) -> None:
        for field in self.fields:
            field["value"] = float(field["default"])
            field["save"](field["value"])

        self.status_message = "Återställde fysisk setup till standardvärden."
        self._reload_fields()

    def _back(self):
        return SceneSwitch(MenuScene())

    # ------------------------------------------------------------
    # Input
    # ------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            return self._back()

        if event.key == pygame.K_UP:
            self.selected_index = (self.selected_index - 1) % len(self.fields)
            return None

        if event.key == pygame.K_DOWN:
            self.selected_index = (self.selected_index + 1) % len(self.fields)
            return None

        mods = pygame.key.get_mods()
        large_step = bool(mods & pygame.KMOD_SHIFT)

        if event.key == pygame.K_LEFT:
            self._adjust_selected(direction=-1, large_step=large_step)
            return None

        if event.key == pygame.K_RIGHT:
            self._adjust_selected(direction=1, large_step=large_step)
            return None

        if event.key == pygame.K_r:
            self._reset_defaults()
            return None

        return None

    # ------------------------------------------------------------
    # Update / render
    # ------------------------------------------------------------

    def update(self, dt: float):
        del dt
        return None

    def render(self, screen: pygame.Surface) -> None:
        screen.fill((18, 18, 18))

        if self.font_title is None or self.font_body is None or self.font_small is None:
            return

        sw = screen.get_width()
        sh = screen.get_height()

        margin_x = 36
        panel_x = margin_x
        panel_y = 110
        panel_w = min(760, sw - (margin_x * 2))
        row_h = 62
        panel_h = row_h * len(self.fields) + 20

        title = self.font_title.render("Fysisk setup", True, WHITE)
        screen.blit(title, (margin_x, 24))

        subtitle = self.font_small.render(
            "Globala motorinställningar för verklig skjutmiljö och projektion.",
            True,
            SOFT,
        )
        screen.blit(subtitle, (margin_x + 2, 66))

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (panel_x, panel_y))

        for i, field in enumerate(self.fields):
            row_top = panel_y + 10 + i * row_h

            if i == self.selected_index:
                hilite = pygame.Surface((panel_w - 16, row_h - 6), pygame.SRCALPHA)
                hilite.fill(HILITE_BG)
                screen.blit(hilite, (panel_x + 8, row_top + 3))

            label_color = YELLOW if i == self.selected_index else WHITE
            label_surf = self.font_body.render(field["label"], True, label_color)
            screen.blit(label_surf, (panel_x + 16, row_top + 6))

            value_text = f"{field['format'].format(field['value'])} {field['unit']}"
            value_surf = self.font_body.render(value_text, True, GREEN)
            value_rect = value_surf.get_rect()
            value_rect.topright = (panel_x + panel_w - 16, row_top + 6)
            screen.blit(value_surf, value_rect)

            hint_surf = self.font_small.render(field["hint"], True, SOFT)
            screen.blit(hint_surf, (panel_x + 16, row_top + 34))

        info_panel_y = panel_y + panel_h + 18
        info_panel_h = 118
        info_panel = pygame.Surface((panel_w, info_panel_h), pygame.SRCALPHA)
        info_panel.fill(PANEL_BG)
        screen.blit(info_panel, (panel_x, info_panel_y))

        controls_1 = self.font_small.render(
            "UP/DOWN = välj rad   LEFT/RIGHT = ändra värde",
            True,
            WHITE,
        )
        controls_2 = self.font_small.render(
            "SHIFT = större steg   R = återställ standard   ESC = tillbaka",
            True,
            WHITE,
        )

        explain_1 = self.font_small.render(
            "Viewport underkant över mark används av helfigursspelet för att placera kroppen rätt i bild.",
            True,
            SOFT,
        )
        explain_2 = self.font_small.render(
            "Exempel: 105 cm betyder att tavlans nederkant sitter 105 cm ovanför marken.",
            True,
            SOFT,
        )

        screen.blit(controls_1, (panel_x + 14, info_panel_y + 12))
        screen.blit(controls_2, (panel_x + 14, info_panel_y + 34))
        screen.blit(explain_1, (panel_x + 14, info_panel_y + 62))
        screen.blit(explain_2, (panel_x + 14, info_panel_y + 82))

        status_color = GREEN if self.status_message else SOFT
        status = self.font_small.render(self.status_message or " ", True, status_color)
        screen.blit(status, (panel_x + 14, info_panel_y + 100))

        footnote = self.font_small.render(
            "Används globalt av motorfunktioner som avståndsskalning och fysisk höjdprojektion.",
            True,
            SOFT,
        )
        screen.blit(footnote, (margin_x, sh - 34))