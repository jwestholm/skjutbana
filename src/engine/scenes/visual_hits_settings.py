from __future__ import annotations

import pygame

from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import (
    load_visual_hits_enabled,
    load_visual_hits_lifetime_ms,
    load_visual_hits_mode,
    load_visual_hits_show_all_planes,
    save_visual_hits_enabled,
    save_visual_hits_lifetime_ms,
    save_visual_hits_mode,
    save_visual_hits_show_all_planes,
)
from src.engine.scenes.menu import MenuScene


WHITE = (240, 240, 240)
SOFT = (190, 190, 190)
YELLOW = (255, 220, 80)
GREEN = (120, 255, 120)
RED = (255, 120, 120)
PANEL_BG = (0, 0, 0, 170)
HILITE_BG = (255, 255, 255, 28)


class VisualHitsSettingsScene(Scene):
    """
    Inställningar för visuella träffmarkeringar.

    Ändrar och sparar direkt:
    - Visa träff
    - Visa träff i alla plan
    - Visningsläge (fade / persistent)
    - Fade-tid i ms

    Kontroller:
    - UP / DOWN välj rad
    - LEFT / RIGHT ändra värde
    - ENTER / SPACE växla valt alternativ
    - R återställ defaults
    - ESC tillbaka till menyn
    """

    wants_hit_scanning = False
    wants_camera_preview = False

    def __init__(self, app=None, bg_color=(18, 18, 18), **kwargs) -> None:
        super().__init__()
        self.app = app
        self.bg_color = bg_color
        self.kwargs = kwargs

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
        self.status_message = "Ändringarna sparas direkt."

    def on_exit(self) -> None:
        pass

    def _reload_fields(self) -> None:
        mode_value = str(load_visual_hits_mode()).strip().lower()
        if mode_value not in {"fade", "persistent"}:
            mode_value = "fade"

        self.fields = [
            {
                "label": "Visa träff",
                "kind": "toggle",
                "value": bool(load_visual_hits_enabled()),
                "default": True,
                "save": save_visual_hits_enabled,
                "hint": "Visar eller döljer visuella träffmarkeringar",
            },
            {
                "label": "Visa träff i alla plan",
                "kind": "toggle",
                "value": bool(load_visual_hits_show_all_planes()),
                "default": False,
                "save": save_visual_hits_show_all_planes,
                "hint": "Ritar debugmarkeringar i flera koordinatplan",
            },
            {
                "label": "Mode",
                "kind": "choice",
                "value": mode_value,
                "choices": ["fade", "persistent"],
                "default": "fade",
                "save": save_visual_hits_mode,
                "hint": "Fade tonar ut över tid, persistent ligger kvar",
            },
            {
                "label": "Fade tid",
                "kind": "int",
                "value": int(load_visual_hits_lifetime_ms()),
                "default": 1000,
                "small_step": 100,
                "large_step": 500,
                "min": 100,
                "max": 10000,
                "unit": "ms",
                "save": save_visual_hits_lifetime_ms,
                "hint": "Hur länge en fade-träff visas",
            },
        ]

    def _selected_field(self) -> dict:
        return self.fields[self.selected_index]

    def _format_value(self, field: dict) -> tuple[str, tuple[int, int, int]]:
        kind = field["kind"]
        value = field["value"]

        if kind == "toggle":
            enabled = bool(value)
            return ("PÅ" if enabled else "AV", GREEN if enabled else RED)

        if kind == "choice":
            if str(value) == "persistent":
                return ("Persistent", WHITE)
            return ("Fade", WHITE)

        if kind == "int":
            return (f"{int(value)} {field.get('unit', '')}".strip(), GREEN)

        return (str(value), WHITE)

    def _save_field(self, field: dict) -> None:
        field["save"](field["value"])

    def _toggle_selected(self) -> None:
        field = self._selected_field()
        kind = field["kind"]

        if kind == "toggle":
            field["value"] = not bool(field["value"])
        elif kind == "choice":
            choices = list(field["choices"])
            current = str(field["value"])
            try:
                idx = choices.index(current)
            except ValueError:
                idx = 0
            field["value"] = choices[(idx + 1) % len(choices)]
        else:
            return

        self._save_field(field)
        value_text, _ = self._format_value(field)
        self.status_message = f"Sparade: {field['label']} = {value_text}"

    def _adjust_selected(self, direction: int, large_step: bool) -> None:
        field = self._selected_field()
        kind = field["kind"]

        if kind == "toggle":
            field["value"] = direction > 0
        elif kind == "choice":
            choices = list(field["choices"])
            current = str(field["value"])
            try:
                idx = choices.index(current)
            except ValueError:
                idx = 0
            step = 1 if direction > 0 else -1
            field["value"] = choices[(idx + step) % len(choices)]
        elif kind == "int":
            step = int(field["large_step"] if large_step else field["small_step"])
            value = int(field["value"]) + (direction * step)
            value = max(int(field["min"]), min(int(field["max"]), value))
            field["value"] = value
        else:
            return

        self._save_field(field)
        value_text, _ = self._format_value(field)
        self.status_message = f"Sparade: {field['label']} = {value_text}"

    def _reset_defaults(self) -> None:
        for field in self.fields:
            field["value"] = field["default"]
            self._save_field(field)
        self.status_message = "Återställde visuella träffar till standardvärden."
        self._reload_fields()

    def _back(self):
        return SceneSwitch(MenuScene())

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

        if event.key in (pygame.K_RETURN, pygame.K_SPACE):
            self._toggle_selected()
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

        if self.font_title is None or self.font_body is None or self.font_small is None:
            return

        sw = screen.get_width()
        sh = screen.get_height()
        margin_x = 36
        panel_x = margin_x
        panel_y = 110
        panel_w = min(820, sw - (margin_x * 2))
        row_h = 62
        panel_h = row_h * len(self.fields) + 20

        title = self.font_title.render("Visuella träffar", True, WHITE)
        screen.blit(title, (margin_x, 24))

        subtitle = self.font_small.render(
            "Debug och feedback för träffmarkeringar i spel- och transformflödet.",
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

            value_text, value_color = self._format_value(field)
            value_surf = self.font_body.render(value_text, True, value_color)
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
            "UP/DOWN = välj rad   LEFT/RIGHT = ändra värde   ENTER/SPACE = växla",
            True,
            WHITE,
        )
        controls_2 = self.font_small.render(
            "SHIFT = större steg   R = återställ standard   ESC = tillbaka",
            True,
            WHITE,
        )
        explain_1 = self.font_small.render(
            "Visa träff i alla plan används för att verifiera transformkedjan visuellt.",
            True,
            SOFT,
        )
        explain_2 = self.font_small.render(
            "Ändringar sparas direkt och används av programmet utan extra bekräftelse.",
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
            "Används av visualizer/debug för träffpresentation och transformverifiering.",
            True,
            SOFT,
        )
        screen.blit(footnote, (margin_x, sh - 34))
