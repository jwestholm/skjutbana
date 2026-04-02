from __future__ import annotations

import pygame

from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import (
    load_visual_hits_enabled,
    load_visual_hits_lifetime_ms,
    load_visual_hits_mode,
    save_visual_hits_enabled,
    save_visual_hits_lifetime_ms,
    save_visual_hits_mode,
)

# Ny flagga kan saknas i äldre settings.py. Då faller vi tillbaka till False/no-op
try:
    from src.engine.settings import (
        load_visual_hits_show_all_planes,
        save_visual_hits_show_all_planes,
    )
except Exception:  # pragma: no cover - kompatibilitet mot äldre settings.py
    def load_visual_hits_show_all_planes() -> bool:
        return False

    def save_visual_hits_show_all_planes(enabled: bool) -> None:
        del enabled
        return None


WHITE = (240, 240, 240)
SOFT_WHITE = (210, 210, 210)
GREEN = (120, 255, 120)
RED = (255, 120, 120)
YELLOW = (255, 215, 120)
PANEL_BG = (0, 0, 0, 165)
ROW_HIGHLIGHT = (255, 255, 255, 28)


class VisualHitsSettingsScene(Scene):
    def __init__(self, bg_color=(0, 0, 0)) -> None:
        self.bg_color = bg_color
        self.font = None
        self.small = None
        self.tiny = None

        self.enabled = True
        self.show_all_planes = False
        self.mode = "fade"
        self.lifetime = 900

        self.selected_index = 0
        self._items = [
            "enabled",
            "show_all_planes",
            "mode",
            "lifetime",
        ]

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 42)
        self.small = pygame.font.Font(None, 28)
        self.tiny = pygame.font.Font(None, 24)

        self.enabled = load_visual_hits_enabled()
        self.show_all_planes = load_visual_hits_show_all_planes()
        self.mode = load_visual_hits_mode()
        self.lifetime = load_visual_hits_lifetime_ms()
        self.selected_index = max(0, min(self.selected_index, len(self._items) - 1))

    def _go_back(self):
        from src.engine.scenes.menu import MenuScene

        return SceneSwitch(MenuScene())

    def _save(self) -> None:
        save_visual_hits_enabled(self.enabled)
        save_visual_hits_show_all_planes(self.show_all_planes)
        save_visual_hits_mode(self.mode)
        save_visual_hits_lifetime_ms(self.lifetime)

    def _current_item(self) -> str:
        return self._items[self.selected_index]

    def _toggle_or_activate_current(self) -> None:
        item = self._current_item()
        if item == "enabled":
            self.enabled = not self.enabled
        elif item == "show_all_planes":
            self.show_all_planes = not self.show_all_planes
        elif item == "mode":
            self.mode = "persistent" if self.mode == "fade" else "fade"
        elif item == "lifetime":
            # ENTER på fade-fältet gör inget destruktivt.
            return None

    def _adjust_current(self, delta: int) -> None:
        item = self._current_item()
        if item == "enabled":
            if delta != 0:
                self.enabled = delta > 0
        elif item == "show_all_planes":
            if delta != 0:
                self.show_all_planes = delta > 0
        elif item == "mode":
            if delta != 0:
                self.mode = "persistent" if self.mode == "fade" else "fade"
        elif item == "lifetime":
            step = 100
            self.lifetime = max(100, self.lifetime + (delta * step))

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            self._save()
            return self._go_back()

        if event.key == pygame.K_UP:
            self.selected_index = (self.selected_index - 1) % len(self._items)
            return None

        if event.key == pygame.K_DOWN:
            self.selected_index = (self.selected_index + 1) % len(self._items)
            return None

        if event.key in (pygame.K_RETURN, pygame.K_SPACE):
            self._toggle_or_activate_current()
            return None

        if event.key == pygame.K_LEFT:
            self._adjust_current(-1)
            return None

        if event.key == pygame.K_RIGHT:
            self._adjust_current(1)
            return None

        return None

    def update(self, dt: float):
        del dt
        return None

    def _row(self, panel: pygame.Surface, y: int, label: str, value: str, selected: bool, value_color) -> None:
        if selected:
            highlight = pygame.Surface((840, 34), pygame.SRCALPHA)
            highlight.fill(ROW_HIGHLIGHT)
            panel.blit(highlight, (20, y - 4))

        label_surf = self.small.render(label, True, YELLOW if selected else SOFT_WHITE)
        value_surf = self.small.render(value, True, value_color)
        panel.blit(label_surf, (40, y))
        panel.blit(value_surf, (500, y))

    def render(self, screen: pygame.Surface):
        screen.fill(self.bg_color)

        panel = pygame.Surface((960, 430), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (40, 40))

        title = self.font.render("Visuella träffar", True, WHITE)
        screen.blit(title, (60, 60))

        rows_y = 120
        state_color = GREEN if self.enabled else RED
        state_text = "PÅ" if self.enabled else "AV"
        self._row(panel, rows_y, "Visa träff:", state_text, self.selected_index == 0, state_color)

        all_planes_color = GREEN if self.show_all_planes else RED
        all_planes_text = "PÅ" if self.show_all_planes else "AV"
        self._row(
            panel,
            rows_y + 44,
            "Visa träff i alla plan:",
            all_planes_text,
            self.selected_index == 1,
            all_planes_color,
        )

        mode_text = "Fade" if self.mode == "fade" else "Persistent"
        self._row(panel, rows_y + 88, "Mode:", mode_text, self.selected_index == 2, SOFT_WHITE)

        self._row(
            panel,
            rows_y + 132,
            "Fade tid:",
            f"{self.lifetime} ms",
            self.selected_index == 3,
            SOFT_WHITE,
        )

        help_lines = [
            "UP / DOWN = navigera i menyn",
            "LEFT / RIGHT = ändra valt värde",
            "ENTER / SPACE = slå av/på eller växla valt alternativ",
            "ESC = spara och gå tillbaka",
        ]
        y = 315
        for line in help_lines:
            surf = self.tiny.render(line, True, SOFT_WHITE)
            screen.blit(surf, (60, y))
            y += 26
