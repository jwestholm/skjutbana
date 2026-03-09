from __future__ import annotations

import pygame

from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import (
    load_visual_hits_enabled,
    load_visual_hits_mode,
    save_visual_hits_enabled,
    save_visual_hits_mode,
)
from src.engine.visual.hit_visualizer import hit_visualizer


WHITE = (240, 240, 240)
SOFT_WHITE = (210, 210, 210)
GREEN = (120, 255, 120)
RED = (255, 120, 120)
PANEL_BG = (0, 0, 0, 165)


class VisualHitsSettingsScene(Scene):
    """
    Inställningar för visuella träffar.

    Kontroller:
    - ENTER / SPACE = slå av / på
    - M = byt läge fade / persistent
    - ESC = tillbaka
    """

    def __init__(self, bg_color=(0, 0, 0)) -> None:
        self.bg_color = bg_color
        self.font = None
        self.small = None
        self.tiny = None
        self.enabled = True
        self.mode = "fade"

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 42)
        self.small = pygame.font.Font(None, 28)
        self.tiny = pygame.font.Font(None, 24)
        self.enabled = load_visual_hits_enabled()
        self.mode = load_visual_hits_mode()

    def _go_back(self):
        from src.engine.scenes.menu import MenuScene
        return SceneSwitch(MenuScene())

    def _save(self) -> None:
        save_visual_hits_enabled(self.enabled)
        save_visual_hits_mode(self.mode)
        hit_visualizer.reload_settings()

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            self._save()
            return self._go_back()

        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self.enabled = not self.enabled
            self._save()
            return None

        if event.key == pygame.K_m:
            self.mode = "persistent" if self.mode == "fade" else "fade"
            self._save()
            return None

        return None

    def update(self, dt: float):
        return None

    def render(self, screen: pygame.Surface) -> None:
        screen.fill(self.bg_color)

        panel = pygame.Surface((900, 260), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (40, 40))

        title = self.font.render("Visuella träffar", True, WHITE)
        screen.blit(title, (60, 60))

        state_color = GREEN if self.enabled else RED
        state_text = "PÅ" if self.enabled else "AV"
        state = self.small.render(f"Status: {state_text}", True, state_color)
        screen.blit(state, (60, 115))

        mode = self.small.render(f"Läge: {self.mode}", True, WHITE)
        screen.blit(mode, (60, 150))

        lines = [
            "ENTER / SPACE: slå av eller på visuella träffar",
            "M: växla mellan fade och persistent",
            "fade = träffmarkering tonar bort efter några sekunder",
            "persistent = träffmarkeringar ligger kvar tills du lämnar innehållet",
            "ESC: tillbaka",
        ]

        y = 200
        for line in lines:
            surf = self.tiny.render(line, True, SOFT_WHITE)
            screen.blit(surf, (60, y))
            y += 26