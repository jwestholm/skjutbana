from __future__ import annotations

import pygame

from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import (
    load_visual_hits_enabled,
    save_visual_hits_enabled,
    load_visual_hits_mode,
    save_visual_hits_mode,
    load_visual_hits_lifetime_ms,
    save_visual_hits_lifetime_ms,
)


WHITE = (240, 240, 240)
SOFT_WHITE = (210, 210, 210)
GREEN = (120, 255, 120)
RED = (255, 120, 120)
PANEL_BG = (0, 0, 0, 165)


class VisualHitsSettingsScene(Scene):

    def __init__(self, bg_color=(0, 0, 0)) -> None:
        self.bg_color = bg_color
        self.font = None
        self.small = None
        self.tiny = None

        self.enabled = True
        self.mode = "fade"
        self.lifetime = 900

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 42)
        self.small = pygame.font.Font(None, 28)
        self.tiny = pygame.font.Font(None, 24)

        self.enabled = load_visual_hits_enabled()
        self.mode = load_visual_hits_mode()
        self.lifetime = load_visual_hits_lifetime_ms()

    def _go_back(self):
        from src.engine.scenes.menu import MenuScene
        return SceneSwitch(MenuScene())

    def _save(self):
        save_visual_hits_enabled(self.enabled)
        save_visual_hits_mode(self.mode)
        save_visual_hits_lifetime_ms(self.lifetime)

    def handle_event(self, event: pygame.event.Event):

        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            self._save()
            return self._go_back()

        if event.key in (pygame.K_RETURN, pygame.K_SPACE):
            self.enabled = not self.enabled

        if event.key == pygame.K_m:
            self.mode = "persistent" if self.mode == "fade" else "fade"

        if event.key == pygame.K_UP:
            self.lifetime += 100

        if event.key == pygame.K_DOWN:
            self.lifetime = max(100, self.lifetime - 100)

        return None

    def update(self, dt: float):
        return None

    def render(self, screen: pygame.Surface):

        screen.fill(self.bg_color)

        panel = pygame.Surface((960, 360), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (40, 40))

        title = self.font.render("Visuella träffar", True, WHITE)
        screen.blit(title, (60, 60))

        state_color = GREEN if self.enabled else RED
        state_text = "PÅ" if self.enabled else "AV"

        line1 = self.small.render(f"Status: {state_text}", True, state_color)
        screen.blit(line1, (60, 120))

        line2 = self.small.render(f"Mode: {self.mode}", True, SOFT_WHITE)
        screen.blit(line2, (60, 160))

        line3 = self.small.render(f"Fade tid: {self.lifetime} ms", True, SOFT_WHITE)
        screen.blit(line3, (60, 200))

        help_lines = [
            "ENTER / SPACE = slå av/på",
            "M = växla fade / persistent",
            "UP / DOWN = ändra fade tid",
            "ESC = tillbaka"
        ]

        y = 260
        for line in help_lines:
            surf = self.tiny.render(line, True, SOFT_WHITE)
            screen.blit(surf, (60, y))
            y += 26