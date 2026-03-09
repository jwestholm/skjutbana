from __future__ import annotations

import pygame

from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import (
    load_visual_hits_enabled,
    load_visual_hits_mode,
    save_visual_hits_enabled,
    save_visual_hits_mode,
    load_scanner_debug_enabled,
    save_scanner_debug_enabled,
)
from src.engine.visual.hit_visualizer import hit_visualizer


WHITE = (240, 240, 240)
SOFT_WHITE = (210, 210, 210)
GREEN = (120, 255, 120)
RED = (255, 120, 120)
PANEL_BG = (0, 0, 0, 165)


class VisualHitsSettingsScene(Scene):
    """
    Inställningar för visuella träffar och scanner-debug.

    Kontroller:
    - ENTER / SPACE = slå av / på visuella träffar
    - M = byt läge fade / persistent
    - D = slå av / på scanner debug overlay
    - ESC = tillbaka
    """

    def __init__(self, bg_color=(0, 0, 0)) -> None:
        self.bg_color = bg_color
        self.font = None
        self.small = None
        self.tiny = None

        self.enabled = True
        self.mode = "fade"
        self.scanner_debug_enabled = False

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 42)
        self.small = pygame.font.Font(None, 28)
        self.tiny = pygame.font.Font(None, 24)

        self.enabled = load_visual_hits_enabled()
        self.mode = load_visual_hits_mode()
        self.scanner_debug_enabled = load_scanner_debug_enabled()

    def _go_back(self):
        from src.engine.scenes.menu import MenuScene
        return SceneSwitch(MenuScene())

    def _save(self) -> None:
        save_visual_hits_enabled(self.enabled)
        save_visual_hits_mode(self.mode)
        save_scanner_debug_enabled(self.scanner_debug_enabled)
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

        if event.key == pygame.K_d:
            self.scanner_debug_enabled = not self.scanner_debug_enabled
            self._save()
            return None

        return None

    def update(self, dt: float):
        return None

    def render(self, screen: pygame.Surface) -> None:
        screen.fill(self.bg_color)

        panel = pygame.Surface((980, 360), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (40, 40))

        title = self.font.render("Visuella träffar / scanner debug", True, WHITE)
        screen.blit(title, (60, 60))

        state_color = GREEN if self.enabled else RED
        state_text = "PÅ" if self.enabled else "AV"
        state = self.small.render(f"Visuella träffar: {state_text}", True, state_color)
        screen.blit(state, (60, 118))

        mode = self.small.render(f"Läge: {self.mode}", True, WHITE)
        screen.blit(mode, (60, 154))

        debug_color = GREEN if self.scanner_debug_enabled else RED
        debug_text = "PÅ" if self.scanner_debug_enabled else "AV"
        debug = self.small.render(f"Scanner debug overlay: {debug_text}", True, debug_color)
        screen.blit(debug, (60, 190))

        lines = [
            "ENTER / SPACE: slå av eller på visuella träffar",
            "M: växla mellan fade och persistent",
            "D: slå av eller på scanner debug overlay",
            "fade = träffmarkering tonar bort efter några sekunder",
            "persistent = träffmarkeringar ligger kvar tills du lämnar innehållet",
            "scanner debug overlay visar scanport crop, warped board, score och mask",
            "overlayn visas ovanpå innehållsscener som använder OverlayScene",
            "ESC: tillbaka",
        ]

        y = 238
        for line in lines:
            surf = self.tiny.render(line, True, SOFT_WHITE)
            screen.blit(surf, (60, y))
            y += 24