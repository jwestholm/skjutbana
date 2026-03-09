from __future__ import annotations

import pygame

from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import (
    load_scanner_debug_overlay_enabled,
    save_scanner_debug_overlay_enabled,
)

WHITE = (240, 240, 240)
SOFT_WHITE = (210, 210, 210)
GREEN = (120, 255, 120)
RED = (255, 120, 120)
PANEL_BG = (0, 0, 0, 165)


class ScannerDebugSettingsScene(Scene):
    """
    Inställningar för scanner-debug overlay.

    Kontroller:
    - ENTER / SPACE = slå av / på
    - ESC = tillbaka
    """

    def __init__(self, bg_color=(0, 0, 0)) -> None:
        self.bg_color = bg_color
        self.font = None
        self.small = None
        self.tiny = None
        self.enabled = False

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 42)
        self.small = pygame.font.Font(None, 28)
        self.tiny = pygame.font.Font(None, 24)
        self.enabled = load_scanner_debug_overlay_enabled()

    def _go_back(self):
        from src.engine.scenes.menu import MenuScene
        return SceneSwitch(MenuScene())

    def _save(self) -> None:
        save_scanner_debug_overlay_enabled(self.enabled)

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

        return None

    def update(self, dt: float):
        return None

    def render(self, screen: pygame.Surface) -> None:
        screen.fill(self.bg_color)

        panel = pygame.Surface((980, 310), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (40, 40))

        title = self.font.render("Scanner debug overlay", True, WHITE)
        screen.blit(title, (60, 60))

        state_color = GREEN if self.enabled else RED
        state_text = "PÅ" if self.enabled else "AV"
        state = self.small.render(f"Status: {state_text}", True, state_color)
        screen.blit(state, (60, 115))

        lines = [
            "ENTER / SPACE: slå av eller på scanner-debug",
            "När den är på visas scannerstatus och debugpaneler ovanpå innehållsscener.",
            "Den visar scanport crop, warped board, score map och binär mask.",
            "Dessutom ritas kandidater, stabila tracks och redan registrerade hål ut.",
            "ESC: tillbaka",
        ]

        y = 165
        for line in lines:
            surf = self.tiny.render(line, True, SOFT_WHITE)
            screen.blit(surf, (60, y))
            y += 26