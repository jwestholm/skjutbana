from __future__ import annotations

import time

import pygame

from src.engine.ai.runtime import ai_runtime
from src.engine.ai.settings import load_ai_settings, save_ai_settings
from src.engine.scene import Scene, SceneSwitch

WHITE = (240, 240, 240)
SOFT_WHITE = (210, 210, 210)
GREEN = (120, 255, 120)
RED = (255, 120, 120)
YELLOW = (255, 220, 100)
PANEL_BG = (0, 0, 0, 170)


class AISettingsScene(Scene):
    def __init__(self, bg_color=(0, 0, 0)) -> None:
        self.bg_color = tuple(bg_color)
        self.font = None
        self.small = None
        self.tiny = None
        self.settings = load_ai_settings()
        self.mode_index = self._mode_values().index(self.settings.get("mode", "train_only"))

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 48)
        self.small = pygame.font.Font(None, 30)
        self.tiny = pygame.font.Font(None, 24)
        self.settings = load_ai_settings()
        self.mode_index = self._mode_values().index(self.settings.get("mode", "train_only"))

    def _mode_values(self) -> list[str]:
        return ["off", "train_only", "advisory", "blended", "ai_priority", "ai_only"]

    def _go_back(self):
        from src.engine.scenes.menu import MenuScene

        menu_state = getattr(self, "return_menu_state", None)
        return SceneSwitch(MenuScene(menu_state=menu_state))

    def _save(self) -> None:
        self.settings["mode"] = self._mode_values()[self.mode_index]
        save_ai_settings(self.settings)

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None
        if event.key == pygame.K_ESCAPE:
            self._save()
            return self._go_back()
        if event.key in (pygame.K_LEFT, pygame.K_a):
            self.settings["blend_percent"] = max(0.0, float(self.settings.get("blend_percent", 0.0)) - 5.0)
            self._save()
            return None
        if event.key in (pygame.K_RIGHT, pygame.K_d):
            self.settings["blend_percent"] = min(100.0, float(self.settings.get("blend_percent", 0.0)) + 5.0)
            self._save()
            return None
        if event.key == pygame.K_UP:
            self.mode_index = (self.mode_index - 1) % len(self._mode_values())
            self._save()
            return None
        if event.key == pygame.K_DOWN:
            self.mode_index = (self.mode_index + 1) % len(self._mode_values())
            self._save()
            return None
        if event.key == pygame.K_c:
            self.settings["min_confidence"] = min(0.99, float(self.settings.get("min_confidence", 0.58)) + 0.02)
            self._save()
            return None
        if event.key == pygame.K_x:
            self.settings["min_confidence"] = max(0.05, float(self.settings.get("min_confidence", 0.58)) - 0.02)
            self._save()
            return None
        if event.key == pygame.K_o:
            self.settings["show_overlay"] = not bool(self.settings.get("show_overlay", True))
            self._save()
            return None
        if event.key == pygame.K_l:
            self.settings["auto_learn"] = not bool(self.settings.get("auto_learn", True))
            self._save()
            return None
        if event.key == pygame.K_r:
            ai_runtime.model.reset()
            return None
        return None

    def update(self, dt: float):
        del dt
        self.settings = load_ai_settings()
        return None

    def render(self, screen: pygame.Surface) -> None:
        screen.fill(self.bg_color)
        panel = pygame.Surface((1120, 620), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (40, 40))

        title = self.font.render("AI – Inställningar och status", True, WHITE)
        screen.blit(title, (60, 60))

        mode = self.small.render(f"Läge: {self.settings.get('mode', 'train_only')}", True, YELLOW)
        blend = self.small.render(f"AI-vikt i detektering: {self.settings.get('blend_percent', 0.0):.0f}%", True, WHITE)
        conf = self.small.render(f"Min confidence: {self.settings.get('min_confidence', 0.58):.2f}", True, WHITE)
        overlay = self.small.render(f"Overlay: {'PÅ' if self.settings.get('show_overlay', True) else 'AV'}", True, GREEN if self.settings.get('show_overlay', True) else RED)
        auto_learn = self.small.render(f"Auto-learn: {'PÅ' if self.settings.get('auto_learn', True) else 'AV'}", True, GREEN if self.settings.get('auto_learn', True) else RED)
        screen.blit(mode, (60, 130))
        screen.blit(blend, (60, 170))
        screen.blit(conf, (60, 210))
        screen.blit(overlay, (60, 250))
        screen.blit(auto_learn, (60, 290))

        summary = ai_runtime.model.summary()
        stats = [
            f"Positiva minnen: {summary['positive_count']}",
            f"Negativa minnen: {summary['negative_count']}",
            f"Uppdateringar: {summary['total_updates']}",
            f"Senaste save: {time.strftime('%H:%M:%S', time.localtime(summary['last_saved_ts'])) if summary['last_saved_ts'] else '-'}",
        ]
        y = 350
        for line in stats:
            surf = self.small.render(line, True, SOFT_WHITE)
            screen.blit(surf, (60, y))
            y += 36

        help_lines = [
            "UP / DOWN: byt AI-läge",
            "LEFT / RIGHT: sänk / höj AI-vikt i träffdetekteringen",
            "X / C: sänk / höj minsta confidence",
            "O: slå overlay av / på",
            "L: slå auto-learn av / på",
            "R: nollställ AI-modellen",
            "ESC: tillbaka",
        ]
        y = 130
        for line in help_lines:
            surf = self.tiny.render(line, True, SOFT_WHITE)
            screen.blit(surf, (610, y))
            y += 34
