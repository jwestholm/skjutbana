from __future__ import annotations

import time

import pygame

from src.engine.ai.runtime import get_ai_runtime
from src.engine.scene import Scene, SceneSwitch

WHITE = (240, 240, 240)
SOFT_WHITE = (210, 210, 210)
GREEN = (120, 255, 120)
RED = (255, 120, 120)
YELLOW = (255, 220, 100)
PANEL_BG = (0, 0, 0, 170)


class AISettingsScene(Scene):
    wants_hit_scanning = False
    wants_camera_preview = False

    def __init__(self, bg_color=(0, 0, 0), **kwargs) -> None:
        super().__init__()
        self.bg_color = tuple(bg_color) if isinstance(bg_color, (list, tuple)) else (0, 0, 0)
        self.font = None
        self.small = None
        self.tiny = None
        self.runtime = get_ai_runtime()
        self.mode_index = 0
        self.status_message = ""

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 48)
        self.small = pygame.font.Font(None, 30)
        self.tiny = pygame.font.Font(None, 24)
        self.runtime = get_ai_runtime()
        mode = str(self.runtime.settings.get("mode", "train_only"))
        modes = self._mode_values()
        self.mode_index = modes.index(mode) if mode in modes else 1

    def on_exit(self) -> None:
        pass

    def _mode_values(self) -> list[str]:
        return ["off", "train_only", "advisory", "blended", "ai_priority", "ai_only"]

    def _go_back(self):
        from src.engine.scenes.menu import MenuScene
        return SceneSwitch(MenuScene())

    def _save(self) -> None:
        self.runtime.settings["mode"] = self._mode_values()[self.mode_index]
        self.runtime.save_settings()

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None
        if event.key == pygame.K_ESCAPE:
            self._save()
            return self._go_back()
        if event.key in (pygame.K_LEFT, pygame.K_a):
            self.runtime.settings["trust_percent"] = max(
                0, int(self.runtime.settings.get("trust_percent", 0)) - 5
            )
            self._save()
            self.status_message = f"AI-vikt: {self.runtime.settings['trust_percent']}%"
            return None
        if event.key in (pygame.K_RIGHT, pygame.K_d):
            self.runtime.settings["trust_percent"] = min(
                100, int(self.runtime.settings.get("trust_percent", 0)) + 5
            )
            self._save()
            self.status_message = f"AI-vikt: {self.runtime.settings['trust_percent']}%"
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
            val = float(self.runtime.settings.get("min_confidence", 0.58))
            self.runtime.settings["min_confidence"] = min(0.99, val + 0.02)
            self._save()
            return None
        if event.key == pygame.K_x:
            val = float(self.runtime.settings.get("min_confidence", 0.58))
            self.runtime.settings["min_confidence"] = max(0.05, val - 0.02)
            self._save()
            return None
        if event.key == pygame.K_o:
            self.runtime.settings["show_overlay"] = not bool(
                self.runtime.settings.get("show_overlay", True)
            )
            self._save()
            return None
        if event.key == pygame.K_l:
            self.runtime.settings["auto_learn"] = not bool(
                self.runtime.settings.get("auto_learn", True)
            )
            self._save()
            return None
        if event.key == pygame.K_r:
            self.runtime.memory.reset()
            self.status_message = "AI-modellen nollställd."
            return None
        if event.key == pygame.K_e:
            self._export_brain()
            return None
        if event.key == pygame.K_i:
            self._import_brain()
            return None
        return None

    def _export_brain(self) -> None:
        from pathlib import Path
        stamp = time.strftime("%Y%m%d_%H%M%S")
        export_dir = Path("content/ai/exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        path = export_dir / f"ai_brain_{stamp}.json"
        try:
            self.runtime.memory.export_brain(path)
            self.status_message = f"Exporterad: {path.name}"
        except Exception as exc:
            self.status_message = f"Export misslyckades: {exc}"

    def _import_brain(self) -> None:
        from pathlib import Path
        export_dir = Path("content/ai/exports")
        if not export_dir.exists():
            self.status_message = "Ingen exports-mapp hittad."
            return
        files = sorted(export_dir.glob("ai_brain_*.json"), reverse=True)
        if not files:
            self.status_message = "Inga exporterade hjärnor hittade."
            return
        # Import the latest export
        latest = files[0]
        local_updates = self.runtime.memory.stats.get("local_updates_since_import", 0)
        if local_updates > 0:
            # Show warning in status — in a real UI this would be a confirmation dialog
            self.status_message = (
                f"VARNING: {local_updates} lokala uppdateringar ersätts. "
                f"Tryck I igen för att bekräfta."
            )
            # Simple two-press confirmation: first press shows warning, second imports
            if not getattr(self, "_import_confirmed", False):
                self._import_confirmed = True
                return
        try:
            result = self.runtime.memory.import_brain(latest)
            self.status_message = (
                f"Importerad: {latest.name} "
                f"(+{result['imported_positive']} pos, +{result['imported_negative']} neg)"
            )
        except Exception as exc:
            self.status_message = f"Import misslyckades: {exc}"
        self._import_confirmed = False

    def update(self, dt: float):
        del dt
        return None

    def render(self, screen: pygame.Surface) -> None:
        screen.fill(self.bg_color)
        if self.font is None or self.small is None or self.tiny is None:
            return

        panel = pygame.Surface((1120, 660), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (40, 40))

        title = self.font.render("AI – Inställningar och status", True, WHITE)
        screen.blit(title, (60, 60))

        mode = self.runtime.settings.get("mode", "train_only")
        trust = int(self.runtime.settings.get("trust_percent", 0))
        min_conf = float(self.runtime.settings.get("min_confidence", 0.58))
        show_overlay = bool(self.runtime.settings.get("show_overlay", True))
        auto_learn = bool(self.runtime.settings.get("auto_learn", True))

        lines_left = [
            (f"Läge: {mode}", YELLOW),
            (f"AI-vikt i detektering: {trust}%", WHITE),
            (f"Min confidence: {min_conf:.2f}", WHITE),
            (f"Overlay: {'PÅ' if show_overlay else 'AV'}", GREEN if show_overlay else RED),
            (f"Auto-learn: {'PÅ' if auto_learn else 'AV'}", GREEN if auto_learn else RED),
        ]
        y = 130
        for text, color in lines_left:
            surf = self.small.render(text, True, color)
            screen.blit(surf, (60, y))
            y += 36

        # Model stats
        summary = self.runtime.memory.summary()
        stats_lines = [
            f"Positiva minnen: {summary['positive_count']}",
            f"Negativa minnen: {summary['negative_count']}",
            f"Totala klick: {summary['total_clicks']}",
            f"Lokala uppdateringar: {summary['local_updates_since_import']}",
            f"Feature-nycklar: {summary['feature_keys']}",
        ]
        last_updated = summary.get("last_updated")
        if last_updated:
            stats_lines.append(
                f"Senaste save: {time.strftime('%H:%M:%S', time.localtime(last_updated))}"
            )
        last_import = summary.get("last_import_ts")
        if last_import:
            stats_lines.append(
                f"Senaste import: {time.strftime('%Y-%m-%d %H:%M', time.localtime(last_import))}"
            )

        y = 350
        for line in stats_lines:
            surf = self.small.render(line, True, SOFT_WHITE)
            screen.blit(surf, (60, y))
            y += 32

        # Help
        help_lines = [
            "UP / DOWN: byt AI-läge",
            "LEFT / RIGHT: sänk / höj AI-vikt",
            "X / C: sänk / höj minsta confidence",
            "O: slå overlay av / på",
            "L: slå auto-learn av / på",
            "R: nollställ AI-modellen",
            "E: exportera AI-hjärna",
            "I: importera AI-hjärna",
            "ESC: tillbaka",
        ]
        y = 130
        for line in help_lines:
            surf = self.tiny.render(line, True, SOFT_WHITE)
            screen.blit(surf, (610, y))
            y += 30

        # Status message
        if self.status_message:
            status = self.small.render(self.status_message, True, GREEN)
            screen.blit(status, (60, 620))
