from __future__ import annotations

import pygame

from src.engine.output.led_service import led_service
from src.engine.output.led_types import LedConnectionConfig, RgbColor
from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import load_led_settings, save_led_settings


WHITE = (240, 240, 240)
SOFT_WHITE = (210, 210, 210)
GREEN = (120, 255, 120)
RED = (255, 120, 120)
YELLOW = (255, 220, 100)
BLUE = (100, 180, 255)
GRAY = (140, 140, 140)
PANEL_BG = (0, 0, 0, 170)


class LedSettingsScene(Scene):
    """
    LED-inställningar för Deltaco/Tuya-list.

    Kontroller:
    - UP / DOWN: välj rad
    - ENTER:
        * toggle på Enabled
        * börja/avsluta textredigering på textfält
        * kör test på test-rader
        * spara på Save-rad
    - BACKSPACE: radera vid textredigering
    - ESC:
        * avbryt redigering om textfält aktivt
        * annars spara + tillbaka till meny
    - LEFT / RIGHT: ändra version
    """

    def __init__(self, bg_color=(0, 0, 0)) -> None:
        self.bg_color = tuple(bg_color)
        self.font = None
        self.small = None
        self.tiny = None

        self.enabled = False
        self.device_id = ""
        self.ip_address = ""
        self.local_key = ""
        self.version = 3.3

        self.selected_index = 0
        self.editing_field: str | None = None
        self.status_text = ""
        self.status_color = SOFT_WHITE

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 42)
        self.small = pygame.font.Font(None, 28)
        self.tiny = pygame.font.Font(None, 22)

        data = load_led_settings()
        self.enabled = bool(data.get("enabled", False))
        self.device_id = str(data.get("device_id", ""))
        self.ip_address = str(data.get("ip_address", ""))
        self.local_key = str(data.get("local_key", ""))
        self.version = float(data.get("version", 3.3))

        self.selected_index = 0
        self.editing_field = None
        self.status_text = "Konfigurera LED-listen och testa färgerna."
        self.status_color = SOFT_WHITE

    def on_exit(self) -> None:
        pass

    # ---------------------------------------------------------
    # navigation
    # ---------------------------------------------------------

    def _rows(self) -> list[dict]:
        return [
            {"kind": "bool", "key": "enabled", "label": "LED aktiverad", "value": "PÅ" if self.enabled else "AV"},
            {"kind": "text", "key": "device_id", "label": "Device ID", "value": self.device_id},
            {"kind": "text", "key": "ip_address", "label": "IP-adress", "value": self.ip_address},
            {"kind": "text", "key": "local_key", "label": "Local key", "value": self._masked_key()},
            {"kind": "version", "key": "version", "label": "Version", "value": f"{self.version:.1f}"},
            {"kind": "action", "key": "test_blue", "label": "Test: blå", "value": ""},
            {"kind": "action", "key": "test_green", "label": "Test: grön", "value": ""},
            {"kind": "action", "key": "test_red", "label": "Test: röd", "value": ""},
            {"kind": "action", "key": "test_white", "label": "Test: vit", "value": ""},
            {"kind": "action", "key": "test_on", "label": "Test: på / default", "value": ""},
            {"kind": "action", "key": "test_off", "label": "Test: av", "value": ""},
            {"kind": "action", "key": "save_back", "label": "Spara och tillbaka", "value": ""},
        ]

    def _masked_key(self) -> str:
        if not self.local_key:
            return ""
        if len(self.local_key) <= 8:
            return "*" * len(self.local_key)
        return f"{self.local_key[:4]}{'*' * (len(self.local_key) - 8)}{self.local_key[-4:]}"

    def _raw_value_for_field(self, key: str) -> str:
        if key == "device_id":
            return self.device_id
        if key == "ip_address":
            return self.ip_address
        if key == "local_key":
            return self.local_key
        return ""

    def _set_field_value(self, key: str, value: str) -> None:
        if key == "device_id":
            self.device_id = value
        elif key == "ip_address":
            self.ip_address = value
        elif key == "local_key":
            self.local_key = value

    def _go_back(self):
        from src.engine.scenes.menu import MenuScene
        return SceneSwitch(MenuScene())

    # ---------------------------------------------------------
    # config + runtime
    # ---------------------------------------------------------

    def _build_config(self, force_enabled: bool | None = None) -> LedConnectionConfig:
        enabled = self.enabled if force_enabled is None else bool(force_enabled)
        return LedConnectionConfig(
            enabled=enabled,
            device_id=self.device_id.strip(),
            ip_address=self.ip_address.strip(),
            local_key=self.local_key.strip(),
            version=float(self.version),
            default_mode="white",
            default_brightness=700,
            default_temperature=450,
            default_colour=RgbColor(255, 255, 255),
        )

    def _save(self) -> None:
        save_led_settings(
            {
                "enabled": bool(self.enabled),
                "device_id": self.device_id.strip(),
                "ip_address": self.ip_address.strip(),
                "local_key": self.local_key.strip(),
                "version": float(self.version),
            }
        )
        led_service.reload(self._build_config())

    def _set_status(self, text: str, color=SOFT_WHITE) -> None:
        self.status_text = text
        self.status_color = color

    # ---------------------------------------------------------
    # tests
    # ---------------------------------------------------------

    def _run_test(self, action_key: str) -> None:
        config = self._build_config(force_enabled=True)

        if not config.is_configured():
            self._set_status("Fyll i Device ID, IP-adress och Local key först.", RED)
            return

        led_service.reload(config)

        if led_service.get_last_error():
            self._set_status(f"LED-fel: {led_service.get_last_error()}", RED)
            return

        if action_key == "test_blue":
            led_service.show_color(RgbColor(0, 0, 255))
            self._set_status("Skickade blå testfärg.", BLUE)
            return

        if action_key == "test_green":
            led_service.show_color(RgbColor(0, 255, 0))
            self._set_status("Skickade grön testfärg.", GREEN)
            return

        if action_key == "test_red":
            led_service.show_color(RgbColor(255, 0, 0))
            self._set_status("Skickade röd testfärg.", RED)
            return

        if action_key == "test_white":
            led_service.show_white(1000, 450)
            self._set_status("Skickade vit testfärg.", WHITE)
            return

        if action_key == "test_on":
            led_service.turn_on()
            led_service.restore_default()
            self._set_status("Skickade ON/default.", GREEN)
            return

        if action_key == "test_off":
            led_service.turn_off()
            self._set_status("Skickade OFF.", GRAY)
            return

    # ---------------------------------------------------------
    # input
    # ---------------------------------------------------------

    def handle_event(self, event: pygame.event.Event):
        rows = self._rows()

        if event.type == pygame.KEYDOWN and self.editing_field is not None:
            field_key = self.editing_field

            if event.key == pygame.K_ESCAPE:
                self.editing_field = None
                self._set_status("Redigering avbruten.", YELLOW)
                return None

            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.editing_field = None
                self._set_status(f"{field_key} uppdaterad.", GREEN)
                return None

            if event.key == pygame.K_BACKSPACE:
                self._set_field_value(field_key, self._raw_value_for_field(field_key)[:-1])
                return None

            if event.unicode and event.unicode.isprintable():
                self._set_field_value(field_key, self._raw_value_for_field(field_key) + event.unicode)
                return None

            return None

        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            self._save()
            return self._go_back()

        if event.key == pygame.K_UP:
            self.selected_index = (self.selected_index - 1) % len(rows)
            return None

        if event.key == pygame.K_DOWN:
            self.selected_index = (self.selected_index + 1) % len(rows)
            return None

        current = rows[self.selected_index]
        kind = current["kind"]
        key = current["key"]

        if kind == "version":
            if event.key == pygame.K_LEFT:
                self.version = max(3.1, round(self.version - 0.1, 1))
                self._set_status(f"Version satt till {self.version:.1f}", SOFT_WHITE)
                return None
            if event.key == pygame.K_RIGHT:
                self.version = min(3.5, round(self.version + 0.1, 1))
                self._set_status(f"Version satt till {self.version:.1f}", SOFT_WHITE)
                return None

        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if kind == "bool" and key == "enabled":
                self.enabled = not self.enabled
                self._set_status(f"LED {'aktiverad' if self.enabled else 'avstängd'} i config.", GREEN if self.enabled else YELLOW)
                return None

            if kind == "text":
                self.editing_field = key
                self._set_status(f"Redigerar {current['label']}. ENTER = klar, ESC = avbryt.", YELLOW)
                return None

            if kind == "action":
                if key == "save_back":
                    self._save()
                    return self._go_back()

                self._run_test(key)
                return None

        return None

    def update(self, dt: float):
        del dt
        return None

    # ---------------------------------------------------------
    # render
    # ---------------------------------------------------------

    def render(self, screen: pygame.Surface) -> None:
        screen.fill(self.bg_color)

        panel = pygame.Surface((screen.get_width() - 80, screen.get_height() - 80), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (40, 40))

        title = self.font.render("LED-list / Deltaco WiFi", True, WHITE)
        screen.blit(title, (60, 60))

        subtitle = self.tiny.render(
            "UP/DOWN: välj  ENTER: redigera/testa  LEFT/RIGHT: version  ESC: spara och tillbaka",
            True,
            SOFT_WHITE,
        )
        screen.blit(subtitle, (60, 98))

        rows = self._rows()
        y = 150

        for index, row in enumerate(rows):
            is_selected = index == self.selected_index
            prefix = "▶ " if is_selected else "  "
            color = WHITE if is_selected else SOFT_WHITE

            label = row["label"]
            value = row["value"]

            if row["kind"] in ("bool", "text", "version"):
                text = f"{prefix}{label}: {value}"
            else:
                text = f"{prefix}{label}"

            surf = self.small.render(text, True, color)
            screen.blit(surf, (70, y))
            y += 36

        info_y = screen.get_height() - 170

        mode_text = "TEXTREDIGERING AKTIV" if self.editing_field else "NAVIGERING"
        mode_color = YELLOW if self.editing_field else BLUE
        mode_surf = self.small.render(f"Läge: {mode_text}", True, mode_color)
        screen.blit(mode_surf, (60, info_y))

        runtime = "ANSLUTEN" if led_service.is_available() else "EJ ANSLUTEN"
        runtime_color = GREEN if led_service.is_available() else GRAY
        runtime_surf = self.small.render(f"Runtime LED: {runtime}", True, runtime_color)
        screen.blit(runtime_surf, (60, info_y + 34))

        status_surf = self.tiny.render(self.status_text, True, self.status_color)
        screen.blit(status_surf, (60, info_y + 72))

        help_lines = [
            "Tips:",
            "- Device ID / Local key hämtas från Tuya/TinyTuya-setup.",
            "- IP-adressen ska vara LED-kontrollerns lokala LAN-adress.",
            "- Testknapparna använder den inmatade konfigurationen direkt.",
        ]

        y = info_y + 108
        for line in help_lines:
            surf = self.tiny.render(line, True, SOFT_WHITE)
            screen.blit(surf, (60, y))
            y += 22