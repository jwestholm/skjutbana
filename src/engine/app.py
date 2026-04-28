from __future__ import annotations

import pygame

from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from src.engine.audio.audio_peak_detector import audio_peak_detector
from src.engine.camera.camera_manager import camera_manager
from src.engine.camera.hit_scanner import hit_scanner
from src.engine.output.led_service import led_service
from src.engine.output.led_types import LedConnectionConfig, RgbColor
from src.engine.scenes.loading import LoadingScene
from src.engine.settings import load_led_settings
from src.engine.visual.hit_visualizer import hit_visualizer


class App:
    def __init__(self) -> None:
        pygame.init()

        self.base_caption = "Skjutbana"
        pygame.display.set_caption(self.base_caption)

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True

        camera_manager.start()
        audio_peak_detector.start()

        self._init_led_runtime()

        self.scene = LoadingScene()
        self.scene.on_enter()

        self._sync_runtime_services(force=True)
        self._apply_scene_led()
        self._update_window_caption()

    def _init_led_runtime(self) -> None:
        led_data = load_led_settings()
        config = LedConnectionConfig(
            enabled=bool(led_data.get("enabled", False)),
            device_id=str(led_data.get("device_id", "")),
            ip_address=str(led_data.get("ip_address", "")),
            local_key=str(led_data.get("local_key", "")),
            version=float(led_data.get("version", 3.3)),
            default_mode=str(led_data.get("default_mode", "colour")),
            default_brightness=int(led_data.get("default_brightness", 1000)),
            default_temperature=int(led_data.get("default_temperature", 500)),
            default_colour=RgbColor(*led_data.get("default_colour", [255, 255, 255])),
        )

        try:
            led_service.configure(config)
            led_service.start()
        except Exception:
            pass

    def _apply_scene_led(self) -> None:
        try:
            scene_led_enabled = bool(getattr(self.scene, "scene_led_enabled", False))
            scene_led_color = tuple(getattr(self.scene, "scene_led_color", (255, 255, 255)))

            if scene_led_enabled:
                led_service.show_color(RgbColor(*scene_led_color))
            else:
                led_service.turn_off()
        except Exception:
            # LED ska aldrig kunna krascha appen
            pass

    def quit(self) -> None:
        self.running = False

    def run(self) -> None:
        try:
            self._main_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _main_loop(self) -> None:
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0

            camera_manager.update()
            audio_peak_detector.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                    break

                switch = self.scene.handle_event(event)
                if switch:
                    self._switch_to(switch.new_scene)
                    break

            if not self.running:
                break

            switch = self.scene.update(dt)
            if switch:
                self._switch_to(switch.new_scene)

            hit_scanner.update(dt)
            self._update_window_caption()

            self.scene.render(self.screen)
            hit_visualizer.update(dt)
            pygame.display.flip()

    def _shutdown(self) -> None:
        try:
            self.scene.on_exit()
        except Exception:
            pass

        try:
            hit_scanner.disable()
        except Exception:
            pass

        try:
            led_service.stop()
        except Exception:
            pass

        try:
            audio_peak_detector.stop()
        except Exception:
            pass

        try:
            camera_manager.stop()
        except Exception:
            pass

        try:
            pygame.quit()
        except Exception:
            pass

    def _sync_runtime_services(self, force: bool = False) -> None:
        wants_scanning = bool(getattr(self.scene, "wants_hit_scanning", False))

        if wants_scanning:
            if force or not hit_scanner.enabled:
                hit_scanner.enable()
        else:
            hit_scanner.disable()

    def _update_window_caption(self) -> None:
        if hit_scanner.enabled and hit_scanner.state != hit_scanner.STATE_OFF:
            caption = f"{self.base_caption} (Scanning)"
        else:
            caption = self.base_caption

        pygame.display.set_caption(caption)

    def _switch_to(self, new_scene) -> None:
        self.scene.on_exit()
        self.scene = new_scene
        self.scene.on_enter()
        self._sync_runtime_services(force=True)
        self._apply_scene_led()
        self._update_window_caption()