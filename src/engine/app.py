from __future__ import annotations

import pygame

from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from src.engine.camera.camera_manager import camera_manager
from src.engine.camera.hit_scanner import hit_scanner
from src.engine.scenes.loading import LoadingScene


class App:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Skjutbana")

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True

        camera_manager.start()

        self.scene = LoadingScene()
        self.scene.on_enter()
        self._sync_runtime_services(force=True)

    def quit(self) -> None:
        self.running = False

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0

            camera_manager.update()

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

            self.scene.render(self.screen)
            pygame.display.flip()

        self.scene.on_exit()
        hit_scanner.disable()
        camera_manager.stop()
        pygame.quit()

    def _sync_runtime_services(self, force: bool = False) -> None:
        wants_scanning = bool(getattr(self.scene, "wants_hit_scanning", False))

        if wants_scanning:
            if force or not hit_scanner.enabled:
                hit_scanner.enable()
        else:
            hit_scanner.disable()

    def _switch_to(self, new_scene) -> None:
        self.scene.on_exit()
        self.scene = new_scene
        self.scene.on_enter()
        self._sync_runtime_services(force=True)