from __future__ import annotations
import pygame

from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from src.engine.scenes.loading import LoadingScene


class App:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Skjutbana")

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True

        self.scene = LoadingScene()
        self.scene.on_enter()

    def quit(self) -> None:
        self.running = False

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                    break

                # Global ESC = avsluta (du kan senare flytta detta till input-mapping)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
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

            self.scene.render(self.screen)
            pygame.display.flip()

        self.scene.on_exit()
        pygame.quit()

    def _switch_to(self, new_scene) -> None:
        self.scene.on_exit()
        self.scene = new_scene
        self.scene.on_enter()