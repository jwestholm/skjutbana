from __future__ import annotations
import pygame

from config import SCREEN_WIDTH, SCREEN_HEIGHT
from src.engine.scene import Scene, SceneSwitch
from src.engine.scenes.menu import MenuScene


class ImageScene(Scene):
    def __init__(self, image_path: str) -> None:
        self.image_path = image_path
        self.image = None

    def on_enter(self) -> None:
        img = pygame.image.load(self.image_path).convert()
        self.image = pygame.transform.smoothscale(img, (SCREEN_WIDTH, SCREEN_HEIGHT))

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return SceneSwitch(MenuScene())
        return None

    def render(self, screen: pygame.Surface) -> None:
        screen.blit(self.image, (0, 0))