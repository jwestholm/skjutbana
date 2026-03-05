from __future__ import annotations
import pygame

from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import load_viewport_rect


class ImageScene(Scene):
    def __init__(self, image_path: str) -> None:
        self.image_path = image_path
        self.viewport = None
        self.image = None

    def on_enter(self) -> None:
        self.viewport = load_viewport_rect()
        img = pygame.image.load(self.image_path).convert()
        self.image = pygame.transform.smoothscale(img, (self.viewport.w, self.viewport.h))

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            from src.engine.scenes.menu import MenuScene
            return SceneSwitch(MenuScene())
        return None

    def render(self, screen: pygame.Surface) -> None:
        # fyll resten (utanför rutan) svart så inget hamnar “i förbjudet område”
        screen.fill((0, 0, 0))
        screen.blit(self.image, (self.viewport.x, self.viewport.y))