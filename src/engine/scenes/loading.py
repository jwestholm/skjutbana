from __future__ import annotations
import pygame

from config import SCREEN_WIDTH, SCREEN_HEIGHT, LOADING_SCREEN_PATH, HOSTAGE_MOVIE_PATH
from src.engine.scene import Scene, SceneSwitch
from src.engine.scenes.video import VideoScene
from src.engine.scenes.menu import MenuScene
from src.engine.scene import SceneSwitch

class LoadingScene(Scene):
    def __init__(self) -> None:
        self.bg = None
        self.font = None

    def on_enter(self) -> None:
        self.bg = pygame.image.load(str(LOADING_SCREEN_PATH)).convert()
        self.bg = pygame.transform.smoothscale(self.bg, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.font = pygame.font.Font(None, 28)

    def handle_event(self, event: pygame.event.Event) -> SceneSwitch | None:
        # Efter loading: valfri tangent (utom ESC som hanteras globalt) eller musklick => spela film
        if event.type == pygame.KEYDOWN:
            return SceneSwitch(MenuScene())
        if event.type == pygame.MOUSEBUTTONDOWN:
            return SceneSwitch(MenuScene())
        return None

    def render(self, screen: pygame.Surface) -> None:
        screen.blit(self.bg, (0, 0))
        hint = self.font.render("Tryck valfri tangent eller klicka för att starta filmtest", True, (255, 255, 255))
        screen.blit(hint, (20, SCREEN_HEIGHT - 40))