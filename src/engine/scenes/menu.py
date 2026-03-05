from __future__ import annotations
import pygame

from config import SCREEN_WIDTH, SCREEN_HEIGHT, HOSTAGE_MOVIE_PATH
from src.engine.scene import Scene, SceneSwitch


class MenuScene(Scene):
    def __init__(self) -> None:
        self.font = None
        self.big = None
        self.items = []
        self.index = 0

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 32)
        self.big = pygame.font.Font(None, 56)

        # Börja hårdkodat (enkelt). Sen flyttar vi till JSON.
        self.items = [("Hostage (video test)", self.start_video),("Måltavla 1 (bild)", self.start_image)]
        self.index = 0

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                # ESC i menyn = avsluta spelet
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                return None

            if event.key in (pygame.K_UP, pygame.K_w):
                self.index = (self.index - 1) % len(self.items)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.index = (self.index + 1) % len(self.items)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                _, make_scene = self.items[self.index]
                return SceneSwitch(make_scene())

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Klick = starta vald
            _, make_scene = self.items[self.index]
            return SceneSwitch(make_scene())

        return None

    def render(self, screen: pygame.Surface) -> None:
        screen.fill((10, 10, 10))

        title = self.big.render("Skjutbana", True, (240, 240, 240))
        screen.blit(title, (40, 40))

        y = 140
        for i, (label, _) in enumerate(self.items):
            selected = (i == self.index)
            prefix = "> " if selected else "  "
            color = (255, 255, 255) if selected else (180, 180, 180)
            text = self.font.render(prefix + label, True, color)
            screen.blit(text, (60, y))
            y += 40

        hint = self.font.render("UP/DOWN + ENTER startar | SPACE pause i video | ESC avslutar", True, (140, 140, 140))
        screen.blit(hint, (40, SCREEN_HEIGHT - 50))
        
    def start_video(self):
        from src.engine.scenes.video import VideoScene
        from config import HOSTAGE_MOVIE_PATH
        return VideoScene(str(HOSTAGE_MOVIE_PATH))
    def start_image(self):
        from src.engine.scenes.image import ImageScene
        return ImageScene("assets/targets/target1.png")