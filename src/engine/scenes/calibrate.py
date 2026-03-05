from __future__ import annotations
import pygame

from config import SCREEN_WIDTH, SCREEN_HEIGHT, LOADING_SCREEN_PATH
from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import load_viewport_rect, save_viewport_rect


class CalibrateViewportScene(Scene):
    """
    Justera grön 'safe area' (viewport).
    - Pilar flyttar
    - + / - ändrar storlek
    - ESC avbryter
    - ENTER sparar
    """

    def __init__(self) -> None:
        self.bg = None
        self.overlay = None
        self.font = None
        self.small = None

        self.original = None
        self.rect = None

        self.move_step = 10
        self.size_step = 20

    def on_enter(self) -> None:
        bg = pygame.image.load(str(LOADING_SCREEN_PATH)).convert()
        self.bg = pygame.transform.smoothscale(bg, (SCREEN_WIDTH, SCREEN_HEIGHT))

        self.overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self.overlay.fill((0, 0, 0, 140))

        self.font = pygame.font.Font(None, 44)
        self.small = pygame.font.Font(None, 26)

        self.original = load_viewport_rect()
        self.rect = self.original.copy()

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            # avbryt (spara ej)
            from src.engine.scenes.menu import MenuScene
            return SceneSwitch(MenuScene())

        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            save_viewport_rect(self.rect)
            from src.engine.scenes.menu import MenuScene
            return SceneSwitch(MenuScene())

        # flytta
        if event.key == pygame.K_LEFT:
            self.rect.x -= self.move_step
        elif event.key == pygame.K_RIGHT:
            self.rect.x += self.move_step
        elif event.key == pygame.K_UP:
            self.rect.y -= self.move_step
        elif event.key == pygame.K_DOWN:
            self.rect.y += self.move_step

        # storlek (+/-)
        # + är ofta K_EQUALS med shift, men keypad + funkar som K_KP_PLUS
        elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
            self.rect.w += self.size_step
            self.rect.h += self.size_step
        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.rect.w -= self.size_step
            self.rect.h -= self.size_step

        # clamp inom skärm
        if self.rect.w < 200:
            self.rect.w = 200
        if self.rect.h < 200:
            self.rect.h = 200
        if self.rect.x < 0:
            self.rect.x = 0
        if self.rect.y < 0:
            self.rect.y = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.x = SCREEN_WIDTH - self.rect.w
        if self.rect.bottom > SCREEN_HEIGHT:
            self.rect.y = SCREEN_HEIGHT - self.rect.h

        return None

    def render(self, screen: pygame.Surface) -> None:
        screen.blit(self.bg, (0, 0))
        screen.blit(self.overlay, (0, 0))

        title = self.font.render("Justera skjutfält / rityta", True, (240, 240, 240))
        screen.blit(title, (40, 30))

        hint = "Pilar: flytta  |  +/-: storlek  |  ENTER: spara  |  ESC: avbryt"
        h = self.small.render(hint, True, (200, 200, 200))
        screen.blit(h, (40, 80))

        # Rita ramen
        pygame.draw.rect(screen, (0, 255, 0), self.rect, 4)

        # Visa siffror
        info = f"x={self.rect.x} y={self.rect.y} w={self.rect.w} h={self.rect.h}"
        i = self.small.render(info, True, (180, 180, 180))
        screen.blit(i, (40, SCREEN_HEIGHT - 40))