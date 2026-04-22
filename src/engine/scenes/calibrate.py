"""
Viewport adjustment scene — manual only.

Lets the user move and resize the green viewport rectangle.
No AR markers, no homography, no camera calibration.
That belongs in calibrate_camera_viewport.py.
"""
from __future__ import annotations

import pygame

from config import LOADING_SCREEN_PATH, SCREEN_HEIGHT, SCREEN_WIDTH
from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import load_viewport_rect, save_viewport_rect


WHITE = (240, 240, 240)
SOFT_WHITE = (210, 210, 210)
GREEN = (120, 255, 120)
YELLOW = (255, 220, 80)
PANEL_BG = (0, 0, 0, 175)


class CalibrateViewportScene(Scene):
    """
    Manuell justering av viewport-rect (skjutgränser / rityta).

    Kontroller:
    - Pilar: flytta viewport
    - +/-: ändra storlek
    - ENTER: spara
    - R: återställ till sparat värde
    - ESC: tillbaka
    """

    wants_camera_preview = False
    wants_hit_scanning = False
    wants_mouse_simulated_hits = False

    def __init__(self) -> None:
        self.bg = None
        self.overlay = None
        self.font = None
        self.small = None
        self.tiny = None

        self.original_viewport: pygame.Rect | None = None
        self.rect: pygame.Rect | None = None
        self.move_step = 10
        self.size_step = 20
        self.status_message = ""

    def on_enter(self) -> None:
        bg = pygame.image.load(str(LOADING_SCREEN_PATH)).convert()
        self.bg = pygame.transform.smoothscale(bg, (SCREEN_WIDTH, SCREEN_HEIGHT))

        self.overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self.overlay.fill((0, 0, 0, 140))

        self.font = pygame.font.Font(None, 44)
        self.small = pygame.font.Font(None, 26)
        self.tiny = pygame.font.Font(None, 22)

        self.original_viewport = load_viewport_rect()
        self.rect = self.original_viewport.copy()
        self.status_message = "Flytta och skala den gröna ramen. ENTER sparar."

    def on_exit(self) -> None:
        pass

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            from src.engine.scenes.menu import MenuScene
            return SceneSwitch(MenuScene())

        if event.key == pygame.K_r:
            assert self.original_viewport is not None
            self.rect = self.original_viewport.copy()
            self.status_message = "Återställde viewport till sparat värde."
            return None

        assert self.rect is not None

        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            save_viewport_rect(self.rect)
            self.original_viewport = self.rect.copy()
            self.status_message = "Viewport sparad."
            return None

        if event.key == pygame.K_LEFT:
            self.rect.x -= self.move_step
        elif event.key == pygame.K_RIGHT:
            self.rect.x += self.move_step
        elif event.key == pygame.K_UP:
            self.rect.y -= self.move_step
        elif event.key == pygame.K_DOWN:
            self.rect.y += self.move_step
        elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
            self.rect.w += self.size_step
            self.rect.h += self.size_step
        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.rect.w -= self.size_step
            self.rect.h -= self.size_step
        else:
            return None

        self.rect.w = max(200, self.rect.w)
        self.rect.h = max(200, self.rect.h)
        self.rect.x = max(0, self.rect.x)
        self.rect.y = max(0, self.rect.y)
        if self.rect.right > SCREEN_WIDTH:
            self.rect.x = SCREEN_WIDTH - self.rect.w
        if self.rect.bottom > SCREEN_HEIGHT:
            self.rect.y = SCREEN_HEIGHT - self.rect.h
        return None

    def update(self, dt: float):
        del dt
        return None

    def render(self, screen: pygame.Surface) -> None:
        assert self.rect is not None
        screen.blit(self.bg, (0, 0))
        screen.blit(self.overlay, (0, 0))

        title = self.font.render("Justera skjutgränser / rityta", True, WHITE)
        screen.blit(title, (40, 28))

        hint = self.small.render(
            "Pilar = flytta | +/- = storlek | ENTER = spara | R = återställ | ESC = tillbaka",
            True,
            SOFT_WHITE,
        )
        screen.blit(hint, (40, 74))

        # Draw viewport rectangle
        pygame.draw.rect(screen, GREEN, self.rect, 4)

        # Center crosshair
        cx, cy = self.rect.centerx, self.rect.centery
        pygame.draw.line(screen, GREEN, (cx - 12, cy), (cx + 12, cy), 2)
        pygame.draw.line(screen, GREEN, (cx, cy - 12), (cx, cy + 12), 2)

        # Info
        info = self.small.render(
            f"x={self.rect.x}  y={self.rect.y}  w={self.rect.w}  h={self.rect.h}",
            True,
            WHITE,
        )
        screen.blit(info, (40, SCREEN_HEIGHT - 42))

        if self.status_message:
            status = self.tiny.render(self.status_message, True, YELLOW)
            screen.blit(status, (40, SCREEN_HEIGHT - 68))
