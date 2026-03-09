from __future__ import annotations

import cv2
import numpy as np
import pygame

from config import SCREEN_WIDTH, SCREEN_HEIGHT
from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import (
    load_scanport_norm,
    load_viewport_rect,
    save_scanport_norm,
    scanport_norm_to_screen_rect,
    screen_rect_to_scanport_norm,
)


ORANGE = (255, 140, 0)
GREEN = (0, 255, 0)
WHITE = (240, 240, 240)
SOFT_WHITE = (210, 210, 210)
RED = (255, 120, 120)
PANEL_BG = (0, 0, 0, 165)


def _draw_dashed_line(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    width: int = 2,
    dash: int = 12,
    gap: int = 8,
) -> None:
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1
    length = max(1, int((dx * dx + dy * dy) ** 0.5))

    for i in range(0, length, dash + gap):
        a = i / length
        b = min(i + dash, length) / length

        sx = int(x1 + dx * a)
        sy = int(y1 + dy * a)
        ex = int(x1 + dx * b)
        ey = int(y1 + dy * b)

        pygame.draw.line(surface, color, (sx, sy), (ex, ey), width)


def _draw_dashed_rect(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    rect: pygame.Rect,
    width: int = 2,
    dash: int = 12,
    gap: int = 8,
) -> None:
    _draw_dashed_line(surface, color, rect.topleft, rect.topright, width, dash, gap)
    _draw_dashed_line(surface, color, rect.topright, rect.bottomright, width, dash, gap)
    _draw_dashed_line(surface, color, rect.bottomright, rect.bottomleft, width, dash, gap)
    _draw_dashed_line(surface, color, rect.bottomleft, rect.topleft, width, dash, gap)


def _clamp_rect_to_screen(rect: pygame.Rect) -> pygame.Rect:
    min_w, min_h = 100, 100
    rect.w = max(min_w, min(rect.w, SCREEN_WIDTH))
    rect.h = max(min_h, min(rect.h, SCREEN_HEIGHT))
    rect.x = max(0, min(rect.x, SCREEN_WIDTH - rect.w))
    rect.y = max(0, min(rect.y, SCREEN_HEIGHT - rect.h))
    return rect


class CameraTestScene(Scene):
    """
    Scanport-kalibrering / kameratest.

    Kontroller:
    - Pilar = flytta
    - + / - = skala proportionellt
    - SHIFT+B = öka bredd
    - B = minska bredd
    - SHIFT+H = öka höjd
    - H = minska höjd
    - ENTER = spara
    - ESC = tillbaka
    """

    def __init__(self, camera_index: int = 0, bg_color=(0, 0, 0)) -> None:
        self.camera_index = camera_index
        self.bg_color = tuple(bg_color)

        self.cap: cv2.VideoCapture | None = None
        self.last_frame: pygame.Surface | None = None
        self.error_message: str | None = None

        self.font = None
        self.small = None
        self.tiny = None

        self.viewport = None
        self.original_rect = None
        self.rect = None

        self.move_step = 10
        self.size_step = 20

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 42)
        self.small = pygame.font.Font(None, 26)
        self.tiny = pygame.font.Font(None, 24)

        self.error_message = None
        self.last_frame = None

        self.viewport = load_viewport_rect()

        scanport = load_scanport_norm()
        self.original_rect = scanport_norm_to_screen_rect(scanport)
        self.rect = self.original_rect.copy()

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap or not self.cap.isOpened():
            self.error_message = f"Kunde inte öppna kamera index {self.camera_index}"
            if self.cap:
                self.cap.release()
            self.cap = None
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def on_exit(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None

    def _go_back(self):
        from src.engine.scenes.menu import MenuScene
        return SceneSwitch(MenuScene())

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            return self._go_back()

        if self.rect is None:
            return None

        mods = pygame.key.get_mods()

        if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
            x, y, w, h = screen_rect_to_scanport_norm(self.rect)
            save_scanport_norm(x, y, w, h)
            return self._go_back()

        if event.key == pygame.K_LEFT:
            self.rect.x -= self.move_step
        elif event.key == pygame.K_RIGHT:
            self.rect.x += self.move_step
        elif event.key == pygame.K_UP:
            self.rect.y -= self.move_step
        elif event.key == pygame.K_DOWN:
            self.rect.y += self.move_step

        elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
            self.rect.inflate_ip(self.size_step, self.size_step)
            self.rect.center = self.original_rect.center if mods & pygame.KMOD_CTRL else self.rect.center

        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.rect.inflate_ip(-self.size_step, -self.size_step)

        elif event.key == pygame.K_b:
            if mods & pygame.KMOD_SHIFT:
                self.rect.w += self.size_step
            else:
                self.rect.w -= self.size_step

        elif event.key == pygame.K_h:
            if mods & pygame.KMOD_SHIFT:
                self.rect.h += self.size_step
            else:
                self.rect.h -= self.size_step

        self.rect = _clamp_rect_to_screen(self.rect)
        return None

    def update(self, dt: float):
        if not self.cap:
            return None

        ok, frame_bgr = self.cap.read()
        if not ok:
            self.error_message = "Kunde inte läsa bild från kameran."
            return None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_LINEAR)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
        self.last_frame = pygame.surfarray.make_surface(frame_rgb).convert()
        return None

    def _draw_overlay(self, screen: pygame.Surface) -> None:
        assert self.rect is not None
        assert self.viewport is not None

        shade = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 120))
        pygame.draw.rect(shade, (0, 0, 0, 0), self.rect)
        screen.blit(shade, (0, 0))

        pygame.draw.rect(screen, ORANGE, self.rect, width=4)
        _draw_dashed_rect(screen, GREEN, self.viewport, width=2, dash=12, gap=8)

    def _draw_info_panel(self, screen: pygame.Surface) -> None:
        panel = pygame.Surface((760, 215), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (20, 20))

        lines = [
            ("Justera scanport", self.font, WHITE),
            ("Orange ram = scanport som analyseras", self.small, WHITE),
            ("Grön streckad ram = viewport / projicerad spelyta", self.small, WHITE),
            ("Pilar: flytta", self.tiny, SOFT_WHITE),
            ("+ / - : öka eller minska proportionellt", self.tiny, SOFT_WHITE),
            ("SHIFT+B / B : öka eller minska bredd", self.tiny, SOFT_WHITE),
            ("SHIFT+H / H : öka eller minska höjd", self.tiny, SOFT_WHITE),
            ("ENTER: spara    ESC: tillbaka", self.tiny, SOFT_WHITE),
        ]

        y = 32
        for text, font, color in lines:
            surf = font.render(text, True, color)
            screen.blit(surf, (36, y))
            y += surf.get_height() + 6

    def _draw_status(self, screen: pygame.Surface) -> None:
        if self.error_message:
            msg = self.small.render(self.error_message, True, RED)
            screen.blit(msg, (24, SCREEN_HEIGHT - 36))
            return

        if self.rect is not None:
            txt = (
                f"Scanport px: x={self.rect.x} y={self.rect.y} "
                f"w={self.rect.w} h={self.rect.h}"
            )
            surf = self.tiny.render(txt, True, WHITE)
            screen.blit(surf, (24, SCREEN_HEIGHT - 36))

    def render(self, screen: pygame.Surface) -> None:
        screen.fill(self.bg_color)

        if self.last_frame is not None:
            screen.blit(self.last_frame, (0, 0))

        if self.rect is not None and self.viewport is not None:
            self._draw_overlay(screen)

        self._draw_info_panel(screen)
        self._draw_status(screen)