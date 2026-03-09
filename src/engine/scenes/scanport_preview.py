from __future__ import annotations

import cv2
import numpy as np
import pygame

from src.engine.camera.camera_manager import camera_manager
from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import load_scanport_rect


WHITE = (240, 240, 240)
SOFT_WHITE = (210, 210, 210)
PANEL_BG = (0, 0, 0, 150)


class ScanportPreview(Scene):
    """
    Visar exakt det scannern ser:
    - rå scanport cropad från kamerabilden
    - uppskalad till hela skärmen

    ESC = tillbaka till menyn
    """

    wants_camera_preview = True

    def __init__(self, bg_color=(0, 0, 0)) -> None:
        self.bg_color = tuple(bg_color)
        self.font = None
        self.small = None

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 36)
        self.small = pygame.font.Font(None, 26)
        camera_manager.start()

    def on_exit(self) -> None:
        pass

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            from src.engine.scenes.menu import MenuScene
            return SceneSwitch(MenuScene())
        return None

    def update(self, dt: float):
        del dt
        return None

    def render(self, screen: pygame.Surface) -> None:
        screen.fill(self.bg_color)

        frame_bgr = camera_manager.get_latest_frame()
        scanport = load_scanport_rect()

        if frame_bgr is not None and scanport is not None:
            frame_h, frame_w = frame_bgr.shape[:2]

            x = max(0, int(scanport.x))
            y = max(0, int(scanport.y))
            w = max(1, int(scanport.w))
            h = max(1, int(scanport.h))

            if x < frame_w and y < frame_h:
                w = min(w, frame_w - x)
                h = min(h, frame_h - y)

                crop = frame_bgr[y:y + h, x:x + w]

                if crop.size > 0:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_rgb = np.transpose(crop_rgb, (1, 0, 2))
                    surf = pygame.surfarray.make_surface(crop_rgb).convert()
                    surf = pygame.transform.smoothscale(surf, screen.get_size())
                    screen.blit(surf, (0, 0))

        panel = pygame.Surface((820, 90), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (20, 20))

        title = self.font.render("Kolla Scanport", True, WHITE)
        screen.blit(title, (35, 30))

        status_lines = []
        status_lines.extend(camera_manager.get_status_lines())

        if scanport is not None:
            status_lines.append(
                f"Scanport: x={scanport.x} y={scanport.y} w={scanport.w} h={scanport.h}"
            )

        y = 60
        for line in status_lines[:2]:
            txt = self.small.render(line, True, SOFT_WHITE)
            screen.blit(txt, (240, y))
            y += 24

        hint = self.small.render("ESC: tillbaka", True, SOFT_WHITE)
        screen.blit(hint, (35, 62))