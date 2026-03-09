import pygame
from src.engine.camera.hit_scanner import hit_scanner


WHITE = (240,240,240)
GREEN = (100,255,120)
RED = (255,120,120)
PANEL_BG = (0,0,0,160)


class ScannerStatusOverlay:

    def __init__(self):
        self.font = None

    def render(self, screen):

        if not hit_scanner.enabled:
            return

        if self.font is None:
            self.font = pygame.font.Font(None, 22)

        snap = hit_scanner.get_debug_snapshot()

        panel = pygame.Surface((240,110), pygame.SRCALPHA)
        panel.fill(PANEL_BG)

        state = snap["state"]
        color = GREEN if state == "ACTIVE" else RED

        lines = [
            f"Scanner: {state}",
            f"Cand: {snap['candidates_count']}",
            f"Stable: {snap['stable_tracks_count']}",
            f"Known: {snap['known_holes_count']}"
        ]

        y = 8

        for i,line in enumerate(lines):

            if i == 0:
                surf = self.font.render(line, True, color)
            else:
                surf = self.font.render(line, True, WHITE)

            panel.blit(surf,(10,y))
            y += 24

        screen.blit(panel,(10,10))


# DENNA RAD SAKNADES
scanner_status_overlay = ScannerStatusOverlay()