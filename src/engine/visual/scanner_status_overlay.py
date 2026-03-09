from __future__ import annotations

import pygame

from src.engine.audio.audio_peak_detector import audio_peak_detector
from src.engine.camera.hit_scanner import hit_scanner


WHITE = (240, 240, 240)
GREEN = (100, 255, 120)
RED = (255, 120, 120)
CYAN = (120, 220, 255)
PANEL_BG = (0, 0, 0, 160)


class ScannerStatusOverlay:
    def __init__(self):
        self.font = None

    def render(self, screen):
        if not hit_scanner.enabled:
            return

        if self.font is None:
            self.font = pygame.font.Font(None, 22)

        snap = hit_scanner.get_debug_snapshot()

        panel = pygame.Surface((340, 160), pygame.SRCALPHA)
        panel.fill(PANEL_BG)

        state = snap["state"]
        state_color = GREEN if state == "ACTIVE" else RED

        latest_audio = audio_peak_detector.get_latest_event()
        if latest_audio is None:
            audio_line = "Audio: no peak yet"
            audio_color = WHITE
        else:
            age = max(0.0, time.time() - latest_audio.timestamp)
            audio_line = f"Audio peak: {age:.2f}s ago  peak={latest_audio.peak:.2f}"
            audio_color = CYAN

        lines = [
            ("Scanner: " + state, state_color),
            (f"Cand: {snap['candidates_count']}", WHITE),
            (f"Stable: {snap['stable_tracks_count']}", WHITE),
            (f"Known: {snap['known_holes_count']}", WHITE),
            (audio_line, audio_color),
        ]

        y = 8
        for line, color in lines:
            surf = self.font.render(line, True, color)
            panel.blit(surf, (10, y))
            y += 24

        screen.blit(panel, (10, 10))


import time
scanner_status_overlay = ScannerStatusOverlay()