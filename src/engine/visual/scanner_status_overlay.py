from __future__ import annotations

import time

import pygame

from src.engine.audio.audio_peak_detector import audio_peak_detector
from src.engine.camera.hit_scanner import hit_scanner
from src.engine.input.hit_input import hit_input
from src.engine.settings import load_audio_status_overlay_enabled

WHITE = (240, 240, 240)
GREEN = (100, 255, 120)
RED = (255, 120, 120)
CYAN = (0, 200, 255)
YELLOW = (255, 220, 80)
SOFT = (190, 190, 190)
PANEL_BG = (0, 0, 0, 160)


class ScannerStatusOverlay:
    def __init__(self):
        self.font = None

    def render(self, screen):
        if not load_audio_status_overlay_enabled():
            return

        if not hit_scanner.enabled:
            return

        if self.font is None:
            self.font = pygame.font.Font(None, 22)

        snap = hit_scanner.get_debug_snapshot()

        state = snap["state"]
        state_color = GREEN if state == "ACTIVE" else RED

        latest_audio = audio_peak_detector.get_latest_event()
        if latest_audio is None:
            audio_line = "Audio: no peak yet"
            audio_color = WHITE
        else:
            age = max(0.0, time.time() - latest_audio.timestamp)
            audio_line = (
                f"Audio peak: {age:.2f}s ago "
                f"peak={latest_audio.peak:.2f} rms={latest_audio.rms:.2f}"
            )
            audio_color = CYAN

        lines: list[tuple[str, tuple[int, int, int]]] = [
            ("Scanner: " + state, state_color),
            (f"Cand: {snap['candidates_count']}", WHITE),
            (f"Stable: {snap['stable_tracks_count']}", WHITE),
            (f"Known: {snap['known_holes_count']}", WHITE),
            (audio_line, audio_color),
        ]

        window_debug = snap.get("window_debug") or {}
        if window_debug:
            pre_count = int(window_debug.get("pre_count", 0))
            post_count = int(window_debug.get("post_count", 0))
            lines.append((f"Window pre/post: {pre_count}/{post_count}", SOFT))

        lines.append(("", WHITE))
        lines.append(("LAST CAMERA HIT", CYAN))

        cam = hit_input.last_camera_hit
        if cam is None:
            lines.append(("camera(scanport): none", SOFT))
            lines.append(("screen(viewport): none", SOFT))
            lines.append(("game(content): none", SOFT))
        else:
            lines.append(
                (
                    f"camera(scanport): x={cam.camera_x:.1f} y={cam.camera_y:.1f}",
                    CYAN,
                )
            )
            lines.append(
                (
                    f"screen(viewport): x={cam.screen_x:.1f} y={cam.screen_y:.1f}",
                    CYAN,
                )
            )
            lines.append(
                (
                    f"game(content): x={cam.game_x:.1f} y={cam.game_y:.1f}",
                    CYAN,
                )
            )

        lines.append(("", WHITE))
        lines.append(("LAST HIT EVENT", YELLOW))

        last_hit = hit_input.last_hit
        if last_hit is None:
            lines.append(("source: none", SOFT))
        else:
            age = max(0.0, time.time() - last_hit.timestamp)
            lines.append((f"source: {last_hit.source}", YELLOW))
            lines.append((f"age: {age:.2f}s", SOFT))

        panel_width = 470
        panel_height = len(lines) * 22 + 18

        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill(PANEL_BG)

        y = 8
        for line, color in lines:
            surf = self.font.render(line, True, color)
            panel.blit(surf, (10, y))
            y += 22

        screen.blit(panel, (10, 10))

        if cam is not None:
            x = int(round(cam.screen_x))
            y = int(round(cam.screen_y))

            pygame.draw.circle(screen, CYAN, (x, y), 20, 3)
            pygame.draw.line(screen, CYAN, (x - 30, y), (x + 30, y), 2)
            pygame.draw.line(screen, CYAN, (x, y - 30), (x, y + 30), 2)


scanner_status_overlay = ScannerStatusOverlay()