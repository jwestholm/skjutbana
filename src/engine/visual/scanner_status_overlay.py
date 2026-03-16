from __future__ import annotations

import time

import pygame

from src.engine.audio.audio_peak_detector import audio_peak_detector
from src.engine.camera.hit_scanner import hit_scanner
from src.engine.input.hit_input import hit_input
from src.engine.settings import (
    load_scanner_debug_overlay_enabled,
    load_scanport_rect,
    load_viewport_rect,
)

WHITE = (240, 240, 240)
GREEN = (100, 255, 120)
RED = (255, 120, 120)
CYAN = (0, 200, 255)
YELLOW = (255, 220, 80)
SOFT = (190, 190, 190)
ORANGE = (255, 170, 80)
PANEL_BG = (0, 0, 0, 160)


class ScannerStatusOverlay:
    def __init__(self):
        self.font = None

    def _fmt_bool(self, value: bool) -> str:
        return "yes" if value else "no"

    def _render_panel(self, screen, lines, panel_x=10, panel_y=10):
        panel_width = 700
        panel_height = len(lines) * 22 + 18

        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill(PANEL_BG)

        y = 8
        for line, color in lines:
            surf = self.font.render(line, True, color)
            panel.blit(surf, (10, y))
            y += 22

        screen.blit(panel, (panel_x, panel_y))

    def render(self, screen):
        # 🔴 viktigt: rita inget om debug overlay är av
        if not load_scanner_debug_overlay_enabled():
            return

        if not hit_scanner.enabled:
            return

        if self.font is None:
            self.font = pygame.font.Font(None, 22)

        snap = hit_scanner.get_debug_snapshot()
        viewport = load_viewport_rect()
        scanport = load_scanport_rect()

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

        lines.append(("", WHITE))
        lines.append(("VIEWPORT / SCANPORT", YELLOW))
        lines.append(
            (
                f"viewport: x={viewport.x} y={viewport.y} w={viewport.w} h={viewport.h}",
                YELLOW,
            )
        )

        if scanport is None:
            lines.append(("scanport: none", RED))
        else:
            lines.append(
                (
                    f"scanport(full camera): x={scanport.x} y={scanport.y} "
                    f"w={scanport.w} h={scanport.h}",
                    YELLOW,
                )
            )

        lines.append(("", WHITE))
        lines.append(("LAST CAMERA HIT", CYAN))

        cam = hit_input.last_camera_hit

        if cam is None:
            lines.append(("full camera: none", SOFT))
            lines.append(("screen(viewport): none", SOFT))
            lines.append(("game(content): none", SOFT))
        else:
            lines.append(
                (
                    f"full camera: x={cam.camera_x:.1f} y={cam.camera_y:.1f}",
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

        self._render_panel(screen, lines)

        # Rita blå ring på senaste camera hit
        if cam is not None:
            x = int(round(cam.screen_x))
            y = int(round(cam.screen_y))

            if 0 <= x < screen.get_width() and 0 <= y < screen.get_height():
                pygame.draw.circle(screen, CYAN, (x, y), 20, 3)
                pygame.draw.line(screen, CYAN, (x - 30, y), (x + 30, y), 2)
                pygame.draw.line(screen, CYAN, (x, y - 30), (x, y + 30), 2)


scanner_status_overlay = ScannerStatusOverlay()