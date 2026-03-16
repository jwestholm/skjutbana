from __future__ import annotations

import time

import pygame

from src.engine.audio.audio_peak_detector import audio_peak_detector
from src.engine.camera.hit_scanner import hit_scanner
from src.engine.input.hit_input import hit_input
from src.engine.settings import (
    load_content_rect,
    load_scanport_rect,
    load_scanner_debug_overlay_enabled,
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
        panel_width = 760
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
        # Viktigt: detta ska styras av scanner debug overlay,
        # inte av audio status overlay.
        if not load_scanner_debug_overlay_enabled():
            return

        if not hit_scanner.enabled:
            return

        if self.font is None:
            self.font = pygame.font.Font(None, 22)

        snap = hit_scanner.get_debug_snapshot()
        viewport = load_viewport_rect()
        scanport = load_scanport_rect()
        content_rect = load_content_rect()

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
        lines.append(("VIEWPORT / SCANPORT / CONTENT", YELLOW))
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
        lines.append(
            (
                f"content rect: x={content_rect.x} y={content_rect.y} "
                f"w={content_rect.w} h={content_rect.h}",
                YELLOW,
            )
        )

        lines.append(("", WHITE))
        lines.append(("LAST CAMERA HIT", CYAN))

        cam = hit_input.last_camera_hit
        camera_ring_visible = False

        if cam is None:
            lines.append(("full camera: none", SOFT))
            lines.append(("scanport local: none", SOFT))
            lines.append(("screen(app): none", SOFT))
            lines.append(("viewport local: none", SOFT))
            lines.append(("content local: none", SOFT))
            lines.append(("content normalized: none", SOFT))
        else:
            full_camera_x = cam.camera_x
            full_camera_y = cam.camera_y

            lines.append(
                (
                    f"full camera: x={full_camera_x:.1f} y={full_camera_y:.1f}",
                    CYAN,
                )
            )

            if scanport is None:
                lines.append(("scanport local: scanport missing", RED))
                inside_scanport = False
            else:
                scanport_local_x = full_camera_x - scanport.x
                scanport_local_y = full_camera_y - scanport.y
                inside_scanport = (
                    0.0 <= scanport_local_x < float(scanport.w)
                    and 0.0 <= scanport_local_y < float(scanport.h)
                )
                scanport_color = CYAN if inside_scanport else ORANGE
                lines.append(
                    (
                        f"scanport local: x={scanport_local_x:.1f} y={scanport_local_y:.1f} "
                        f"in_scanport={self._fmt_bool(inside_scanport)}",
                        scanport_color,
                    )
                )
                if scanport.w > 0 and scanport.h > 0:
                    scan_norm_x = scanport_local_x / float(scanport.w)
                    scan_norm_y = scanport_local_y / float(scanport.h)
                    lines.append(
                        (
                            f"scanport normalized: x={scan_norm_x:.4f} y={scan_norm_y:.4f}",
                            SOFT,
                        )
                    )

            inside_viewport = (
                viewport.x <= cam.screen_x < (viewport.x + viewport.w)
                and viewport.y <= cam.screen_y < (viewport.y + viewport.h)
            )

            inside_content = (
                content_rect.x <= cam.screen_x < (content_rect.x + content_rect.w)
                and content_rect.y <= cam.screen_y < (content_rect.y + content_rect.h)
            )

            screen_color = CYAN if inside_viewport else RED
            content_color = CYAN if inside_content else RED

            lines.append(
                (
                    f"screen(app): x={cam.screen_x:.1f} y={cam.screen_y:.1f} "
                    f"in_viewport={self._fmt_bool(inside_viewport)}",
                    screen_color,
                )
            )
            lines.append(
                (
                    f"viewport local: x={cam.viewport_x:.1f} y={cam.viewport_y:.1f}",
                    SOFT,
                )
            )
            lines.append(
                (
                    f"content local: x={cam.content_x:.1f} y={cam.content_y:.1f} "
                    f"in_content={self._fmt_bool(inside_content)}",
                    content_color,
                )
            )
            lines.append(
                (
                    f"content normalized: x={cam.content_norm_x:.4f} y={cam.content_norm_y:.4f}",
                    SOFT,
                )
            )

            camera_ring_visible = (
                0 <= int(round(cam.screen_x)) < screen.get_width()
                and 0 <= int(round(cam.screen_y)) < screen.get_height()
            )
            lines.append(
                (
                    f"ring visible on display: {self._fmt_bool(camera_ring_visible)}",
                    GREEN if camera_ring_visible else RED,
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

        self._render_panel(screen, lines, 10, 10)

        if cam is not None and camera_ring_visible:
            x = int(round(cam.screen_x))
            y = int(round(cam.screen_y))

            pygame.draw.circle(screen, CYAN, (x, y), 20, 3)
            pygame.draw.line(screen, CYAN, (x - 30, y), (x + 30, y), 2)
            pygame.draw.line(screen, CYAN, (x, y - 30), (x, y + 30), 2)


scanner_status_overlay = ScannerStatusOverlay()