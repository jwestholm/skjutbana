from __future__ import annotations

import cv2
import numpy as np
import pygame

from src.engine.camera.hit_scanner import hit_scanner
from src.engine.settings import load_scanner_debug_overlay_enabled


WHITE = (235, 235, 235)
SOFT_WHITE = (200, 200, 200)
GREEN = (80, 255, 120)
YELLOW = (255, 220, 100)
CYAN = (100, 220, 255)
RED = (255, 90, 90)
PANEL_BG = (0, 0, 0, 170)
BORDER = (110, 110, 110)


class ScannerDebugOverlay:
    """
    Kompakt debug-overlay för träffscannern.

    Visar bara det viktigaste:
    - state/status
    - candidates/stable/known
    - warped board
    - binary mask

    Avsikt:
    - vara liten nog att kunna ligga kvar under test
    - ge snabb signal om varför detection fungerar eller inte
    """

    def __init__(self) -> None:
        self.font = None
        self.small = None
        self.tiny = None

    def _ensure_fonts(self) -> None:
        if self.font is None:
            self.font = pygame.font.Font(None, 24)
        if self.small is None:
            self.small = pygame.font.Font(None, 20)
        if self.tiny is None:
            self.tiny = pygame.font.Font(None, 17)

    def render(self, screen: pygame.Surface) -> None:
        if not load_scanner_debug_overlay_enabled():
            return

        self._ensure_fonts()
        snapshot = hit_scanner.get_debug_snapshot()

        panel_w = 420
        panel_h = 300
        margin = 14
        panel_x = screen.get_width() - panel_w - margin
        panel_y = margin

        overlay = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        overlay.fill(PANEL_BG)
        pygame.draw.rect(overlay, BORDER, overlay.get_rect(), 2)

        title = self.font.render("Scanner debug", True, WHITE)
        overlay.blit(title, (10, 8))

        state = snapshot.get("state", "OFF")
        last_status = str(snapshot.get("last_status", ""))
        if len(last_status) > 34:
            last_status = last_status[:31] + "..."

        status_lines = [
            f"State: {state}",
            f"Status: {last_status}",
            (
                f"Cand: {snapshot.get('candidates_count', 0)}   "
                f"Stable: {snapshot.get('stable_tracks_count', 0)}   "
                f"Known: {snapshot.get('known_holes_count', 0)}"
            ),
        ]

        y = 34
        for line in status_lines:
            surf = self.small.render(line, True, SOFT_WHITE)
            overlay.blit(surf, (10, y))
            y += 20

        warped_box = pygame.Rect(10, 98, 195, 150)
        mask_box = pygame.Rect(215, 98, 195, 150)

        self._draw_frame_panel(
            target=overlay,
            box=warped_box,
            title="Warped",
            frame=snapshot["debug_frames"].get("warped_gray"),
            overlays={
                "candidates": snapshot.get("candidates", []),
                "stable_tracks": snapshot.get("stable_tracks", []),
                "known_holes": snapshot.get("known_holes", []),
                "board_size": snapshot.get("board_size", (1, 1)),
            },
        )

        self._draw_frame_panel(
            target=overlay,
            box=mask_box,
            title="Mask",
            frame=snapshot["debug_frames"].get("mask"),
            overlays=None,
        )

        legend = "gul=kandidat  grön=stabil  cyan=känt hål"
        legend_surf = self.tiny.render(legend, True, SOFT_WHITE)
        overlay.blit(legend_surf, (10, 272))

        screen.blit(overlay, (panel_x, panel_y))

    def _draw_frame_panel(
        self,
        target: pygame.Surface,
        box: pygame.Rect,
        title: str,
        frame: np.ndarray | None,
        overlays: dict | None,
    ) -> None:
        pygame.draw.rect(target, (70, 70, 70), box, 1)

        title_surf = self.small.render(title, True, WHITE)
        target.blit(title_surf, (box.x + 6, box.y + 4))

        inner = pygame.Rect(box.x + 6, box.y + 24, box.w - 12, box.h - 30)
        pygame.draw.rect(target, (35, 35, 35), inner)

        if frame is None:
            txt = self.tiny.render("ingen data", True, RED)
            target.blit(txt, (inner.x + 8, inner.y + 8))
            return

        surface = self._numpy_to_surface(frame)
        if surface is None:
            txt = self.tiny.render("renderfel", True, RED)
            target.blit(txt, (inner.x + 8, inner.y + 8))
            return

        draw_rect = self._fit_rect(surface.get_size(), inner)
        scaled = pygame.transform.smoothscale(surface, draw_rect.size)
        target.blit(scaled, draw_rect.topleft)
        pygame.draw.rect(target, (120, 120, 120), draw_rect, 1)

        if overlays:
            self._draw_board_overlays(target, draw_rect, overlays)

    def _draw_board_overlays(
        self,
        target: pygame.Surface,
        draw_rect: pygame.Rect,
        overlays: dict,
    ) -> None:
        board_w, board_h = overlays.get("board_size", (1, 1))
        board_w = max(1, int(board_w))
        board_h = max(1, int(board_h))

        def map_point(x: float, y: float) -> tuple[int, int]:
            sx = draw_rect.x + int(round((x / board_w) * draw_rect.w))
            sy = draw_rect.y + int(round((y / board_h) * draw_rect.h))
            return sx, sy

        for hx, hy, _score in overlays.get("known_holes", []):
            px, py = map_point(hx, hy)
            pygame.draw.circle(target, CYAN, (px, py), 5, 1)

        for item in overlays.get("candidates", []):
            px, py = map_point(item["board_x"], item["board_y"])
            pygame.draw.circle(target, YELLOW, (px, py), 4, 1)

        for item in overlays.get("stable_tracks", []):
            px, py = map_point(item["board_x"], item["board_y"])
            pygame.draw.circle(target, GREEN, (px, py), 6, 2)

    def _numpy_to_surface(self, arr: np.ndarray) -> pygame.Surface | None:
        if arr is None:
            return None

        try:
            if arr.ndim == 2:
                rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.ndim == 3 and arr.shape[2] == 3:
                rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            else:
                return None

            rgb = np.transpose(rgb, (1, 0, 2))
            surf = pygame.surfarray.make_surface(rgb)
            return surf.convert()
        except Exception:
            return None

    def _fit_rect(self, src_size: tuple[int, int], dst_rect: pygame.Rect) -> pygame.Rect:
        src_w, src_h = src_size
        dst_w, dst_h = dst_rect.size

        if src_w <= 0 or src_h <= 0:
            return dst_rect.copy()

        scale = min(dst_w / src_w, dst_h / src_h)
        w = max(1, int(src_w * scale))
        h = max(1, int(src_h * scale))

        x = dst_rect.x + (dst_w - w) // 2
        y = dst_rect.y + (dst_h - h) // 2
        return pygame.Rect(x, y, w, h)


scanner_debug_overlay = ScannerDebugOverlay()