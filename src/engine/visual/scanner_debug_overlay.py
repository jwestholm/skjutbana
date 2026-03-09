from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pygame

from src.engine.camera.hit_scanner import hit_scanner
from src.engine.settings import load_scanner_debug_overlay_enabled


WHITE = (240, 240, 240)
SOFT_WHITE = (205, 205, 205)
GREEN = (80, 255, 120)
YELLOW = (255, 220, 100)
RED = (255, 90, 90)
CYAN = (100, 220, 255)
PANEL_BG = (0, 0, 0, 180)
BORDER = (120, 120, 120)


@dataclass
class PanelBox:
    x: int
    y: int
    w: int
    h: int


class ScannerDebugOverlay:
    """
    Visar vad träffscannern faktiskt analyserar:

    - crop_bgr
    - warped_gray
    - score
    - mask
    - kandidater / stabila tracks / kända hål
    - scanner-status
    """

    def __init__(self) -> None:
        self.font = None
        self.small = None
        self.tiny = None

    def _ensure_fonts(self) -> None:
        if self.font is None:
            self.font = pygame.font.Font(None, 28)
        if self.small is None:
            self.small = pygame.font.Font(None, 22)
        if self.tiny is None:
            self.tiny = pygame.font.Font(None, 18)

    def render(self, screen: pygame.Surface) -> None:
        if not load_scanner_debug_overlay_enabled():
            return

        self._ensure_fonts()

        snapshot = hit_scanner.get_debug_snapshot()

        panel_w = 820
        panel_h = 700
        panel_x = 18
        panel_y = 18

        overlay = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        overlay.fill(PANEL_BG)

        pygame.draw.rect(overlay, BORDER, overlay.get_rect(), 2)

        title = self.font.render("Scanner debug overlay", True, WHITE)
        overlay.blit(title, (14, 12))

        status_lines = [
            f"State: {snapshot['state']}",
            f"Enabled: {snapshot['enabled']}",
            f"Status: {snapshot['last_status']}",
            f"Known holes: {snapshot['known_holes_count']}",
            f"Tracks: {snapshot['tracks_count']}",
            f"Candidates: {snapshot['candidates_count']}",
            f"Stable tracks: {snapshot['stable_tracks_count']}",
        ]

        y = 48
        for line in status_lines:
            surf = self.small.render(line, True, SOFT_WHITE)
            overlay.blit(surf, (14, y))
            y += 22

        boxes = {
            "crop_bgr": PanelBox(14, 215, 380, 210),
            "warped_gray": PanelBox(410, 215, 380, 210),
            "score": PanelBox(14, 455, 380, 210),
            "mask": PanelBox(410, 455, 380, 210),
        }

        self._draw_frame_panel(
            overlay,
            box=boxes["crop_bgr"],
            title="Scanport crop",
            frame=snapshot["debug_frames"].get("crop_bgr"),
            overlays=None,
        )

        board_overlays = {
            "candidates": snapshot["candidates"],
            "stable_tracks": snapshot["stable_tracks"],
            "known_holes": snapshot["known_holes"],
            "board_size": snapshot["board_size"],
        }

        self._draw_frame_panel(
            overlay,
            box=boxes["warped_gray"],
            title="Warped board",
            frame=snapshot["debug_frames"].get("warped_gray"),
            overlays=board_overlays,
        )

        self._draw_frame_panel(
            overlay,
            box=boxes["score"],
            title="Score map",
            frame=snapshot["debug_frames"].get("score"),
            overlays=None,
        )

        self._draw_frame_panel(
            overlay,
            box=boxes["mask"],
            title="Binary mask",
            frame=snapshot["debug_frames"].get("mask"),
            overlays=None,
        )

        legend_lines = [
            "Legend:",
            "gul ring = kandidat",
            "grön ring = stabil track",
            "cyan ring = redan registrerat hål",
        ]
        ly = 170
        for line in legend_lines:
            surf = self.tiny.render(line, True, SOFT_WHITE)
            overlay.blit(surf, (520, ly))
            ly += 18

        screen.blit(overlay, (panel_x, panel_y))

    def _draw_frame_panel(
        self,
        target: pygame.Surface,
        box: PanelBox,
        title: str,
        frame: np.ndarray | None,
        overlays: dict | None,
    ) -> None:
        pygame.draw.rect(target, (65, 65, 65), (box.x, box.y, box.w, box.h), 1)

        title_surf = self.small.render(title, True, WHITE)
        target.blit(title_surf, (box.x + 8, box.y + 6))

        inner = pygame.Rect(box.x + 8, box.y + 30, box.w - 16, box.h - 38)
        pygame.draw.rect(target, (35, 35, 35), inner)

        if frame is None:
            txt = self.small.render("ingen data", True, SOFT_WHITE)
            target.blit(txt, (inner.x + 10, inner.y + 10))
            return

        surface = self._numpy_to_surface(frame)
        if surface is None:
            txt = self.small.render("kunde inte rendera frame", True, RED)
            target.blit(txt, (inner.x + 10, inner.y + 10))
            return

        draw_rect = self._fit_rect(surface.get_size(), inner)
        scaled = pygame.transform.smoothscale(surface, draw_rect.size)
        target.blit(scaled, draw_rect.topleft)

        pygame.draw.rect(target, (120, 120, 120), draw_rect, 1)

        if overlays:
            self._draw_board_overlays(target, draw_rect, overlays)

    def _draw_board_overlays(self, target: pygame.Surface, draw_rect: pygame.Rect, overlays: dict) -> None:
        board_w, board_h = overlays.get("board_size", (1, 1))
        board_w = max(1, int(board_w))
        board_h = max(1, int(board_h))

        def map_point(x: float, y: float) -> tuple[int, int]:
            sx = draw_rect.x + int(round((x / board_w) * draw_rect.w))
            sy = draw_rect.y + int(round((y / board_h) * draw_rect.h))
            return sx, sy

        for hx, hy, _score in overlays.get("known_holes", []):
            px, py = map_point(hx, hy)
            pygame.draw.circle(target, CYAN, (px, py), 8, 2)

        for item in overlays.get("candidates", []):
            px, py = map_point(item["board_x"], item["board_y"])
            pygame.draw.circle(target, YELLOW, (px, py), 7, 2)

        for item in overlays.get("stable_tracks", []):
            px, py = map_point(item["board_x"], item["board_y"])
            pygame.draw.circle(target, GREEN, (px, py), 10, 2)

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