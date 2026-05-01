from __future__ import annotations

import math
import time
from dataclasses import dataclass

import pygame

from src.engine.input.hit_input import HitEvent, hit_input
from src.engine.camera.hit_scanner import hit_scanner
from src.engine.settings import (
    load_content_rect,
    load_viewport_rect,
    load_visual_hits_enabled,
    load_visual_hits_lifetime_ms,
    load_visual_hits_mode,
    load_visual_hits_radius,
    load_visual_hits_show_all_planes,
    load_visual_hits_show_candidates,
    load_visual_hits_candidates_count,
    load_visual_hits_candidates_lifetime_s,
)


@dataclass
class VisualMarker:
    x: float
    y: float
    timestamp: float
    color: tuple[int, int, int]
    label: str
    style: str


@dataclass
class CandidateSnapshot:
    """En snapshot av kandidater tagen vid ett specifikt skott."""
    candidates: list[dict]
    timestamp: float
    shot_id: int


class HitVisualizer:
    COLOR_MOUSE = (255, 60, 60)
    COLOR_CAMERA = (0, 200, 255)
    COLOR_VIEWPORT = (255, 220, 80)
    COLOR_CONTENT = (255, 120, 220)
    COLOR_SCANPORT = (120, 255, 120)
    COLOR_HOMOGRAPHY = (190, 120, 255)
    COLOR_DEFAULT = (255, 255, 255)

    def __init__(self):
        self.hits: list[VisualMarker] = []
        self._candidate_snapshots: list[CandidateSnapshot] = []
        self._last_seen_shot_id: int = -1
        hit_input.subscribe(self._on_hit)

    def reload_settings(self):
        """
        Behålls för kompatibilitet med befintliga settings-scener.
        Inställningarna läses dynamiskt i render/update.
        """
        return None

    def clear(self):
        self.hits.clear()
        # Behåll _candidate_snapshots — de ska överleva scenbyte

    def clear_all(self):
        """Full reset — clear everything including candidate snapshots."""
        self.hits.clear()
        self._candidate_snapshots.clear()

    def _color_for_source(self, source: str):
        if source == "mouse":
            return self.COLOR_MOUSE
        if source == "camera":
            return self.COLOR_CAMERA
        return self.COLOR_DEFAULT

    def _append_marker(
        self,
        markers: list[VisualMarker],
        *,
        x: float,
        y: float,
        color: tuple[int, int, int],
        label: str,
        style: str,
        timestamp: float,
    ):
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        markers.append(
            VisualMarker(
                x=float(x),
                y=float(y),
                timestamp=timestamp,
                color=color,
                label=label,
                style=style,
            )
        )

    def _markers_for_event(self, event: HitEvent) -> list[VisualMarker]:
        now = time.time()
        markers: list[VisualMarker] = []
        base_color = self._color_for_source(event.source)

        if not load_visual_hits_show_all_planes():
            self._append_marker(
                markers,
                x=event.screen_x,
                y=event.screen_y,
                color=base_color,
                label=event.source,
                style="cross",
                timestamp=now,
            )
            return markers

        viewport = load_viewport_rect()
        content = load_content_rect()

        viewport_screen_x = float(viewport.x + event.viewport_x)
        viewport_screen_y = float(viewport.y + event.viewport_y)
        content_screen_x = float(content.x + event.content_x)
        content_screen_y = float(content.y + event.content_y)

        self._append_marker(
            markers,
            x=event.requested_screen_x,
            y=event.requested_screen_y,
            color=base_color,
            label="click" if event.source == "mouse" else "requested",
            style="ring",
            timestamp=now,
        )
        self._append_marker(
            markers,
            x=event.screen_x,
            y=event.screen_y,
            color=base_color,
            label="screen",
            style="cross",
            timestamp=now,
        )
        self._append_marker(
            markers,
            x=viewport_screen_x,
            y=viewport_screen_y,
            color=self.COLOR_VIEWPORT,
            label="viewport",
            style="square",
            timestamp=now,
        )
        self._append_marker(
            markers,
            x=content_screen_x,
            y=content_screen_y,
            color=self.COLOR_CONTENT,
            label="content",
            style="diamond",
            timestamp=now,
        )
        self._append_marker(
            markers,
            x=event.scanport_screen_x,
            y=event.scanport_screen_y,
            color=self.COLOR_SCANPORT,
            label="scanport",
            style="triangle",
            timestamp=now,
        )
        self._append_marker(
            markers,
            x=event.homography_screen_x,
            y=event.homography_screen_y,
            color=self.COLOR_HOMOGRAPHY,
            label="homography",
            style="x",
            timestamp=now,
        )
        return markers

    def _on_hit(self, event: HitEvent):
        if not load_visual_hits_enabled():
            return
        self.hits.extend(self._markers_for_event(event))

    def update(self, dt: float):
        del dt
        now = time.time()

        mode = load_visual_hits_mode()
        if mode != "persistent":
            lifetime = load_visual_hits_lifetime_ms() / 1000.0
            self.hits = [hit for hit in self.hits if now - hit.timestamp <= lifetime]

        # Snapshotta kandidater från hit_scanner vid nya skott
        if load_visual_hits_show_candidates():
            self._capture_candidate_snapshots(now)
            self._prune_candidate_snapshots(now)

    def _draw_cross(self, overlay: pygame.Surface, color, x: int, y: int, radius: int):
        pygame.draw.circle(overlay, color, (x, y), radius, 2)
        pygame.draw.line(overlay, color, (x - radius - 4, y), (x + radius + 4, y), 2)
        pygame.draw.line(overlay, color, (x, y - radius - 4), (x, y + radius + 4), 2)

    def _draw_ring(self, overlay: pygame.Surface, color, x: int, y: int, radius: int):
        pygame.draw.circle(overlay, color, (x, y), radius + 6, 2)
        pygame.draw.circle(overlay, color, (x, y), max(2, radius - 4), 1)

    def _draw_square(self, overlay: pygame.Surface, color, x: int, y: int, radius: int):
        size = radius * 2
        rect = pygame.Rect(x - radius, y - radius, size, size)
        pygame.draw.rect(overlay, color, rect, 2)

    def _draw_diamond(self, overlay: pygame.Surface, color, x: int, y: int, radius: int):
        points = [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]
        pygame.draw.polygon(overlay, color, points, 2)

    def _draw_triangle(self, overlay: pygame.Surface, color, x: int, y: int, radius: int):
        points = [(x, y - radius), (x + radius, y + radius), (x - radius, y + radius)]
        pygame.draw.polygon(overlay, color, points, 2)

    def _draw_x(self, overlay: pygame.Surface, color, x: int, y: int, radius: int):
        pygame.draw.line(overlay, color, (x - radius, y - radius), (x + radius, y + radius), 2)
        pygame.draw.line(overlay, color, (x - radius, y + radius), (x + radius, y - radius), 2)

    def _draw_marker(self, overlay: pygame.Surface, marker: VisualMarker, radius: int):
        rgba = (marker.color[0], marker.color[1], marker.color[2], 255)
        x = int(round(marker.x))
        y = int(round(marker.y))

        if marker.style == "ring":
            self._draw_ring(overlay, rgba, x, y, radius)
        elif marker.style == "square":
            self._draw_square(overlay, rgba, x, y, radius)
        elif marker.style == "diamond":
            self._draw_diamond(overlay, rgba, x, y, radius)
        elif marker.style == "triangle":
            self._draw_triangle(overlay, rgba, x, y, radius)
        elif marker.style == "x":
            self._draw_x(overlay, rgba, x, y, radius)
        else:
            self._draw_cross(overlay, rgba, x, y, radius)

    def _capture_candidate_snapshots(self, now: float) -> None:
        """Fånga nya kandidater från hit_scanner när ett nytt skott har bearbetats."""
        candidates = list(hit_scanner.last_candidates)
        if not candidates:
            return

        # Använd senaste audio_event_count som shot-id för att detektera nya skott.
        # audio_event_count kan nollställas vid scenbyte (enable/disable), så vi
        # jämför med _last_seen_shot_id och accepterar även lägre värden som nya.
        current_shot_id = hit_scanner.audio_event_count
        if current_shot_id == self._last_seen_shot_id:
            return
        if current_shot_id < self._last_seen_shot_id:
            # Scanner har startats om — nollställ vår räknare
            self._last_seen_shot_id = 0

        self._last_seen_shot_id = current_shot_id
        max_count = load_visual_hits_candidates_count()

        snapshot = CandidateSnapshot(
            candidates=candidates[:max_count],
            timestamp=now,
            shot_id=current_shot_id,
        )

        lifetime_s = load_visual_hits_candidates_lifetime_s()
        if lifetime_s == 0:
            # "Till nästa skott" — behåll bara den senaste
            self._candidate_snapshots = [snapshot]
        else:
            self._candidate_snapshots.append(snapshot)

    def _prune_candidate_snapshots(self, now: float) -> None:
        """Ta bort utgångna snapshots baserat på livstidsinställningen."""
        lifetime_s = load_visual_hits_candidates_lifetime_s()
        if lifetime_s == 0:
            # Behåll bara senaste (hanteras redan i capture)
            return
        self._candidate_snapshots = [
            s for s in self._candidate_snapshots
            if now - s.timestamp <= lifetime_s
        ]

    def render(self, screen: pygame.Surface):
        if not load_visual_hits_enabled():
            return

        radius = load_visual_hits_radius()
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        show_all_planes = load_visual_hits_show_all_planes()
        label_font = pygame.font.Font(None, 18) if show_all_planes else None

        for hit in self.hits:
            self._draw_marker(overlay, hit, radius)
            if show_all_planes and label_font is not None:
                text = label_font.render(hit.label, True, hit.color)
                overlay.blit(text, (int(round(hit.x)) + radius + 8, int(round(hit.y)) - 10))

        if load_visual_hits_show_candidates():
            self._render_candidates(overlay, screen)

        screen.blit(overlay, (0, 0))

    def _render_candidates(self, overlay: pygame.Surface, screen: pygame.Surface):
        """
        Ritar ut kandidat-snapshots på overlayen.
        Varje kandidat visas med numrering, score och position.
        Färgkodas från grön (hög score) till röd (låg score).
        """
        del screen
        if not self._candidate_snapshots:
            return

        max_count = load_visual_hits_candidates_count()
        font = pygame.font.Font(None, 16)
        font_panel = pygame.font.Font(None, 18)
        lifetime_s = load_visual_hits_candidates_lifetime_s()
        now = time.time()

        # Rita alla aktiva snapshots
        for snap in self._candidate_snapshots:
            candidates = snap.candidates[:max_count]
            if not candidates:
                continue

            # Beräkna alpha-fade om livstid > 0
            if lifetime_s > 0:
                age = now - snap.timestamp
                fade = max(0.0, min(1.0, 1.0 - (age / lifetime_s)))
            else:
                fade = 1.0

            if fade <= 0.01:
                continue

            max_score = max((c.get("score", 0.0) for c in candidates), default=1.0)
            if max_score <= 0:
                max_score = 1.0

            for i, cand in enumerate(candidates):
                cam_x = cand.get("camera_x", 0.0)
                cam_y = cand.get("camera_y", 0.0)
                score = cand.get("score", 0.0)

                try:
                    sx, sy = hit_input._canonical_camera_to_screen(cam_x, cam_y)
                except Exception:
                    continue

                if not (math.isfinite(sx) and math.isfinite(sy)):
                    continue

                ratio = score / max_score
                if ratio > 0.5:
                    r = int(255 * (1.0 - ratio) * 2)
                    g = 255
                else:
                    r = 255
                    g = int(255 * ratio * 2)
                alpha = int(200 * fade)
                color = (r, g, 60, alpha)

                ix = int(round(sx))
                iy = int(round(sy))

                pygame.draw.circle(overlay, color, (ix, iy), 14, 2)
                num_text = font.render(str(i + 1), True, (255, 255, 255, int(240 * fade)))
                num_rect = num_text.get_rect(center=(ix, iy))
                overlay.blit(num_text, num_rect)

                score_label = f"{score:.1f}"
                score_surf = font.render(score_label, True, color)
                overlay.blit(score_surf, (ix + 17, iy - 6))

        # Visa infopanel för senaste snapshot
        latest = self._candidate_snapshots[-1]
        latest_candidates = latest.candidates[:max_count]
        if latest_candidates:
            self._render_candidate_panel(overlay, latest_candidates, font_panel, max_count, latest.shot_id)

    def _render_candidate_panel(
        self,
        overlay: pygame.Surface,
        candidates: list[dict],
        font: pygame.font.Font,
        max_count: int,
        shot_id: int = 0,
    ):
        """Kompakt panel med kandidatinfo i hörnet."""
        panel_w = 370
        line_h = 17
        header_h = 24
        panel_h = header_h + line_h * min(len(candidates), max_count) + 8
        margin = 10
        panel_x = margin
        panel_y = overlay.get_height() - panel_h - margin

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 170))
        pygame.draw.rect(panel, (110, 110, 110, 200), panel.get_rect(), 1)

        header = font.render(
            f"Skott #{shot_id} — Kandidater (top {min(len(candidates), max_count)})",
            True,
            (255, 220, 80, 255),
        )
        panel.blit(header, (8, 4))

        max_score = max((c.get("score", 0.0) for c in candidates), default=1.0)
        if max_score <= 0:
            max_score = 1.0

        for i, cand in enumerate(candidates[:max_count]):
            score = cand.get("score", 0.0)
            cx = cand.get("camera_x", 0.0)
            cy = cand.get("camera_y", 0.0)
            cd = cand.get("center_darkening", 0.0)
            lcg = cand.get("local_contrast_gain", 0.0)
            psc = cand.get("pre_shot_change", 0.0)

            ratio = score / max_score
            if ratio > 0.5:
                r = int(255 * (1.0 - ratio) * 2)
                g = 255
            else:
                r = 255
                g = int(255 * ratio * 2)
            color = (r, g, 60, 230)

            line = f"#{i+1}  scr:{score:.1f}  psc:{psc:.1f}  cd:{cd:.1f}  lcg:{lcg:.1f}  ({cx:.0f},{cy:.0f})"
            text = font.render(line, True, color)
            panel.blit(text, (8, header_h + i * line_h))

        overlay.blit(panel, (panel_x, panel_y))


hit_visualizer = HitVisualizer()
