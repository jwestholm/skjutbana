from __future__ import annotations

"""V2.24.4 Game Hit Context verification scene.

This is deliberately a diagnostic scene rather than a reusable object engine.
It exercises the V2.24 HitRegion contract and V2.24.1 local physical search with:

* stationary target and no-shoot regions,
* a moving target,
* overlapping target/no-shoot regions,
* an edge target,
* a dedicated outside-region challenge,
* an EMPTY-regions mode that forces the legacy global path.

HitRegions are only detector search context.  Exact classification below uses the
returned HitEvent XY and, when available, the frozen shot-time game regions.
"""

from dataclasses import dataclass
import time
from typing import Iterable

import pygame

from src.engine.input.hit_input import HitEvent, hit_input
from src.engine.input.hit_regions import HitRegion, latest_hit_context_snapshot


BG = (15, 18, 24)
GRID = (35, 42, 54)
TEXT = (238, 242, 248)
MUTED = (155, 166, 181)
TARGET = (55, 196, 112)
NO_SHOOT = (231, 72, 72)
EDGE = (78, 142, 235)
OVERLAP = (195, 96, 222)
AMBER = (241, 177, 63)
CYAN = (72, 215, 235)
WHITE = (250, 250, 250)
BLACK = (12, 12, 14)
HUD_H = 118
FROZEN_GHOST_SECONDS = 3.0


@dataclass
class TestBox:
    object_id: str
    role: str
    x: float
    y: float
    w: float
    h: float
    color: tuple[int, int, int]
    label: str
    vx: float = 0.0
    moving: bool = False

    def hit_region(self) -> HitRegion:
        return HitRegion(
            object_id=self.object_id,
            x=float(self.x),
            y=float(self.y),
            width=float(self.w),
            height=float(self.h),
            role=self.role,
            metadata={"v242_test": True, "label": self.label, "moving": self.moving},
        )

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h


class HitContextTestV244:
    def __init__(self, game_root: str, viewport: pygame.Rect) -> None:
        del game_root
        self.viewport = viewport.copy()
        self.boxes: list[TestBox] = []
        self.empty_regions = False
        self.pause_motion = False
        self._subscribed = False
        self.hit_count = 0
        self.target_hits = 0
        self.no_shoot_hits = 0
        self.outside_hits = 0
        self.snapshot_mismatch = 0
        self.last_result = "Inga träffar ännu"
        self.last_detail = ""
        self.last_xy: tuple[float, float] | None = None
        self.last_frozen_regions: tuple = ()
        self.last_frozen_until = 0.0
        self.last_snapshot_id = None
        self.last_motion_px = 0.0
        self._font_big = None
        self._font = None
        self._font_small = None

    # ------------------------------------------------------------------
    # Lifecycle / V2.24 contract
    # ------------------------------------------------------------------
    def on_enter(self) -> None:
        self._font_big = pygame.font.Font(None, 34)
        self._font = pygame.font.Font(None, 26)
        self._font_small = pygame.font.Font(None, 21)
        self._build_layout()
        if not self._subscribed:
            hit_input.subscribe(self._on_hit)
            self._subscribed = True
        print("[V2.24.4 TESTSCENE] entered; E=empty/global P=pause R=reset")

    def on_exit(self) -> None:
        if self._subscribed:
            hit_input.unsubscribe(self._on_hit)
            self._subscribed = False

    def get_hit_regions(self):
        if self.empty_regions:
            return ()
        return tuple(box.hit_region() for box in self.boxes)

    # ------------------------------------------------------------------
    # Layout / movement
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        w = float(self.viewport.w)
        h = float(self.viewport.h)
        play_top = float(HUD_H + 18)
        play_h = max(260.0, h - play_top - 18.0)

        bw = max(90.0, min(150.0, w * 0.14))
        bh = max(68.0, min(112.0, play_h * 0.20))

        self.boxes = [
            TestBox("static_target", "target", w * 0.08, play_top + 18, bw, bh,
                    TARGET, "TARGET / STILLA"),
            TestBox("static_no_shoot", "no_shoot", w * 0.77, play_top + 18, bw, bh,
                    NO_SHOOT, "NO SHOOT"),
            TestBox("moving_target", "target", w * 0.35, play_top + play_h * 0.34,
                    bw, bh, TARGET, "TARGET / RÖRLIG", vx=max(180.0, w * 0.18), moving=True),
            TestBox("overlap_target", "target", w * 0.39, play_top + play_h * 0.68,
                    bw, bh, TARGET, "OVERLAP TARGET"),
            TestBox("overlap_no_shoot", "no_shoot", w * 0.46, play_top + play_h * 0.72,
                    bw, bh, OVERLAP, "OVERLAP NO SHOOT"),
            TestBox("edge_target", "target", max(4.0, w - bw - 8.0),
                    max(play_top, h - bh - 8.0), bw, bh, EDGE, "EDGE TARGET"),
            TestBox("outside_challenge", "target", w * 0.07,
                    play_top + play_h * 0.70, bw, bh, TARGET, "SKJUT PRECIS UTANFÖR"),
        ]

    def update(self, dt: float):
        if self.pause_motion:
            return None
        moving = next((b for b in self.boxes if b.object_id == "moving_target"), None)
        if moving is None:
            return None
        left = self.viewport.w * 0.25
        right = self.viewport.w * 0.72 - moving.w
        moving.x += moving.vx * max(0.0, min(float(dt), 0.1))
        if moving.x <= left:
            moving.x = left
            moving.vx = abs(moving.vx)
        elif moving.x >= right:
            moving.x = right
            moving.vx = -abs(moving.vx)
        return None

    # ------------------------------------------------------------------
    # Hit diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _regions_containing(regions: Iterable, x: float, y: float) -> list:
        result = []
        for r in regions:
            try:
                if float(r.x) <= x <= float(r.x + r.width) and float(r.y) <= y <= float(r.y + r.height):
                    result.append(r)
            except Exception:
                continue
        return result

    def _current_regions_containing(self, x: float, y: float) -> list[HitRegion]:
        if self.empty_regions:
            return []
        return [b.hit_region() for b in self.boxes if b.contains(x, y)]

    @staticmethod
    def _describe(regions: Iterable) -> str:
        items = []
        for r in regions:
            items.append(f"{getattr(r, 'object_id', '?')}:{getattr(r, 'role', '?')}")
        return ", ".join(items) if items else "UTANFÖR ALLA REGIONS"

    def _on_hit(self, hit: HitEvent) -> None:
        x = float(hit.game_x)
        y = float(hit.game_y)
        self.hit_count += 1
        self.last_xy = (x, y)

        snapshot = latest_hit_context_snapshot()
        event_shot_id = getattr(hit, "shot_id", None)
        source = str(getattr(hit, "source", getattr(hit, "kind", "unknown")))

        frozen_regions = ()
        snapshot_id = None
        if snapshot is not None:
            snapshot_id = getattr(snapshot, "shot_id", None)
            candidate_regions = tuple(getattr(snapshot, "game_regions", ()) or ())
            # If HitEvent exposes shot_id, do not accidentally attribute an old snapshot.
            if event_shot_id is None or snapshot_id == event_shot_id:
                frozen_regions = candidate_regions

        frozen_hits = self._regions_containing(frozen_regions, x, y)
        current_hits = self._current_regions_containing(x, y)

        # Measure how far matching objects moved between PANG and HitEvent.
        # This is more useful than only counting role/containment changes: the
        # moving target may still contain the same XY after moving 20-60 px.
        current_by_id = {b.object_id: b for b in self.boxes}
        motion_px = 0.0
        for region in frozen_regions:
            box = current_by_id.get(str(getattr(region, "object_id", "")))
            if box is None:
                continue
            try:
                frozen_cx = float(region.x) + float(region.width) * 0.5
                frozen_cy = float(region.y) + float(region.height) * 0.5
                current_cx = float(box.x) + float(box.w) * 0.5
                current_cy = float(box.y) + float(box.h) * 0.5
                motion_px = max(motion_px, ((current_cx - frozen_cx) ** 2 + (current_cy - frozen_cy) ** 2) ** 0.5)
            except Exception:
                pass
        self.last_motion_px = motion_px

        # Camera/audio hits should have a frozen snapshot. Mouse hits may not.
        authoritative_regions = frozen_hits if frozen_regions else current_hits
        roles = {str(getattr(r, "role", "")) for r in authoritative_regions}

        if "no_shoot" in roles:
            self.no_shoot_hits += 1
            verdict = "NO SHOOT"
            verdict_color = "RED"
        elif "target" in roles:
            self.target_hits += 1
            verdict = "TARGET"
            verdict_color = "GREEN"
        else:
            self.outside_hits += 1
            verdict = "UTANFÖR ALLA REGIONS"
            verdict_color = "AMBER"

        frozen_desc = self._describe(frozen_hits)
        current_desc = self._describe(current_hits)
        if frozen_regions and frozen_desc != current_desc:
            self.snapshot_mismatch += 1

        self.last_frozen_regions = tuple(frozen_regions)
        self.last_frozen_until = time.monotonic() + FROZEN_GHOST_SECONDS
        self.last_snapshot_id = snapshot_id
        self.last_result = f"{verdict} @ ({x:.0f}, {y:.0f})"
        self.last_detail = f"frozen: {frozen_desc} | current: {current_desc}"

        print(
            "[V2.24.4 TEST-HIT] "
            f"source={source} event_shot={event_shot_id} snapshot={snapshot_id} "
            f"xy=({x:.1f},{y:.1f}) empty={self.empty_regions} motion={self.last_motion_px:.1f}px "
            f"verdict={verdict_color}:{verdict} frozen=[{frozen_desc}] current=[{current_desc}]"
        )

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None
        if event.key == pygame.K_e:
            self.empty_regions = not self.empty_regions
            mode = "EMPTY/GLOBAL" if self.empty_regions else "REGION-FIRST"
            print(f"[V2.24.4 TESTSCENE] mode={mode}")
        elif event.key == pygame.K_p:
            self.pause_motion = not self.pause_motion
        elif event.key == pygame.K_r:
            self.hit_count = self.target_hits = self.no_shoot_hits = self.outside_hits = 0
            self.snapshot_mismatch = 0
            self.last_result = "Räknare nollställda"
            self.last_detail = ""
            self.last_xy = None
            self.last_motion_px = 0.0
            self.last_frozen_regions = ()
        return None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, screen: pygame.Surface) -> None:
        surface = pygame.Surface((self.viewport.w, self.viewport.h))
        surface.fill(BG)
        self._draw_grid(surface)
        self._draw_hud(surface)
        self._draw_boxes(surface)
        self._draw_frozen_snapshot(surface)
        self._draw_last_hit(surface)
        screen.blit(surface, self.viewport.topleft)

    def _draw_grid(self, surface: pygame.Surface) -> None:
        for x in range(0, surface.get_width(), 40):
            pygame.draw.line(surface, GRID, (x, HUD_H), (x, surface.get_height()), 1)
        for y in range(HUD_H, surface.get_height(), 40):
            pygame.draw.line(surface, GRID, (0, y), (surface.get_width(), y), 1)

    def _draw_hud(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, (21, 26, 35), (0, 0, surface.get_width(), HUD_H))
        mode = "EMPTY REGIONS → GLOBAL" if self.empty_regions else "HITREGIONS → LOCAL FIRST"
        mode_color = AMBER if self.empty_regions else CYAN
        title = self._font_big.render(f"V2.24.4 HIT CONTEXT TEST   |   {mode}", True, mode_color)
        surface.blit(title, (14, 10))
        stats = (
            f"Hits {self.hit_count}   target {self.target_hits}   no-shoot {self.no_shoot_hits}   "
            f"outside {self.outside_hits}   role diff {self.snapshot_mismatch}   last motion {self.last_motion_px:.0f}px"
        )
        surface.blit(self._font.render(stats, True, TEXT), (14, 45))
        surface.blit(self._font_small.render(
            "E = empty/global   P = pausa rörelse   R = nollställ   ESC = tillbaka   |   "
            "Skjut även precis UTANFÖR den markerade targeten.", True, MUTED), (14, 75))
        if self.last_result:
            surface.blit(self._font_small.render(self.last_result, True, WHITE), (14, 96))
        if self.last_detail:
            # Keep the long diagnostic on the right when there is room.
            text = self.last_detail[:120]
            img = self._font_small.render(text, True, MUTED)
            x = max(360, surface.get_width() - img.get_width() - 12)
            surface.blit(img, (x, 96))

    def _draw_boxes(self, surface: pygame.Surface) -> None:
        for box in self.boxes:
            rect = pygame.Rect(int(box.x), int(box.y), int(box.w), int(box.h))
            fill = box.color
            if self.empty_regions:
                fill = tuple(max(28, int(c * 0.35)) for c in fill)
            pygame.draw.rect(surface, fill, rect, border_radius=7)
            pygame.draw.rect(surface, WHITE, rect, 2, border_radius=7)
            label = self._font_small.render(box.label, True, BLACK if not self.empty_regions else TEXT)
            surface.blit(label, label.get_rect(center=rect.center))

            if box.object_id == "outside_challenge":
                outer = rect.inflate(34, 34)
                pygame.draw.rect(surface, AMBER, outer, 3, border_radius=9)
                hint = self._font_small.render("skjut i gula området UTANFÖR grön ruta", True, AMBER)
                surface.blit(hint, (outer.x, max(HUD_H + 2, outer.y - 22)))

    def _draw_frozen_snapshot(self, surface: pygame.Surface) -> None:
        if time.monotonic() > self.last_frozen_until:
            return
        for region in self.last_frozen_regions:
            try:
                rect = pygame.Rect(int(region.x), int(region.y), int(region.width), int(region.height))
            except Exception:
                continue
            pygame.draw.rect(surface, CYAN, rect, 3)
        if self.last_frozen_regions:
            txt = self._font_small.render("CYAN = fryst position vid PANG", True, CYAN)
            surface.blit(txt, (surface.get_width() - txt.get_width() - 12, HUD_H + 7))

    def _draw_last_hit(self, surface: pygame.Surface) -> None:
        if self.last_xy is None:
            return
        x, y = int(self.last_xy[0]), int(self.last_xy[1])
        pygame.draw.circle(surface, WHITE, (x, y), 12, 2)
        pygame.draw.line(surface, WHITE, (x - 17, y), (x + 17, y), 2)
        pygame.draw.line(surface, WHITE, (x, y - 17), (x, y + 17), 2)


def create_game(game_root: str, viewport: pygame.Rect):
    return HitContextTestV244(game_root=game_root, viewport=viewport)
