"""Physical capture target — a static board for recording real shots.

Purpose: collect physical evidence (source recording + ground truth) without game
content that changes between shots. The board is plain light grey with numbered thin
aim crosses; nothing on the board changes while shooting unless the operator presses a
key. Detected hits are NOT drawn on the board by default (a changing board contaminates
detection, cf. the AI-test HUD counter); they are printed to stdout and logged.

The weapon/sequence text is shown for 3 s after a key press only (board text
attracts false emissions). Keys (on the PC):
  1 / 2 / 3   weapon = airgun 4.5 mm / CO2 AR15 / CO2 pistol (starts a new sequence)
  C           mark a clap / no-impact control now
  N           new sequence with the same weapon
  H           toggle drawing of detected hits on the board (default off)
  ESC         back to the menu (handled by the engine)

Marks are written to <active source recording>/operator_marks.jsonl and to
/data/skjutbana/physical_targets/<UTC>/marks.jsonl (hits included in the latter).
Ground truth (exact hole XY) is labelled afterwards from the recorded camera frames.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pygame

from src.engine.input.hit_input import HitEvent, hit_input

BOARD = (232, 232, 232)
INK = (40, 40, 40)
LABEL = (90, 90, 90)
HIT = (220, 30, 30)
WEAPONS = {pygame.K_1: "airgun_4_5mm", pygame.K_2: "co2_ar15", pygame.K_3: "co2_pistol",
           pygame.K_KP1: "airgun_4_5mm", pygame.K_KP2: "co2_ar15", pygame.K_KP3: "co2_pistol"}
LOG_ROOT = Path("/data/skjutbana/physical_targets")


class PhysicalCaptureTarget:
    hit_precision = "SHOT"          # no precision flash: the board must stay static
    owns_hit_presentation = True

    def __init__(self, game_root: str, viewport: pygame.Rect) -> None:
        self.viewport = viewport.copy()
        self.weapon = "unlabelled"
        self.sequence = 0
        self.hits: list[tuple[float, float]] = []
        self.show_hits = False
        self.font = None
        self.small = None
        self._log_dir: Path | None = None
        self.label_until = 0.0       # weapon text shown only briefly: static board text attracted
                                     # 3/3 false live emissions (2026-11-18 AR15 round 2)
        self.offset = (0.0, 0.0)   # grid offset (fraction of cell), new per sequence: spread wear

    # ------------------------------------------------------------------
    def on_enter(self) -> None:
        pygame.font.init()
        self.font = pygame.font.Font(None, 34)
        self.small = pygame.font.Font(None, 26)
        try:
            self._log_dir = LOG_ROOT / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            self._log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"[TARGET] no log dir: {exc}")
            self._log_dir = None
        self.label_until = time.time() + 3.0
        hit_input.subscribe(self._on_hit)
        self._mark("session_start")
        print("[TARGET] 1/2/3 = luftgevär/AR15/pistol, C = klapp, N = ny serie, H = visa träffar")

    def on_exit(self) -> None:
        try:
            hit_input.unsubscribe(self._on_hit)
        except Exception:
            pass
        self._mark("session_end")

    # ------------------------------------------------------------------
    def _recording_dir(self) -> Path | None:
        try:
            from src.engine.source_recording import get_source_recorder
            recorder = get_source_recorder()
            directory = getattr(recorder, "directory", None) if recorder is not None else None
            return Path(directory) if directory is not None else None
        except Exception:
            return None

    def _mark(self, kind: str, **fields) -> None:
        row = dict(wall_ts=time.time(), kind=kind, weapon=self.weapon, sequence=self.sequence, **fields)
        line = json.dumps(row) + "\n"
        targets = [self._log_dir / "marks.jsonl"] if self._log_dir else []
        rec = self._recording_dir()
        if rec is not None and kind != "hit":
            targets.append(rec / "operator_marks.jsonl")
        for path in targets:
            try:
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(line)
            except Exception as exc:
                print(f"[TARGET] could not write {path}: {exc}")

    def _on_hit(self, hit: HitEvent) -> None:
        self.hits.append((float(hit.game_x), float(hit.game_y)))
        self._mark("hit", camera_x=float(hit.camera_x), camera_y=float(hit.camera_y),
                   game_x=float(hit.game_x), game_y=float(hit.game_y), hit_ts=float(hit.timestamp),
                   source=str(hit.source))
        print(f"[TARGET] hit #{len(self.hits)} {self.weapon} seq {self.sequence}: "
              f"game=({hit.game_x:.0f},{hit.game_y:.0f}) cam=({hit.camera_x:.1f},{hit.camera_y:.1f})")

    # ------------------------------------------------------------------
    def _new_sequence(self) -> None:
        # Move the whole cross grid between sequences so the same paper spots are not
        # shot out (ROADMAP "spread the aim points"). Never moves during a sequence.
        self.sequence += 1
        self.offset = (random.uniform(-0.35, 0.35), random.uniform(-0.35, 0.35))
        self._mark("sequence_start", aims_game_xy=self.aim_points())
        print(f"[TARGET] serie {self.sequence}: {self.weapon}")

    def aim_points(self) -> list[tuple[int, int]]:
        w, h = self.viewport.size
        ox, oy = self.offset
        return [(int(w * (col + 0.5 + ox) / 4), int(h * (row + 0.5 + oy) / 3))
                for row in range(3) for col in range(4)]

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None
        self.label_until = time.time() + 3.0
        if event.key in WEAPONS:
            self.weapon = WEAPONS[event.key]
            self._new_sequence()
        elif event.key == pygame.K_n:
            self._new_sequence()
        elif event.key == pygame.K_c:
            self._mark("clap_or_no_impact")
            print("[TARGET] klapp/kontroll markerad")
        elif event.key == pygame.K_h:
            self.show_hits = not self.show_hits
        return None

    def update(self, dt: float):
        return None

    # ------------------------------------------------------------------
    def render(self, screen: pygame.Surface) -> None:
        vp = self.viewport
        surface = pygame.Surface(vp.size)
        surface.fill(BOARD)
        w, h = vp.size
        for n, (x, y) in enumerate(self.aim_points(), start=1):
            pygame.draw.line(surface, INK, (x - 14, y), (x + 14, y), 1)
            pygame.draw.line(surface, INK, (x, y - 14), (x, y + 14), 1)
            label = self.small.render(str(n), True, LABEL)
            surface.blit(label, (x + 18, y + 12))
        if time.time() < self.label_until:
            text = self.font.render(f"{self.weapon}  serie {self.sequence}", True, LABEL)
            surface.blit(text, (12, h - 34))
        if self.show_hits:
            for gx, gy in self.hits:
                pygame.draw.circle(surface, HIT, (int(gx), int(gy)), 9, 2)
        screen.blit(surface, vp.topleft)


def create_game(game_root: str, viewport: pygame.Rect):
    return PhysicalCaptureTarget(game_root, viewport)
