from __future__ import annotations

"""Kompetensprov AK – BAS / BAS Hv simulator.

The game intentionally keeps the official BAS rule model separate from the
physical hit detector:
- Physical/game hit coordinates arrive through normal HitInput.
- Search context never replaces the final physical HitEvent coordinate.
- Exact zone-mask classification uses that point without target assist,
  ballistic displacement or expansion outside the silhouette.

The fixed 45-second cease-fire is a simulator rule requested for this game.
Official BAS result time is still measured from ELD/start signal to last shot.
"""

from dataclasses import dataclass, asdict
import importlib.util
import json
import math
import os
import hashlib
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any

import pygame
import cv2
import numpy as np

from src.engine.input.hit_input import HitEvent, hit_input
from src.engine.camera.camera_manager import camera_manager
from src.engine.input.hit_regions import HitRegion, hit_context_snapshot_for_shot
from src.engine.range_projection import (
    RangeProjectionGeometry,
    projected_ground_anchor_y_px,
    projected_target_height_px,
)
from src.engine.settings import load_range_projection_settings, load_scanport_rect

# Dynamic game loading does not guarantee that this file has a conventional
# package __name__. Load the pure sibling rules module explicitly and keep the
# dependency game-local.
_RULES_PATH = Path(__file__).with_name("rules.py")
_RULES_SPEC = importlib.util.spec_from_file_location("kompetensprov_ak_rules", _RULES_PATH)
if _RULES_SPEC is None or _RULES_SPEC.loader is None:
    raise ImportError(f"Could not load competence rules: {_RULES_PATH}")
rules = importlib.util.module_from_spec(_RULES_SPEC)
sys.modules[_RULES_SPEC.name] = rules
_RULES_SPEC.loader.exec_module(rules)

PROFILES = rules.PROFILES
ZONE_POINTS = rules.ZONE_POINTS
ShotRecord = rules.ShotRecord
CompetitionResult = rules.CompetitionResult
ManualMark = rules.ManualMark
SIMULATOR_WINDOW_S = rules.SIMULATOR_WINDOW_S
TARGET_REAL_HEIGHT_CM = rules.TARGET_REAL_HEIGHT_CM
BULLET_DIAMETER_CM = rules.BULLET_DIAMETER_CM
COUNTED_HITS = rules.COUNTED_HITS
expected_posture = rules.expected_posture
ak4b_banheight_cm = rules.ak4b_banheight_cm
evaluate_result = rules.evaluate_result
evaluate_manual_result = rules.evaluate_manual_result

TARGET_OBJECT_ID = "kompetensprov_hf2020"
TARGET_PRIORITY = 8.0
HORIZON_Y_NORM = 0.50
RESULT_DELAY_S = 2.20
COMMAND_KLAR_S = 1.65
COMMAND_FARDIG_S = 1.65
ELD_OVERLAY_S = 0.70
SHOT_DEBOUNCE_S = 0.075

# Pre-start target positioning / manual verification.
TARGET_SHIFT_STEP_CM = 5.0
TARGET_EDGE_MARGIN_PX = 10
REVIEW_REMOVE_RADIUS_PX = 42.0
REVIEW_DATA_ROOT = Path("/data/skjutbana/kompetensprov_manual_reviews")

ZONE_ID_TO_LETTER = {
    1: "H",
    2: "A",
    3: "X",
    4: "B",
    5: "C",
    6: "D",
}

# Presentation
WHITE = (245, 247, 244)
MUTED = (185, 193, 184)
GREEN = (98, 221, 128)
RED = (244, 91, 82)
AMBER = (248, 193, 78)
CYAN = (100, 212, 236)
PANEL = (10, 15, 17, 224)
PANEL_SOFT = (14, 20, 22, 190)
BLACK = (7, 9, 10)
COUNTED = (255, 220, 82)
EXTRA = (156, 164, 162)
MISS = (237, 91, 85)


@dataclass(slots=True)
class CommandState:
    text: str
    sound_key: str
    started: float
    duration: float


class KompetensprovAKGame:
    def __init__(self, game_root: str, viewport: pygame.Rect) -> None:
        self.game_root = Path(game_root)
        self.viewport = viewport.copy()
        self.local_viewport = pygame.Rect(0, 0, self.viewport.w, self.viewport.h)

        self.physical_test = os.environ.get("SKJUTBANA_KOMPETENSPROV_TEST") == "1"
        self.owns_hit_presentation = self.physical_test
        self.profile_index = 0  # BAS Hv first; easiest match to user's recent physical test.
        self.state = "setup"
        self.command: CommandState | None = None
        self.fire_started: float | None = None
        self.ceasefire_started: float | None = None
        self.eld_overlay_until = 0.0
        self.next_beep_s = 5.0
        self.shots: list[ShotRecord] = []
        self._accepted_shot_ids: set[int] = set()
        self._last_accept_monotonic = -999.0
        self._subscribed = False
        self._show_zone_overlay = True
        self.target_offset_cm = 0.0

        # Post-fire human verification is deliberately separate from automatic
        # physical-hit output. It is allowed to drive the displayed verified
        # competition result, but never mutates self.shots.
        self.manual_marks: list[ManualMark] = []
        self.manual_confirmed = False
        self.review_surface: pygame.Surface | None = None
        self.review_source = "none"
        self.review_message = ""
        self._review_crop_local = pygame.Rect(0, 0, 1, 1)
        self._review_display_screen = pygame.Rect(0, 0, 1, 1)

        self.font_title: pygame.font.Font | None = None
        self.font_big: pygame.font.Font | None = None
        self.font_medium: pygame.font.Font | None = None
        self.font_small: pygame.font.Font | None = None
        self.font_tiny: pygame.font.Font | None = None

        self.background: pygame.Surface | None = None
        self.background_scaled: pygame.Surface | None = None
        self.target: pygame.Surface | None = None
        self.target_zones: pygame.Surface | None = None
        self.zone_mask: pygame.Surface | None = None
        self.preview: pygame.Surface | None = None
        self.sounds: dict[str, pygame.mixer.Sound] = {}
        self._audio_ok = False

        self._target_local_rect = pygame.Rect(0, 0, 1, 1)
        self._target_screen_rect = pygame.Rect(0, 0, 1, 1)
        self._load_assets()

    # ------------------------------------------------------------------
    # Lifecycle / assets
    # ------------------------------------------------------------------
    @property
    def profile(self):
        return PROFILES[self.profile_index]

    def _load_assets(self) -> None:
        required = {
            "background": self.game_root / "background.png",
            "target": self.game_root / "hf2020.png",
            "target_zones": self.game_root / "hf2020_zones.png",
            "zone_mask": self.game_root / "hf2020_zone_mask.png",
        }
        for label, path in required.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {label} asset: {path}")

        self.background = pygame.image.load(str(required["background"])).convert()
        self.target = pygame.image.load(str(required["target"])).convert_alpha()
        self.target_zones = pygame.image.load(str(required["target_zones"])).convert_alpha()
        self.zone_mask = pygame.image.load(str(required["zone_mask"])).convert()
        self.zone_mask_sha256 = hashlib.sha256(required["zone_mask"].read_bytes()).hexdigest()
        self._rebuild_background()

    def _load_audio(self) -> None:
        self.sounds.clear()
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init()
            audio_dir = self.game_root / "audio"
            for key, filename in {
                "klar": "skytt_klar.wav",
                "fardig": "skytt_fardig.wav",
                "eld": "eld.wav",
                "upphor": "eld_upphor.wav",
                "beep": "beep.wav",
            }.items():
                path = audio_dir / filename
                if path.exists():
                    self.sounds[key] = pygame.mixer.Sound(str(path))
            self._audio_ok = bool(self.sounds)
        except Exception as exc:
            self._audio_ok = False
            print(f"[KOMPETENSPROV] audio unavailable: {exc}")

    def _play(self, key: str) -> None:
        if self.physical_test:
            return
        sound = self.sounds.get(key)
        if sound is None:
            return
        try:
            sound.play()
        except Exception:
            pass

    def _rebuild_background(self) -> None:
        if self.background is None:
            self.background_scaled = None
            return
        self.background_scaled = pygame.transform.smoothscale(
            self.background, (self.viewport.w, self.viewport.h)
        )

    def on_enter(self) -> None:
        self.font_title = pygame.font.Font(None, 64)
        self.font_big = pygame.font.Font(None, 46)
        self.font_medium = pygame.font.Font(None, 31)
        self.font_small = pygame.font.Font(None, 23)
        self.font_tiny = pygame.font.Font(None, 18)
        self._load_audio()
        if not self._subscribed:
            hit_input.subscribe(self._on_hit)
            self._subscribed = True
        self._reset_to_select()

    def on_exit(self) -> None:
        if self._subscribed:
            hit_input.unsubscribe(self._on_hit)
            self._subscribed = False

    # ------------------------------------------------------------------
    # Physical projection
    # ------------------------------------------------------------------
    def _geometry(self) -> RangeProjectionGeometry:
        settings = load_range_projection_settings()
        return RangeProjectionGeometry(
            wall_distance_m=float(settings.get("wall_distance_m", 6.3)),
            viewport_physical_width_cm=float(settings.get("viewport_physical_width_cm", 90.0)),
            viewport_physical_height_cm=float(settings.get("viewport_physical_height_cm", 70.0)),
            viewport_bottom_world_cm=float(settings.get("viewport_bottom_world_cm", 105.0)),
        )

    def _update_target_rect(self) -> pygame.Rect:
        if self.target is None:
            return self._target_local_rect
        geometry = self._geometry()
        distance_m = float(self.profile.distance_m)
        height_px = max(
            2,
            int(round(projected_target_height_px(
                real_height_cm=TARGET_REAL_HEIGHT_CM,
                virtual_distance_m=distance_m,
                viewport=self.local_viewport,
                geometry=geometry,
            ))),
        )
        src_w, src_h = self.target.get_size()
        aspect = src_w / max(1.0, float(src_h))
        width_px = max(2, int(round(height_px * aspect)))

        foot_screen_y = projected_ground_anchor_y_px(
            distance_m=distance_m,
            viewport=self.viewport,
            geometry=geometry,
            horizon_y_norm=HORIZON_Y_NORM,
        )
        foot_local_y = foot_screen_y - self.viewport.y
        offset_px = (
            float(self.target_offset_cm)
            / max(0.01, float(geometry.viewport_physical_width_cm))
            * float(self.viewport.w)
        )
        max_offset_px = max(
            0.0,
            float(self.viewport.w) * 0.5 - float(width_px) * 0.5 - TARGET_EDGE_MARGIN_PX,
        )
        offset_px = max(-max_offset_px, min(max_offset_px, offset_px))
        # Persist the actually usable physical-wall offset after clamping.
        self.target_offset_cm = (
            offset_px
            / max(1.0, float(self.viewport.w))
            * float(geometry.viewport_physical_width_cm)
        )
        center_local_x = self.viewport.w * 0.5 + offset_px
        local = pygame.Rect(
            int(round(center_local_x - width_px * 0.5)),
            int(round(foot_local_y - height_px)),
            width_px,
            height_px,
        )
        screen = local.move(self.viewport.x, self.viewport.y)
        self._target_local_rect = local
        self._target_screen_rect = screen
        return local

    def _move_target(self, delta_cm: float) -> None:
        """Move the whole projected target physically on the wall before start."""
        if self.state != "setup":
            return
        self.target_offset_cm += float(delta_cm)
        self._update_target_rect()
        self.review_message = f"Figur sidled: {self.target_offset_cm:+.0f} cm"

    def get_hit_regions(self):
        if self.state not in {"klar", "fardig", "firing", "ceasefire"}:
            return ()
        rect = self._update_target_rect()
        # One image AABB is search context; transparent silhouette gaps remain
        # misses under exact downstream mask classification. No per-zone quotas.
        return (HitRegion(
            object_id=TARGET_OBJECT_ID, x=float(rect.x), y=float(rect.y),
            width=float(rect.w), height=float(rect.h), role="target",
            priority=TARGET_PRIORITY, owner="kompetensprov_ak",
            metadata={"game": "kompetensprov_ak", "profile": self.profile.profile_id,
                      "distance_m": float(self.profile.distance_m),
                      "target": "Helfigur 2020", "target_rect": tuple(rect),
                      "zone_mask_sha256": self.zone_mask_sha256,
                      "classification": "exact_mask_at_unchanged_physical_xy"},
        ),)

    # ------------------------------------------------------------------
    # State machine / commands
    # ------------------------------------------------------------------
    def _reset_to_select(self) -> None:
        self.state = "setup"
        self.command = None
        self.fire_started = None
        self.ceasefire_started = None
        self.eld_overlay_until = 0.0
        self.next_beep_s = 5.0
        self.shots.clear()
        self._accepted_shot_ids.clear()
        self._last_accept_monotonic = -999.0
        self.manual_marks.clear()
        self.manual_confirmed = False
        self.review_surface = None
        self.review_source = "none"
        self.review_message = ""
        self.manual_marks.clear()
        self.manual_confirmed = False
        self.review_surface = None
        self.review_source = "none"
        self.review_message = ""
        self._update_target_rect()

    def _start_competition(self) -> None:
        self.shots.clear()
        self._accepted_shot_ids.clear()
        self._last_accept_monotonic = -999.0
        self.fire_started = None
        self.ceasefire_started = None
        self.next_beep_s = 5.0
        if self.physical_test:
            self._begin_fire()
        else:
            self._start_command("klar", "SKYTT KLAR", COMMAND_KLAR_S)
        print(
            f"[KOMPETENSPROV] start profile={self.profile.profile_id} "
            f"distance={self.profile.distance_m:.0f}m pk_req={self.profile.min_pk:.2f}"
        )

    def _start_command(self, state: str, text: str, duration: float) -> None:
        now = time.monotonic()
        key = "klar" if state == "klar" else "fardig"
        self.state = state
        self.command = CommandState(text=text, sound_key=key, started=now, duration=duration)
        self._play(key)

    def _begin_fire(self) -> None:
        now = time.monotonic()
        self.state = "firing"
        self.command = None
        self.fire_started = now
        self.next_beep_s = 5.0
        self.eld_overlay_until = 0.0 if self.physical_test else now + ELD_OVERLAY_S
        self._play("eld")
        print("[KOMPETENSPROV] ELD t=0.000")

    def _begin_ceasefire(self) -> None:
        if self.state != "firing":
            return
        now = time.monotonic()
        self.state = "ceasefire"
        self.ceasefire_started = now
        self._play("upphor")
        self._capture_review_snapshot()
        result = evaluate_result(self.shots, self.profile)
        print(
            "[KOMPETENSPROV] ELD UPPHÖR "
            f"shots={result.total_shots} hits={result.total_hits} "
            f"last={result.last_shot_time_s} points={result.counted_points} "
            f"pk={result.pk:.2f} passed={result.passed}"
        )

    # ------------------------------------------------------------------
    # Input / shot capture
    # ------------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event):
        if self.state == "review" and event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self._manual_add_at_screen(event.pos)
            elif event.button == 3:
                self._manual_remove_at_screen(event.pos)
            return None

        if event.type != pygame.KEYDOWN:
            return None

        if self.state == "setup":
            if event.key in (pygame.K_UP, pygame.K_TAB):
                self.profile_index = (self.profile_index - 1) % len(PROFILES)
                self._update_target_rect()
            elif event.key == pygame.K_DOWN:
                self.profile_index = (self.profile_index + 1) % len(PROFILES)
                self._update_target_rect()
            elif event.key == pygame.K_1:
                self.profile_index = 0
                self._update_target_rect()
            elif event.key == pygame.K_2:
                self.profile_index = min(1, len(PROFILES) - 1)
                self._update_target_rect()
            elif event.key == pygame.K_LEFT:
                self._move_target(-TARGET_SHIFT_STEP_CM)
            elif event.key == pygame.K_RIGHT:
                self._move_target(TARGET_SHIFT_STEP_CM)
            elif event.key == pygame.K_r:
                self.target_offset_cm = 0.0
                self._update_target_rect()
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self._start_competition()
            return None

        if self.state == "review":
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._confirm_manual_review()
            elif event.key == pygame.K_r:
                self.manual_marks.clear()
                self.review_message = "Manuella markeringar återställda"
            elif event.key == pygame.K_a:
                self._seed_manual_from_detected()
            elif event.key == pygame.K_SPACE:
                # Explicit escape hatch: preserve AUTO result and skip human GT.
                self.manual_confirmed = False
                self._save_review_record()
                self.state = "results"
            return None

        if self.state == "results":
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                # Always return to setup so target position can be changed before
                # every new physical series.
                self._reset_to_select()
            elif event.key == pygame.K_m:
                self._reset_to_select()
            elif event.key in (pygame.K_TAB, pygame.K_z):
                self._show_zone_overlay = not self._show_zone_overlay
            return None

        if self.physical_test and self.state == "firing" and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._begin_ceasefire()
        elif event.key == pygame.K_r:
            # During commands/firing, restart only by going back to setup.
            self._reset_to_select()
        elif event.key == pygame.K_m:
            self._reset_to_select()
        return None

    def _mask_zone_at(self, x: float, y: float, *, rect=None) -> tuple[str | None, int]:
        """Exact pixel classification; the region AABB is never hit authority."""
        if self.zone_mask is None or not all(math.isfinite(v) for v in (x, y)):
            return None, 0
        rect = self._target_local_rect if rect is None else rect
        if rect.w <= 0 or rect.h <= 0:
            return None, 0
        nx, ny = (x - rect.x) / rect.w, (y - rect.y) / rect.h
        if not (0.0 <= nx < 1.0 and 0.0 <= ny < 1.0):
            return None, 0
        mw, mh = self.zone_mask.get_size()
        zone = ZONE_ID_TO_LETTER.get(int(self.zone_mask.get_at(
            (int(nx * mw), int(ny * mh))).r))
        return zone, int(ZONE_POINTS.get(zone, 0))

    def _zone_for_raw_point(self, raw_x: float, raw_y: float, *, rect=None):
        zone, points = self._mask_zone_at(raw_x, raw_y, rect=rect)
        return float(raw_x), float(raw_y), 0.0, zone, points

    def _rect_for_hit(self, hit):
        """Reuse the PANG-time region, never resolve against a moved target."""
        sid = getattr(hit, "shot_id", None)
        snapshot = hit_context_snapshot_for_shot(sid) if sid is not None else None
        if snapshot is not None:
            for region in snapshot.game_regions:
                if region.object_id == TARGET_OBJECT_ID:
                    return pygame.Rect(region.metadata.get("target_rect", region.rect))
        return self._target_local_rect.copy()

    def _on_hit(self, hit: HitEvent) -> None:
        if self.state != "firing" or self.fire_started is None:
            return
        now = time.monotonic()
        elapsed = now - self.fire_started
        if elapsed < 0.0 or (not self.physical_test and elapsed > SIMULATOR_WINDOW_S):
            return

        raw_shot_id = getattr(hit, "shot_id", None)
        shot_id = int(raw_shot_id) if raw_shot_id is not None else None
        if shot_id is not None:
            if shot_id in self._accepted_shot_ids:
                return
        elif now - self._last_accept_monotonic < SHOT_DEBOUNCE_S:
            return

        self._update_target_rect()
        raw_x = float(hit.game_x)
        raw_y = float(hit.game_y)
        rect = self._rect_for_hit(hit)
        scored_x, scored_y, ballistic_cm, zone, points = self._zone_for_raw_point(raw_x, raw_y, rect=rect)
        detection_mode = "DIRECT"

        norm_x = (scored_x - rect.x) / max(1.0, float(rect.w))
        norm_y = (scored_y - rect.y) / max(1.0, float(rect.h))
        shot_number = len(self.shots) + 1
        record = ShotRecord(
            shot_number=shot_number,
            shot_id=shot_id,
            time_s=float(elapsed),
            raw_x=raw_x,
            raw_y=raw_y,
            scored_x=scored_x,
            scored_y=scored_y,
            norm_x=float(norm_x),
            norm_y=float(norm_y),
            zone=zone,
            points=int(points),
            expected_posture=expected_posture(shot_number),
            detection_mode=detection_mode,
            ballistic_cm=float(ballistic_cm),
            camera_x=float(hit.camera_x), camera_y=float(hit.camera_y),
            source=str(hit.source), hit_timestamp=float(hit.timestamp),
            target_rect=tuple(rect),
        )
        self.shots.append(record)
        if shot_id is not None:
            self._accepted_shot_ids.add(shot_id)
        self._last_accept_monotonic = now
        if self.physical_test:
            print("[KOMPETENSPROV-PHYSICAL] " + json.dumps(asdict(record), separators=(",", ":")))

        print(
            "[KOMPETENSPROV] "
            f"shot={shot_number} shot_id={shot_id} t={elapsed:.3f}s "
            f"posture={record.expected_posture} raw=({record.raw_x:.1f},{record.raw_y:.1f}) "
            f"scored=({record.scored_x:.1f},{record.scored_y:.1f}) "
            f"ak4={record.ballistic_cm:+.1f}cm zone={record.zone or 'MISS'} "
            f"points={record.points} mode={record.detection_mode}"
        )


    # ------------------------------------------------------------------
    # Post-fire manual verification
    # ------------------------------------------------------------------
    def _fallback_review_surface(self) -> pygame.Surface:
        surf = pygame.Surface(self.local_viewport.size)
        if self.background is not None:
            bg = pygame.transform.smoothscale(self.background, self.local_viewport.size)
            surf.blit(bg, (0, 0))
        else:
            surf.fill((76, 95, 78))
        if self.target is not None:
            rect = self._target_local_rect
            scaled = pygame.transform.smoothscale(self.target, rect.size)
            surf.blit(scaled, rect.topleft)
        return surf

    def _camera_frame_to_viewport_surface(self, frame_bgr: np.ndarray) -> pygame.Surface | None:
        try:
            if frame_bgr is None or frame_bgr.size == 0:
                return None

            warped = None
            H = getattr(hit_input, "homography", None)
            if H is not None:
                H = np.asarray(H, dtype=np.float64)
                translate = np.array(
                    [
                        [1.0, 0.0, -float(self.viewport.x)],
                        [0.0, 1.0, -float(self.viewport.y)],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                local_h = translate @ H
                warped = cv2.warpPerspective(
                    frame_bgr,
                    local_h,
                    (int(self.viewport.w), int(self.viewport.h)),
                    flags=cv2.INTER_LINEAR,
                )
                self.review_source = "camera/homography"
            else:
                scanport = load_scanport_rect()
                if scanport is not None:
                    h, w = frame_bgr.shape[:2]
                    x0 = max(0, int(scanport.x))
                    y0 = max(0, int(scanport.y))
                    x1 = min(w, int(scanport.x + scanport.w))
                    y1 = min(h, int(scanport.y + scanport.h))
                    if x1 > x0 and y1 > y0:
                        crop = frame_bgr[y0:y1, x0:x1]
                        warped = cv2.resize(
                            crop,
                            (int(self.viewport.w), int(self.viewport.h)),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        self.review_source = "camera/scanport"

            if warped is None:
                return None
            rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            arr = np.transpose(rgb, (1, 0, 2))
            return pygame.surfarray.make_surface(arr.copy())
        except Exception as exc:
            print(f"[KOMPETENSPROV] manual-review camera conversion failed: {exc}")
            return None

    def _capture_review_snapshot(self) -> None:
        self._update_target_rect()
        frame = None
        try:
            frame = camera_manager.capture_fresh_frame()
            if frame is None:
                frame = camera_manager.get_latest_frame()
        except Exception as exc:
            print(f"[KOMPETENSPROV] review camera capture failed: {exc}")

        surface = self._camera_frame_to_viewport_surface(frame) if frame is not None else None
        if surface is None:
            surface = self._fallback_review_surface()
            self.review_source = "render-fallback"
        self.review_surface = surface

        rect = self._target_local_rect
        pad_x = max(35, int(rect.w * 0.55))
        pad_y = max(30, int(rect.h * 0.12))
        crop = rect.inflate(pad_x * 2, pad_y * 2)
        crop.clamp_ip(self.local_viewport)
        self._review_crop_local = crop
        print(
            "[KOMPETENSPROV] manual-review snapshot "
            f"source={self.review_source} crop={tuple(crop)} offset_cm={self.target_offset_cm:+.1f}"
        )

    def _fit_review_display_rect(self, crop: pygame.Rect) -> pygame.Rect:
        max_w = max(320, int(self.viewport.w * 0.64))
        max_h = max(260, int(self.viewport.h * 0.78))
        aspect = crop.w / max(1.0, float(crop.h))
        w = max_w
        h = int(round(w / max(0.01, aspect)))
        if h > max_h:
            h = max_h
            w = int(round(h * aspect))
        x = self.viewport.x + 24
        y = self.viewport.y + max(64, (self.viewport.h - h) // 2)
        return pygame.Rect(x, y, max(1, w), max(1, h))

    def _review_screen_to_game(self, pos: tuple[int, int]) -> tuple[float, float] | None:
        display = self._review_display_screen
        crop = self._review_crop_local
        if not display.collidepoint(pos):
            return None
        nx = (float(pos[0]) - display.x) / max(1.0, float(display.w))
        ny = (float(pos[1]) - display.y) / max(1.0, float(display.h))
        return (
            float(crop.x) + nx * float(crop.w),
            float(crop.y) + ny * float(crop.h),
        )

    def _manual_add_at_screen(self, pos: tuple[int, int]) -> None:
        point = self._review_screen_to_game(pos)
        if point is None:
            return
        if len(self.manual_marks) >= len(self.shots):
            self.review_message = "Kan inte markera fler träffar än registrerade skott"
            return

        raw_x, raw_y = point
        scored_x, scored_y, _ballistic_cm, zone, points = self._zone_for_raw_point(raw_x, raw_y)
        rect = self._target_local_rect
        norm_x = (scored_x - rect.x) / max(1.0, float(rect.w))
        norm_y = (scored_y - rect.y) / max(1.0, float(rect.h))
        mark = ManualMark(
            mark_number=len(self.manual_marks) + 1,
            raw_x=float(raw_x),
            raw_y=float(raw_y),
            scored_x=float(scored_x),
            scored_y=float(scored_y),
            norm_x=float(norm_x),
            norm_y=float(norm_y),
            zone=zone,
            points=int(points),
        )
        self.manual_marks.append(mark)
        self.review_message = (
            f"M{mark.mark_number}: {mark.zone or 'UTANFÖR'} "
            f"{mark.points}p"
        )

    def _manual_remove_at_screen(self, pos: tuple[int, int]) -> None:
        point = self._review_screen_to_game(pos)
        if point is None or not self.manual_marks:
            return
        gx, gy = point
        nearest_i = min(
            range(len(self.manual_marks)),
            key=lambda i: math.hypot(
                self.manual_marks[i].raw_x - gx,
                self.manual_marks[i].raw_y - gy,
            ),
        )
        mark = self.manual_marks[nearest_i]
        threshold = max(
            REVIEW_REMOVE_RADIUS_PX,
            0.12 * max(self._target_local_rect.w, self._target_local_rect.h),
        )
        if math.hypot(mark.raw_x - gx, mark.raw_y - gy) <= threshold:
            self.manual_marks.pop(nearest_i)
            # Re-number only manual marks; detector ShotRecords stay untouched.
            self.manual_marks = [
                ManualMark(
                    mark_number=i + 1,
                    raw_x=m.raw_x,
                    raw_y=m.raw_y,
                    scored_x=m.scored_x,
                    scored_y=m.scored_y,
                    norm_x=m.norm_x,
                    norm_y=m.norm_y,
                    zone=m.zone,
                    points=m.points,
                )
                for i, m in enumerate(self.manual_marks)
            ]
            self.review_message = "Markering borttagen"

    def _seed_manual_from_detected(self) -> None:
        """Convenience only: copy AUTO hits into the manual layer for editing."""
        marks: list[ManualMark] = []
        for shot in self.shots:
            if not shot.is_hit:
                continue
            marks.append(
                ManualMark(
                    mark_number=len(marks) + 1,
                    raw_x=shot.raw_x,
                    raw_y=shot.raw_y,
                    scored_x=shot.scored_x,
                    scored_y=shot.scored_y,
                    norm_x=shot.norm_x,
                    norm_y=shot.norm_y,
                    zone=shot.zone,
                    points=shot.points,
                )
            )
        self.manual_marks = marks
        self.review_message = f"{len(marks)} AUTO-träffar kopierade – justera manuellt"

    def _save_review_record(self) -> None:
        try:
            auto_result = evaluate_result(self.shots, self.profile)
            manual_result = (
                evaluate_manual_result(
                    self.manual_marks,
                    total_shots=len(self.shots),
                    last_shot_time_s=auto_result.last_shot_time_s,
                    profile=self.profile,
                )
                if self.manual_confirmed
                else None
            )
            payload = {
                "schema": "kompetensprov_manual_review_v1",
                "saved_at": datetime.now().astimezone().isoformat(),
                "profile": {
                    "id": self.profile.profile_id,
                    "title": self.profile.title,
                    "distance_m": self.profile.distance_m,
                    "min_pk": self.profile.min_pk,
                },
                "target_offset_cm": float(self.target_offset_cm),
                "review_source": self.review_source,
                "manual_confirmed": bool(self.manual_confirmed),
                "detected_hits": [asdict(s) for s in self.shots],
                "automatic_result": asdict(auto_result),
                "manual_verified_hits": [asdict(m) for m in self.manual_marks],
                "manual_result": None if manual_result is None else asdict(manual_result),
            }
            root = REVIEW_DATA_ROOT
            try:
                root.mkdir(parents=True, exist_ok=True)
            except Exception:
                root = Path("data/kompetensprov_manual_reviews")
                root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = root / f"kompetensprov_{self.profile.profile_id}_{stamp}.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[KOMPETENSPROV] manual review saved: {path}")
        except Exception as exc:
            print(f"[KOMPETENSPROV] manual review save failed: {exc}")

    def _confirm_manual_review(self) -> None:
        self.manual_confirmed = True
        self._save_review_record()
        self.state = "results"

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, dt: float):
        del dt
        now = time.monotonic()
        self._update_target_rect()

        if self.state == "klar" and self.command is not None:
            if now - self.command.started >= self.command.duration:
                self._start_command("fardig", "SKYTT FÄRDIG", COMMAND_FARDIG_S)
        elif self.state == "fardig" and self.command is not None:
            if now - self.command.started >= self.command.duration:
                self._begin_fire()
        elif self.state == "firing" and self.fire_started is not None:
            elapsed = now - self.fire_started
            while not self.physical_test and self.next_beep_s < SIMULATOR_WINDOW_S and elapsed >= self.next_beep_s:
                self._play("beep")
                print(f"[KOMPETENSPROV] time beep {self.next_beep_s:.0f}s")
                self.next_beep_s += 5.0
            if not self.physical_test and elapsed >= SIMULATOR_WINDOW_S:
                self._begin_ceasefire()
        elif self.state == "ceasefire" and self.ceasefire_started is not None:
            if now - self.ceasefire_started >= RESULT_DELAY_S:
                self.state = "review"
        return None

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _render_background(self, screen: pygame.Surface) -> None:
        if self.background_scaled is not None:
            screen.blit(self.background_scaled, self.viewport.topleft)
        else:
            pygame.draw.rect(screen, (76, 95, 78), self.viewport)

    def _render_target(self, screen: pygame.Surface, *, zones: bool = False) -> None:
        source = self.target_zones if zones else self.target
        if source is None:
            return
        rect = self._target_screen_rect
        if rect.w <= 1 or rect.h <= 1:
            return
        scaled = pygame.transform.smoothscale(source, rect.size)
        screen.blit(scaled, rect.topleft)

    @staticmethod
    def _panel(size: tuple[int, int], alpha: int = 220) -> pygame.Surface:
        surf = pygame.Surface(size, pygame.SRCALPHA)
        surf.fill((10, 15, 17, alpha))
        return surf

    def _center_text(self, screen, text, font, color, y, shadow=True):
        if font is None:
            return
        img = font.render(text, True, color)
        x = self.viewport.centerx - img.get_width() // 2
        if shadow:
            shadow_img = font.render(text, True, BLACK)
            screen.blit(shadow_img, (x + 3, y + 3))
        screen.blit(img, (x, y))

    def _render_select(self, screen: pygame.Surface) -> None:
        """Pre-start setup: choose BAS profile and move the physical aim area."""
        if None in (self.font_title, self.font_big, self.font_medium, self.font_small):
            return

        # Keep the actual target clearly visible during setup. Only small panels
        # are used; do not veil the whole firing area.
        title_panel = pygame.Surface((min(780, self.viewport.w - 40), 114), pygame.SRCALPHA)
        title_panel.fill((8, 13, 15, 218))
        tx = self.viewport.centerx - title_panel.get_width() // 2
        ty = self.viewport.y + 16
        screen.blit(title_panel, (tx, ty))
        self._center_text(screen, "KOMPETENSPROV AK", self.font_title, WHITE, ty + 10)
        self._center_text(
            screen,
            f"{self.profile.title} • {self.profile.distance_m:.0f} m • Pk ≥ {self.profile.min_pk:.2f}",
            self.font_small,
            AMBER,
            ty + 70,
        )

        info_w = min(760, self.viewport.w - 50)
        info_h = 136
        panel = pygame.Surface((info_w, info_h), pygame.SRCALPHA)
        panel.fill((8, 13, 15, 218))
        px = self.viewport.centerx - info_w // 2
        py = self.viewport.bottom - info_h - 18
        screen.blit(panel, (px, py))

        lines = [
            f"Figur sidled: {self.target_offset_cm:+.0f} cm på den fysiska projektionsytan",
            "← / → flytta figur 5 cm     ↑ / ↓ välj BAS Hv 30 m eller BAS 50 m",
            "R centrera figur            ENTER startar SKYTT KLAR-sekvensen",
            ("FYSIKTEST: ENTER start/stopp, ljud av; vänta 5 s före/efter skott."
             if self.physical_test else "Inga skott eller tider registreras före ELD."),
        ]
        y = py + 14
        for i, line in enumerate(lines):
            img = self.font_small.render(line, True, WHITE if i == 2 else MUTED)
            screen.blit(img, (self.viewport.centerx - img.get_width() // 2, y))
            y += 27

    def _render_command(self, screen: pygame.Surface, text: str) -> None:
        veil = pygame.Surface(self.viewport.size, pygame.SRCALPHA)
        veil.fill((0, 0, 0, 62))
        screen.blit(veil, self.viewport.topleft)
        panel = self._panel((min(800, self.viewport.w - 80), 150), 205)
        px = self.viewport.centerx - panel.get_width() // 2
        py = self.viewport.centery - panel.get_height() // 2
        screen.blit(panel, (px, py))
        self._center_text(screen, text, self.font_title, AMBER, py + 42)

    def _render_review(self, screen: pygame.Surface) -> None:
        if self.review_surface is None:
            self.review_surface = self._fallback_review_surface()
            self.review_source = "render-fallback"

        dim = pygame.Surface(self.viewport.size, pygame.SRCALPHA)
        dim.fill((4, 7, 8, 224))
        screen.blit(dim, self.viewport.topleft)

        crop = self._review_crop_local
        display = self._fit_review_display_rect(crop)
        self._review_display_screen = display

        try:
            crop_surface = self.review_surface.subsurface(crop).copy()
        except Exception:
            crop_surface = self.review_surface.copy()
            crop = self.local_viewport.copy()
            self._review_crop_local = crop

        scaled = pygame.transform.smoothscale(crop_surface, display.size)
        screen.blit(scaled, display.topleft)
        pygame.draw.rect(screen, WHITE, display, 2)

        # Raw physical manual marks are drawn on the camera image. Scoring uses
        # the same unchanged physical point in the zone mask.
        for mark in self.manual_marks:
            nx = (mark.raw_x - crop.x) / max(1.0, float(crop.w))
            ny = (mark.raw_y - crop.y) / max(1.0, float(crop.h))
            x = display.x + nx * display.w
            y = display.y + ny * display.h
            color = COUNTED if mark.is_hit else RED
            pygame.draw.circle(screen, BLACK, (int(x), int(y)), 12)
            pygame.draw.circle(screen, color, (int(x), int(y)), 9, 3)
            if self.font_tiny:
                label = self.font_tiny.render(
                    f"{mark.mark_number}:{mark.zone or '-'} {mark.points}p",
                    True,
                    color,
                )
                screen.blit(label, (int(x) + 12, int(y) - 9))

        right_x = display.right + 22
        right_w = max(260, self.viewport.right - right_x - 18)
        panel = pygame.Surface((right_w, min(display.h, self.viewport.h - 80)), pygame.SRCALPHA)
        panel.fill(PANEL)
        panel_y = display.y
        screen.blit(panel, (right_x, panel_y))

        y = panel_y + 18
        def line(text: str, font, color=WHITE, gap=29):
            nonlocal y
            if font is not None:
                img = font.render(text, True, color)
                screen.blit(img, (right_x + 18, y))
                y += gap

        line("MANUELL KONTROLL", self.font_big, AMBER, 44)
        line(f"Bild: {self.review_source}", self.font_tiny, MUTED, 25)
        line(f"Registrerade skott: {len(self.shots)}", self.font_small, WHITE, 28)
        line(f"Manuella markeringar: {len(self.manual_marks)}", self.font_small, WHITE, 35)
        line("Vänsterklick  lägg till verkligt kulhål", self.font_tiny, WHITE, 23)
        line("Högerklick   ta bort markering", self.font_tiny, WHITE, 23)
        line("R             återställ manuella", self.font_tiny, WHITE, 23)
        line("A             kopiera AUTO som start", self.font_tiny, MUTED, 23)
        line("ENTER         godkänn manuell kontroll", self.font_tiny, GREEN, 23)
        line("SPACE         hoppa över / använd AUTO", self.font_tiny, MUTED, 34)

        if self.review_message:
            line(self.review_message, self.font_tiny, CYAN, 26)

        line(
            "Manuell kontroll sparas separat.",
            self.font_tiny,
            MUTED,
            21,
        )
        line(
            "Den ändrar aldrig detectorns emitted truth.",
            self.font_tiny,
            MUTED,
            21,
        )

    def _result_target_rect(self) -> pygame.Rect:
        result_h = int(self.viewport.h * 0.76)
        if self.target is None:
            aspect = 0.42
        else:
            tw, th = self.target.get_size()
            aspect = tw / max(1.0, float(th))
        result_w = int(result_h * aspect)
        x = self.viewport.x + max(28, int(self.viewport.w * 0.07))
        y = self.viewport.y + (self.viewport.h - result_h) // 2
        return pygame.Rect(x, y, result_w, result_h)

    def _render_results(self, screen: pygame.Surface) -> None:
        auto_result: CompetitionResult = evaluate_result(self.shots, self.profile)
        if self.manual_confirmed:
            result = evaluate_manual_result(
                self.manual_marks,
                total_shots=len(self.shots),
                last_shot_time_s=auto_result.last_shot_time_s,
                profile=self.profile,
            )
            counted = set(result.counted_shot_numbers)
            result_marks = tuple(self.manual_marks)
        else:
            result = auto_result
            counted = set(result.counted_shot_numbers)
            result_marks = ()

        dim = pygame.Surface(self.viewport.size, pygame.SRCALPHA)
        dim.fill((4, 7, 8, 205))
        screen.blit(dim, self.viewport.topleft)

        target_rect = self._result_target_rect()
        source = self.target_zones if self._show_zone_overlay else self.target
        if source is not None:
            scaled = pygame.transform.smoothscale(source, target_rect.size)
            screen.blit(scaled, target_rect.topleft)

        # Result close-up. Manual verification, when confirmed, is the displayed
        # competition score; automatic detector output remains preserved in
        # self.shots and the saved review JSON.
        if self.manual_confirmed:
            for mark in result_marks:
                x = target_rect.x + mark.norm_x * target_rect.w
                y = target_rect.y + mark.norm_y * target_rect.h
                if not (
                    target_rect.left - 28 <= x <= target_rect.right + 28
                    and target_rect.top - 28 <= y <= target_rect.bottom + 28
                ):
                    continue
                is_counted = mark.mark_number in counted
                color = COUNTED if is_counted else EXTRA
                radius = 12 if is_counted else 8
                pygame.draw.circle(screen, BLACK, (int(x), int(y)), radius + 3)
                pygame.draw.circle(screen, color, (int(x), int(y)), radius, 3)
                if self.font_tiny:
                    num = self.font_tiny.render(f"M{mark.mark_number}", True, color)
                    screen.blit(
                        num,
                        (int(x - num.get_width() / 2), int(y - num.get_height() / 2)),
                    )
        else:
            for shot in self.shots:
                x = target_rect.x + shot.norm_x * target_rect.w
                y = target_rect.y + shot.norm_y * target_rect.h
                if not (
                    target_rect.left - 28 <= x <= target_rect.right + 28
                    and target_rect.top - 28 <= y <= target_rect.bottom + 28
                ):
                    continue

                is_counted = shot.shot_number in counted
                if shot.is_hit and is_counted:
                    color = COUNTED
                    radius = 12
                elif shot.is_hit:
                    color = EXTRA
                    radius = 8
                else:
                    color = MISS
                    radius = 9

                if shot.is_hit:
                    pygame.draw.circle(screen, BLACK, (int(x), int(y)), radius + 3)
                    pygame.draw.circle(screen, color, (int(x), int(y)), radius, 3)
                    if self.font_tiny:
                        num = self.font_tiny.render(str(shot.shot_number), True, color)
                        screen.blit(
                            num,
                            (int(x - num.get_width() / 2), int(y - num.get_height() / 2)),
                        )
                else:
                    pygame.draw.line(
                        screen,
                        color,
                        (int(x - radius), int(y - radius)),
                        (int(x + radius), int(y + radius)),
                        3,
                    )
                    pygame.draw.line(
                        screen,
                        color,
                        (int(x + radius), int(y - radius)),
                        (int(x - radius), int(y + radius)),
                        3,
                    )

        panel_x = target_rect.right + 34
        panel_w = self.viewport.right - panel_x - 26
        panel_w = max(380, panel_w)
        panel = pygame.Surface((panel_w, int(self.viewport.h * 0.88)), pygame.SRCALPHA)
        panel.fill(PANEL)
        panel_y = self.viewport.y + int(self.viewport.h * 0.06)
        screen.blit(panel, (panel_x, panel_y))

        y = panel_y + 20
        def line(text: str, font, color=WHITE, gap=28):
            nonlocal y
            if font is not None:
                img = font.render(text, True, color)
                screen.blit(img, (panel_x + 22, y))
                y += gap

        line(self.profile.title, self.font_big, AMBER, 46)
        line(f"{self.profile.distance_m:.0f} m • krav Pk {self.profile.min_pk:.2f}", self.font_small, MUTED, 30)
        line("AK4B/C • Ptr 10 • RPS EHV", self.font_small, MUTED, 30)
        line(
            f"Resultatkälla: {'MANUELL VERIFIERING' if self.manual_confirmed else 'AUTO'}",
            self.font_small,
            GREEN if self.manual_confirmed else CYAN,
            36,
        )

        last_text = "–" if result.last_shot_time_s is None else f"{result.last_shot_time_s:.2f} s"
        line(f"Avfyrade skott: {result.total_shots}", self.font_medium, WHITE, 34)
        line(f"Träffar: {result.total_hits}", self.font_medium, WHITE, 34)
        line(f"9 bästa: {result.counted_points} p", self.font_medium, WHITE, 34)
        line(f"Tid till sista skott: {last_text}", self.font_medium, WHITE, 34)
        line(f"Poängkvot: {result.pk:.2f}", self.font_medium, AMBER, 44)

        status = "GODKÄND" if result.passed else "UNDERKÄND"
        line(status, self.font_title, GREEN if result.passed else RED, 58)
        line(result.reason, self.font_small, WHITE, 34)

        line("Zon = fysisk XY; ingen virtuell banhöjdsförskjutning", self.font_small, CYAN, 34)
        line("Räknade träffar", self.font_medium, WHITE, 34)

        if self.manual_confirmed:
            selected_manual = [m for m in self.manual_marks if m.mark_number in counted]
            selected_manual.sort(key=lambda m: m.mark_number)
            for mark in selected_manual[:COUNTED_HITS]:
                line(
                    f"M{mark.mark_number:02d}  {mark.zone or '-'}  {mark.points}p",
                    self.font_tiny,
                    COUNTED,
                    22,
                )
        else:
            selected = [s for s in self.shots if s.shot_number in counted]
            selected.sort(key=lambda s: s.shot_number)
            for shot in selected[:COUNTED_HITS]:
                line(
                    f"#{shot.shot_number:02d}  {shot.zone}  {shot.points}p   {shot.time_s:5.2f}s   {shot.expected_posture}",
                    self.font_tiny,
                    COUNTED,
                    22,
                )

        footer_y = self.viewport.bottom - 34
        footer = "ENTER = nytt förberedelseläge • M = menyval • TAB = zoner av/på"
        if self.font_small is not None:
            img = self.font_small.render(footer, True, MUTED)
            screen.blit(img, (self.viewport.centerx - img.get_width() // 2, footer_y))

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render(self, screen: pygame.Surface) -> None:
        self._render_background(screen)
        self._update_target_rect()

        if self.state == "results":
            self._render_results(screen)
            return
        if self.state == "review":
            self._render_review(screen)
            return

        self._render_target(screen, zones=False)

        if self.state == "setup":
            self._render_select(screen)
            return

        if self.state in {"klar", "fardig"} and self.command is not None:
            self._render_command(screen, self.command.text)
            return

        if self.state == "firing":
            # No timer, shot count, score or HUD during fire. Only the mandatory
            # ELD command is visible briefly at the beginning.
            if time.monotonic() < self.eld_overlay_until:
                self._render_command(screen, "ELD!")
            return

        if self.state == "ceasefire":
            self._render_command(screen, "ELD UPPHÖR")


def create_game(game_root: str, viewport: pygame.Rect):
    return KompetensprovAKGame(game_root=game_root, viewport=viewport)
