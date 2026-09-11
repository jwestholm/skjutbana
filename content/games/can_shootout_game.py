from __future__ import annotations

"""
Skjutprylar - physically scaled two-player target game.

Default virtual distance: 20 m
Distance controls: + / - in 5 m steps (starts a fresh match).

The game deliberately keeps physical-hit authority honest:
- Every active target is a V2.25 GameObject.
- Targets expose HitRegions through ObjectManager.
- Direct gameplay score first uses the canonical emitted HitEvent.game_x/game_y.
- If that misses, this game alone may use a bounded same-shot SHADOW candidate
  that still resolves against the exact frozen target shape (temporary assist).
- The emitted coordinate is always retained/marked; assisted XY is marked separately.

Target dimensions below are the explicit physical game-world dimensions used for
projection. Standard beverage-can dimensions are representative 33 cl dimensions;
the humorous objects use intentionally explicit approximate game dimensions.
"""

from dataclasses import dataclass
import math
import random
import time
from types import SimpleNamespace

import pygame

from src.engine.game_objects import (
    HitShapeSpec,
    ObjectGeometry,
    ObjectManager,
    PenetrationMode,
    ProjectileProfile,
    WorldPlacement,
    make_breakable_object,
)
from src.engine.input.hit_input import HitEvent, hit_input
from src.engine.input.object_hit_v2223 import object_hit_registry_v2223
from src.engine.range_projection import (
    RangeProjectionGeometry,
    cm_to_viewport_px_x,
    cm_to_viewport_px_y,
    projected_ground_anchor_y_px,
    projected_lateral_offset_px,
    projected_size_on_wall_cm,
    projected_target_height_px,
)
from src.engine.settings import load_range_projection_settings


# ---------------------------------------------------------------------------
# Match / range
# ---------------------------------------------------------------------------

DEFAULT_DISTANCE_M = 20.0
DISTANCE_STEP_M = 5.0
MAX_DISTANCE_M = 100.0
SHOTS_PER_PLAYER = 5
PLAYERS = (1, 2)

SHELF_HEIGHT_CM = 135.0
HORIZON_Y_NORM = 0.43
HUD_HEIGHT = 132

INTRO_SECONDS = 1.55
SHOT_RESULT_SECONDS = 1.15
PLAYER_RESULT_SECONDS = 2.15

PERFECT_FIVE_BONUS = 500
STREAK_BONUS_STEP = 25
MAX_STREAK_BONUS = 100

# Seven targets but only five shots: choose your priorities.
TARGET_LATERAL_OFFSETS_M = (-0.90, -0.60, -0.30, 0.0, 0.30, 0.60, 0.90)

GAME_PROJECTILE = ProjectileProfile(
    profile_id="shoot_stuff",
    damage=1.0,
    penetration_power=0.0,
    max_object_hits=1,
    tags=frozenset({"shoot_stuff_game"}),
)

# ---------------------------------------------------------------------------
# GAME-LOCAL temporary aim/hit assist
# ---------------------------------------------------------------------------
#
# This is intentionally NOT detector authority and must not be promoted to the
# engine.  It exists only so this game remains fun while final physical XY
# selection is still under active research.
#
# Flow:
#   final HitEvent misses -> inspect SAME-SHOT object shadow candidates ->
#   require a bounded/high candidate -> convert candidate camera XY to game XY ->
#   resolve again against the exact frozen V2.25 object shape.
#
# The original emitted XY is kept and displayed.  Assisted hits are logged and
# marked separately, so they cannot be confused with real detector accuracy.
GAMEPLAY_ASSIST_ENABLED = True
GAMEPLAY_ASSIST_MAX_CANDIDATE_RANK = 25
GAMEPLAY_ASSIST_MIN_CONFIDENCE = 0.18



# ---------------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    title: str
    subtitle: str
    width_cm: float
    height_cm: float
    points: int
    draw_kind: str
    primary: tuple[int, int, int]
    secondary: tuple[int, int, int]
    shape_kind: str = "rect"


TARGET_SPECS = (
    TargetSpec(
        "mini_can", "MINIBURK", "Liten men dryg",
        5.0, 8.0, 225, "can",
        (75, 153, 210), (29, 78, 120), "can",
    ),
    TargetSpec(
        "soda_can", "LÄSKBURK", "Klassikern",
        6.6, 11.5, 125, "can",
        (205, 74, 62), (121, 31, 27), "can",
    ),
    TargetSpec(
        "toilet_roll", "TOARULLE", "Krisberedskap",
        10.5, 10.0, 175, "roll",
        (235, 229, 210), (176, 165, 146), "ellipse",
    ),
    TargetSpec(
        "gold_can", "GULDBURK", "Helt onödigt blank",
        6.6, 11.5, 300, "can",
        (245, 191, 57), (153, 102, 16), "can",
    ),
    TargetSpec(
        "rubber_duck", "GUMMIANKA", "Kvack +250",
        11.5, 10.0, 250, "duck",
        (250, 211, 56), (235, 130, 31), "ellipse",
    ),
    TargetSpec(
        "beans_xxl", "BÖNBURK XXL", "Stor. Inte smart.",
        10.0, 15.0, 100, "can",
        (115, 173, 87), (50, 93, 42), "can",
    ),
    TargetSpec(
        "sardines", "SARDINER", "Låg profil",
        12.5, 3.5, 275, "tin",
        (143, 177, 191), (59, 91, 106), "rect",
    ),
)


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

SKY_TOP = (53, 91, 128)
SKY_BOTTOM = (167, 191, 199)
GROUND_TOP = (89, 115, 73)
GROUND_BOTTOM = (53, 73, 48)
TEXT = (247, 246, 239)
MUTED = (202, 206, 201)
PANEL = (12, 18, 22)
PANEL_2 = (23, 31, 36)
AMBER = (255, 190, 70)
GOOD = (105, 235, 145)
BAD = (255, 91, 82)
CYAN = (100, 220, 245)
WHITE = (255, 255, 255)
BLACK = (12, 14, 15)
WOOD = (109, 70, 42)
WOOD_LIGHT = (151, 100, 59)
WOOD_DARK = (70, 43, 29)


@dataclass
class ImpactMarker:
    x: float
    y: float
    hit: bool
    label: str
    created: float
    lifetime: float = 1.45


@dataclass
class ScorePop:
    x: float
    y: float
    text: str
    created: float
    lifetime: float = 1.25


@dataclass
class KnockedTarget:
    x: float
    y: float
    width: float
    height: float
    spec_index: int
    vx: float
    vy: float
    angle: float
    angular_velocity: float
    age: float = 0.0
    lifetime: float = 1.75


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    life: float
    max_life: float
    kind: str = "spark"


class ShootStuffGame:
    def __init__(self, game_root: str, viewport: pygame.Rect) -> None:
        del game_root
        self.viewport = viewport.copy()
        self.local_viewport = pygame.Rect(0, 0, self.viewport.w, self.viewport.h)
        self.manager = ObjectManager()
        self.rng = random.Random()

        self.distance_m = DEFAULT_DISTANCE_M

        self.font_huge = None
        self.font_big = None
        self.font_medium = None
        self.font_small = None
        self.font_tiny = None

        self.player = 1
        self.shots_used = {1: 0, 2: 0}
        self.score = {1: 0, 2: 0}
        self.hits = {1: 0, 2: 0}
        self.streak = {1: 0, 2: 0}
        self.best_streak = {1: 0, 2: 0}
        self.perfect_bonus_given = {1: False, 2: False}

        self.phase = "intro"
        self.phase_until = 0.0
        self.overlay_title = ""
        self.overlay_subtitle = ""
        self.overlay_color = TEXT

        self.markers: list[ImpactMarker] = []
        self.score_pops: list[ScorePop] = []
        self.knocked_targets: list[KnockedTarget] = []
        self.particles: list[Particle] = []

        self._subscribed = False
        self._target_geometry: dict[str, tuple[float, float, float, float]] = {}
        self._shelf_y = float(self.viewport.h * 0.66)
        self._last_resolution_line = ""
        self._last_hit_shot_id = None
        self._background_cache = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        self.font_huge = pygame.font.Font(None, 92)
        self.font_big = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 30)
        self.font_small = pygame.font.Font(None, 23)
        self.font_tiny = pygame.font.Font(None, 17)

        if not self._subscribed:
            hit_input.subscribe(self._on_hit)
            self._subscribed = True

        self.distance_m = max(self._minimum_distance_m(), DEFAULT_DISTANCE_M)
        self.distance_m = self._snap_distance(self.distance_m)
        self._start_match()

    def on_exit(self) -> None:
        if self._subscribed:
            hit_input.unsubscribe(self._on_hit)
            self._subscribed = False

    # ------------------------------------------------------------------
    # Projection / range
    # ------------------------------------------------------------------

    def _projection_geometry(self) -> RangeProjectionGeometry:
        settings = load_range_projection_settings()
        return RangeProjectionGeometry(
            wall_distance_m=float(settings.get("wall_distance_m", 6.3)),
            viewport_physical_width_cm=float(
                settings.get("viewport_physical_width_cm", 90.0)
            ),
            viewport_physical_height_cm=float(
                settings.get("viewport_physical_height_cm", 70.0)
            ),
            viewport_bottom_world_cm=float(
                settings.get("viewport_bottom_world_cm", 105.0)
            ),
        )

    def _minimum_distance_m(self) -> float:
        wall = self._projection_geometry().wall_distance_m
        return max(
            DISTANCE_STEP_M,
            math.ceil(float(wall) / DISTANCE_STEP_M) * DISTANCE_STEP_M,
        )

    @staticmethod
    def _snap_distance(value: float) -> float:
        return round(float(value) / DISTANCE_STEP_M) * DISTANCE_STEP_M

    def _change_distance(self, delta: float) -> None:
        minimum = self._minimum_distance_m()
        requested = self._snap_distance(self.distance_m + delta)
        new_distance = max(minimum, min(MAX_DISTANCE_M, requested))
        if abs(new_distance - self.distance_m) < 0.01:
            return
        self.distance_m = new_distance

        # Distance is a match setting. Changing it deliberately starts a fresh,
        # fair two-player match instead of silently moving already-shot targets.
        self._start_match()
        self.overlay_title = f"{self.distance_m:.0f} METER"
        self.overlay_subtitle = "Ny match • målen är fysiskt omskalade"
        self.overlay_color = AMBER
        self.phase = "intro"
        self.phase_until = time.monotonic() + INTRO_SECONDS

        print(f"[SHOOT-STUFF] distance changed -> {self.distance_m:.0f}m; fresh match")

    def _projected_size_px(self, spec: TargetSpec) -> tuple[float, float]:
        geometry = self._projection_geometry()
        width_wall_cm = projected_size_on_wall_cm(
            spec.width_cm,
            self.distance_m,
            geometry.wall_distance_m,
        )
        width_px = cm_to_viewport_px_x(
            width_wall_cm,
            self.local_viewport,
            geometry.viewport_physical_width_cm,
        )
        height_px = projected_target_height_px(
            spec.height_cm,
            self.distance_m,
            self.local_viewport,
            geometry,
        )
        return max(2.0, width_px), max(2.0, height_px)

    def _projected_shelf_y(self) -> float:
        geometry = self._projection_geometry()
        ground_y = projected_ground_anchor_y_px(
            self.distance_m,
            self.local_viewport,
            geometry,
            horizon_y_norm=HORIZON_Y_NORM,
        )
        shelf_height_wall_cm = projected_size_on_wall_cm(
            SHELF_HEIGHT_CM,
            self.distance_m,
            geometry.wall_distance_m,
        )
        shelf_height_px = cm_to_viewport_px_y(
            shelf_height_wall_cm,
            self.local_viewport,
            geometry.viewport_physical_height_cm,
        )
        max_target_h = max(self._projected_size_px(s)[1] for s in TARGET_SPECS)
        shelf_y = ground_y - shelf_height_px
        return max(
            float(HUD_HEIGHT + max_target_h + 34),
            min(float(self.viewport.h - 64), float(shelf_y)),
        )

    def _target_center_x(self, lateral_offset_m: float) -> float:
        geometry = self._projection_geometry()
        offset_px = projected_lateral_offset_px(
            lateral_offset_m,
            self.distance_m,
            self.local_viewport,
            geometry,
        )
        return float(self.viewport.w * 0.5 + offset_px)

    # ------------------------------------------------------------------
    # GameObject rack
    # ------------------------------------------------------------------

    def _shape_for_spec(self, spec: TargetSpec) -> HitShapeSpec:
        if spec.shape_kind == "ellipse":
            return HitShapeSpec.ellipse(x=0.04, y=0.04, width=0.92, height=0.92)
        if spec.shape_kind == "can":
            return HitShapeSpec.polygon(
                (
                    (0.12, 0.00),
                    (0.88, 0.00),
                    (0.97, 0.08),
                    (0.97, 0.92),
                    (0.88, 1.00),
                    (0.12, 1.00),
                    (0.03, 0.92),
                    (0.03, 0.08),
                )
            )
        return HitShapeSpec.rect(x=0.03, y=0.04, width=0.94, height=0.92)

    def _build_rack(self) -> None:
        self.manager = ObjectManager()
        self._target_geometry.clear()
        self._shelf_y = self._projected_shelf_y()

        for index, (spec, lateral_m) in enumerate(
            zip(TARGET_SPECS, TARGET_LATERAL_OFFSETS_M)
        ):
            target_w, target_h = self._projected_size_px(spec)
            center_x = self._target_center_x(lateral_m)
            geometry = ObjectGeometry(
                x=center_x - target_w * 0.5,
                y=self._shelf_y - target_h,
                width=target_w,
                height=target_h,
            )

            obj = make_breakable_object(
                spec.target_id,
                geometry,
                integrity=1.0,
                role="target",
                owner=f"player_{self.player}",
                entity_id=f"rack_p{self.player}",
                part_id=spec.target_id,
                shape=self._shape_for_spec(spec),
                tags={
                    "shoot_stuff",
                    f"distance_{int(self.distance_m)}m",
                    f"points_{spec.points}",
                    spec.draw_kind,
                },
                material_id="range_target",
                penetration_mode=PenetrationMode.NEVER,
                penetration_resistance=0.0,
                break_sound="",
                break_effect="",
                hide_on_break=True,
                z_index=5,
                hit_depth=float(self.distance_m),
            )
            obj.world = WorldPlacement(
                distance_m=self.distance_m,
                world_x_m=float(lateral_m),
                world_y_m=SHELF_HEIGHT_CM / 100.0,
                world_z_m=self.distance_m,
                metadata={
                    "physical_width_cm": spec.width_cm,
                    "physical_height_cm": spec.height_cm,
                    "points": spec.points,
                    "title": spec.title,
                },
            )
            obj.metadata.update(
                {
                    "spec_index": index,
                    "title": spec.title,
                    "subtitle": spec.subtitle,
                    "points": spec.points,
                    "physical_width_cm": spec.width_cm,
                    "physical_height_cm": spec.height_cm,
                    "virtual_distance_m": self.distance_m,
                }
            )
            self.manager.add(obj)
            self._target_geometry[obj.object_id] = geometry.aabb

    def get_hit_regions(self):
        return self.manager.get_hit_regions()

    # ------------------------------------------------------------------
    # Match flow
    # ------------------------------------------------------------------

    def _start_match(self) -> None:
        self.player = 1
        self.shots_used = {1: 0, 2: 0}
        self.score = {1: 0, 2: 0}
        self.hits = {1: 0, 2: 0}
        self.streak = {1: 0, 2: 0}
        self.best_streak = {1: 0, 2: 0}
        self.perfect_bonus_given = {1: False, 2: False}
        self.markers.clear()
        self.score_pops.clear()
        self.knocked_targets.clear()
        self.particles.clear()
        self._last_resolution_line = ""
        self._rebuild_visual_world()
        self._start_player(1)

    def _start_player(self, player: int) -> None:
        self.player = player
        self._build_rack()
        self.phase = "intro"
        self.phase_until = time.monotonic() + INTRO_SECONDS
        self.overlay_title = f"SPELARE {player}"
        self.overlay_subtitle = (
            f"5 skott • 7 mål • {self.distance_m:.0f} meter • välj smart"
        )
        self.overlay_color = CYAN if player == 1 else AMBER

    def _finalize_current_player(self) -> None:
        if (
            self.hits[self.player] == SHOTS_PER_PLAYER
            and not self.perfect_bonus_given[self.player]
        ):
            self.perfect_bonus_given[self.player] = True
            self.score[self.player] += PERFECT_FIVE_BONUS
            self.score_pops.append(
                ScorePop(
                    self.viewport.w * 0.5,
                    self.viewport.h * 0.45,
                    f"5/5! +{PERFECT_FIVE_BONUS}",
                    time.monotonic(),
                    lifetime=1.8,
                )
            )

        self.phase = "player_result"
        self.phase_until = time.monotonic() + PLAYER_RESULT_SECONDS
        self.overlay_title = f"SPELARE {self.player}: {self.score[self.player]} P"
        self.overlay_subtitle = (
            f"{self.hits[self.player]}/5 träffar • "
            f"bästa svit {self.best_streak[self.player]}"
        )
        self.overlay_color = CYAN if self.player == 1 else AMBER

    def _advance_after_player_result(self) -> None:
        if self.player == 1:
            self._start_player(2)
        else:
            self._end_game()

    def _end_game(self) -> None:
        self.phase = "game_over"
        p1, p2 = self.score[1], self.score[2]
        if p1 > p2:
            self.overlay_title = "SPELARE 1 VINNER!"
            self.overlay_color = CYAN
        elif p2 > p1:
            self.overlay_title = "SPELARE 2 VINNER!"
            self.overlay_color = AMBER
        else:
            self.overlay_title = "OAVGJORT!"
            self.overlay_color = TEXT
        self.overlay_subtitle = f"{p1}  -  {p2}"

        for _ in range(90):
            self._spawn_celebration_particle()

    # ------------------------------------------------------------------
    # Input / exact gameplay resolution
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_r:
            self._start_match()
            return None

        unicode_value = str(getattr(event, "unicode", "") or "")
        if unicode_value == "+" or event.key == pygame.K_KP_PLUS:
            self._change_distance(+DISTANCE_STEP_M)
            return None
        if unicode_value == "-" or event.key == pygame.K_KP_MINUS:
            self._change_distance(-DISTANCE_STEP_M)
            return None

        return None

    def _gameplay_assist_for_miss(self, hit: HitEvent):
        """Return a candidate-backed assisted resolution or None.

        IMPORTANT:
        - game-only convenience, never physical detector truth;
        - same shot_id only;
        - active target only;
        - bounded by candidate rank/confidence;
        - candidate XY must still collide with the target's exact frozen shape.
        """
        if not GAMEPLAY_ASSIST_ENABLED:
            return None

        shot_id = getattr(hit, "shot_id", None)
        if shot_id is None:
            # Mouse/debug hits have no physical candidate snapshot.
            return None

        snap = object_hit_registry_v2223.snapshot_for_shot(int(shot_id))
        if snap is None or not snap.results:
            return None

        eligible = []
        for object_id, shadow in snap.results.items():
            if object_id not in self._target_geometry:
                continue

            obj = self.manager.get(object_id)
            if obj is None or not obj.hittable:
                continue

            if not bool(getattr(shadow, "hit", False)):
                continue

            rank = getattr(shadow, "candidate_rank", None)
            confidence = float(getattr(shadow, "confidence", 0.0) or 0.0)
            if rank is None or int(rank) > GAMEPLAY_ASSIST_MAX_CANDIDATE_RANK:
                continue
            if confidence < GAMEPLAY_ASSIST_MIN_CONFIDENCE:
                continue

            camera_x = getattr(shadow, "camera_x", None)
            camera_y = getattr(shadow, "camera_y", None)
            if camera_x is None or camera_y is None:
                continue

            game_xy = object_hit_registry_v2223.camera_to_game_point(
                float(camera_x),
                float(camera_y),
            )
            if game_xy is None:
                continue

            assist_x, assist_y = float(game_xy[0]), float(game_xy[1])

            # Reuse ObjectManager and the shot's frozen GameObject snapshots.
            # We never alter the real HitEvent or engine state.
            proxy_hit = SimpleNamespace(
                game_x=assist_x,
                game_y=assist_y,
                shot_id=int(shot_id),
            )
            assisted_resolution = self.manager.resolve_hit(proxy_hit, GAME_PROJECTILE)
            assisted_interactions = [
                item
                for item in assisted_resolution.interactions
                if item.object_id == object_id
                and item.object_id in self._target_geometry
            ]
            if not assisted_interactions:
                continue

            eligible.append(
                (
                    confidence,
                    -int(rank),
                    float(getattr(shadow, "shot_novelty", 0.0) or 0.0),
                    float(getattr(shadow, "hole_likeness", 0.0) or 0.0),
                    object_id,
                    shadow,
                    assisted_resolution,
                    assisted_interactions,
                    assist_x,
                    assist_y,
                )
            )

        if not eligible:
            return None

        eligible.sort(key=lambda row: row[:4], reverse=True)
        (
            _confidence,
            _neg_rank,
            _novelty,
            _likeness,
            object_id,
            shadow,
            assisted_resolution,
            assisted_interactions,
            assist_x,
            assist_y,
        ) = eligible[0]

        return {
            "object_id": object_id,
            "shadow": shadow,
            "resolution": assisted_resolution,
            "interactions": assisted_interactions,
            "x": assist_x,
            "y": assist_y,
        }

    def _on_hit(self, hit: HitEvent) -> None:
        now = time.monotonic()
        if self.phase != "live":
            print(
                "[SHOOT-STUFF] ignored hit outside live window "
                f"phase={self.phase} shot_id={getattr(hit, 'shot_id', None)}"
            )
            return

        self.phase = "processing"
        self._last_hit_shot_id = getattr(hit, "shot_id", None)

        geometry_before = {}
        for obj in self.manager.objects:
            geometry_before[obj.object_id] = (
                float(obj.geometry.x),
                float(obj.geometry.y),
                float(obj.geometry.width),
                float(obj.geometry.height),
                int(obj.metadata.get("spec_index", 0)),
            )

        # First and always preferred path: canonical final emitted HitEvent.
        result = self.manager.resolve_hit(hit, GAME_PROJECTILE)
        interactions = [
            interaction
            for interaction in result.interactions
            if interaction.object_id in self._target_geometry
        ]

        emitted_x = float(result.game_x)
        emitted_y = float(result.game_y)
        gameplay_x = emitted_x
        gameplay_y = emitted_y
        assisted = False
        assist_shadow = None

        # Temporary game-local assist: only after a genuine final-XY miss.
        if not interactions:
            assist = self._gameplay_assist_for_miss(hit)
            if assist is not None:
                assisted = True
                assist_shadow = assist["shadow"]
                result = assist["resolution"]
                interactions = assist["interactions"]
                gameplay_x = float(assist["x"])
                gameplay_y = float(assist["y"])

                # Show BOTH facts: the detector-selected XY missed, while a
                # same-shot strong candidate was used by this game.
                self.markers.append(
                    ImpactMarker(
                        emitted_x,
                        emitted_y,
                        False,
                        "VALD XY",
                        now,
                    )
                )

        self.shots_used[self.player] += 1

        if interactions:
            interaction = interactions[0]
            obj = self.manager.get(interaction.object_id)
            spec_index = int(obj.metadata.get("spec_index", 0)) if obj else 0
            spec = TARGET_SPECS[spec_index]
            base_points = int(obj.metadata.get("points", spec.points)) if obj else spec.points

            self.streak[self.player] += 1
            self.best_streak[self.player] = max(
                self.best_streak[self.player],
                self.streak[self.player],
            )
            streak_bonus = min(
                MAX_STREAK_BONUS,
                max(0, self.streak[self.player] - 1) * STREAK_BONUS_STEP,
            )
            gained = base_points + streak_bonus
            self.score[self.player] += gained
            self.hits[self.player] += 1

            marker_label = (
                f"ASSIST {spec.title} +{gained}"
                if assisted
                else f"{spec.title} +{gained}"
            )
            self.markers.append(
                ImpactMarker(
                    gameplay_x,
                    gameplay_y,
                    True,
                    marker_label,
                    now,
                )
            )
            self.score_pops.append(
                ScorePop(
                    gameplay_x,
                    gameplay_y - max(
                        12.0,
                        geometry_before[interaction.object_id][3] * 0.7,
                    ),
                    f"+{gained}",
                    now,
                )
            )

            geom = geometry_before.get(interaction.object_id)
            if geom is not None:
                x, y, w, h, idx = geom
                self._spawn_knock_animation(
                    x,
                    y,
                    w,
                    h,
                    idx,
                    gameplay_x,
                    gameplay_y,
                )

            subtitle = spec.subtitle + f" • +{base_points}"
            if streak_bonus:
                subtitle += f" • svit +{streak_bonus}"

            if assisted and assist_shadow is not None:
                rank = int(getattr(assist_shadow, "candidate_rank", 0) or 0)
                confidence = float(getattr(assist_shadow, "confidence", 0.0) or 0.0)
                self.overlay_title = "ASSIST! " + self._hit_exclamation(spec)
                self.overlay_subtitle = (
                    subtitle
                    + f" • kandidat #{rank} conf {confidence:.2f}"
                )
                self.overlay_color = AMBER
                self._last_resolution_line = (
                    f"ASSISTED {spec.title} | vald XY {emitted_x:.0f},{emitted_y:.0f} "
                    f"-> kandidat XY {gameplay_x:.0f},{gameplay_y:.0f} "
                    f"| rank={rank} conf={confidence:.2f}"
                )
            else:
                self.overlay_title = self._hit_exclamation(spec)
                self.overlay_subtitle = subtitle
                self.overlay_color = GOOD
                self._last_resolution_line = (
                    f"TRÄFF {spec.title} | XY {gameplay_x:.0f},{gameplay_y:.0f} | "
                    f"frozen={result.used_frozen_snapshot}"
                )
        else:
            self.streak[self.player] = 0
            self.markers.append(
                ImpactMarker(
                    emitted_x,
                    emitted_y,
                    False,
                    "MISS",
                    now,
                )
            )
            self._spawn_miss_particles(emitted_x, emitted_y)
            self.overlay_title = self.rng.choice(
                ("MISS!", "NÄRA...", "LUFT!", "INGEN PRYL!")
            )
            self.overlay_subtitle = (
                "Ingen aktiv mål-shape och ingen tillräckligt stark assist-kandidat"
            )
            self.overlay_color = BAD
            self._last_resolution_line = (
                f"MISS | XY {emitted_x:.0f},{emitted_y:.0f} | "
                f"assist=none"
            )

        mode = "ASSISTED" if assisted and interactions else "DIRECT"
        print(
            "[SHOOT-STUFF] "
            f"shot_id={self._last_hit_shot_id} player={self.player} "
            f"shot={self.shots_used[self.player]}/{SHOTS_PER_PLAYER} "
            f"distance={self.distance_m:.0f}m "
            f"mode={mode} "
            f"emitted_xy=({emitted_x:.1f},{emitted_y:.1f}) "
            f"gameplay_xy=({gameplay_x:.1f},{gameplay_y:.1f}) "
            f"objects={[i.object_id for i in interactions]} "
            f"score={self.score[self.player]}"
        )

        self.phase = "result"
        self.phase_until = now + SHOT_RESULT_SECONDS

    def _hit_exclamation(self, spec: TargetSpec) -> str:
        if spec.target_id == "rubber_duck":
            return "KVACK!"
        if spec.target_id == "toilet_roll":
            return "PAPPERSKRIS!"
        if spec.target_id == "beans_xxl":
            return "BÖNOR ÖVERALLT!"
        if spec.target_id == "sardines":
            return "SARDINTRÄFF!"
        if spec.target_id == "gold_can":
            return "GULD!"
        return self.rng.choice(("KLONK!", "TRÄFF!", "PANG!", "SNYGGT!"))

    # ------------------------------------------------------------------
    # Update / animations
    # ------------------------------------------------------------------

    def update(self, dt: float):
        now = time.monotonic()
        dt = max(0.0, min(0.05, float(dt)))
        self._update_animations(dt, now)

        if self.phase == "intro" and now >= self.phase_until:
            self.phase = "live"
            self.overlay_title = ""
            self.overlay_subtitle = ""
            return None

        if self.phase == "result" and now >= self.phase_until:
            if self.shots_used[self.player] >= SHOTS_PER_PLAYER:
                self._finalize_current_player()
            else:
                self.phase = "live"
                self.overlay_title = ""
                self.overlay_subtitle = ""
            return None

        if self.phase == "player_result" and now >= self.phase_until:
            self._advance_after_player_result()

        return None

    def _spawn_knock_animation(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        spec_index: int,
        hit_x: float,
        hit_y: float,
    ) -> None:
        center_x = x + w * 0.5
        side = -1.0 if hit_x >= center_x else 1.0

        self.knocked_targets.append(
            KnockedTarget(
                x=x,
                y=y,
                width=w,
                height=h,
                spec_index=spec_index,
                vx=side * self.rng.uniform(120.0, 215.0),
                vy=-self.rng.uniform(175.0, 260.0),
                angle=0.0,
                angular_velocity=side * self.rng.uniform(260.0, 540.0),
            )
        )

        for _ in range(16):
            angle = self.rng.uniform(-math.pi, math.pi)
            speed = self.rng.uniform(55.0, 180.0)
            self.particles.append(
                Particle(
                    hit_x,
                    hit_y,
                    math.cos(angle) * speed,
                    math.sin(angle) * speed - 40.0,
                    self.rng.uniform(1.5, 3.2),
                    self.rng.uniform(0.35, 0.8),
                    0.8,
                    "spark",
                )
            )

    def _spawn_miss_particles(self, x: float, y: float) -> None:
        for _ in range(8):
            angle = self.rng.uniform(0.0, math.tau)
            speed = self.rng.uniform(25.0, 70.0)
            self.particles.append(
                Particle(
                    x,
                    y,
                    math.cos(angle) * speed,
                    math.sin(angle) * speed,
                    self.rng.uniform(1.0, 2.2),
                    self.rng.uniform(0.25, 0.55),
                    0.55,
                    "dust",
                )
            )

    def _spawn_celebration_particle(self) -> None:
        self.particles.append(
            Particle(
                self.rng.uniform(0, self.viewport.w),
                self.rng.uniform(-40, self.viewport.h * 0.25),
                self.rng.uniform(-45, 45),
                self.rng.uniform(60, 180),
                self.rng.uniform(2.0, 5.0),
                self.rng.uniform(1.0, 2.3),
                2.3,
                "confetti",
            )
        )

    def _update_animations(self, dt: float, now: float) -> None:
        gravity = 430.0
        for target in self.knocked_targets:
            target.age += dt
            target.x += target.vx * dt
            target.y += target.vy * dt
            target.vy += gravity * dt
            target.angle += target.angular_velocity * dt
        self.knocked_targets = [
            target for target in self.knocked_targets
            if target.age <= target.lifetime
        ]

        for p in self.particles:
            p.x += p.vx * dt
            p.y += p.vy * dt
            if p.kind != "confetti":
                p.vy += 160.0 * dt
            p.life -= dt
        self.particles = [p for p in self.particles if p.life > 0.0]

        self.markers = [m for m in self.markers if now - m.created <= m.lifetime]
        self.score_pops = [p for p in self.score_pops if now - p.created <= p.lifetime]

    # ------------------------------------------------------------------
    # Procedural range background
    # ------------------------------------------------------------------

    def _rebuild_visual_world(self) -> None:
        self._shelf_y = self._projected_shelf_y()
        self._build_background()

    def _build_background(self) -> None:
        surf = pygame.Surface((self.viewport.w, self.viewport.h)).convert()
        w, h = surf.get_size()
        horizon = int(h * HORIZON_Y_NORM)

        self._vertical_gradient(surf, pygame.Rect(0, 0, w, horizon + 1), SKY_TOP, SKY_BOTTOM)
        self._vertical_gradient(
            surf,
            pygame.Rect(0, horizon, w, h - horizon),
            GROUND_TOP,
            GROUND_BOTTOM,
        )

        hill_y = int(horizon * 0.88)
        pygame.draw.polygon(
            surf,
            (85, 115, 102),
            [
                (0, horizon),
                (0, hill_y + 30),
                (int(w * 0.13), hill_y - 20),
                (int(w * 0.27), hill_y + 10),
                (int(w * 0.43), hill_y - 34),
                (int(w * 0.58), hill_y + 5),
                (int(w * 0.74), hill_y - 27),
                (w, hill_y + 12),
                (w, horizon),
            ],
        )

        for x in range(-10, w + 20, max(18, w // 70)):
            tw = self.rng.randint(12, 25)
            th = self.rng.randint(28, 55)
            pygame.draw.polygon(
                surf,
                (43, 76, 55),
                [(x, horizon + 6), (x + tw // 2, horizon - th), (x + tw, horizon + 6)],
            )

        lane_color = (190, 186, 146)
        center_x = w // 2
        for spread in (0.16, 0.31, 0.46):
            pygame.draw.line(
                surf, lane_color,
                (center_x, horizon),
                (int(center_x - w * spread), h),
                2,
            )
            pygame.draw.line(
                surf, lane_color,
                (center_x, horizon),
                (int(center_x + w * spread), h),
                2,
            )

        # Distance board changes with the configured virtual distance.
        sign_w = max(180, int(w * 0.23))
        sign_h = max(58, int(h * 0.08))
        sign_rect = pygame.Rect(
            center_x - sign_w // 2,
            int(self._shelf_y - max(self._projected_size_px(s)[1] for s in TARGET_SPECS) - sign_h - 34),
            sign_w,
            sign_h,
        )
        pygame.draw.rect(surf, (63, 74, 68), sign_rect, border_radius=8)
        pygame.draw.rect(surf, (182, 184, 166), sign_rect, 2, border_radius=8)
        if self.font_big is not None:
            txt = self.font_big.render(f"{self.distance_m:.0f} m", True, (231, 226, 202))
            surf.blit(txt, txt.get_rect(center=sign_rect.center))

        shelf_rect = pygame.Rect(
            int(w * 0.08),
            int(self._shelf_y),
            int(w * 0.84),
            16,
        )
        pygame.draw.rect(surf, WOOD_DARK, shelf_rect.move(0, 6), border_radius=4)
        pygame.draw.rect(surf, WOOD, shelf_rect, border_radius=4)
        pygame.draw.line(
            surf,
            WOOD_LIGHT,
            (shelf_rect.left + 5, shelf_rect.top + 3),
            (shelf_rect.right - 5, shelf_rect.top + 3),
            2,
        )

        for leg_x in (shelf_rect.left + 35, shelf_rect.right - 50):
            pygame.draw.polygon(
                surf,
                WOOD_DARK,
                [
                    (leg_x, shelf_rect.bottom),
                    (leg_x + 18, shelf_rect.bottom),
                    (leg_x + 34, h),
                    (leg_x + 12, h),
                ],
            )

        if self.font_tiny is not None:
            joke = self.font_tiny.render(
                "SKJUTBANA • inga ankor har godkänt denna aktivitet",
                True,
                (225, 218, 192),
            )
            surf.blit(joke, joke.get_rect(center=(w // 2, h - 18)))

        self._background_cache = surf

    @staticmethod
    def _vertical_gradient(surface, rect, top, bottom) -> None:
        height = max(1, rect.height)
        for i in range(height):
            t = i / max(1, height - 1)
            color = tuple(
                int(round(top[c] + (bottom[c] - top[c]) * t))
                for c in range(3)
            )
            pygame.draw.line(
                surface,
                color,
                (rect.left, rect.top + i),
                (rect.right - 1, rect.top + i),
            )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, screen: pygame.Surface) -> None:
        surf = pygame.Surface((self.viewport.w, self.viewport.h)).convert_alpha()
        if self._background_cache is not None:
            surf.blit(self._background_cache, (0, 0))
        else:
            surf.fill((55, 80, 60))

        self._draw_targets(surf)
        self._draw_knocked_targets(surf)
        self._draw_particles(surf)
        self._draw_markers(surf)
        self._draw_score_pops(surf)
        self._draw_hud(surf)

        if self.phase in {"intro", "result", "player_result"}:
            self._draw_overlay(surf)
        elif self.phase == "game_over":
            self._draw_game_over(surf)

        screen.blit(surf, self.viewport.topleft)

    def _draw_targets(self, surf: pygame.Surface) -> None:
        for obj in self.manager.objects:
            if not obj.visible:
                continue
            spec_index = int(obj.metadata.get("spec_index", 0))
            spec = TARGET_SPECS[spec_index]
            g = obj.geometry
            rect = pygame.Rect(
                int(round(g.x)),
                int(round(g.y)),
                max(2, int(round(g.width))),
                max(2, int(round(g.height))),
            )
            self._draw_target(surf, rect, spec)

            if self.font_tiny is not None:
                points = self.font_tiny.render(str(spec.points), True, (241, 231, 196))
                plate_w = max(24, points.get_width() + 8)
                plate = pygame.Rect(
                    rect.centerx - plate_w // 2,
                    int(self._shelf_y + 6),
                    plate_w,
                    max(17, points.get_height() + 3),
                )
                pygame.draw.rect(surf, (50, 43, 33), plate, border_radius=3)
                surf.blit(points, points.get_rect(center=plate.center))

    def _draw_target(self, surf: pygame.Surface, rect: pygame.Rect, spec: TargetSpec) -> None:
        if spec.draw_kind == "can":
            self._draw_can(surf, rect, spec)
        elif spec.draw_kind == "roll":
            self._draw_roll(surf, rect, spec)
        elif spec.draw_kind == "duck":
            self._draw_duck(surf, rect, spec)
        elif spec.draw_kind == "tin":
            self._draw_tin(surf, rect, spec)
        else:
            pygame.draw.rect(surf, spec.primary, rect, border_radius=max(2, rect.width // 6))
            pygame.draw.rect(surf, WHITE, rect, 1, border_radius=max(2, rect.width // 6))

    def _draw_can(self, surf, rect, spec) -> None:
        radius = max(2, rect.width // 5)
        shadow = rect.move(max(1, rect.width // 8), max(1, rect.height // 18))
        pygame.draw.rect(surf, (26, 30, 29), shadow, border_radius=radius)
        pygame.draw.rect(surf, spec.secondary, rect, border_radius=radius)
        inner = rect.inflate(-max(2, rect.width // 7), -max(2, rect.height // 18))
        if inner.width > 1 and inner.height > 1:
            pygame.draw.rect(surf, spec.primary, inner, border_radius=max(1, radius - 1))
        rim_h = max(2, rect.height // 11)
        pygame.draw.ellipse(surf, (215, 222, 218), pygame.Rect(rect.x, rect.y, rect.width, rim_h))
        pygame.draw.ellipse(
            surf,
            (118, 129, 128),
            pygame.Rect(rect.x, rect.bottom - rim_h, rect.width, rim_h),
        )
        stripe_h = max(2, rect.height // 5)
        pygame.draw.rect(
            surf,
            spec.secondary,
            pygame.Rect(rect.x + 1, rect.centery - stripe_h // 2, max(1, rect.width - 2), stripe_h),
        )

    def _draw_roll(self, surf, rect, spec) -> None:
        pygame.draw.ellipse(surf, (60, 61, 57), rect.move(2, 2))
        pygame.draw.ellipse(surf, spec.primary, rect)
        inner = rect.inflate(-max(4, rect.width // 3), -max(4, rect.height // 3))
        if inner.width > 0 and inner.height > 0:
            pygame.draw.ellipse(surf, (91, 82, 68), inner)
            pygame.draw.ellipse(surf, (35, 32, 28), inner, 1)
        pygame.draw.arc(surf, spec.secondary, rect.inflate(-2, -2), 0.5, 4.7, 2)

    def _draw_duck(self, surf, rect, spec) -> None:
        body = pygame.Rect(rect.x, rect.y + rect.height // 3, rect.width, rect.height * 2 // 3)
        head_d = max(3, int(min(rect.width, rect.height) * 0.52))
        head = pygame.Rect(
            rect.x + int(rect.width * 0.52),
            rect.y,
            head_d,
            head_d,
        )
        pygame.draw.ellipse(surf, (55, 53, 40), body.move(2, 2))
        pygame.draw.ellipse(surf, spec.primary, body)
        pygame.draw.ellipse(surf, spec.primary, head)
        beak = [
            (head.right - 1, head.centery - 2),
            (head.right + max(2, rect.width // 5), head.centery),
            (head.right - 1, head.centery + 3),
        ]
        pygame.draw.polygon(surf, spec.secondary, beak)
        eye_r = max(1, rect.width // 18)
        pygame.draw.circle(
            surf,
            BLACK,
            (head.x + int(head.width * 0.65), head.y + int(head.height * 0.35)),
            eye_r,
        )

    def _draw_tin(self, surf, rect, spec) -> None:
        pygame.draw.rect(surf, (35, 40, 42), rect.move(2, 2), border_radius=3)
        pygame.draw.rect(surf, spec.primary, rect, border_radius=3)
        pygame.draw.rect(surf, spec.secondary, rect, 2, border_radius=3)
        inset = rect.inflate(-max(2, rect.width // 10), -max(2, rect.height // 4))
        if inset.width > 0 and inset.height > 0:
            pygame.draw.rect(surf, (193, 206, 209), inset, 1, border_radius=2)
        pygame.draw.line(surf, spec.secondary, (rect.centerx, rect.top + 2), (rect.centerx, rect.bottom - 2), 1)

    def _draw_knocked_targets(self, surf) -> None:
        for target in self.knocked_targets:
            spec = TARGET_SPECS[target.spec_index]
            base = pygame.Surface(
                (max(4, int(target.width)), max(4, int(target.height))),
                pygame.SRCALPHA,
            )
            self._draw_target(
                base,
                pygame.Rect(0, 0, base.get_width(), base.get_height()),
                spec,
            )
            rotated = pygame.transform.rotozoom(base, -target.angle, 1.0)
            center = (
                int(target.x + target.width * 0.5),
                int(target.y + target.height * 0.5),
            )
            surf.blit(rotated, rotated.get_rect(center=center))

    def _draw_particles(self, surf) -> None:
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p.life / max(0.001, p.max_life)))))
            if p.kind == "spark":
                color = (255, 224, 112, alpha)
            elif p.kind == "confetti":
                choices = ((255, 88, 80), (95, 220, 255), (255, 210, 65), (130, 240, 145))
                rgb = choices[int(abs(p.x + p.y)) % len(choices)]
                color = (*rgb, alpha)
            else:
                color = (194, 178, 148, alpha)
            r = max(1, int(p.radius))
            particle_surf = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, color, (r + 1, r + 1), r)
            surf.blit(particle_surf, (int(p.x - r), int(p.y - r)))

    def _draw_markers(self, surf) -> None:
        now = time.monotonic()
        for marker in self.markers:
            age = now - marker.created
            t = max(0.0, min(1.0, age / marker.lifetime))
            color = GOOD if marker.hit else BAD
            radius = int(10 + 16 * t)
            x, y = int(round(marker.x)), int(round(marker.y))
            pygame.draw.circle(surf, color, (x, y), radius, 2)
            pygame.draw.line(surf, color, (x - 8, y), (x + 8, y), 2)
            pygame.draw.line(surf, color, (x, y - 8), (x, y + 8), 2)
            if self.font_tiny is not None:
                txt = self.font_tiny.render(marker.label, True, color)
                surf.blit(txt, (x + 12, y - txt.get_height() - 5))

    def _draw_score_pops(self, surf) -> None:
        if self.font_medium is None:
            return
        now = time.monotonic()
        for pop in self.score_pops:
            age = now - pop.created
            t = max(0.0, min(1.0, age / pop.lifetime))
            txt = self.font_medium.render(pop.text, True, AMBER)
            surf.blit(
                txt,
                txt.get_rect(center=(int(pop.x), int(pop.y - 38.0 * t))),
            )

    def _draw_hud(self, surf) -> None:
        if self.font_medium is None or self.font_small is None or self.font_tiny is None:
            return

        panel = pygame.Surface((surf.get_width(), HUD_HEIGHT), pygame.SRCALPHA)
        panel.fill((*PANEL, 235))
        surf.blit(panel, (0, 0))

        left_w = int(surf.get_width() * 0.30)
        right_x = int(surf.get_width() * 0.70)

        self._draw_player_hud(surf, 1, pygame.Rect(10, 9, left_w - 18, HUD_HEIGHT - 18))
        self._draw_player_hud(
            surf,
            2,
            pygame.Rect(right_x + 8, 9, surf.get_width() - right_x - 18, HUD_HEIGHT - 18),
        )

        center_rect = pygame.Rect(left_w, 8, right_x - left_w, HUD_HEIGHT - 16)
        pygame.draw.rect(surf, PANEL_2, center_rect, border_radius=10)

        title = self.font_medium.render(
            f"SKJUTPRYLAR • {self.distance_m:.0f} m",
            True,
            TEXT,
        )
        surf.blit(title, title.get_rect(center=(center_rect.centerx, center_rect.y + 20)))

        if self.phase == "live":
            shot_no = self.shots_used[self.player] + 1
            status = f"SPELARE {self.player} • SKOTT {shot_no}/5 • SKJUT!"
            status_color = GOOD
        elif self.phase == "game_over":
            status = "MATCH KLAR"
            status_color = AMBER
        else:
            status = f"SPELARE {self.player}"
            status_color = CYAN if self.player == 1 else AMBER

        status_surf = self.font_small.render(status, True, status_color)
        surf.blit(status_surf, status_surf.get_rect(center=(center_rect.centerx, center_rect.y + 47)))

        controls = self.font_tiny.render(
            "- / + = 5 m   •   7 mål / 5 skott   •   R = ny match",
            True,
            AMBER,
        )
        surf.blit(controls, controls.get_rect(center=(center_rect.centerx, center_rect.y + 70)))

        auto_color = GOOD if self._last_resolution_line.startswith("TRÄFF") else MUTED
        auto = self._last_resolution_line or "Automarkering: väntar på första registrerade träffen"
        auto_surf = self.font_tiny.render(auto[:100], True, auto_color)
        surf.blit(auto_surf, auto_surf.get_rect(center=(center_rect.centerx, center_rect.y + 94)))

    def _draw_player_hud(self, surf, player, rect) -> None:
        active = player == self.player and self.phase != "game_over"
        color = CYAN if player == 1 else AMBER
        pygame.draw.rect(surf, PANEL_2, rect, border_radius=10)
        if active:
            pygame.draw.rect(surf, color, rect, 3, border_radius=10)

        name = self.font_medium.render(f"SPELARE {player}", True, color)
        surf.blit(name, (rect.x + 10, rect.y + 7))

        score = self.font_big.render(str(self.score[player]), True, TEXT)
        surf.blit(score, (rect.x + 10, rect.y + 34))

        info = self.font_tiny.render(
            f"Skott {self.shots_used[player]}/5   Träff {self.hits[player]}/5   Svit {self.streak[player]}",
            True,
            MUTED,
        )
        surf.blit(info, (rect.x + 10, rect.bottom - 21))

    def _draw_overlay(self, surf) -> None:
        if not self.overlay_title or self.font_huge is None or self.font_medium is None:
            return
        shade = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 92))
        surf.blit(shade, (0, 0))
        y = max(HUD_HEIGHT + 100, surf.get_height() // 2 - 10)
        title = self.font_huge.render(self.overlay_title, True, self.overlay_color)
        surf.blit(title, title.get_rect(center=(surf.get_width() // 2, y)))
        if self.overlay_subtitle:
            sub = self.font_medium.render(self.overlay_subtitle, True, TEXT)
            surf.blit(sub, sub.get_rect(center=(surf.get_width() // 2, y + 62)))

    def _draw_game_over(self, surf) -> None:
        if self.font_huge is None or self.font_big is None or self.font_medium is None:
            return
        shade = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 205))
        surf.blit(shade, (0, 0))
        center_y = surf.get_height() // 2 - 70

        title = self.font_huge.render(self.overlay_title, True, self.overlay_color)
        surf.blit(title, title.get_rect(center=(surf.get_width() // 2, center_y)))

        score = self.font_big.render(self.overlay_subtitle, True, TEXT)
        surf.blit(score, score.get_rect(center=(surf.get_width() // 2, center_y + 72)))

        result = self.font_medium.render(
            f"Träffar {self.hits[1]}/5  -  {self.hits[2]}/5",
            True,
            MUTED,
        )
        surf.blit(result, result.get_rect(center=(surf.get_width() // 2, center_y + 122)))

        hint = self.font_small.render(
            "- / + = ändra avstånd och börja om    R = ny match    ESC = tillbaka",
            True,
            MUTED,
        )
        surf.blit(hint, hint.get_rect(center=(surf.get_width() // 2, center_y + 175)))


def create_game(game_root: str, viewport: pygame.Rect):
    return ShootStuffGame(game_root=game_root, viewport=viewport)
