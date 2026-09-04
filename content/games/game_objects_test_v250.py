from __future__ import annotations

"""V2.25 GameObject foundation diagnostic scene.

This scene is intentionally small and explicit.  It demonstrates composition:
static, living, breakable and penetrable objects all use the same GameObject
model while exact hit resolution uses frozen V2.25 shape snapshots.
"""

import pygame

from src.engine.game_objects import (
    BallisticBody,
    DamageModel,
    DurabilityLayer,
    EffectAction,
    GameObject,
    HitShapeSpec,
    MotionKind,
    ObjectGeometry,
    ObjectManager,
    PenetrationMode,
    ProjectileProfile,
    ReactionRule,
    make_breakable_object,
    make_living_object,
    make_static_object,
)
from src.engine.input.hit_input import HitEvent, hit_input

BG = (15, 18, 24)
GRID = (35, 42, 54)
TEXT = (239, 243, 249)
MUTED = (155, 166, 181)
GREEN = (58, 196, 112)
RED = (230, 72, 72)
BLUE = (75, 145, 235)
AMBER = (241, 177, 63)
PURPLE = (190, 99, 225)
CYAN = (72, 215, 235)
WHITE = (250, 250, 250)
BLACK = (15, 15, 17)
HUD_H = 132


class GameObjectsTestV250:
    def __init__(self, game_root: str, viewport: pygame.Rect) -> None:
        del game_root
        self.viewport = viewport.copy()
        self.manager = ObjectManager()
        self._subscribed = False
        self._font_big = None
        self._font = None
        self._small = None
        self._projectiles = [
            ProjectileProfile("low", damage=1.0, penetration_power=0.5, max_object_hits=2),
            ProjectileProfile("medium", damage=1.0, penetration_power=1.5, max_object_hits=4),
            ProjectileProfile("high", damage=2.0, penetration_power=3.0, max_object_hits=6),
        ]
        self._projectile_index = 1
        self.last_line = "Skjut på objekten. 1/2/3 byter gameplay-projektilprofil."
        self.last_effects = ""
        self.last_shot_id = None

    @property
    def projectile(self) -> ProjectileProfile:
        return self._projectiles[self._projectile_index]

    def on_enter(self) -> None:
        self._font_big = pygame.font.Font(None, 34)
        self._font = pygame.font.Font(None, 25)
        self._small = pygame.font.Font(None, 20)
        self._build_objects()
        if not self._subscribed:
            hit_input.subscribe(self._on_hit)
            self._subscribed = True
        print("[V2.25.0 OBJECT-TEST] entered; 1/2/3 projectile profile, R reset")

    def on_exit(self) -> None:
        if self._subscribed:
            hit_input.unsubscribe(self._on_hit)
            self._subscribed = False

    def _build_objects(self) -> None:
        self.manager = ObjectManager()
        w = float(self.viewport.w)
        h = float(self.viewport.h)
        top = HUD_H + 28
        box_w = max(110.0, min(160.0, w * 0.16))
        box_h = max(92.0, min(130.0, (h - top) * 0.25))

        crate = make_breakable_object(
            "crate",
            ObjectGeometry(w * 0.08, top, box_w, box_h),
            integrity=2.0,
            penetration_mode=PenetrationMode.IF_POWER,
            penetration_resistance=1.0,
            break_sound="crate_break",
            break_effect="dust",
            z_index=3,
        )
        crate.metadata["label"] = "BREAKABLE / integrity 2"
        self.manager.add(crate)

        living = make_living_object(
            "living_target",
            ObjectGeometry(w * 0.38, top, box_w, box_h),
            health=3.0,
            shape=HitShapeSpec.ellipse(x=0.12, y=0.02, width=0.76, height=0.96),
            penetration_mode=PenetrationMode.IF_POWER,
            penetration_resistance=1.2,
            death_sound="target_down",
            death_effect="death",
            z_index=3,
        )
        living.metadata["label"] = "LIVING / health 3"
        self.manager.add(living)

        hard = make_static_object(
            "hard_no_shoot",
            ObjectGeometry(w * 0.72, top, box_w, box_h),
            object_type="hard_block",
            role="no_shoot",
            tags={"solid", "no_shoot"},
            ballistic_body=BallisticBody(
                penetration_mode=PenetrationMode.NEVER,
                penetration_resistance=99.0,
                receives_damage=False,
            ),
            z_index=3,
        )
        hard.metadata["label"] = "STATIC NO-SHOOT / BLOCKS"
        self.manager.add(hard)

        # Penetration stack: glass is visually/front-most.  A sufficiently
        # penetrative gameplay projectile continues to the rear target at the
        # exact same XY.
        back = make_living_object(
            "rear_target",
            ObjectGeometry(w * 0.49, top + box_h + 90, box_w, box_h),
            health=3.0,
            penetration_mode=PenetrationMode.NEVER,
            penetration_resistance=2.0,
            death_effect="rear_down",
            z_index=4,
        )
        back.metadata["label"] = "TARGET BAKOM GLAS"
        self.manager.add(back)

        glass = GameObject(
            object_id="glass_panel",
            geometry=ObjectGeometry(w * 0.49, top + box_h + 90, box_w, box_h),
            object_type="glass",
            role="target",
            tags={"breakable", "glass", "penetrable"},
            z_index=10,
            hit_shape=HitShapeSpec.rect(),
            ballistic_body=BallisticBody(
                penetration_mode=PenetrationMode.ALWAYS,
                penetration_resistance=0.4,
            ),
            damage_model=DamageModel(
                layers=[DurabilityLayer("integrity", 1.0)],
                terminal_event="object.broken",
            ),
            reactions=[ReactionRule.on(
                "object.broken",
                EffectAction.spawn_effect("glass_shards"),
                EffectAction.play_sound("glass_break"),
                EffectAction.set_state("broken"),
                EffectAction.set_visible(False),
                EffectAction.set_active(False),
                once=True,
            )],
        )
        glass.metadata["label"] = "GLAS FRAMFÖR TARGET / PENETRERBAR"
        self.manager.add(glass)

        moving = make_living_object(
            "moving_living",
            ObjectGeometry(w * 0.08, top + box_h + 105, box_w * 0.85, box_h * 0.85),
            health=2.0,
            penetration_mode=PenetrationMode.IF_POWER,
            penetration_resistance=0.8,
            death_effect="moving_down",
            z_index=5,
        )
        moving.motion_kind = MotionKind.KINEMATIC
        moving.velocity_x = max(120.0, w * 0.14)
        moving.metadata["label"] = "MOVING LIVING / shot_id snapshot"
        moving.update_callback = self._bounce_moving
        self.manager.add(moving)

    def _bounce_moving(self, obj: GameObject, dt: float) -> None:
        del dt
        left = self.viewport.w * 0.05
        right = self.viewport.w * 0.40 - obj.geometry.width
        if obj.geometry.x <= left:
            obj.geometry.x = left
            obj.velocity_x = abs(obj.velocity_x)
        elif obj.geometry.x >= right:
            obj.geometry.x = right
            obj.velocity_x = -abs(obj.velocity_x)

    def get_hit_regions(self):
        return self.manager.get_hit_regions()

    def update(self, dt: float):
        self.manager.update(dt)
        return None

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None
        if event.key == pygame.K_1:
            self._projectile_index = 0
        elif event.key == pygame.K_2:
            self._projectile_index = 1
        elif event.key == pygame.K_3:
            self._projectile_index = 2
        elif event.key == pygame.K_r:
            self._build_objects()
            self.last_line = "Objekten nollställda"
            self.last_effects = ""
        return None

    def _on_hit(self, hit: HitEvent) -> None:
        before_effects = len(self.manager.effect_requests)
        result = self.manager.resolve_hit(hit, self.projectile)
        self.last_shot_id = getattr(hit, "shot_id", None)
        if not result.interactions:
            self.last_line = (
                f"shot={self.last_shot_id} ({result.game_x:.0f},{result.game_y:.0f}) -> INGET OBJEKT "
                f"frozen={result.used_frozen_snapshot}"
            )
        else:
            parts = []
            for item in result.interactions:
                verb = "PEN" if item.penetrated else "STOP"
                parts.append(f"{item.object_id}:{item.role}:{verb}:dmg={item.damage_applied:.1f}")
            self.last_line = (
                f"shot={self.last_shot_id} frozen={result.used_frozen_snapshot} -> "
                + " | ".join(parts)
            )
        new_effects = self.manager.effect_requests[before_effects:]
        self.last_effects = ", ".join(
            f"{e.payload.get('kind')}:{e.payload.get('name')}" for e in new_effects
        ) or "inga effect requests"
        print(
            "[V2.25.0 OBJECT-HIT] "
            f"shot={self.last_shot_id} frozen={result.used_frozen_snapshot} "
            f"xy=({result.game_x:.1f},{result.game_y:.1f}) "
            f"projectile={result.projectile_profile_id} "
            f"objects={[i.object_id for i in result.interactions]} "
            f"stopped={result.stopped}"
        )

    def render(self, screen: pygame.Surface) -> None:
        surf = pygame.Surface((self.viewport.w, self.viewport.h))
        surf.fill(BG)
        self._draw_grid(surf)
        self._draw_hud(surf)
        self._draw_objects(surf)
        screen.blit(surf, self.viewport.topleft)

    def _draw_grid(self, surf: pygame.Surface) -> None:
        for x in range(0, surf.get_width(), 40):
            pygame.draw.line(surf, GRID, (x, HUD_H), (x, surf.get_height()), 1)
        for y in range(HUD_H, surf.get_height(), 40):
            pygame.draw.line(surf, GRID, (0, y), (surf.get_width(), y), 1)

    def _draw_hud(self, surf: pygame.Surface) -> None:
        pygame.draw.rect(surf, (21, 26, 35), (0, 0, surf.get_width(), HUD_H))
        profile = self.projectile
        title = self._font_big.render("V2.25.0 GAME OBJECT SYSTEM", True, CYAN)
        surf.blit(title, (14, 10))
        ptxt = (
            f"Projectile {profile.profile_id}: damage={profile.damage:g} penetration={profile.penetration_power:g} "
            f"max_hits={profile.max_object_hits}   [1/2/3]"
        )
        surf.blit(self._font.render(ptxt, True, TEXT), (14, 43))
        surf.blit(self._small.render(self.last_line[:150], True, WHITE), (14, 75))
        surf.blit(self._small.render(("effects: " + self.last_effects)[:150], True, MUTED), (14, 99))
        surf.blit(self._small.render("R = reset   ESC = tillbaka   |   caliber_label är metadata; gameplay-värden styr penetration/skada", True, MUTED), (14, 116))

    def _draw_objects(self, surf: pygame.Surface) -> None:
        for obj in sorted(self.manager.objects, key=lambda o: (o.z_index, o.object_id)):
            if not obj.visible:
                continue
            g = obj.geometry
            rect = pygame.Rect(int(g.x), int(g.y), int(g.width), int(g.height))
            if obj.object_type == "living":
                color = GREEN
            elif obj.object_type == "breakable":
                color = AMBER
            elif obj.object_type == "glass":
                color = CYAN
            elif obj.role == "no_shoot":
                color = RED
            else:
                color = BLUE

            if obj.hit_shape is not None and obj.hit_shape.kind in {"ellipse", "circle"}:
                pygame.draw.ellipse(surf, color, rect)
                pygame.draw.ellipse(surf, WHITE, rect, 2)
            else:
                pygame.draw.rect(surf, color, rect, border_radius=7)
                pygame.draw.rect(surf, WHITE, rect, 2, border_radius=7)

            label = str(obj.metadata.get("label", obj.object_id))
            img = self._small.render(label, True, BLACK if obj.object_type != "glass" else BLACK)
            surf.blit(img, img.get_rect(center=rect.center))

            if obj.damage_model is not None:
                status = " ".join(
                    f"{layer.name}:{float(layer.current or 0):.0f}/{layer.maximum:.0f}"
                    for layer in obj.damage_model.layers
                )
                info = self._small.render(status, True, WHITE)
                surf.blit(info, (rect.x, rect.bottom + 4))


def create_game(game_root: str, viewport: pygame.Rect):
    return GameObjectsTestV250(game_root=game_root, viewport=viewport)
