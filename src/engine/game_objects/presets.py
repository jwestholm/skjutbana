"""Small convenience constructors; composition remains the canonical model."""
from __future__ import annotations

from typing import Iterable

from .damage import DamageModel, DurabilityLayer
from .events import EffectAction, ReactionRule
from .geometry import HitShapeSpec, ObjectGeometry
from .model import GameObject, MotionKind
from .projectiles import BallisticBody, PenetrationMode


def make_static_object(
    object_id: str,
    geometry: ObjectGeometry,
    *,
    object_type: str = "static",
    role: str = "target",
    owner: str = "",
    entity_id: str = "",
    part_id: str = "",
    shape: HitShapeSpec | None = None,
    tags: Iterable[str] = (),
    ballistic_body: BallisticBody | None = None,
    z_index: int = 0,
    hit_depth: float | None = None,
) -> GameObject:
    return GameObject(
        object_id=object_id,
        geometry=geometry,
        object_type=object_type,
        role=role,
        owner=owner,
        entity_id=entity_id,
        part_id=part_id,
        tags=set(tags) | {"static"},
        motion_kind=MotionKind.STATIC,
        hit_shape=shape or HitShapeSpec.rect(),
        ballistic_body=ballistic_body or BallisticBody(),
        z_index=z_index,
        hit_depth=hit_depth,
    )


def make_breakable_object(
    object_id: str,
    geometry: ObjectGeometry,
    *,
    integrity: float = 1.0,
    role: str = "target",
    owner: str = "",
    entity_id: str = "",
    part_id: str = "",
    shape: HitShapeSpec | None = None,
    tags: Iterable[str] = (),
    material_id: str = "generic",
    penetration_mode: PenetrationMode = PenetrationMode.IF_POWER,
    penetration_resistance: float = 1.0,
    break_sound: str = "",
    break_effect: str = "dust",
    hide_on_break: bool = True,
    z_index: int = 0,
    hit_depth: float | None = None,
) -> GameObject:
    actions = []
    if break_effect:
        actions.append(EffectAction.spawn_effect(break_effect))
    if break_sound:
        actions.append(EffectAction.play_sound(break_sound))
    actions.extend((EffectAction.set_state("broken"), EffectAction.set_active(False)))
    if hide_on_break:
        actions.append(EffectAction.set_visible(False))
    return GameObject(
        object_id=object_id,
        geometry=geometry,
        object_type="breakable",
        role=role,
        owner=owner,
        entity_id=entity_id,
        part_id=part_id,
        tags=set(tags) | {"breakable"},
        hit_shape=shape or HitShapeSpec.rect(),
        ballistic_body=BallisticBody(
            material_id=material_id,
            penetration_mode=penetration_mode,
            penetration_resistance=float(penetration_resistance),
        ),
        damage_model=DamageModel(
            layers=[DurabilityLayer("integrity", float(integrity))],
            terminal_event="object.broken",
        ),
        reactions=[ReactionRule.on("object.broken", *actions, once=True)],
        z_index=z_index,
        hit_depth=hit_depth,
    )


def make_living_object(
    object_id: str,
    geometry: ObjectGeometry,
    *,
    health: float = 100.0,
    role: str = "target",
    owner: str = "",
    entity_id: str = "",
    part_id: str = "",
    shape: HitShapeSpec | None = None,
    tags: Iterable[str] = (),
    material_id: str = "generic",
    penetration_mode: PenetrationMode = PenetrationMode.NEVER,
    penetration_resistance: float = 1.0,
    death_sound: str = "",
    death_effect: str = "death",
    hide_on_death: bool = False,
    z_index: int = 0,
    hit_depth: float | None = None,
) -> GameObject:
    actions = []
    if death_effect:
        actions.append(EffectAction.spawn_effect(death_effect))
    if death_sound:
        actions.append(EffectAction.play_sound(death_sound))
    actions.extend((EffectAction.set_state("dead"), EffectAction.set_active(False)))
    if hide_on_death:
        actions.append(EffectAction.set_visible(False))
    return GameObject(
        object_id=object_id,
        geometry=geometry,
        object_type="living",
        role=role,
        owner=owner,
        entity_id=entity_id,
        part_id=part_id,
        tags=set(tags) | {"living"},
        hit_shape=shape or HitShapeSpec.ellipse(),
        ballistic_body=BallisticBody(
            material_id=material_id,
            penetration_mode=penetration_mode,
            penetration_resistance=float(penetration_resistance),
        ),
        damage_model=DamageModel(
            layers=[DurabilityLayer("health", float(health))],
            terminal_event="object.died",
        ),
        reactions=[ReactionRule.on("object.died", *actions, once=True)],
        z_index=z_index,
        hit_depth=hit_depth,
    )


__all__ = ["make_static_object", "make_breakable_object", "make_living_object"]
