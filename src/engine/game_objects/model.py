"""Core V2.25 GameObject model.

The canonical model is capability/composition based.  ``object_type`` and tags
express semantic identity; hit shape, ballistic body, durability and reactions
express independent behaviours.  This avoids an inheritance dead-end for
objects that are simultaneously living, armored, breakable and penetrable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from src.engine.input.hit_regions import HitRegion

from .damage import DamageModel
from .events import ReactionRule
from .geometry import HitShapeSpec, ObjectGeometry, WorldPlacement, shape_bounds
from .projectiles import BallisticBody

SCHEMA_VERSION = "2.25.0"


class LifecycleState(str, Enum):
    ACTIVE = "active"
    DISABLED = "disabled"
    REMOVED = "removed"


class MotionKind(str, Enum):
    STATIC = "static"
    KINEMATIC = "kinematic"
    DYNAMIC = "dynamic"  # reserved for a future physics service


@dataclass(slots=True)
class GameObject:
    object_id: str
    geometry: ObjectGeometry
    object_type: str = "generic"
    role: str = "target"
    owner: str = ""
    entity_id: str = ""
    part_id: str = ""
    tags: set[str] = field(default_factory=set)
    lifecycle: LifecycleState = LifecycleState.ACTIVE
    visible: bool = True
    state: str = "default"
    motion_kind: MotionKind = MotionKind.STATIC
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    z_index: int = 0
    hit_depth: float | None = None
    search_priority: float = 1.0
    hit_shape: HitShapeSpec | None = field(default_factory=HitShapeSpec.rect)
    ballistic_body: BallisticBody = field(default_factory=BallisticBody)
    damage_model: DamageModel | None = None
    reactions: list[ReactionRule] = field(default_factory=list)
    world: WorldPlacement | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    update_callback: Callable[["GameObject", float], None] | None = None
    render_callback: Callable[[Any, "GameObject"], None] | None = None
    generation: int = 0
    _fired_reactions: set[int] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self.object_id = str(self.object_id).strip()
        if not self.object_id:
            raise ValueError("GameObject.object_id must be non-empty")
        self.object_type = str(self.object_type or "generic")
        self.role = str(self.role or "target")
        self.owner = str(self.owner or "")
        self.entity_id = str(self.entity_id or self.object_id)
        self.part_id = str(self.part_id or "")
        self.tags = {str(v) for v in self.tags}
        self.lifecycle = LifecycleState(self.lifecycle)
        self.motion_kind = MotionKind(self.motion_kind)

    @property
    def active(self) -> bool:
        return self.lifecycle == LifecycleState.ACTIVE

    @active.setter
    def active(self, value: bool) -> None:
        self.lifecycle = LifecycleState.ACTIVE if value else LifecycleState.DISABLED

    @property
    def removed(self) -> bool:
        return self.lifecycle == LifecycleState.REMOVED

    @property
    def hittable(self) -> bool:
        return self.active and self.hit_shape is not None

    @property
    def depth(self) -> float:
        return float(self.hit_depth if self.hit_depth is not None else self.z_index)

    def update(self, dt: float) -> None:
        if not self.active:
            return
        step = max(0.0, float(dt))
        if self.motion_kind == MotionKind.KINEMATIC:
            self.geometry.move(float(self.velocity_x) * step, float(self.velocity_y) * step)
        if self.update_callback is not None:
            self.update_callback(self, step)

    def render(self, surface: Any) -> None:
        if self.visible and not self.removed and self.render_callback is not None:
            self.render_callback(surface, self)

    def shape_snapshot(self) -> dict[str, Any] | None:
        if self.hit_shape is None:
            return None
        return self.hit_shape.snapshot(self.geometry)

    def object_snapshot_metadata(self) -> dict[str, Any] | None:
        shape = self.shape_snapshot()
        if shape is None:
            return None
        return {
            "schema": SCHEMA_VERSION,
            "object_id": self.object_id,
            "generation": int(self.generation),
            "object_type": self.object_type,
            "role": self.role,
            "owner": self.owner,
            "entity_id": self.entity_id,
            "part_id": self.part_id,
            "tags": sorted(self.tags),
            "state": self.state,
            "lifecycle": self.lifecycle.value,
            "z_index": int(self.z_index),
            "hit_depth": float(self.depth),
            "motion_kind": self.motion_kind.value,
            "shape": shape,
            "ballistic_body": self.ballistic_body.to_snapshot(),
            "world": None if self.world is None else {
                "distance_m": self.world.distance_m,
                "world_x_m": self.world.world_x_m,
                "world_y_m": self.world.world_y_m,
                "world_z_m": self.world.world_z_m,
                "metadata": dict(self.world.metadata or {}),
            },
            "metadata": dict(self.metadata),
        }

    def get_hit_region(self) -> HitRegion | None:
        """Return fast approximate search AABB with exact frozen shape metadata."""
        if not self.hittable:
            return None
        snapshot = self.object_snapshot_metadata()
        if snapshot is None:
            return None
        bounds = shape_bounds(snapshot["shape"])
        if bounds is None:
            return None
        x, y, w, h = bounds
        return HitRegion(
            object_id=self.object_id,
            x=float(x),
            y=float(y),
            width=float(w),
            height=float(h),
            role=self.role,
            priority=max(0.0, float(self.search_priority)),
            enabled=True,
            owner=self.owner,
            metadata={
                "v250_game_object": True,
                "v250_object_snapshot": snapshot,
            },
        )

    def reaction_actions(self, event_type: str, payload: Mapping[str, Any] | None = None) -> list[tuple[int, ReactionRule]]:
        result: list[tuple[int, ReactionRule]] = []
        payload_map = dict(payload or {})
        projectile_tags = {str(v) for v in payload_map.get("projectile_tags", ()) or ()}
        for index, rule in enumerate(self.reactions):
            if rule.trigger != str(event_type):
                continue
            if rule.once and index in self._fired_reactions:
                continue
            if rule.require_state is not None and self.state != rule.require_state:
                continue
            if rule.require_tags and not rule.require_tags.issubset(self.tags):
                continue
            if rule.require_projectile_tags and not rule.require_projectile_tags.issubset(projectile_tags):
                continue
            if any(payload_map.get(key) != value for key, value in dict(rule.require_payload).items()):
                continue
            result.append((index, rule))
        return result

    def mark_reaction_fired(self, index: int) -> None:
        self._fired_reactions.add(int(index))

    def reset_runtime_state(self) -> None:
        self._fired_reactions.clear()
        if self.damage_model is not None:
            self.damage_model.reset()


__all__ = [
    "SCHEMA_VERSION",
    "LifecycleState",
    "MotionKind",
    "GameObject",
]
