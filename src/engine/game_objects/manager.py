"""V2.25 ObjectManager: hit regions, exact frozen collision, damage and reactions."""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Iterable, Mapping

from src.engine.input.hit_regions import latest_hit_context_snapshot

from .damage import DamageReport
from .events import EffectAction, GameObjectEvent, ObjectEventBus
from .geometry import shape_contains
from .model import GameObject, LifecycleState
from .projectiles import BallisticBody, DEFAULT_PROJECTILE, PenetrationMode, ProjectileProfile


@dataclass(frozen=True, slots=True)
class ObjectInteraction:
    object_id: str
    object_type: str
    role: str
    entity_id: str
    part_id: str
    depth: float
    penetrated: bool
    blocked: bool
    damage_requested: float
    damage_applied: float
    damage_report: DamageReport | None
    remaining_penetration: float
    snapshot_generation: int | None
    live_generation: int | None


@dataclass(frozen=True, slots=True)
class ShotResolution:
    shot_id: int | None
    game_x: float
    game_y: float
    projectile_profile_id: str
    used_frozen_snapshot: bool
    snapshot_id: int | None
    interactions: tuple[ObjectInteraction, ...]
    stopped: bool
    unclaimed: bool
    timestamp: float = field(default_factory=time.time)


class ObjectManager:
    """Owns GameObjects and resolves gameplay consequences from HitEvents.

    The manager never moves the hit coordinate.  It consumes the already-resolved
    physical ``HitEvent.game_x/game_y`` and optionally uses the exact V2.25 shape
    snapshots embedded in the V2.24 shot-time HitRegions.
    """

    def __init__(
        self,
        *,
        snapshot_provider: Callable[[int], Any | None] | None = None,
        event_history_limit: int = 256,
        effect_request_limit: int = 256,
    ) -> None:
        self._objects: dict[str, GameObject] = {}
        self._order: list[str] = []
        self._generations: dict[str, int] = {}
        self.events = ObjectEventBus(event_history_limit)
        self.effect_requests: list[GameObjectEvent] = []
        self._effect_request_limit = max(1, int(effect_request_limit))
        self._snapshot_provider = snapshot_provider
        self.last_resolution: ShotResolution | None = None
        self._reaction_depth = 0

    # ------------------------------------------------------------------
    # Object lifecycle
    # ------------------------------------------------------------------
    def add(self, obj: GameObject) -> GameObject:
        key = str(obj.object_id)
        if key in self._objects and not self._objects[key].removed:
            raise ValueError(f"GameObject id already exists: {key}")
        generation = self._generations.get(key, 0) + 1
        self._generations[key] = generation
        obj.generation = generation
        obj.lifecycle = LifecycleState.ACTIVE
        self._objects[key] = obj
        if key not in self._order:
            self._order.append(key)
        self._emit(obj, "object.spawned", payload={"generation": generation})
        return obj

    def get(self, object_id: str) -> GameObject | None:
        return self._objects.get(str(object_id))

    def remove(self, object_id: str) -> GameObject | None:
        obj = self.get(object_id)
        if obj is None or obj.removed:
            return obj
        obj.lifecycle = LifecycleState.REMOVED
        self._emit(obj, "object.removed")
        return obj

    def clear(self) -> None:
        for key in list(self._order):
            self.remove(key)
        self._objects.clear()
        self._order.clear()

    @property
    def objects(self) -> tuple[GameObject, ...]:
        return tuple(
            self._objects[key]
            for key in self._order
            if key in self._objects and not self._objects[key].removed
        )

    def update(self, dt: float) -> None:
        for obj in self.objects:
            obj.update(dt)

    def render(self, surface: Any) -> None:
        for obj in sorted(self.objects, key=lambda o: (o.z_index, o.object_id)):
            obj.render(surface)

    def get_hit_regions(self):
        regions = []
        for obj in self.objects:
            region = obj.get_hit_region()
            if region is not None:
                regions.append(region)
        return tuple(regions)

    # ------------------------------------------------------------------
    # Frozen shot context
    # ------------------------------------------------------------------
    def _snapshot_for_shot(self, shot_id: int | None):
        if shot_id is None:
            return None
        if self._snapshot_provider is not None:
            try:
                return self._snapshot_provider(int(shot_id))
            except Exception:
                return None
        try:
            from src.engine.input.object_hit_v2223 import object_hit_registry_v2223
            return object_hit_registry_v2223.snapshot_for_shot(int(shot_id))
        except Exception:
            return None

    @staticmethod
    def _object_snapshot_from_region(region: Any) -> Mapping[str, Any] | None:
        try:
            meta = dict(getattr(region, "metadata", {}) or {})
            if not meta.get("v250_game_object"):
                return None
            snap = meta.get("v250_object_snapshot")
            return snap if isinstance(snap, Mapping) else None
        except Exception:
            return None

    def _frozen_hits(self, shot_id: int, x: float, y: float) -> tuple[Any | None, list[Mapping[str, Any]]]:
        snapshot = self._snapshot_for_shot(shot_id)
        if snapshot is None:
            return None, []
        hits: list[Mapping[str, Any]] = []
        for region in tuple(getattr(snapshot, "game_regions", ()) or ()):
            obj_snapshot = self._object_snapshot_from_region(region)
            if obj_snapshot is None:
                continue
            shape = obj_snapshot.get("shape")
            if isinstance(shape, Mapping) and shape_contains(shape, x, y):
                hits.append(obj_snapshot)
        return snapshot, hits

    def _current_hits(self, x: float, y: float) -> list[Mapping[str, Any]]:
        hits: list[Mapping[str, Any]] = []
        for obj in self.objects:
            if not obj.hittable:
                continue
            snap = obj.object_snapshot_metadata()
            if snap is None:
                continue
            shape = snap.get("shape")
            if isinstance(shape, Mapping) and shape_contains(shape, x, y):
                hits.append(snap)
        return hits

    @staticmethod
    def _body_from_snapshot(snapshot: Mapping[str, Any], fallback: BallisticBody) -> BallisticBody:
        raw = snapshot.get("ballistic_body")
        if not isinstance(raw, Mapping):
            return fallback
        try:
            return BallisticBody(
                material_id=str(raw.get("material_id", fallback.material_id)),
                penetration_mode=PenetrationMode(str(raw.get("penetration_mode", fallback.penetration_mode.value))),
                penetration_resistance=float(raw.get("penetration_resistance", fallback.penetration_resistance)),
                damage_multiplier=float(raw.get("damage_multiplier", fallback.damage_multiplier)),
                receives_damage=bool(raw.get("receives_damage", fallback.receives_damage)),
                metadata=dict(raw.get("metadata", {}) or {}),
            )
        except Exception:
            return fallback

    # ------------------------------------------------------------------
    # Hit resolution
    # ------------------------------------------------------------------
    def resolve_hit(self, hit: Any, projectile: ProjectileProfile | None = None) -> ShotResolution:
        projectile = projectile or DEFAULT_PROJECTILE
        state = projectile.start()
        x = float(getattr(hit, "game_x"))
        y = float(getattr(hit, "game_y"))
        raw_shot_id = getattr(hit, "shot_id", None)
        shot_id = int(raw_shot_id) if raw_shot_id is not None else None

        snapshot = None
        frozen: list[Mapping[str, Any]] = []
        if shot_id is not None:
            snapshot, frozen = self._frozen_hits(shot_id, x, y)
        used_frozen = snapshot is not None
        candidates = frozen if used_frozen else self._current_hits(x, y)
        candidates.sort(
            key=lambda s: (float(s.get("hit_depth", s.get("z_index", 0.0))), str(s.get("object_id", ""))),
            reverse=True,
        )

        interactions: list[ObjectInteraction] = []
        for object_snapshot in candidates:
            if state.stopped or state.object_hits >= max(1, int(projectile.max_object_hits)):
                state.stopped = True
                break

            object_id = str(object_snapshot.get("object_id", ""))
            obj = self.get(object_id)
            snapshot_generation = int(object_snapshot.get("generation", 0) or 0)
            live_generation = None if obj is None else int(obj.generation)
            if obj is None or obj.removed or (snapshot_generation and live_generation != snapshot_generation):
                # Do not apply a delayed shot to a newly-spawned object that reused
                # the same id.  Emit a diagnostic event instead.
                self.events.emit(GameObjectEvent(
                    "shot.stale_object",
                    object_id=object_id,
                    shot_id=shot_id,
                    payload={
                        "snapshot_generation": snapshot_generation,
                        "live_generation": live_generation,
                    },
                ))
                continue

            body = self._body_from_snapshot(object_snapshot, obj.ballistic_body)
            damage_requested = max(0.0, float(projectile.damage) * max(0.0, float(body.damage_multiplier)))
            damage_report = None
            damage_applied = 0.0

            self._emit(obj, "shot.hit", shot_id=shot_id, payload={
                "game_x": x,
                "game_y": y,
                "projectile_profile": projectile.profile_id,
                "caliber_label": projectile.caliber_label,
                "damage_type": projectile.damage_type,
                "projectile_tags": sorted(projectile.tags),
                "penetration_before": state.remaining_penetration,
            })

            if body.receives_damage and obj.damage_model is not None and damage_requested > 0.0:
                damage_report = obj.damage_model.apply_damage(damage_requested, projectile.damage_type)
                damage_applied = float(damage_report.applied)
                self._emit(obj, "damage.applied", shot_id=shot_id, payload={
                    "requested": damage_report.requested,
                    "applied": damage_report.applied,
                    "remaining": damage_report.remaining,
                    "damage_type": projectile.damage_type,
                    "caliber_label": projectile.caliber_label,
                    "projectile_tags": sorted(projectile.tags),
                    "layers": [
                        {
                            "layer": r.layer,
                            "applied": r.applied,
                            "current": r.current,
                            "maximum": r.maximum,
                            "depleted_now": r.depleted_now,
                        }
                        for r in damage_report.layers
                    ],
                })
                for layer_result in damage_report.layers:
                    if layer_result.depleted_now:
                        self._emit(obj, "durability.depleted", shot_id=shot_id, payload={"layer": layer_result.layer})
                if damage_report.terminal_now and damage_report.terminal_event:
                    self._emit(obj, damage_report.terminal_event, shot_id=shot_id, payload={
                        "damage_type": projectile.damage_type,
                        "projectile_profile": projectile.profile_id,
                        "caliber_label": projectile.caliber_label,
                        "projectile_tags": sorted(projectile.tags),
                    })

            penetrated, remaining = body.penetration_result(state.remaining_penetration)
            state.object_hits += 1
            state.remaining_penetration = float(remaining)
            blocked = not penetrated
            self._emit(
                obj,
                "shot.penetrated" if penetrated else "shot.blocked",
                shot_id=shot_id,
                payload={
                    "projectile_profile": projectile.profile_id,
                    "caliber_label": projectile.caliber_label,
                    "projectile_tags": sorted(projectile.tags),
                    "remaining_penetration": state.remaining_penetration,
                },
            )

            interactions.append(ObjectInteraction(
                object_id=obj.object_id,
                object_type=obj.object_type,
                role=str(object_snapshot.get("role", obj.role)),
                entity_id=str(object_snapshot.get("entity_id", obj.entity_id)),
                part_id=str(object_snapshot.get("part_id", obj.part_id)),
                depth=float(object_snapshot.get("hit_depth", object_snapshot.get("z_index", obj.depth))),
                penetrated=bool(penetrated),
                blocked=bool(blocked),
                damage_requested=float(damage_requested),
                damage_applied=float(damage_applied),
                damage_report=damage_report,
                remaining_penetration=float(state.remaining_penetration),
                snapshot_generation=snapshot_generation or None,
                live_generation=live_generation,
            ))

            if blocked:
                state.stopped = True
                break

        resolution = ShotResolution(
            shot_id=shot_id,
            game_x=x,
            game_y=y,
            projectile_profile_id=projectile.profile_id,
            used_frozen_snapshot=bool(used_frozen),
            snapshot_id=None if snapshot is None else int(getattr(snapshot, "shot_id", 0) or 0),
            interactions=tuple(interactions),
            stopped=bool(state.stopped),
            unclaimed=not bool(interactions),
        )
        self.last_resolution = resolution
        return resolution

    # ------------------------------------------------------------------
    # Event/reaction engine
    # ------------------------------------------------------------------
    def _emit(self, obj: GameObject, event_type: str, *, shot_id: int | None = None,
              payload: Mapping[str, Any] | None = None) -> GameObjectEvent:
        event = self.events.emit(GameObjectEvent(
            event_type=str(event_type),
            object_id=obj.object_id,
            shot_id=shot_id,
            payload=dict(payload or {}),
        ))
        self._run_reactions(obj, event)
        return event

    def _run_reactions(self, obj: GameObject, event: GameObjectEvent) -> None:
        if self._reaction_depth >= 8:
            self.events.emit(GameObjectEvent(
                "reaction.depth_limit",
                object_id=obj.object_id,
                shot_id=event.shot_id,
                payload={"trigger": event.event_type},
            ))
            return

        matches = obj.reaction_actions(event.event_type, event.payload)
        if not matches:
            return
        self._reaction_depth += 1
        try:
            for index, rule in matches:
                if rule.once:
                    obj.mark_reaction_fired(index)
                for action in rule.actions:
                    self._execute_action(obj, event, action)
        finally:
            self._reaction_depth -= 1

    def _execute_action(self, obj: GameObject, source_event: GameObjectEvent, action: EffectAction) -> None:
        kind = str(action.kind)
        if kind == "set_state":
            obj.state = str(action.name)
            self.events.emit(GameObjectEvent(
                "object.state_changed", obj.object_id, source_event.shot_id,
                payload={"state": obj.state, "trigger": source_event.event_type},
            ))
            return
        if kind == "set_visible":
            obj.visible = bool(action.params.get("visible", True))
            return
        if kind == "set_active":
            obj.active = bool(action.params.get("active", True))
            return
        if kind == "remove":
            self.remove(obj.object_id)
            return
        if kind == "emit_event":
            self._emit(obj, action.name, shot_id=source_event.shot_id, payload=dict(action.params))
            return

        # External effect request: sound, particles, animation, spawn, impulse,
        # screen shake, decal, scoring hook, or a future game-specific action.
        effect = GameObjectEvent(
            "effect.requested",
            object_id=obj.object_id,
            shot_id=source_event.shot_id,
            payload={
                "kind": kind,
                "name": action.name,
                "params": dict(action.params),
                "trigger": source_event.event_type,
            },
        )
        self.effect_requests.append(effect)
        if len(self.effect_requests) > self._effect_request_limit:
            del self.effect_requests[:-self._effect_request_limit]
        self.events.emit(effect)


__all__ = ["ObjectInteraction", "ShotResolution", "ObjectManager"]
