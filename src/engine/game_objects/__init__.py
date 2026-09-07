"""Stable public API for the V2.25 GameObject foundation."""
from .damage import DamageModel, DamageReport, DurabilityLayer, LayerDamageResult
from .events import EffectAction, GameObjectEvent, ObjectEventBus, ReactionRule
from .geometry import HitShapeSpec, ObjectGeometry, WorldPlacement, shape_bounds, shape_contains
from .manager import ObjectInteraction, ObjectManager, ShotResolution
from .model import GameObject, LifecycleState, MotionKind
from .presets import make_breakable_object, make_living_object, make_static_object
from .projectiles import (
    BallisticBody,
    DEFAULT_PROJECTILE,
    PenetrationMode,
    ProjectileProfile,
    ProjectileState,
)

__all__ = [
    "GameObject",
    "LifecycleState",
    "MotionKind",
    "ObjectGeometry",
    "WorldPlacement",
    "HitShapeSpec",
    "shape_bounds",
    "shape_contains",
    "BallisticBody",
    "PenetrationMode",
    "ProjectileProfile",
    "ProjectileState",
    "DEFAULT_PROJECTILE",
    "DurabilityLayer",
    "DamageModel",
    "DamageReport",
    "LayerDamageResult",
    "GameObjectEvent",
    "EffectAction",
    "ReactionRule",
    "ObjectEventBus",
    "ObjectInteraction",
    "ShotResolution",
    "ObjectManager",
    "make_static_object",
    "make_breakable_object",
    "make_living_object",
]
