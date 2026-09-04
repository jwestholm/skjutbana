"""Gameplay projectile and penetration contracts for V2.25.

This is intentionally a GAMEPLAY model, not a real-world ballistic calculator.
A caliber label may be carried for game/config selection, but penetration and
damage are explicit game parameters supplied by the game.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class PenetrationMode(str, Enum):
    NEVER = "never"
    IF_POWER = "if_power"
    ALWAYS = "always"


@dataclass(frozen=True, slots=True)
class ProjectileProfile:
    profile_id: str = "default"
    caliber_label: str = ""
    diameter_mm: float | None = None
    damage: float = 1.0
    damage_type: str = "projectile"
    penetration_power: float = 0.0
    max_object_hits: int = 1
    tags: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def start(self) -> "ProjectileState":
        return ProjectileState(
            profile=self,
            remaining_penetration=max(0.0, float(self.penetration_power)),
            object_hits=0,
        )


@dataclass(slots=True)
class ProjectileState:
    profile: ProjectileProfile
    remaining_penetration: float
    object_hits: int = 0
    stopped: bool = False


@dataclass(frozen=True, slots=True)
class BallisticBody:
    """How a game object interacts with a projectile at the same game XY."""

    material_id: str = "generic"
    penetration_mode: PenetrationMode = PenetrationMode.NEVER
    penetration_resistance: float = 1.0
    damage_multiplier: float = 1.0
    receives_damage: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def penetration_result(self, remaining_power: float) -> tuple[bool, float]:
        resistance = max(0.0, float(self.penetration_resistance))
        mode = PenetrationMode(self.penetration_mode)
        if mode == PenetrationMode.NEVER:
            return False, max(0.0, float(remaining_power))
        if mode == PenetrationMode.ALWAYS:
            return True, max(0.0, float(remaining_power) - resistance)
        if float(remaining_power) + 1e-9 >= resistance:
            return True, max(0.0, float(remaining_power) - resistance)
        return False, max(0.0, float(remaining_power))

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "material_id": str(self.material_id),
            "penetration_mode": str(self.penetration_mode.value),
            "penetration_resistance": float(self.penetration_resistance),
            "damage_multiplier": float(self.damage_multiplier),
            "receives_damage": bool(self.receives_damage),
            "metadata": dict(self.metadata),
        }


DEFAULT_PROJECTILE = ProjectileProfile()


__all__ = [
    "PenetrationMode",
    "ProjectileProfile",
    "ProjectileState",
    "BallisticBody",
    "DEFAULT_PROJECTILE",
]
