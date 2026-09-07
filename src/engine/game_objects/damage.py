"""Composable durability/damage layers for V2.25 GameObjects."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(slots=True)
class DurabilityLayer:
    name: str
    maximum: float
    current: float | None = None
    damage_multiplier: float = 1.0
    damage_type_multipliers: Mapping[str, float] = field(default_factory=dict)
    spillover: bool = True

    def __post_init__(self) -> None:
        self.maximum = max(0.0, float(self.maximum))
        if self.current is None:
            self.current = self.maximum
        else:
            self.current = max(0.0, min(self.maximum, float(self.current)))

    @property
    def depleted(self) -> bool:
        return float(self.current or 0.0) <= 1e-9

    def reset(self) -> None:
        self.current = float(self.maximum)

    def apply(self, incoming_damage: float, damage_type: str) -> tuple[float, float, bool]:
        """Return (applied_to_layer, remaining_incoming, depleted_now)."""
        incoming = max(0.0, float(incoming_damage))
        if incoming <= 0.0 or self.depleted:
            return 0.0, incoming, False

        type_mul = float(self.damage_type_multipliers.get(str(damage_type), 1.0))
        mul = max(0.0, float(self.damage_multiplier) * type_mul)
        if mul <= 1e-12:
            return 0.0, 0.0 if not self.spillover else incoming, False

        effective = incoming * mul
        before = float(self.current or 0.0)
        applied = min(before, effective)
        self.current = max(0.0, before - applied)
        depleted_now = before > 1e-9 and self.depleted

        if not self.spillover:
            remaining = 0.0
        else:
            effective_left = max(0.0, effective - applied)
            remaining = effective_left / mul
        return float(applied), float(remaining), bool(depleted_now)


@dataclass(frozen=True, slots=True)
class LayerDamageResult:
    layer: str
    applied: float
    remaining: float
    current: float
    maximum: float
    depleted_now: bool


@dataclass(frozen=True, slots=True)
class DamageReport:
    requested: float
    applied: float
    remaining: float
    layers: tuple[LayerDamageResult, ...]
    terminal_now: bool
    terminal_event: str | None


@dataclass(slots=True)
class DamageModel:
    layers: list[DurabilityLayer] = field(default_factory=list)
    terminal_event: str = "object.destroyed"

    @property
    def depleted(self) -> bool:
        return bool(self.layers) and all(layer.depleted for layer in self.layers)

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def layer(self, name: str) -> DurabilityLayer | None:
        key = str(name)
        return next((layer for layer in self.layers if layer.name == key), None)

    def apply_damage(self, amount: float, damage_type: str = "projectile") -> DamageReport:
        requested = max(0.0, float(amount))
        remaining = requested
        results: list[LayerDamageResult] = []
        was_depleted = self.depleted

        for layer in self.layers:
            if remaining <= 1e-12:
                break
            applied, remaining, depleted_now = layer.apply(remaining, damage_type)
            results.append(LayerDamageResult(
                layer=layer.name,
                applied=float(applied),
                remaining=float(remaining),
                current=float(layer.current or 0.0),
                maximum=float(layer.maximum),
                depleted_now=bool(depleted_now),
            ))

        terminal_now = bool(not was_depleted and self.depleted)
        return DamageReport(
            requested=requested,
            applied=float(sum(r.applied for r in results)),
            remaining=float(remaining),
            layers=tuple(results),
            terminal_now=terminal_now,
            terminal_event=self.terminal_event if terminal_now else None,
        )


__all__ = [
    "DurabilityLayer",
    "LayerDamageResult",
    "DamageReport",
    "DamageModel",
]
