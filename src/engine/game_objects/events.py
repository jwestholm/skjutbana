"""Object event / reaction primitives for V2.25."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class GameObjectEvent:
    event_type: str
    object_id: str
    shot_id: int | None = None
    timestamp: float = field(default_factory=time.time)
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EffectAction:
    """Declarative response to an object event.

    Core actions (state/visibility/active/remove/emit_event) are applied by the
    ObjectManager.  Media/visual actions such as play_sound/spawn_effect/
    animation are emitted as effect requests for a future renderer/audio layer.
    """

    kind: str
    name: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def play_sound(cls, cue: str, **params: Any) -> "EffectAction":
        return cls("play_sound", cue, params)

    @classmethod
    def spawn_effect(cls, effect: str, **params: Any) -> "EffectAction":
        return cls("spawn_effect", effect, params)

    @classmethod
    def animation(cls, animation: str, **params: Any) -> "EffectAction":
        return cls("animation", animation, params)

    @classmethod
    def set_state(cls, state: str) -> "EffectAction":
        return cls("set_state", state, {})

    @classmethod
    def set_visible(cls, visible: bool) -> "EffectAction":
        return cls("set_visible", "", {"visible": bool(visible)})

    @classmethod
    def set_active(cls, active: bool) -> "EffectAction":
        return cls("set_active", "", {"active": bool(active)})

    @classmethod
    def remove(cls) -> "EffectAction":
        return cls("remove", "", {})

    @classmethod
    def emit_event(cls, event_type: str, **payload: Any) -> "EffectAction":
        return cls("emit_event", event_type, payload)


@dataclass(frozen=True, slots=True)
class ReactionRule:
    trigger: str
    actions: tuple[EffectAction, ...]
    once: bool = False
    require_state: str | None = None
    require_tags: frozenset[str] = frozenset()
    require_payload: Mapping[str, Any] = field(default_factory=dict)
    require_projectile_tags: frozenset[str] = frozenset()

    @classmethod
    def on(cls, trigger: str, *actions: EffectAction, once: bool = False,
           require_state: str | None = None, require_tags: Iterable[str] = (),
           require_payload: Mapping[str, Any] | None = None,
           require_projectile_tags: Iterable[str] = ()) -> "ReactionRule":
        return cls(
            trigger=str(trigger),
            actions=tuple(actions),
            once=bool(once),
            require_state=require_state,
            require_tags=frozenset(str(v) for v in require_tags),
            require_payload=dict(require_payload or {}),
            require_projectile_tags=frozenset(str(v) for v in require_projectile_tags),
        )


class ObjectEventBus:
    def __init__(self, history_limit: int = 256) -> None:
        self._subscribers: list[Callable[[GameObjectEvent], None]] = []
        self.history: deque[GameObjectEvent] = deque(maxlen=max(1, int(history_limit)))

    def subscribe(self, callback: Callable[[GameObjectEvent], None]) -> None:
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[GameObjectEvent], None]) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def emit(self, event: GameObjectEvent) -> GameObjectEvent:
        self.history.append(event)
        for callback in list(self._subscribers):
            try:
                callback(event)
            except Exception:
                # Object/effect listeners are game-layer observers and must not
                # crash hit processing.
                pass
        return event


__all__ = ["GameObjectEvent", "EffectAction", "ReactionRule", "ObjectEventBus"]
