from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any


EventCallback = Callable[[dict[str, Any]], None]


class EventBus:
    """Small in-process publish/subscribe bus for engine events."""

    def __init__(self) -> None:
        self._subscribers: list[EventCallback] = []
        self._lock = threading.RLock()
        self._sequence = 0

    def subscribe(self, callback: EventCallback) -> None:
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: EventCallback) -> None:
        with self._lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

    def emit(
        self,
        event: str,
        data: dict[str, Any] | None = None,
        *,
        source: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(event, str) or not event:
            raise ValueError("event must be a non-empty string")

        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            subscribers = list(self._subscribers)

        message: dict[str, Any] = {
            "type": "event",
            "event": event,
            "sequence": sequence,
            "timestamp": time.time(),
            "data": data or {},
        }

        if source:
            message["source"] = source

        for callback in subscribers:
            try:
                callback(message)
            except Exception as exc:
                print(f"[EventBus] Subscriber error for {event}: {exc}")

        return message


event_bus = EventBus()
