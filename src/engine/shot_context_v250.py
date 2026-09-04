"""V2.25 shot identity bridge.

V2.24 proved that frozen object geometry works, but HitEvent did not carry the
scanner's shot_id.  This runtime patch preserves backward compatibility while
attaching shot identity BEFORE HitInput notifies game subscribers.

No detector authority changes are made here.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

SCHEMA_VERSION = "2.25.0"


@dataclass(frozen=True, slots=True)
class ShotEmissionContextV250:
    shot_id: int
    peak_ts: float
    scanner_state: str = "pending"


_tls = threading.local()


def current_shot_context_v250() -> ShotEmissionContextV250 | None:
    value = getattr(_tls, "shot_context_v250", None)
    return value if isinstance(value, ShotEmissionContextV250) else None


def annotate_hit_event_v250(event: Any, context: ShotEmissionContextV250 | None = None) -> Any:
    """Attach backward-compatible dynamic fields to the existing HitEvent."""
    ctx = context if context is not None else current_shot_context_v250()
    if ctx is None:
        # Make the contract explicit even for mouse/debug/non-scanner hits.
        if not hasattr(event, "shot_id"):
            setattr(event, "shot_id", None)
        if not hasattr(event, "shot_peak_ts"):
            setattr(event, "shot_peak_ts", None)
        if not hasattr(event, "shot_context_schema"):
            setattr(event, "shot_context_schema", SCHEMA_VERSION)
        return event

    setattr(event, "shot_id", int(ctx.shot_id))
    setattr(event, "shot_peak_ts", float(ctx.peak_ts))
    setattr(event, "shot_scanner_state", str(ctx.scanner_state))
    setattr(event, "shot_context_schema", SCHEMA_VERSION)
    return event


def _install_hit_input_bridge(HitInputClass: type) -> None:
    if getattr(HitInputClass, "_v250_shot_context_installed", False):
        return
    previous = HitInputClass._build_event_from_camera
    HitInputClass._v250_previous_build_event_from_camera = previous

    def wrapped_build_event(self, *args, **kwargs):
        event = previous(self, *args, **kwargs)
        # _build_event_from_camera returns before HitInput.push_camera_hit calls
        # _notify(), so subscribers see shot_id on first delivery.
        return annotate_hit_event_v250(event)

    HitInputClass._build_event_from_camera = wrapped_build_event

    # Also normalize non-camera/debug events at the notification boundary when
    # the current HitInput implementation exposes _notify(). This gives game
    # code a consistent getattr/direct-attribute contract: mouse hits carry
    # shot_id=None while camera hits retain the scanner context above.
    previous_notify = getattr(HitInputClass, "_notify", None)
    if callable(previous_notify):
        HitInputClass._v250_previous_notify = previous_notify

        def wrapped_notify(self, event, *args, **kwargs):
            return previous_notify(self, annotate_hit_event_v250(event), *args, **kwargs)

        HitInputClass._notify = wrapped_notify

    HitInputClass._v250_shot_context_installed = True


def _install_scanner_bridge(HitScannerClass: type) -> None:
    if getattr(HitScannerClass, "_v250_shot_context_installed", False):
        return
    previous = HitScannerClass._emit_track_result
    HitScannerClass._v250_previous_emit_track_result = previous

    def wrapped_emit(self, track, event):
        old = getattr(_tls, "shot_context_v250", None)
        _tls.shot_context_v250 = ShotEmissionContextV250(
            shot_id=int(getattr(event, "shot_id", 0) or 0),
            peak_ts=float(getattr(event, "peak_ts", 0.0) or 0.0),
            scanner_state=str(getattr(event, "state", "pending") or "pending"),
        )
        try:
            return previous(self, track, event)
        finally:
            if old is None:
                try:
                    delattr(_tls, "shot_context_v250")
                except AttributeError:
                    pass
            else:
                _tls.shot_context_v250 = old

    HitScannerClass._emit_track_result = wrapped_emit
    HitScannerClass._v250_shot_context_installed = True


def install_v250_runtime(AppClass=None) -> None:
    del AppClass  # kept for the same installer signature as V2.22-V2.24 patches
    from src.engine.input.hit_input import HitInput
    from src.engine.camera.hit_scanner import HitScanner

    _install_hit_input_bridge(HitInput)
    _install_scanner_bridge(HitScanner)
    print("[V2.25.0] shot-id HitEvent bridge + GameObject foundation installed")


__all__ = [
    "SCHEMA_VERSION",
    "ShotEmissionContextV250",
    "current_shot_context_v250",
    "annotate_hit_event_v250",
    "install_v250_runtime",
]
