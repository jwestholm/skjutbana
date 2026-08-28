"""V2.22 runtime integration for the fast ShotResolver.

Installed from ``src.engine.ai.bootstrap.apply_bootstrap`` before the AIRuntime
singleton is normally created.  The patch is intentionally fail-open:
- existing detector behaviour remains authoritative in off/train_only/advisory
- V2.22 never interpolates coordinates between two candidates
- external/heavy experts publish already-computed votes; the resolver itself
  stays small and synchronous
"""
from __future__ import annotations

from collections import deque
import math
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.engine.ai.shot_resolver_v222 import ShotResolverV222

SCHEMA_VERSION = "2.22"
_INSTALLED = False

V222_DEFAULT_SETTINGS: Dict[str, Any] = {
    "resolver_v222_enabled": True,
    "resolver_v222_log": False,
    "resolver_v222_cluster_radius_px": 18.0,
    "resolver_v222_max_external_votes": 96,
    "resolver_v222_game_prior_weight": 0.06,
    "resolver_v222_latency_history": 512,
    # Existing settings remain authoritative for mode/min/override/trust.
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if not math.isfinite(result):
            return default
        return result
    except Exception:
        return default


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = max(0.0, min(1.0, float(fraction))) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _ensure_v222_state(runtime: Any) -> None:
    if getattr(runtime, "_v222_state_ready", False):
        return
    settings = getattr(runtime, "settings", {}) or {}
    resolver_config = {
        "cluster_radius_px": _safe_float(settings.get("resolver_v222_cluster_radius_px", 18.0), 18.0),
        "max_external_votes_per_source": int(settings.get("resolver_v222_max_external_votes", 96) or 96),
        "game_prior_max_weight": _safe_float(settings.get("resolver_v222_game_prior_weight", 0.06), 0.06),
    }
    runtime.shot_resolver_v222 = ShotResolverV222(resolver_config)
    history_size = max(32, min(5000, int(settings.get("resolver_v222_latency_history", 512) or 512)))
    runtime._v222_resolver_latencies_ms = deque(maxlen=history_size)
    runtime._v222_end_to_end_latencies_ms = deque(maxlen=history_size)
    runtime._v222_external_votes: Dict[int, Dict[str, Dict[str, Any]]] = {}
    runtime._v222_vote_lock = threading.RLock()
    runtime._v222_game_context_by_shot: Dict[int, Dict[str, Any]] = {}
    runtime._v222_game_context_provider: Optional[Callable[..., Optional[Mapping[str, Any]]]] = None
    runtime._v222_shot_start_hooks: Dict[str, Callable[..., Any]] = {}
    runtime._v222_last_decision: Optional[Dict[str, Any]] = None
    runtime._v222_state_ready = True
    try:
        runtime.session_stats.setdefault("resolver_v222_decisions", 0)
        runtime.session_stats.setdefault("resolver_v222_overrides", 0)
        runtime.session_stats.setdefault("resolver_v222_agrees_camera", 0)
        runtime.session_stats.setdefault("resolver_v222_external_vote_batches", 0)
        runtime.session_stats.setdefault("resolver_v222_game_snapshots", 0)
    except Exception:
        pass


def _enrich_candidates(runtime: Any, candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Attach cheap shot-local evidence used by the resolver.

    The methods already exist in AIRuntime.  Failures are intentionally neutral.
    """
    enriched: List[Dict[str, Any]] = []
    post_frames = list(getattr(runtime, "_post_shot_frames", []) or [])
    for candidate in candidates:
        item = dict(candidate)
        if item.get("persistence") is None:
            try:
                item["persistence"] = float(runtime.compute_persistence(item)) if post_frames else 0.5
            except Exception:
                item["persistence"] = 0.5
        if item.get("existed_before") is None:
            try:
                item["existed_before"] = float(runtime.existed_before_shot(item))
            except Exception:
                item["existed_before"] = 0.0
        enriched.append(item)
    return enriched


def _call_game_provider(runtime: Any, shot_id: int, peak_ts: float) -> Optional[Dict[str, Any]]:
    provider = getattr(runtime, "_v222_game_context_provider", None)
    if provider is None:
        return None
    try:
        value = provider(shot_id=shot_id, peak_ts=peak_ts, runtime=runtime)
    except TypeError:
        try:
            value = provider(shot_id, peak_ts)
        except TypeError:
            value = provider()
    except Exception:
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _snapshot_game_context(runtime: Any, ctx: Any) -> None:
    _ensure_v222_state(runtime)
    shot_id = int(getattr(ctx, "shot_id", 0) or 0)
    if shot_id <= 0 or shot_id in runtime._v222_game_context_by_shot:
        return
    value = _call_game_provider(runtime, shot_id, _safe_float(getattr(ctx, "peak_ts", 0.0), 0.0))
    if value is not None:
        runtime._v222_game_context_by_shot[shot_id] = value
        try:
            runtime.session_stats["resolver_v222_game_snapshots"] = int(
                runtime.session_stats.get("resolver_v222_game_snapshots", 0) or 0
            ) + 1
        except Exception:
            pass


def _notify_shot_start_hooks(runtime: Any, ctx: Any) -> None:
    _ensure_v222_state(runtime)
    # Hooks MUST be non-blocking.  They exist so a future physical-image worker
    # can start parallel work at audio-peak time and later publish compact votes.
    for name, callback in list(runtime._v222_shot_start_hooks.items()):
        try:
            callback(runtime=runtime, shot_context=ctx)
        except TypeError:
            try:
                callback(runtime, ctx)
            except Exception:
                pass
        except Exception:
            pass


def _external_votes_for_shot(runtime: Any, shot_id: int) -> Dict[str, List[Dict[str, Any]]]:
    _ensure_v222_state(runtime)
    result: Dict[str, List[Dict[str, Any]]] = {}
    with runtime._v222_vote_lock:
        per_source = dict(runtime._v222_external_votes.get(int(shot_id), {}))
        for source, record in per_source.items():
            weight = max(0.0, _safe_float(record.get("weight", 1.0), 1.0))
            votes: List[Dict[str, Any]] = []
            for vote in list(record.get("votes", []) or []):
                item = dict(vote)
                item["expert_weight"] = weight
                votes.append(item)
            result[str(source)] = votes
    return result


def _pending_vote_shots(runtime: Any) -> List[int]:
    _ensure_v222_state(runtime)
    with runtime._v222_vote_lock:
        return sorted(int(key) for key in runtime._v222_external_votes.keys())


def _decision_log_line(decision: Mapping[str, Any], *, apply: bool, mode: str, e2e_ms: float) -> str:
    return (
        "[V2.22 RESOLVER] "
        f"shot={decision.get('shot_id')} mode={mode} apply={apply} "
        f"xy=({float(decision.get('camera_x', 0.0)):.1f},{float(decision.get('camera_y', 0.0)):.1f}) "
        f"conf={float(decision.get('confidence', 0.0)):.3f} "
        f"score={float(decision.get('score', 0.0)):.3f} "
        f"margin={float(decision.get('margin', 0.0)):.3f} "
        f"clusters={int(decision.get('cluster_count', 0) or 0)} "
        f"resolver={float(decision.get('resolver_ms', 0.0)):.2f}ms e2e={e2e_ms:.1f}ms"
    )


def install_v222_runtime_patch() -> None:
    """Patch AIRuntime once. Safe to call repeatedly."""
    global _INSTALLED
    if _INSTALLED:
        return

    import src.engine.ai.runtime as runtime_module

    # Important: update defaults before get_ai_runtime() creates the singleton.
    runtime_module.DEFAULT_SETTINGS.update(V222_DEFAULT_SETTINGS)
    AIRuntime = runtime_module.AIRuntime

    original_init = AIRuntime.__init__
    original_create_shot_context = AIRuntime._create_shot_context
    original_mark_shot_finished = AIRuntime.mark_shot_finished
    original_choose_for_emission = AIRuntime.choose_for_emission

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _ensure_v222_state(self)

    def patched_create_shot_context(self, scanner, scanner_event):
        ctx = original_create_shot_context(self, scanner, scanner_event)
        try:
            _snapshot_game_context(self, ctx)
            _notify_shot_start_hooks(self, ctx)
        except Exception:
            pass
        return ctx

    def patched_mark_shot_finished(self, shot_id: int, state: str = "finished") -> None:
        try:
            original_mark_shot_finished(self, shot_id, state)
        finally:
            try:
                _ensure_v222_state(self)
                sid = int(shot_id)
                # Keep the most recently finished shot's resolver data briefly;
                # aggressively remove older shot-owned data to prevent leaks.
                with self._v222_vote_lock:
                    for old_sid in list(self._v222_external_votes.keys()):
                        if old_sid != sid:
                            self._v222_external_votes.pop(old_sid, None)
                for old_sid in list(self._v222_game_context_by_shot.keys()):
                    if old_sid != sid:
                        self._v222_game_context_by_shot.pop(old_sid, None)
            except Exception:
                pass

    def publish_resolver_votes(
        self,
        shot_id: int,
        source: str,
        votes: Sequence[Mapping[str, Any]],
        *,
        weight: float = 1.0,
        produced_ts: Optional[float] = None,
    ) -> None:
        """Publish compact votes from a fast/parallel expert.

        Each vote must contain camera_x/camera_y (or x/y) and score/confidence.
        Heavy full-frame work must NOT be done inside choose_for_emission().
        """
        _ensure_v222_state(self)
        sid = int(shot_id)
        if sid <= 0:
            return
        max_votes = max(1, int(self.settings.get("resolver_v222_max_external_votes", 96) or 96))
        clean: List[Dict[str, Any]] = []
        for vote in votes:
            if not isinstance(vote, Mapping):
                continue
            x = _safe_float(vote.get("camera_x", vote.get("x", float("nan"))), float("nan"))
            y = _safe_float(vote.get("camera_y", vote.get("y", float("nan"))), float("nan"))
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            item = dict(vote)
            item["camera_x"] = x
            item["camera_y"] = y
            item["score"] = max(0.0, min(1.0, _safe_float(vote.get("score", vote.get("confidence", 0.0)), 0.0)))
            clean.append(item)
        clean.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        with self._v222_vote_lock:
            self._v222_external_votes.setdefault(sid, {})[str(source)] = {
                "votes": clean[:max_votes],
                "weight": max(0.0, float(weight)),
                "produced_ts": float(produced_ts if produced_ts is not None else time.time()),
            }
        try:
            self.session_stats["resolver_v222_external_vote_batches"] = int(
                self.session_stats.get("resolver_v222_external_vote_batches", 0) or 0
            ) + 1
        except Exception:
            pass

    def clear_resolver_votes(self, shot_id: Optional[int] = None) -> None:
        _ensure_v222_state(self)
        with self._v222_vote_lock:
            if shot_id is None:
                self._v222_external_votes.clear()
            else:
                self._v222_external_votes.pop(int(shot_id), None)

    def set_game_context_provider(self, provider: Optional[Callable[..., Optional[Mapping[str, Any]]]]) -> None:
        """Register a fast provider sampled once when an audio shot context is created."""
        _ensure_v222_state(self)
        self._v222_game_context_provider = provider

    def set_shot_game_context(self, shot_id: int, context: Optional[Mapping[str, Any]]) -> None:
        """Explicitly publish an already-frozen game snapshot for one shot."""
        _ensure_v222_state(self)
        sid = int(shot_id)
        if context is None:
            self._v222_game_context_by_shot.pop(sid, None)
        elif isinstance(context, Mapping):
            self._v222_game_context_by_shot[sid] = dict(context)

    def register_shot_start_hook(self, name: str, callback: Callable[..., Any]) -> None:
        """Register a NON-BLOCKING shot-start hook for future parallel experts."""
        _ensure_v222_state(self)
        self._v222_shot_start_hooks[str(name)] = callback

    def unregister_shot_start_hook(self, name: str) -> None:
        _ensure_v222_state(self)
        self._v222_shot_start_hooks.pop(str(name), None)

    def resolver_status(self) -> Dict[str, Any]:
        _ensure_v222_state(self)
        resolver_values = list(self._v222_resolver_latencies_ms)
        e2e_values = list(self._v222_end_to_end_latencies_ms)
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": bool(self.settings.get("resolver_v222_enabled", True)),
            "last_decision": dict(self._v222_last_decision) if isinstance(self._v222_last_decision, dict) else None,
            "resolver_latency_ms": {
                "n": len(resolver_values),
                "p50": _percentile(resolver_values, 0.50),
                "p95": _percentile(resolver_values, 0.95),
                "p99": _percentile(resolver_values, 0.99),
            },
            "end_to_end_latency_ms": {
                "n": len(e2e_values),
                "p50": _percentile(e2e_values, 0.50),
                "p95": _percentile(e2e_values, 0.95),
                "p99": _percentile(e2e_values, 0.99),
            },
            "pending_external_vote_shots": (
                sorted(int(key) for key in self._v222_external_votes.keys())
                if not hasattr(self, "_v222_vote_lock")
                else _pending_vote_shots(self)
            ),
            "game_context_shots": sorted(int(key) for key in self._v222_game_context_by_shot.keys()),
            "shot_start_hooks": sorted(str(key) for key in self._v222_shot_start_hooks.keys()),
        }

    def patched_choose_for_emission(
        self,
        default_x: float,
        default_y: float,
        shot_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        _ensure_v222_state(self)
        if not bool(self.settings.get("resolver_v222_enabled", True)):
            return original_choose_for_emission(self, default_x, default_y, shot_id=shot_id)

        mode = str(self.settings.get("mode", "train_only") or "train_only").strip().lower()
        # Preserve the zero-cost historical path during ordinary train_only mode.
        # Use advisory for live/shadow resolver verification without authority.
        if mode in {"off", "train_only"}:
            return original_choose_for_emission(self, default_x, default_y, shot_id=shot_id)

        result: Dict[str, Any] = {
            "apply": False,
            "camera_x": float(default_x),
            "camera_y": float(default_y),
            "confidence": 0.0,
            "confidence_calibrated": False,
            "reason": "resolver_passthrough",
            "shot_id": shot_id,
            "selection_mode": "discrete_candidate",
        }

        try:
            ctx = self._select_context(shot_id)
        except Exception:
            ctx = None
        if ctx is None or (shot_id is not None and int(getattr(ctx, "shot_id", 0) or 0) != int(shot_id)):
            result["reason"] = "resolver_missing_shot_context"
            return result

        sid = int(getattr(ctx, "shot_id", 0) or 0)
        try:
            self._sync_legacy_state(ctx)
        except Exception:
            pass
        try:
            _snapshot_game_context(self, ctx)
        except Exception:
            pass

        shot_candidates = [
            dict(candidate) for candidate in list(getattr(ctx, "candidates", []) or [])
            if int(candidate.get("_ai_shot_id", sid) or sid) == sid
        ]
        enriched = _enrich_candidates(self, shot_candidates)
        try:
            ranked = self.rank_candidates(enriched, limit=max(1, len(enriched))) if enriched else []
        except Exception:
            ranked = []

        external_votes = _external_votes_for_shot(self, sid)
        game_context = self._v222_game_context_by_shot.get(sid)
        try:
            decision_obj = self.shot_resolver_v222.resolve(
                default_xy=(float(default_x), float(default_y)),
                camera_candidates=enriched,
                ranked_candidates=ranked,
                external_votes=external_votes,
                game_context=game_context,
                trust_percent=_safe_float(self.settings.get("trust_percent", 0.0), 0.0),
                mode=mode,
                shot_id=sid,
            )
            decision = decision_obj.as_dict()
        except Exception:
            result["reason"] = "resolver_exception_passthrough"
            return result

        peak_ts = _safe_float(getattr(ctx, "peak_ts", 0.0), 0.0)
        e2e_ms = max(0.0, (time.time() - peak_ts) * 1000.0) if peak_ts > 0.0 else 0.0
        self._v222_resolver_latencies_ms.append(float(decision.get("resolver_ms", 0.0)))
        self._v222_end_to_end_latencies_ms.append(float(e2e_ms))
        decision["end_to_end_ms"] = float(e2e_ms)
        decision["mode"] = mode
        decision["external_sources"] = sorted(external_votes.keys())
        decision["game_context_present"] = bool(game_context)
        self._v222_last_decision = dict(decision)
        try:
            self.session_stats["resolver_v222_decisions"] = int(
                self.session_stats.get("resolver_v222_decisions", 0) or 0
            ) + 1
        except Exception:
            pass

        confidence = _safe_float(decision.get("confidence", 0.0), 0.0)
        min_conf = _safe_float(self.settings.get("min_confidence", 0.58), 0.58)
        override_conf = _safe_float(self.settings.get("override_confidence", 0.92), 0.92)
        trust = max(0.0, min(1.0, _safe_float(self.settings.get("trust_percent", 0.0), 0.0) / 100.0))

        selected_x = _safe_float(decision.get("camera_x", default_x), default_x)
        selected_y = _safe_float(decision.get("camera_y", default_y), default_y)
        agrees_camera = math.hypot(selected_x - default_x, selected_y - default_y) <= 1.0

        apply = False
        reason = "resolver_advisory"
        if mode == "advisory":
            apply = False
            reason = "resolver_advisory"
        elif mode == "blended":
            # V2.22 semantic change: trust_percent is evidence trust, NEVER an XY blend.
            apply = trust > 0.0 and confidence >= min_conf and not agrees_camera
            reason = "resolver_blended_discrete" if apply else "resolver_blended_passthrough"
        elif mode == "ai_priority":
            apply = confidence >= override_conf and not agrees_camera
            reason = "resolver_ai_priority" if apply else "resolver_ai_priority_passthrough"
        elif mode == "ai_only":
            apply = confidence >= min_conf and not agrees_camera
            reason = "resolver_ai_only" if apply else "resolver_ai_only_passthrough"
        else:
            # Unknown modes remain fail-open.
            apply = False
            reason = "resolver_unknown_mode_passthrough"

        if agrees_camera:
            try:
                self.session_stats["resolver_v222_agrees_camera"] = int(
                    self.session_stats.get("resolver_v222_agrees_camera", 0) or 0
                ) + 1
            except Exception:
                pass

        if apply:
            try:
                self.session_stats["resolver_v222_overrides"] = int(
                    self.session_stats.get("resolver_v222_overrides", 0) or 0
                ) + 1
            except Exception:
                pass

        result.update({
            "apply": bool(apply),
            "camera_x": float(selected_x if apply else default_x),
            "camera_y": float(selected_y if apply else default_y),
            "confidence": float(confidence),
            "confidence_calibrated": False,
            "reason": reason,
            "shot_id": sid,
            "selection_mode": "discrete_candidate",
            "resolver_decision": decision,
            # Kept only for backwards-compatible diagnostics. It no longer means
            # coordinate interpolation in V2.22.
            "blend": 1.0 if apply else 0.0,
        })

        if bool(self.settings.get("resolver_v222_log", False)):
            try:
                print(_decision_log_line(decision, apply=apply, mode=mode, e2e_ms=e2e_ms))
            except Exception:
                pass
        return result

    AIRuntime.__init__ = patched_init
    AIRuntime._create_shot_context = patched_create_shot_context
    AIRuntime.mark_shot_finished = patched_mark_shot_finished
    AIRuntime.choose_for_emission = patched_choose_for_emission
    AIRuntime.publish_resolver_votes = publish_resolver_votes
    AIRuntime.clear_resolver_votes = clear_resolver_votes
    AIRuntime.set_game_context_provider = set_game_context_provider
    AIRuntime.set_shot_game_context = set_shot_game_context
    AIRuntime.register_shot_start_hook = register_shot_start_hook
    AIRuntime.unregister_shot_start_hook = unregister_shot_start_hook
    AIRuntime.resolver_status = resolver_status
    AIRuntime._v222_runtime_patch = True

    # If somebody created the singleton before bootstrap, make it usable too.
    existing = getattr(runtime_module, "_RUNTIME", None)
    if existing is not None:
        try:
            _ensure_v222_state(existing)
        except Exception:
            pass

    _INSTALLED = True


__all__ = [
    "SCHEMA_VERSION",
    "V222_DEFAULT_SETTINGS",
    "install_v222_runtime_patch",
]
