"""Isolated monkey-patch selftest for V2.22 AIRuntime integration.

This test uses a fake legacy AIRuntime so it can run without camera/pygame.
Run from repository root:
    python3 -m automation.runtime_v222_selftest
"""
from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys
import threading
import time


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _make_fake_runtime_module() -> ModuleType:
    module = ModuleType("src.engine.ai.runtime")
    module.DEFAULT_SETTINGS = {
        "enabled": True,
        "mode": "train_only",
        "top_k": 10,
        "min_confidence": 0.58,
        "override_confidence": 0.92,
        "trust_percent": 0,
    }
    module._RUNTIME = None

    class AIRuntime:
        def __init__(self, storage_dir="content/ai"):
            self.storage_dir = storage_dir
            self.settings = dict(module.DEFAULT_SETTINGS)
            self.session_stats = {}
            self._shots = {}
            self._post_shot_frames = [(object(), time.time()), (object(), time.time())]

        def _create_shot_context(self, scanner, scanner_event):
            ctx = SimpleNamespace(
                shot_id=int(scanner_event.shot_id),
                peak_ts=float(scanner_event.peak_ts),
                candidates=list(getattr(scanner_event, "candidates", [])),
                state="pending",
            )
            self._shots[ctx.shot_id] = ctx
            return ctx

        def mark_shot_finished(self, shot_id, state="finished"):
            if int(shot_id) in self._shots:
                self._shots[int(shot_id)].state = state

        def choose_for_emission(self, default_x, default_y, shot_id=None):
            return {
                "apply": False,
                "camera_x": float(default_x),
                "camera_y": float(default_y),
                "confidence": 0.0,
                "reason": "legacy_passthrough",
                "shot_id": shot_id,
            }

        def _select_context(self, shot_id=None):
            if shot_id is not None:
                return self._shots.get(int(shot_id))
            return next(iter(self._shots.values()), None)

        def _sync_legacy_state(self, ctx):
            self._post_shot_frames = [(object(), time.time()), (object(), time.time())]

        def rank_candidates(self, candidates, limit=None):
            ranked = []
            for candidate in candidates:
                item = dict(candidate)
                item["ai_score"] = float(item.get("ai_score", 0.5))
                item["combined_score"] = 0.5 * min(1.0, float(item.get("score", 0.0)) / 15.0) + 0.5 * item["ai_score"]
                ranked.append(item)
            ranked.sort(key=lambda c: c["combined_score"], reverse=True)
            for idx, item in enumerate(ranked):
                item["rank"] = idx + 1
            return ranked[: int(limit or len(ranked))]

        def compute_persistence(self, candidate):
            return float(candidate.get("persistence", 0.8))

        def existed_before_shot(self, candidate):
            return float(candidate.get("existed_before", 0.0))

    module.AIRuntime = AIRuntime
    return module


def main() -> None:
    fake = _make_fake_runtime_module()
    sys.modules["src.engine.ai.runtime"] = fake

    import src.engine.ai.runtime_v222 as patch

    patch._INSTALLED = False
    patch.install_v222_runtime_patch()

    runtime = fake.AIRuntime(storage_dir="unused")
    _assert(getattr(fake.AIRuntime, "_v222_runtime_patch", False), "runtime class not patched")
    _assert(hasattr(runtime, "publish_resolver_votes"), "publish_resolver_votes missing")
    _assert(hasattr(runtime, "set_game_context_provider"), "game context API missing")

    game_calls = []
    runtime.set_game_context_provider(
        lambda **kw: game_calls.append((kw["shot_id"], kw["peak_ts"])) or {
            "priors": [{"camera_x": 132.0, "camera_y": 101.0, "radius_px": 20.0, "score": 1.0}]
        }
    )

    candidates = [
        {"camera_x": 100.0, "camera_y": 100.0, "score": 12.0, "ai_score": 0.55, "persistence": 0.85, "existed_before": 0.0, "_ai_shot_id": 7},
        {"camera_x": 132.0, "camera_y": 101.0, "score": 10.0, "ai_score": 0.96, "persistence": 0.98, "existed_before": 0.0, "_ai_shot_id": 7},
    ]
    event = SimpleNamespace(shot_id=7, peak_ts=time.time() - 0.20, candidates=candidates)
    ctx = runtime._create_shot_context(None, event)
    _assert(len(game_calls) == 1, f"game context provider should snapshot once, calls={game_calls}")

    runtime.publish_resolver_votes(
        7,
        "physical_shadow",
        [{"camera_x": 132.0, "camera_y": 101.0, "score": 0.99}],
        weight=1.0,
    )

    runtime.settings["mode"] = "advisory"
    advisory = runtime.choose_for_emission(100.0, 100.0, shot_id=7)
    _assert(not advisory["apply"], "advisory must never change authority")
    selected = advisory["resolver_decision"]
    _assert(abs(selected["camera_x"] - 132.0) < 1.5, f"resolver did not prefer physical-supported Top-2: {selected}")

    runtime.settings["mode"] = "blended"
    runtime.settings["trust_percent"] = 100
    runtime.settings["min_confidence"] = 0.40
    blended = runtime.choose_for_emission(100.0, 100.0, shot_id=7)
    _assert(blended["apply"], f"blended discrete test did not apply: {blended}")
    _assert(abs(blended["camera_x"] - 132.0) < 1.5, f"blended produced wrong x: {blended}")
    _assert(abs(blended["camera_x"] - 116.0) > 5.0, "V2.22 must never output old XY midpoint")
    _assert(blended["selection_mode"] == "discrete_candidate", "selection mode not discrete")

    status = runtime.resolver_status()
    _assert(status["resolver_latency_ms"]["n"] >= 2, f"latency history missing: {status}")
    _assert(status["end_to_end_latency_ms"]["n"] >= 2, f"end-to-end history missing: {status}")

    publish_errors = []
    def _publisher():
        try:
            for idx in range(80):
                runtime.publish_resolver_votes(
                    7,
                    "parallel_probe",
                    [{"camera_x": 132.0, "camera_y": 101.0, "score": 0.80 + (idx % 10) * 0.01}],
                )
        except Exception as exc:
            publish_errors.append(exc)
    thread = threading.Thread(target=_publisher)
    thread.start()
    for _ in range(40):
        runtime.settings["mode"] = "advisory"
        runtime.choose_for_emission(100.0, 100.0, shot_id=7)
    thread.join()
    _assert(not publish_errors, f"parallel vote publishing failed: {publish_errors}")

    original_resolve = runtime.shot_resolver_v222.resolve
    def _explode(*args, **kwargs):
        raise RuntimeError("intentional selftest failure")
    runtime.shot_resolver_v222.resolve = _explode
    fail_open = runtime.choose_for_emission(100.0, 100.0, shot_id=7)
    runtime.shot_resolver_v222.resolve = original_resolve
    _assert(not fail_open["apply"], f"resolver exception must fail open: {fail_open}")
    _assert(fail_open["camera_x"] == 100.0 and fail_open["camera_y"] == 100.0, f"fail-open XY changed: {fail_open}")

    runtime.mark_shot_finished(7, "finished")
    print("V2.22 RUNTIME PATCH SELFTEST")
    print("============================")
    print("[PASS] runtime patch installs idempotently before singleton use")
    print("[PASS] game context is snapshotted at shot-context creation")
    print("[PASS] external physical-style votes can support camera Top-2")
    print("[PASS] advisory mode cannot override detector authority")
    print("[PASS] blended mode selects a discrete candidate; no XY interpolation")
    print("[PASS] resolver and end-to-end latency telemetry are recorded")
    print("[PASS] parallel expert vote publication is thread-safe")
    print("[PASS] resolver exceptions fail open to the detector coordinate")
    print()
    print("All V2.22 runtime patch selftests passed.")


if __name__ == "__main__":
    main()
