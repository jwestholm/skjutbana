"""Selftests for V2.22 fast ShotResolver.

Run from repository root:
    python3 -m automation.offline_v222_selftest

No camera/projector is required.
"""
from __future__ import annotations

import math
import statistics
import time

from src.engine.ai.shot_resolver_v222 import ShotResolverV222


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _candidate(x, y, score, ai, persistence=0.9, existed=0.0, source="camera_detector"):
    return {
        "camera_x": float(x),
        "camera_y": float(y),
        "score": float(score),
        "ai_score": float(ai),
        "persistence": float(persistence),
        "existed_before": float(existed),
        "source": source,
    }


def test_external_expert_can_resolve_camera_top2() -> None:
    resolver = ShotResolverV222()
    candidates = [
        _candidate(100, 100, 12.0, 0.55, persistence=0.85),
        _candidate(132, 101, 10.0, 0.96, persistence=0.98),
        _candidate(210, 160, 8.0, 0.40, persistence=0.65),
    ]
    ranked = [dict(c, combined_score=(0.8 if i == 0 else 0.95 if i == 1 else 0.4), rank=i + 1) for i, c in enumerate(candidates)]
    decision = resolver.resolve(
        default_xy=(100, 100),
        camera_candidates=candidates,
        ranked_candidates=ranked,
        external_votes={"physical_v2215": [{"camera_x": 132, "camera_y": 101, "score": 0.99}]},
        trust_percent=100,
        mode="advisory",
        shot_id=1,
    )
    _assert(math.hypot(decision.camera_x - 132, decision.camera_y - 101) <= 1.5, f"expected top2 cluster, got {decision.as_dict()}")
    _assert("physical_v2215" in decision.source_names, "physical expert source missing")


def test_never_interpolates_between_holes() -> None:
    resolver = ShotResolverV222({"cluster_radius_px": 12.0})
    supplied = {(100.0, 100.0), (140.0, 100.0), (141.0, 101.0)}
    decision = resolver.resolve(
        default_xy=(100, 100),
        camera_candidates=[
            _candidate(100, 100, 10.0, 0.60),
            _candidate(140, 100, 9.0, 0.97),
        ],
        external_votes={"physical": [{"camera_x": 141, "camera_y": 101, "score": 0.98}]},
        trust_percent=100,
        mode="advisory",
        shot_id=2,
    )
    actual = (float(decision.camera_x), float(decision.camera_y))
    _assert(actual in supplied, f"resolver emitted interpolated/non-candidate coordinate: {actual}")


def test_known_hole_penalty_prefers_new_hole() -> None:
    resolver = ShotResolverV222()
    old_hole = _candidate(100, 100, 15.0, 0.94, persistence=1.0, existed=0.99)
    new_hole = _candidate(155, 105, 10.0, 0.88, persistence=0.95, existed=0.0)
    decision = resolver.resolve(
        default_xy=(100, 100),
        camera_candidates=[old_hole, new_hole],
        external_votes={"physical": [{"camera_x": 155, "camera_y": 105, "score": 0.92}]},
        trust_percent=100,
        mode="advisory",
        shot_id=3,
    )
    _assert(math.hypot(decision.camera_x - 155, decision.camera_y - 105) < 2.0, f"old hole won: {decision.as_dict()}")


def test_game_context_is_tiebreaker_not_facit() -> None:
    resolver = ShotResolverV222()
    strong = _candidate(100, 100, 13.0, 0.98, persistence=0.99, existed=0.0)
    weak_target = _candidate(180, 100, 6.0, 0.35, persistence=0.45, existed=0.0)
    decision = resolver.resolve(
        default_xy=(100, 100),
        camera_candidates=[strong, weak_target],
        external_votes={"physical": [{"camera_x": 100, "camera_y": 100, "score": 0.99}]},
        game_context={
            "priors": [{"camera_x": 180, "camera_y": 100, "radius_px": 20, "score": 1.0, "kind": "target"}]
        },
        trust_percent=100,
        mode="advisory",
        shot_id=4,
    )
    _assert(math.hypot(decision.camera_x - 100, decision.camera_y - 100) < 2.0, f"game context overrode dominant physical evidence: {decision.as_dict()}")


def test_confidence_margin_behaves_sensibly() -> None:
    resolver = ShotResolverV222()
    clear = resolver.resolve(
        default_xy=(100, 100),
        camera_candidates=[
            _candidate(100, 100, 12.0, 0.98, persistence=0.98),
            _candidate(220, 100, 4.0, 0.20, persistence=0.30),
        ],
        external_votes={"physical": [{"camera_x": 100, "camera_y": 100, "score": 0.99}]},
        mode="advisory",
        shot_id=5,
    )
    ambiguous = resolver.resolve(
        default_xy=(100, 100),
        camera_candidates=[
            _candidate(100, 100, 9.5, 0.78, persistence=0.80),
            _candidate(140, 100, 9.4, 0.77, persistence=0.80),
        ],
        mode="advisory",
        shot_id=6,
    )
    _assert(clear.confidence > ambiguous.confidence, f"clear confidence {clear.confidence:.3f} <= ambiguous {ambiguous.confidence:.3f}")
    _assert(not clear.confidence_calibrated, "V2.22 score must not claim calibrated probability")


def benchmark_resolver() -> tuple[float, float, float]:
    resolver = ShotResolverV222()
    candidates = []
    for i in range(256):
        x = 100.0 + (i % 32) * 40.0
        y = 100.0 + (i // 32) * 40.0
        candidates.append(_candidate(x, y, 4.0 + (i % 12), 0.2 + (i % 7) / 10.0, persistence=0.5 + (i % 5) / 10.0))
    votes = [
        {"camera_x": c["camera_x"] + 1.0, "camera_y": c["camera_y"] + 1.0, "score": 0.4 + (i % 6) / 10.0}
        for i, c in enumerate(candidates[:96])
    ]
    timings = []
    for _ in range(120):
        t0 = time.perf_counter()
        resolver.resolve(
            default_xy=(100.0, 100.0),
            camera_candidates=candidates,
            ranked_candidates=candidates,
            external_votes={"physical_shadow": votes},
            mode="advisory",
            shot_id=99,
        )
        timings.append((time.perf_counter() - t0) * 1000.0)
    timings.sort()
    p50 = timings[len(timings) // 2]
    p95 = timings[int(0.95 * (len(timings) - 1))]
    worst = timings[-1]
    # Generous hard ceiling for slow CI/VMs. The practical target printed below
    # is much tighter and should be checked on the actual range PC.
    _assert(p95 < 100.0, f"resolver p95 too slow even for safety ceiling: {p95:.2f} ms")
    return p50, p95, worst


def main() -> None:
    print("V2.22 SHOT RESOLVER SELFTEST")
    print("===========================")
    tests = [
        ("camera Top-2 can win when independent physical evidence agrees", test_external_expert_can_resolve_camera_top2),
        ("resolver always emits a real discrete candidate coordinate", test_never_interpolates_between_holes),
        ("known-hole evidence penalises old holes", test_known_hole_penalty_prefers_new_hole),
        ("game context is only a weak prior / tie-breaker", test_game_context_is_tiebreaker_not_facit),
        ("confidence reacts to decision margin and is explicitly uncalibrated", test_confidence_margin_behaves_sensibly),
    ]
    for label, fn in tests:
        fn()
        print(f"[PASS] {label}")

    p50, p95, worst = benchmark_resolver()
    print(f"[PASS] bounded resolver benchmark: p50={p50:.2f} ms p95={p95:.2f} ms worst={worst:.2f} ms")
    print("       target on range PC: resolver p95 < 10 ms; end-to-end shot -> HitEvent p95 < 500 ms")
    print()
    print("All V2.22 ShotResolver selftests passed.")


if __name__ == "__main__":
    main()
