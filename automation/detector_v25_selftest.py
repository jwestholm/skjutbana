from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_detector_modules():
    # Load the extension files without importing src.engine.camera.__init__, so
    # this self-test never starts camera/game runtime code.
    for name in ("src", "src.engine", "src.engine.camera"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module

    v24_name = "src.engine.camera.detector_v24_extension"
    if v24_name not in sys.modules:
        path = ROOT / "src/engine/camera/detector_v24_extension.py"
        spec = importlib.util.spec_from_file_location(v24_name, path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[v24_name] = module
        spec.loader.exec_module(module)

    v25_name = "src.engine.camera.detector_v25_extension"
    path = ROOT / "src/engine/camera/detector_v25_extension.py"
    spec = importlib.util.spec_from_file_location(v25_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[v25_name] = module
    spec.loader.exec_module(module)
    return module


v25 = _load_detector_modules()


def _gaussian(shape: tuple[int, int], cx: float, cy: float, sigma: float, amplitude: float) -> np.ndarray:
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (amplitude * np.exp(-((xx-cx)**2 + (yy-cy)**2)/(2.0*sigma*sigma))).astype(np.float32)


def test_localisation_probe() -> None:
    shape = (120, 140)
    gt_x, gt_y = 70.0, 60.0
    signal_x, signal_y = 81.0, 54.0
    absdiff = _gaussian(shape, signal_x, signal_y, 2.0, 8.0)
    zscore = _gaussian(shape, signal_x, signal_y, 2.2, 4.0)
    context = {
        "bbox": (0, 0, shape[1], shape[0]),
        "absdiff": absdiff,
        "zscore": zscore,
        "dog": np.zeros(shape, np.float32),
        "valid": np.ones(shape, bool),
    }
    probe = v25._gt_local_probe(context, gt_x, gt_y, dict(v25.DEFAULT_CONFIG))
    assert probe and probe.get("found")
    assert abs(float(probe["dx"]) - 11.0) < 2.5, probe
    assert abs(float(probe["dy"]) + 6.0) < 2.5, probe


def test_weighted_refine() -> None:
    shape = (90, 90)
    true_x, true_y = 44.0, 43.0
    temporal = _gaussian(shape, true_x, true_y, 2.2, 10.0)
    seed_x, seed_y = 48, 43
    rx, ry = v25._weighted_refine(temporal, temporal, temporal/2.0, seed_x, seed_y, 7)
    before = math.hypot(seed_x-true_x, seed_y-true_y)
    after = math.hypot(rx-true_x, ry-true_y)
    assert after < before * 0.5, (before, after, rx, ry)


def test_shadow_accumulator() -> None:
    class Dummy:
        pass

    engine = Dummy()
    cfg = dict(v25.DEFAULT_CONFIG)
    frames = [
        {"camera_x": 98.5, "camera_y": 80.5, "score": 5.0, "v24_patch_prior": 0.5, "v24_tile_probe": 1.0},
        {"camera_x": 100.0, "camera_y": 79.0, "score": 5.5, "v24_patch_prior": 0.6, "v24_tile_probe": 1.0},
        {"camera_x": 101.0, "camera_y": 80.0, "score": 6.0, "v24_patch_prior": 0.7, "v25_refined_tile": 1.0},
    ]
    for index, candidate in enumerate(frames):
        v25._shadow_accumulate(engine, 1, [candidate], 10.0 + 0.03*index, cfg)
    summary = v25._shadow_summary(engine, 1, 100.0, 80.0, cfg)
    block = summary.get("gt_cluster")
    assert isinstance(block, dict), summary
    assert int(block.get("hits", 0)) >= 3, summary
    assert float(block.get("jitter_px", 99.0)) < 3.0, summary


def test_shadow_mode_config() -> None:
    data = json.loads((ROOT / "content/ai/ranker_v4_config.json").read_text(encoding="utf-8"))
    assert data.get("shadow_mode") is True


def test_shadow_ranker_contract() -> None:
    # Functional contract: V4 may disagree strongly with the base ranker, but
    # shadow_mode must return the untouched base order.
    ai_pkg = types.ModuleType("src.engine.ai")
    ai_pkg.__path__ = []
    sys.modules["src.engine.ai"] = ai_pkg

    ranker_module = types.ModuleType("src.engine.ai.ranker_v4")

    class FakeRankerV4:
        def __init__(self, *args, **kwargs):
            self.config = {
                "enabled": True,
                "shadow_mode": True,
                "shadow_training_enabled": False,
                "initial_patch_weight": 0.2,
                "full_patch_weight": 0.2,
                "max_model_weight": 0.8,
            }

        def patch_prior(self, candidate):
            return 1.0 - float(candidate["score"]) / 10.0

        def raw_score(self, candidate):
            return -float(candidate["score"])

        def effective_weight(self):
            return 0.8

        def score(self, candidate):
            return 1.0 / (1.0 + math.exp(float(candidate["score"])))

        def learn_from_ground_truth(self, *args, **kwargs):
            return {"trained": False}

        def reset(self):
            pass

    ranker_module.RankerV4 = FakeRankerV4
    sys.modules["src.engine.ai.ranker_v4"] = ranker_module

    runtime_module = types.ModuleType("src.engine.ai.runtime")

    class Memory:
        def reset(self):
            pass

    class AIRuntime:
        def __init__(self):
            self.settings = {"top_k": 3, "benchmark_mode": True}
            self.memory = Memory()

        def rank_candidates(self, candidates, limit=None):
            rows = [dict(c) for c in sorted(candidates, key=lambda c: c["score"], reverse=True)]
            for index, row in enumerate(rows, start=1):
                row["combined_score"] = float(row["score"])
                row["rank"] = index
            return rows[:limit]

        def rank_with_funnel(self, *args, **kwargs):
            return {}

    runtime_module.AIRuntime = AIRuntime
    sys.modules["src.engine.ai.runtime"] = runtime_module

    path = ROOT / "src/engine/ai/ranker_v4_extension.py"
    name = "v25_selftest_ranker_extension"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    extension.install_ranker_v4_extension()

    runtime = AIRuntime()
    candidates = [
        {"camera_x": 1.0, "camera_y": 1.0, "score": 9.0},
        {"camera_x": 2.0, "camera_y": 2.0, "score": 5.0},
        {"camera_x": 3.0, "camera_y": 3.0, "score": 1.0},
    ]
    actual = runtime.rank_candidates(candidates, limit=3)
    shadow = runtime._v25_shadow_rank_pool
    assert [row["score"] for row in actual] == [9.0, 5.0, 1.0], actual
    assert [row["score"] for row in shadow] == [1.0, 5.0, 9.0], shadow
    assert actual[0].get("v25_v4_shadow_rank") == 3, actual


def main() -> None:
    tests = [
        test_localisation_probe,
        test_weighted_refine,
        test_shadow_accumulator,
        test_shadow_mode_config,
        test_shadow_ranker_contract,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("Detector V2.5 self-test: PASS")


if __name__ == "__main__":
    main()
