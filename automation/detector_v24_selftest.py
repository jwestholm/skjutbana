from __future__ import annotations

import importlib.util
import math
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_descriptor() -> None:
    mod = _load("detector_v24_extension_selftest", "src/engine/camera/detector_v24_extension.py")
    size = 33
    center = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)

    # Compact dark core + bright ring signal.
    hole_abs = np.zeros((size, size), dtype=np.float32)
    hole_dark = np.zeros_like(hole_abs)
    hole_z = np.zeros_like(hole_abs)
    hole_abs[dist <= 2.2] = 5.0
    hole_dark[dist <= 2.2] = 4.5
    hole_z[dist <= 2.2] = 2.8
    ring = (dist >= 3.0) & (dist <= 5.0)
    hole_abs[ring] = 3.8
    hole_z[ring] = 2.0

    hole = mod._patch_descriptor(
        px=center,
        py=center,
        absdiff=hole_abs,
        darkening=hole_dark,
        zscore=hole_z,
        cfg={"patch_radius_px": 8},
    )
    hole["v24_patch_prior"] = mod._patch_prior(hole)

    # Long edge residual through the same point.
    edge_abs = np.zeros((size, size), dtype=np.float32)
    edge_dark = np.zeros_like(edge_abs)
    edge_z = np.zeros_like(edge_abs)
    edge_abs[center - 1:center + 2, 4:-4] = 5.0
    edge_dark[center - 1:center + 2, 4:-4] = 2.5
    edge_z[center - 1:center + 2, 4:-4] = 2.8

    edge = mod._patch_descriptor(
        px=center,
        py=center,
        absdiff=edge_abs,
        darkening=edge_dark,
        zscore=edge_z,
        cfg={"patch_radius_px": 8},
    )
    edge["v24_patch_prior"] = mod._patch_prior(edge)

    assert hole["v24_patch_isotropy"] > edge["v24_patch_isotropy"], (hole, edge)
    assert hole["v24_patch_prior"] > edge["v24_patch_prior"], (hole, edge)
    print(
        "patch descriptor: OK  "
        f"hole_prior={hole['v24_patch_prior']:.3f} "
        f"edge_prior={edge['v24_patch_prior']:.3f}"
    )


def test_shot_accumulator() -> None:
    mod = _load("detector_v24_accumulator_selftest", "src/engine/camera/detector_v24_extension.py")

    class Engine:
        pass

    engine = Engine()
    cfg = dict(mod.DEFAULT_CONFIG)
    first = {
        "camera_x": 100.0,
        "camera_y": 80.0,
        "score": 8.0,
        "detector_v1": 1.0,
        "detector_v2": 0.0,
        "v24_patch_prior": 0.70,
        "v24_patch_local_snr": 0.75,
    }
    out1 = mod._update_shot_accumulator(
        engine,
        shot_id=1,
        candidates=[first],
        frame_ts=10.0,
        cfg=cfg,
    )
    assert out1

    out2 = mod._update_shot_accumulator(
        engine,
        shot_id=1,
        candidates=[],
        frame_ts=10.08,
        cfg=cfg,
    )
    carried = [c for c in out2 if float(c.get("shot_accumulator_carried", 0.0)) > 0.5]
    assert carried, out2
    assert math.hypot(carried[0]["camera_x"] - 100.0, carried[0]["camera_y"] - 80.0) < 1.0
    print("shot accumulator: OK  one-frame strong V1 candidate survives disappearance")


def test_tile_probe() -> None:
    mod = _load("detector_v24_tile_selftest", "src/engine/camera/detector_v24_extension.py")

    class FakeEngine:
        @staticmethod
        def _candidate_features(*, px, py, saliency, absdiff, darkening, dog, zscore):
            return {
                "area": 8.0,
                "radius": 2.0,
                "circularity": 0.8,
                "score": 7.0,
                "center_change": float(absdiff[py, px]),
                "local_contrast": float(absdiff[py, px]),
                "dog_value": float(dog[py, px]),
            }

        @staticmethod
        def _apply_known_hole_penalty(scanner, candidate):
            return None

    shape = (96, 128)
    absdiff = np.zeros(shape, dtype=np.float32)
    darkening = np.zeros(shape, dtype=np.float32)
    zscore = np.zeros(shape, dtype=np.float32)
    dog = np.zeros(shape, dtype=np.float32)
    saliency = np.zeros(shape, dtype=np.float32)
    valid = np.ones(shape, dtype=bool)

    gx, gy = 71, 42
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    dist = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2)
    blob = dist <= 2.3
    absdiff[blob] = 4.2
    darkening[blob] = 3.7
    zscore[blob] = 2.1
    dog[blob] = 1.2
    # Deliberately keep saliency below a hypothetical global V2 threshold.
    saliency[blob] = 4.0

    cfg = dict(mod.DEFAULT_CONFIG)
    probes = mod._tile_probe_candidates(
        FakeEngine(),
        scanner=object(),
        saliency=saliency,
        absdiff=absdiff,
        darkening=darkening,
        dog=dog,
        zscore=zscore,
        valid=valid,
        bbox=(0, 0, shape[1], shape[0]),
        frame_ts=1.0,
        cfg=cfg,
    )
    nearest = min(
        math.hypot(float(c["camera_x"]) - gx, float(c["camera_y"]) - gy)
        for c in probes
    )
    assert nearest <= 3.0, (nearest, probes[:5])
    print(f"tile probe: OK  weak local hole recovered at {nearest:.2f}px")


def test_ranker_learning() -> None:
    mod = _load("ranker_v4_selftest", "src/engine/ai/ranker_v4.py")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        config = dict(mod.DEFAULT_CONFIG)
        config.update(
            {
                "min_positive_shots": 1,
                "full_weight_shots": 10,
                "save_every_shots": 99999,
                "training_log_enabled": False,
            }
        )
        config_path = tmp_path / "config.json"
        config_path.write_text(__import__("json").dumps(config), encoding="utf-8")
        ranker = mod.RankerV4(
            model_path=tmp_path / "model.json",
            config_path=config_path,
            log_path=tmp_path / "pairs.jsonl",
        )

        positive = {
            "camera_x": 50.0,
            "camera_y": 50.0,
            "score": 5.0,
            "v24_patch_core_to_outer": 0.85,
            "v24_patch_compactness": 0.82,
            "v24_patch_centeredness": 0.92,
            "v24_patch_isotropy": 0.88,
            "v24_patch_bipolar": 0.65,
            "v24_patch_local_snr": 0.80,
            "detector_v1": 1.0,
        }
        negative = {
            "camera_x": 150.0,
            "camera_y": 120.0,
            "score": 18.0,
            "v24_patch_core_to_outer": 0.25,
            "v24_patch_compactness": 0.28,
            "v24_patch_centeredness": 0.35,
            "v24_patch_isotropy": 0.10,
            "v24_patch_bipolar": 0.05,
            "v24_patch_local_snr": 0.55,
            "detector_v1": 1.0,
        }

        before = ranker.raw_score(positive) - ranker.raw_score(negative)
        for _ in range(25):
            ranker.learn_from_ground_truth((50.0, 50.0), [positive, negative])
        after = ranker.raw_score(positive) - ranker.raw_score(negative)
        assert after > before + 0.25, (before, after, ranker.summary())

        # Exact-GT patch supervision must also work when NO detector candidate
        # lies near GT. This is the key V2.4 label-noise fix.
        distant_negative = dict(negative)
        distant_negative["camera_x"] = 180.0
        distant_negative["camera_y"] = 140.0
        override = dict(positive)
        override["camera_x"] = 50.0
        override["camera_y"] = 50.0
        result = ranker.learn_from_ground_truth(
            (50.0, 50.0),
            [distant_negative],
            positive_override=override,
        )
        assert result.get("trained") is True, result
        assert result.get("positive_source") == "ground_truth_patch", result
        print(
            f"ranker learning: OK  margin {before:.3f} -> {after:.3f} "
            "+ exact-GT patch supervision"
        )


def main() -> None:
    print("Detector V2.4 self-test")
    print("=" * 60)
    test_patch_descriptor()
    test_shot_accumulator()
    test_tile_probe()
    test_ranker_learning()
    print("=" * 60)
    print("ALL V2.4 SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
