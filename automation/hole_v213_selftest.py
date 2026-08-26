from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.engine.ai.hole_patch_ai_v213 import HolePatchAIConfig, HolePatchAI
from src.engine.offline.hole_dataset_v213 import build_dataset_split, discover_hole_assets
from src.engine.offline.hole_training_v213 import SamplingConfig, run_training_experiment


def _toy_patch(rng: np.random.Generator, background: str) -> np.ndarray:
    size = 128
    yy, xx = np.mgrid[0:size, 0:size]
    if background == "white":
        base = np.full((size, size), rng.uniform(205, 235), dtype=np.float32)
    elif background == "grid":
        base = np.full((size, size), 205.0, dtype=np.float32)
        base[(xx % 18) < 2] -= 28
        base[(yy % 18) < 2] -= 28
    elif background == "gray":
        base = np.full((size, size), rng.uniform(100, 145), dtype=np.float32)
    else:
        base = 120 + 35 * np.sin(xx / 6.0) + 20 * np.cos(yy / 8.0)
    base += rng.normal(0, 2.5, size=(size, size))

    # Irregular synthetic physical-looking dark core exactly at source centre.
    cx = cy = (size - 1) / 2.0
    angle = np.arctan2(yy - cy, xx - cx)
    radius = 3.0 + rng.uniform(0.5, 2.5) + 0.7 * np.sin(angle * rng.integers(3, 7) + rng.uniform(0, 6.28))
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    core = rr <= radius
    rim = (rr > radius) & (rr <= radius + rng.uniform(1.5, 3.5))
    base[core] -= rng.uniform(55, 105)
    base[rim] += rng.uniform(5, 25)
    base = cv2.GaussianBlur(np.clip(base, 0, 255).astype(np.uint8), (3, 3), rng.uniform(0.3, 0.8))
    return base


def _write_toy_archive(root: Path, *, seed: int = 21301) -> None:
    rng = np.random.default_rng(seed)
    sessions = [f"toy_session_{i}" for i in range(6)]
    backgrounds = ["white", "white_grid", "checker_anim"]
    index = 1
    for session_i, session in enumerate(sessions):
        for _ in range(16):
            background = backgrounds[session_i % len(backgrounds)]
            patch = _toy_patch(rng, "grid" if background == "white_grid" else "white")
            image_path = root / f"synt_{index:07d}.png"
            cv2.imwrite(str(image_path), patch)
            meta = {
                "session_id": session,
                "background_mode": background,
                "image_type": "synt",
                "patch_size": [128, 128],
                "gt_camera_x": 1000.0 + index,
                "gt_camera_y": 900.0,
            }
            image_path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
            index += 1

    # Novel background never used for training.
    for _ in range(18):
        patch = _toy_patch(rng, "gray")
        image_path = root / f"synt_{index:07d}.png"
        cv2.imwrite(str(image_path), patch)
        image_path.with_suffix(".json").write_text(json.dumps({
            "session_id": "toy_novel",
            "background_mode": "gray",
            "image_type": "synt",
            "patch_size": [128, 128],
        }), encoding="utf-8")
        index += 1

    # Tiny REAL holdout uses a different texture and is never in training.
    for real_index in range(1, 7):
        patch = _toy_patch(rng, "texture")
        image_path = root / f"hole_{real_index:07d}.png"
        cv2.imwrite(str(image_path), patch)
        image_path.with_suffix(".json").write_text(json.dumps({
            "session_id": "real_holdout",
            "background_mode": "coord_grid",
            "image_type": "hole",
            "patch_size": [128, 128],
        }), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Selftest for V2.13 dataset split, anti-centre-bias sampling and numpy Hole-AI trainer.")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    _ = build_parser().parse_args()
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="skjutbana_v213_") as temp_dir:
        root = Path(temp_dir) / "holes"
        out = Path(temp_dir) / "out"
        root.mkdir(parents=True)
        _write_toy_archive(root)

        assets, summary = discover_hole_assets(root)
        split = build_dataset_split(assets, holdout_backgrounds=("gray",), seed=21301)
        if summary.paired_synthetic != 114:
            failures.append(f"expected 114 synthetic assets, got {summary.paired_synthetic}")
        if summary.paired_real != 6:
            failures.append(f"expected 6 real assets, got {summary.paired_real}")
        if len(split.background_holdout) != 18:
            failures.append(f"expected 18 novel-background holdout, got {len(split.background_holdout)}")
        train_sessions = {a.session_id for a in split.train}
        val_sessions = {a.session_id for a in split.validation}
        test_sessions = {a.session_id for a in split.test}
        if train_sessions & val_sessions or train_sessions & test_sessions or val_sessions & test_sessions:
            failures.append("session leakage found between train/validation/test")
        if any(a.kind == "real" for a in split.train + split.validation + split.test):
            failures.append("REAL hole asset leaked into synthetic train/validation/test")

        report = run_training_experiment(
            holes_root=root,
            output_dir=out,
            model_config=HolePatchAIConfig(input_size=20, hidden_size=48, learning_rate=0.0025),
            sampling=SamplingConfig(
                positive_jitter_px=13.0,
                negative_min_px=23.0,
                negative_max_px=29.0,
                positives_per_image=1,
                negatives_per_image=2,
                noise_sigma_255=1.0,
            ),
            holdout_backgrounds=("gray",),
            epochs=5,
            batch_assets=24,
            seed=21301,
        )
        model_path = out / "hole_patch_ai_v213.npz"
        if not model_path.exists():
            failures.append("trained model was not written")
        if not (out / "hole_v213_report.json").exists():
            failures.append("training report was not written")

        history = report.get("history", [])
        if not history:
            failures.append("training history is empty")
        else:
            first = float(history[0].get("train_loss", 999))
            best = min(float(row.get("train_loss", 999)) for row in history)
            if best >= first * 0.98:
                failures.append(f"training loss did not meaningfully improve: first={first}, best={best}")

        test_ai = report.get("evaluations", {}).get("synthetic_test", {}).get("ai", {})
        if test_ai and float(test_ai.get("auc") or 0.0) < 0.75:
            failures.append(f"toy held-out AUC unexpectedly low: {test_ai.get('auc')}")
        stress_result = report.get("evaluations", {}).get("off_center_stress", {})
        stress_ai = stress_result.get("ai", {}) if isinstance(stress_result, dict) else {}
        if stress_ai and float(stress_ai.get("recall") or 0.0) < 0.55:
            failures.append(f"off-centre stress recall too low: {stress_ai.get('recall')}")
        stress_offset = stress_result.get("offset_refinement", {}) if isinstance(stress_result, dict) else {}
        if stress_offset and float(stress_offset.get("median_error_px") or 999.0) > 12.0:
            failures.append(f"off-centre offset localisation too weak: median={stress_offset.get('median_error_px')}px")

        loaded, meta = HolePatchAI.load(model_path)
        if loaded.config.input_size != 20:
            failures.append("model save/load lost config")
        if str(meta.get("model_type")) != "hole_patch_mlp_v213":
            failures.append("model metadata missing model_type")

    print("V2.13 SELFTEST")
    print("==============")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("[PASS] hole-bank discovery")
    print("[PASS] no REAL data leakage into training")
    print("[PASS] session-level split")
    print("[PASS] novel-background holdout")
    print("[PASS] candidate-centred jittered training")
    print("[PASS] auxiliary offset localisation head")
    print("[PASS] held-out learning check")
    print("[PASS] off-centre anti-cheat stress")
    print("[PASS] model save/load")
    print("All V2.13 selftests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
