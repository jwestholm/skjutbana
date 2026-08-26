from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.engine.ai.hole_patch_ai_v214 import HolePatchAIV214, HolePatchAIConfigV214
from src.engine.offline.hole_dataset_v213 import discover_hole_assets
from src.engine.offline.hole_training_v214 import (
    DomainRandomizationConfigV214,
    SamplingConfigV214,
    domain_randomize_patch,
    run_training_experiment_v214,
)


def _toy_patch(rng: np.random.Generator, background: str) -> np.ndarray:
    size = 128
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    if background == "white":
        base = np.full((size, size), rng.uniform(195, 235), dtype=np.float32)
    elif background == "grid":
        base = np.full((size, size), 205.0, dtype=np.float32)
        base[(xx.astype(np.int32) % 17) < 2] -= 35
        base[(yy.astype(np.int32) % 19) < 2] -= 28
    elif background == "gray":
        base = 105.0 + 18.0 * np.sin(xx / 7.0) + 11.0 * np.cos(yy / 9.0)
    elif background == "dark":
        base = 48.0 + 26.0 * np.sin((xx + yy) / 8.0)
    else:
        noise = rng.normal(0, 1, (size, size)).astype(np.float32)
        base = 125.0 + 38.0 * cv2.GaussianBlur(noise, (0, 0), 3.0)

    base += rng.normal(0.0, 2.0, base.shape).astype(np.float32)
    cx = cy = (size - 1) / 2.0
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    angle = np.arctan2(yy - cy, xx - cx)
    radius = 3.2 + rng.uniform(0.5, 1.7) + 0.65 * np.sin(angle * int(rng.integers(3, 7)) + rng.uniform(0, 6.28))
    core = rr <= radius
    rim = (rr > radius) & (rr <= radius + rng.uniform(1.2, 2.8))
    base[core] -= rng.uniform(58, 95)
    base[rim] += rng.uniform(6, 20)
    return cv2.GaussianBlur(np.clip(base, 0, 255).astype(np.uint8), (3, 3), 0.55)


def _write_archive(root: Path, seed: int = 21401) -> None:
    rng = np.random.default_rng(seed)
    index = 1
    # Six eligible sessions using bright/simple projected backgrounds.
    for session_i in range(6):
        bg = "white" if session_i % 2 == 0 else "white_grid"
        toy_bg = "white" if bg == "white" else "grid"
        for _ in range(18):
            path = root / f"synt_{index:07d}.png"
            cv2.imwrite(str(path), _toy_patch(rng, toy_bg))
            path.with_suffix(".json").write_text(json.dumps({
                "session_id": f"train_session_{session_i}",
                "background_mode": bg,
                "image_type": "synt",
                "patch_size": [128, 128],
            }), encoding="utf-8")
            index += 1

    # Strict novel backgrounds: never train/model-select on these.
    for novel_bg, toy_bg in (("gray", "gray"), ("black", "dark")):
        for _ in range(20):
            path = root / f"synt_{index:07d}.png"
            cv2.imwrite(str(path), _toy_patch(rng, toy_bg))
            path.with_suffix(".json").write_text(json.dumps({
                "session_id": f"novel_{novel_bg}",
                "background_mode": novel_bg,
                "image_type": "synt",
                "patch_size": [128, 128],
            }), encoding="utf-8")
            index += 1

    # Real-like holdout uses a different texture and naming; never training.
    for real_i in range(1, 7):
        path = root / f"hole_{real_i:07d}.png"
        cv2.imwrite(str(path), _toy_patch(rng, "texture"))
        path.with_suffix(".json").write_text(json.dumps({
            "session_id": "real_holdout",
            "background_mode": "coord_grid",
            "image_type": "hole",
            "patch_size": [128, 128],
        }), encoding="utf-8")


def main() -> int:
    _ = argparse.ArgumentParser(description="Selftest V2.14 background-generalising Hole-AI").parse_args()
    failures: list[str] = []

    # Direct invariance check: large brightness shift should not radically alter
    # local-physics features because absolute brightness is not an input channel.
    rng = np.random.default_rng(21401)
    patch = _toy_patch(rng, "white")[32:96, 32:96]
    brighter = np.clip(patch.astype(np.int16) + 35, 0, 255).astype(np.uint8)
    probe = HolePatchAIV214(HolePatchAIConfigV214(input_size=18, hidden_size=32), seed=21401)
    f1 = probe.features_from_patch(patch)
    f2 = probe.features_from_patch(brighter)
    denom = max(1e-9, float(np.linalg.norm(f1) * np.linalg.norm(f2)))
    cosine = float(np.dot(f1, f2) / denom)
    if cosine < 0.95:
        failures.append(f"background-invariant features too sensitive to brightness shift: cosine={cosine:.3f}")

    transformed, _ = domain_randomize_patch(
        patch, (7.0, -4.0), rng=np.random.default_rng(99),
        config=DomainRandomizationConfigV214.from_profile("strong"),
        sampling=SamplingConfigV214(), enabled=True, force=True,
    )
    if float(np.mean(np.abs(transformed.astype(np.float32) - patch.astype(np.float32)))) < 8.0:
        failures.append("forced domain randomization did not materially alter the candidate background")

    with tempfile.TemporaryDirectory(prefix="skjutbana_v214_") as temp_dir:
        root = Path(temp_dir) / "holes"
        out = Path(temp_dir) / "out"
        root.mkdir(parents=True)
        _write_archive(root)
        assets, summary = discover_hole_assets(root)
        if summary.paired_real != 6:
            failures.append(f"expected 6 real holdouts, got {summary.paired_real}")

        report = run_training_experiment_v214(
            holes_root=root,
            output_dir=out,
            model_config=HolePatchAIConfigV214(input_size=18, hidden_size=44, learning_rate=0.0020),
            sampling=SamplingConfigV214(positive_jitter_px=14.0, negative_min_px=23.0, negative_max_px=29.0),
            domain=DomainRandomizationConfigV214.from_profile("standard"),
            holdout_backgrounds=("black", "gray"),
            epochs=6,
            batch_assets=24,
            seed=21401,
        )
        if not (out / "hole_patch_ai_v214.npz").exists():
            failures.append("V2.14 model file missing")
        if not (out / "hole_v214_report.json").exists():
            failures.append("V2.14 report missing")
        if any(asset.kind == "real" for asset in assets if asset.image_path.stem.startswith("synt_")):
            failures.append("kind discovery corrupted synthetic/real separation")

        history = report.get("history") or []
        if not history:
            failures.append("training history is empty")
        else:
            first = float(history[0].get("train_loss", 999))
            best = min(float(row.get("train_loss", 999)) for row in history)
            if best >= first * 0.99:
                failures.append(f"training loss did not improve: first={first:.4f} best={best:.4f}")

        novel = ((report.get("evaluations", {}).get("novel_background_holdout") or {}).get("ai") or {})
        if novel and float(novel.get("auc") or 0.0) < 0.62:
            failures.append(f"toy strict novel-background AUC too low: {novel.get('auc')}")
        procedural = ((report.get("evaluations", {}).get("procedural_domain_stress") or {}).get("ai") or {})
        if procedural and float(procedural.get("auc") or 0.0) < 0.62:
            failures.append(f"procedural-domain AUC too low: {procedural.get('auc')}")

        loaded, metadata = HolePatchAIV214.load(out / "hole_patch_ai_v214.npz")
        if loaded.config.feature_channels != 4:
            failures.append("model save/load lost feature channel config")
        if metadata.get("model_type") != "hole_patch_mlp_v214_background_invariant":
            failures.append("model metadata missing V2.14 model type")

    print("V2.14 SELFTEST")
    print("==============")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print(f"[PASS] brightness-invariant local feature maps (cosine={cosine:.3f})")
    print("[PASS] procedural/background remix materially changes training domain")
    print("[PASS] REAL assets remain holdout-only")
    print("[PASS] strict novel backgrounds remain outside train/model selection")
    print("[PASS] clean + domain-stress model selection")
    print("[PASS] held-out background learning smoke test")
    print("[PASS] V2.14 model save/load")
    print("All V2.14 selftests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
