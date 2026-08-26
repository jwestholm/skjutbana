from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np

from src.engine.ai.hole_patch_ai_v213 import HolePatchAI, HolePatchAIConfig
from .hole_dataset_v213 import (
    DatasetSplit,
    HoleAsset,
    build_dataset_split,
    center_contrast_score,
    crop_candidate,
    discover_hole_assets,
    iter_assets_limited,
    read_gray,
    sample_candidate_center,
)




EVALUATION_SEED_OFFSETS = {
    "validation": 5000,
    "synthetic_test": 6000,
    "novel_background_holdout": 7000,
    "real_holdout": 8000,
    "off_center_stress": 9000,
}


def evaluation_seed(base_seed: int, name: str) -> int:
    return int(base_seed) + int(EVALUATION_SEED_OFFSETS.get(name, 12000))


@dataclass(frozen=True)
class SamplingConfig:
    positive_jitter_px: float = 14.0
    negative_min_px: float = 24.0
    negative_max_px: float = 30.0
    positives_per_image: int = 1
    negatives_per_image: int = 2
    horizontal_flip_probability: float = 0.50
    vertical_flip_probability: float = 0.20
    brightness_jitter: float = 0.10
    contrast_jitter: float = 0.12
    noise_sigma_255: float = 1.5


@dataclass
class ExampleBatch:
    patches: list[np.ndarray]
    labels: np.ndarray
    offsets_px: np.ndarray
    backgrounds: list[str]
    asset_stems: list[str]
    candidate_distance_px: np.ndarray
    baseline_scores: np.ndarray


@dataclass
class EvaluationRows:
    probabilities: np.ndarray
    labels: np.ndarray
    predicted_offsets_px: np.ndarray
    target_offsets_px: np.ndarray
    baseline_scores: np.ndarray
    backgrounds: list[str]
    candidate_distance_px: np.ndarray
    asset_stems: list[str]


def _augment_patch(
    patch: np.ndarray,
    offset_px: tuple[float, float],
    *,
    rng: np.random.Generator,
    sampling: SamplingConfig,
    enabled: bool,
) -> tuple[np.ndarray, tuple[float, float]]:
    if not enabled:
        return patch, offset_px
    result = patch.astype(np.float32)
    dx, dy = float(offset_px[0]), float(offset_px[1])

    if rng.random() < float(sampling.horizontal_flip_probability):
        result = np.fliplr(result)
        dx = -dx
    if rng.random() < float(sampling.vertical_flip_probability):
        result = np.flipud(result)
        dy = -dy

    contrast = 1.0 + rng.uniform(-float(sampling.contrast_jitter), float(sampling.contrast_jitter))
    brightness = 255.0 * rng.uniform(-float(sampling.brightness_jitter), float(sampling.brightness_jitter))
    mean = float(np.mean(result))
    result = (result - mean) * contrast + mean + brightness
    sigma = max(0.0, float(sampling.noise_sigma_255))
    if sigma > 0.0:
        result += rng.normal(0.0, sigma, size=result.shape).astype(np.float32)
    return np.ascontiguousarray(np.clip(result, 0.0, 255.0).astype(np.uint8)), (dx, dy)


def _make_examples_for_asset(
    asset: HoleAsset,
    *,
    rng: np.random.Generator,
    model_config: HolePatchAIConfig,
    sampling: SamplingConfig,
    augment: bool,
    positive_jitter_override: float | None = None,
    positive_count_override: int | None = None,
    negative_count_override: int | None = None,
) -> ExampleBatch:
    image = read_gray(asset)
    patches: list[np.ndarray] = []
    labels: list[float] = []
    offsets: list[tuple[float, float]] = []
    backgrounds: list[str] = []
    asset_stems: list[str] = []
    distances: list[float] = []
    baseline_scores: list[float] = []

    positive_count = int(positive_count_override if positive_count_override is not None else sampling.positives_per_image)
    negative_count = int(negative_count_override if negative_count_override is not None else sampling.negatives_per_image)
    positive_jitter = float(positive_jitter_override if positive_jitter_override is not None else sampling.positive_jitter_px)

    for label, count in ((1, positive_count), (0, negative_count)):
        for _ in range(max(0, count)):
            center, hole_minus_candidate = sample_candidate_center(
                image,
                rng=rng,
                label=label,
                crop_size=int(model_config.crop_size),
                positive_jitter_px=positive_jitter,
                negative_min_px=float(sampling.negative_min_px),
                negative_max_px=float(sampling.negative_max_px),
            )
            patch = crop_candidate(image, center, int(model_config.crop_size))
            patch, hole_minus_candidate = _augment_patch(
                patch,
                hole_minus_candidate,
                rng=rng,
                sampling=sampling,
                enabled=augment,
            )
            patches.append(patch)
            labels.append(float(label))
            offsets.append(hole_minus_candidate if label else (0.0, 0.0))
            backgrounds.append(asset.background_mode)
            asset_stems.append(asset.stem)
            distances.append(math.hypot(*hole_minus_candidate) if label else math.hypot(center[0] - (image.shape[1] - 1) / 2.0, center[1] - (image.shape[0] - 1) / 2.0))
            baseline_scores.append(center_contrast_score(patch))

    return ExampleBatch(
        patches=patches,
        labels=np.asarray(labels, dtype=np.float32),
        offsets_px=np.asarray(offsets, dtype=np.float32).reshape(-1, 2),
        backgrounds=backgrounds,
        asset_stems=asset_stems,
        candidate_distance_px=np.asarray(distances, dtype=np.float32),
        baseline_scores=np.asarray(baseline_scores, dtype=np.float32),
    )


def _iter_training_batches(
    assets: Sequence[HoleAsset],
    *,
    model: HolePatchAI,
    sampling: SamplingConfig,
    batch_assets: int,
    seed: int,
    epoch: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(int(seed) + 1009 * int(epoch))
    indices = np.arange(len(assets))
    rng.shuffle(indices)
    batch_assets = max(1, int(batch_assets))

    for start in range(0, len(indices), batch_assets):
        patches: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        offsets: list[np.ndarray] = []
        for index in indices[start : start + batch_assets]:
            examples = _make_examples_for_asset(
                assets[int(index)],
                rng=rng,
                model_config=model.config,
                sampling=sampling,
                augment=True,
            )
            patches.extend(examples.patches)
            labels.append(examples.labels)
            offsets.append(examples.offsets_px)
        if not patches:
            continue
        order = np.arange(len(patches))
        rng.shuffle(order)
        features = model.feature_batch([patches[int(i)] for i in order])
        y = np.concatenate(labels, axis=0)[order]
        off = np.concatenate(offsets, axis=0)[order]
        yield features, y, off


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels).astype(np.int32).reshape(-1)
    scores = np.asarray(scores).astype(np.float64).reshape(-1)
    pos = int(np.sum(labels == 1))
    neg = int(np.sum(labels == 0))
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        average_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = average_rank
        i = j
    sum_pos_ranks = float(np.sum(ranks[labels == 1]))
    return float((sum_pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg))


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    labels = np.asarray(labels).astype(np.int32).reshape(-1)
    scores = np.asarray(scores).astype(np.float64).reshape(-1)
    pred = scores >= float(threshold)
    pos = labels == 1
    neg = ~pos
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & neg))
    tn = int(np.sum((~pred) & neg))
    fn = int(np.sum((~pred) & pos))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    accuracy = _safe_div(tp + tn, len(labels))
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {
        "threshold": float(threshold),
        "count": int(len(labels)),
        "positive_count": int(np.sum(pos)),
        "negative_count": int(np.sum(neg)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "f1": round(f1, 6),
        "auc": None if (auc := roc_auc(labels, scores)) is None else round(float(auc), 6),
    }


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    labels = np.asarray(labels).astype(np.int32).reshape(-1)
    scores = np.asarray(scores).astype(np.float64).reshape(-1)
    if labels.size == 0:
        return 0.5, threshold_metrics(labels, scores, 0.5)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], np.quantile(scores, np.linspace(0.02, 0.98, 97)))))
    best_threshold = 0.5
    best_metrics = threshold_metrics(labels, scores, best_threshold)
    best_key = (float(best_metrics["f1"]), float(best_metrics["recall"]), float(best_metrics["specificity"]))
    for threshold in candidates:
        metrics = threshold_metrics(labels, scores, float(threshold))
        key = (float(metrics["f1"]), float(metrics["recall"]), float(metrics["specificity"]))
        if key > best_key:
            best_threshold, best_metrics, best_key = float(threshold), metrics, key
    return best_threshold, best_metrics


def choose_baseline_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    if len(scores) == 0:
        return 0.0, threshold_metrics(labels, scores, 0.0)
    unique = np.unique(np.quantile(np.asarray(scores, dtype=np.float64), np.linspace(0.02, 0.98, 97)))
    best_threshold = float(np.median(scores))
    best_metrics = threshold_metrics(labels, scores, best_threshold)
    best_key = (float(best_metrics["f1"]), float(best_metrics["recall"]), float(best_metrics["specificity"]))
    for threshold in unique:
        metrics = threshold_metrics(labels, scores, float(threshold))
        key = (float(metrics["f1"]), float(metrics["recall"]), float(metrics["specificity"]))
        if key > best_key:
            best_threshold, best_metrics, best_key = float(threshold), metrics, key
    return best_threshold, best_metrics


def evaluate_assets(
    model: HolePatchAI,
    assets: Sequence[HoleAsset],
    *,
    sampling: SamplingConfig,
    seed: int,
    positives_per_image: int = 1,
    negatives_per_image: int = 2,
    positive_jitter_px: float | None = None,
    augment: bool = False,
) -> EvaluationRows:
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    predicted_offsets: list[np.ndarray] = []
    target_offsets: list[np.ndarray] = []
    baseline_scores: list[np.ndarray] = []
    backgrounds: list[str] = []
    distances: list[np.ndarray] = []
    stems: list[str] = []

    rng = np.random.default_rng(int(seed))
    for asset in assets:
        examples = _make_examples_for_asset(
            asset,
            rng=rng,
            model_config=model.config,
            sampling=sampling,
            augment=augment,
            positive_jitter_override=positive_jitter_px,
            positive_count_override=positives_per_image,
            negative_count_override=negatives_per_image,
        )
        if not examples.patches:
            continue
        p, off = model.predict_patches(examples.patches)
        probabilities.append(p)
        labels.append(examples.labels)
        predicted_offsets.append(off)
        target_offsets.append(examples.offsets_px)
        baseline_scores.append(examples.baseline_scores)
        backgrounds.extend(examples.backgrounds)
        distances.append(examples.candidate_distance_px)
        stems.extend(examples.asset_stems)

    if not probabilities:
        return EvaluationRows(
            probabilities=np.empty((0,), dtype=np.float32),
            labels=np.empty((0,), dtype=np.float32),
            predicted_offsets_px=np.empty((0, 2), dtype=np.float32),
            target_offsets_px=np.empty((0, 2), dtype=np.float32),
            baseline_scores=np.empty((0,), dtype=np.float32),
            backgrounds=[],
            candidate_distance_px=np.empty((0,), dtype=np.float32),
            asset_stems=[],
        )
    return EvaluationRows(
        probabilities=np.concatenate(probabilities),
        labels=np.concatenate(labels),
        predicted_offsets_px=np.concatenate(predicted_offsets),
        target_offsets_px=np.concatenate(target_offsets),
        baseline_scores=np.concatenate(baseline_scores),
        backgrounds=backgrounds,
        candidate_distance_px=np.concatenate(distances),
        asset_stems=stems,
    )


def _offset_metrics(rows: EvaluationRows) -> dict[str, Any]:
    positive = rows.labels >= 0.5
    if not np.any(positive):
        return {"positive_count": 0, "median_error_px": None, "p95_error_px": None, "mean_error_px": None}
    delta = rows.predicted_offsets_px[positive] - rows.target_offsets_px[positive]
    error = np.sqrt(np.sum(delta * delta, axis=1))
    return {
        "positive_count": int(error.size),
        "mean_error_px": round(float(np.mean(error)), 4),
        "median_error_px": round(float(np.median(error)), 4),
        "p95_error_px": round(float(np.percentile(error, 95)), 4),
    }


def summarize_evaluation(
    rows: EvaluationRows,
    *,
    ai_threshold: float,
    baseline_threshold: float,
) -> dict[str, Any]:
    result = {
        "examples": int(len(rows.labels)),
        "ai": threshold_metrics(rows.labels, rows.probabilities, ai_threshold),
        "center_contrast_baseline": threshold_metrics(rows.labels, rows.baseline_scores, baseline_threshold),
        "offset_refinement": _offset_metrics(rows),
        "score_distribution": {},
        "by_background": {},
    }
    for name, mask in (
        ("positive", rows.labels >= 0.5),
        ("negative", rows.labels < 0.5),
    ):
        values = rows.probabilities[mask]
        if values.size:
            result["score_distribution"][name] = {
                "mean": round(float(np.mean(values)), 6),
                "p10": round(float(np.percentile(values, 10)), 6),
                "median": round(float(np.median(values)), 6),
                "p90": round(float(np.percentile(values, 90)), 6),
            }
    backgrounds = np.asarray(rows.backgrounds, dtype=object)
    for background in sorted(set(rows.backgrounds)):
        mask = backgrounds == background
        if np.any(mask):
            result["by_background"][background] = {
                "ai": threshold_metrics(rows.labels[mask], rows.probabilities[mask], ai_threshold),
                "baseline": threshold_metrics(rows.labels[mask], rows.baseline_scores[mask], baseline_threshold),
            }
    return result


def train_model(
    model: HolePatchAI,
    train_assets: Sequence[HoleAsset],
    validation_assets: Sequence[HoleAsset],
    *,
    sampling: SamplingConfig,
    epochs: int,
    batch_assets: int,
    seed: int,
    validation_limit: int | None = None,
) -> tuple[list[dict[str, Any]], float, float]:
    history: list[dict[str, Any]] = []
    val_assets = iter_assets_limited(validation_assets, validation_limit, seed + 77)
    best_state: list[np.ndarray] | None = None
    best_auc = -1.0
    best_epoch = -1
    best_threshold = 0.5
    best_baseline_threshold = 0.0

    for epoch in range(1, int(epochs) + 1):
        started = time.perf_counter()
        batch_losses: list[float] = []
        batch_acc: list[float] = []
        for features, labels, offsets in _iter_training_batches(
            train_assets,
            model=model,
            sampling=sampling,
            batch_assets=batch_assets,
            seed=seed,
            epoch=epoch,
        ):
            metrics = model.train_batch(features, labels, offsets)
            batch_losses.append(metrics.loss)
            batch_acc.append(metrics.accuracy)

        val_rows = evaluate_assets(
            model,
            val_assets,
            sampling=sampling,
            seed=evaluation_seed(seed, "validation"),
            positives_per_image=1,
            negatives_per_image=2,
            positive_jitter_px=sampling.positive_jitter_px,
            augment=False,
        )
        threshold, val_metrics = choose_threshold(val_rows.labels, val_rows.probabilities)
        baseline_threshold, baseline_metrics = choose_baseline_threshold(val_rows.labels, val_rows.baseline_scores)
        val_auc = val_metrics.get("auc")
        auc_value = float(val_auc) if val_auc is not None else float(val_metrics.get("f1", 0.0))
        if auc_value > best_auc:
            best_auc = auc_value
            best_epoch = epoch
            best_threshold = threshold
            best_baseline_threshold = baseline_threshold
            best_state = [value.copy() for value in model.parameters()]

        row = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(batch_losses)) if batch_losses else 0.0, 6),
            "train_accuracy": round(float(np.mean(batch_acc)) if batch_acc else 0.0, 6),
            "validation": val_metrics,
            "baseline_validation": baseline_metrics,
            "ai_threshold": round(float(threshold), 6),
            "baseline_threshold": round(float(baseline_threshold), 6),
            "seconds": round(time.perf_counter() - started, 3),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs}: loss={row['train_loss']:.4f} "
            f"val_auc={val_metrics.get('auc')} val_f1={val_metrics.get('f1')} "
            f"baseline_f1={baseline_metrics.get('f1')} time={row['seconds']:.1f}s"
        )

    if best_state is not None:
        for target, source in zip(model.parameters(), best_state):
            target[...] = source
    for row in history:
        row["selected_epoch"] = row["epoch"] == best_epoch
    return history, float(best_threshold), float(best_baseline_threshold)


def run_training_experiment(
    *,
    holes_root: Path,
    output_dir: Path,
    model_config: HolePatchAIConfig,
    sampling: SamplingConfig,
    holdout_backgrounds: Iterable[str],
    epochs: int,
    batch_assets: int,
    seed: int,
    max_train_assets: int | None = None,
    max_eval_assets: int | None = None,
) -> dict[str, Any]:
    holes_root = Path(holes_root).expanduser().resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assets, archive_summary = discover_hole_assets(holes_root, inspect_images=True)
    split = build_dataset_split(
        assets,
        holdout_backgrounds=holdout_backgrounds,
        seed=seed,
    )
    train_assets = iter_assets_limited(split.train, max_train_assets, seed + 1)
    validation_assets = iter_assets_limited(split.validation, max_eval_assets, seed + 2)
    test_assets = iter_assets_limited(split.test, max_eval_assets, seed + 3)
    background_assets = iter_assets_limited(split.background_holdout, max_eval_assets, seed + 4)
    real_assets = tuple(split.real_holdout)  # never truncate the tiny real holdout by default

    if len(train_assets) < 10:
        raise RuntimeError(f"Need at least 10 synthetic training assets, found {len(train_assets)}")
    if len(validation_assets) < 2:
        raise RuntimeError(f"Need at least 2 validation assets from held-out sessions, found {len(validation_assets)}")

    model = HolePatchAI(model_config, seed=seed)
    print("V2.13 HOLE-AI TRAINING")
    print("======================")
    print(f"Hole archive          : {holes_root}")
    print(f"Synthetic train       : {len(train_assets)}")
    print(f"Synthetic validation  : {len(validation_assets)}")
    print(f"Synthetic test        : {len(test_assets)}")
    print(f"Novel-background test : {len(background_assets)} {list(split.holdout_backgrounds)}")
    print(f"REAL holdout          : {len(real_assets)} (never used for training)")
    print(f"Positive jitter       : <= {sampling.positive_jitter_px:.1f}px")
    print(f"Negative candidate    : {sampling.negative_min_px:.1f}..{sampling.negative_max_px:.1f}px from hole")
    print("Model input           : candidate-centred crop; source 128x128 centre is NEVER fed directly")
    print("Auxiliary task        : predict hole offset from candidate centre")
    print()

    history, ai_threshold, baseline_threshold = train_model(
        model,
        train_assets,
        validation_assets,
        sampling=sampling,
        epochs=epochs,
        batch_assets=batch_assets,
        seed=seed,
        validation_limit=max_eval_assets,
    )

    eval_groups: dict[str, tuple[HoleAsset, ...]] = {
        "validation": tuple(validation_assets),
        "synthetic_test": tuple(test_assets),
        "novel_background_holdout": tuple(background_assets),
        "real_holdout": tuple(real_assets),
    }
    evaluations: dict[str, Any] = {}
    for index, (name, group) in enumerate(eval_groups.items()):
        if not group:
            evaluations[name] = {"examples": 0, "note": "no assets in this split"}
            continue
        rows = evaluate_assets(
            model,
            group,
            sampling=sampling,
            seed=evaluation_seed(seed, name),
            positives_per_image=2 if name == "real_holdout" else 1,
            negatives_per_image=4 if name == "real_holdout" else 2,
            positive_jitter_px=sampling.positive_jitter_px,
            augment=False,
        )
        evaluations[name] = summarize_evaluation(
            rows,
            ai_threshold=ai_threshold,
            baseline_threshold=baseline_threshold,
        )

    # Explicit anti-centre-cheat stress: positives are farther from candidate
    # centre than normal training samples, while negatives may still contain the
    # hole near the edge of the candidate crop.
    stress_assets = tuple(test_assets[: max(1, min(len(test_assets), 1000))]) or tuple(validation_assets[:1000])
    if stress_assets:
        stress_rows = evaluate_assets(
            model,
            stress_assets,
            sampling=sampling,
            seed=evaluation_seed(seed, "off_center_stress"),
            positives_per_image=2,
            negatives_per_image=2,
            positive_jitter_px=min(sampling.negative_min_px - 2.0, sampling.positive_jitter_px + 4.0),
            augment=False,
        )
        evaluations["off_center_stress"] = summarize_evaluation(
            stress_rows,
            ai_threshold=ai_threshold,
            baseline_threshold=baseline_threshold,
        )

    model_path = output_dir / "hole_patch_ai_v213.npz"
    report_path = output_dir / "hole_v213_report.json"
    model.save(
        model_path,
        metadata={
            "ai_threshold": ai_threshold,
            "baseline_threshold": baseline_threshold,
            "sampling": asdict(sampling),
            "holdout_backgrounds": list(split.holdout_backgrounds),
            "session_assignment": split.session_assignment,
            "seed": int(seed),
        },
    )

    report = {
        "schema_version": "2.13",
        "purpose": "first_pixel_hole_ai_learning_proof_candidate_centered_with_offset_refinement",
        "archive": archive_summary.to_dict(),
        "split": split.to_dict(),
        "effective_counts": {
            "train": len(train_assets),
            "validation": len(validation_assets),
            "test": len(test_assets),
            "background_holdout": len(background_assets),
            "real_holdout": len(real_assets),
        },
        "model_config": asdict(model_config),
        "sampling_config": asdict(sampling),
        "selected_thresholds": {
            "hole_ai": ai_threshold,
            "center_contrast_baseline": baseline_threshold,
        },
        "history": history,
        "evaluations": evaluations,
        "model_path": str(model_path),
        "important_interpretation": [
            "real_holdout is never used for training or threshold selection",
            "source hole images are centred by archive format, but model examples are candidate-centred crops with random candidate offsets",
            "offset regression forces the model to localise the hole relative to the candidate instead of only classifying a centred 128x128 source image",
            "local negatives are a first learning proof; full-frame detector hard negatives remain a later V2.14 input",
        ],
    }
    temp = report_path.with_suffix(report_path.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(report_path)
    return report
