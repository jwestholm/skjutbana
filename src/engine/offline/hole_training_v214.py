from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np

from src.engine.ai.hole_patch_ai_v214 import HolePatchAIV214, HolePatchAIConfigV214
from src.engine.offline.hole_dataset_v213 import (
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
from src.engine.offline.hole_training_v213 import (
    EvaluationRows,
    choose_baseline_threshold,
    choose_threshold,
    evaluation_seed,
    summarize_evaluation,
)


@dataclass(frozen=True)
class SamplingConfigV214:
    positive_jitter_px: float = 16.0
    negative_min_px: float = 24.0
    negative_max_px: float = 30.0
    positives_per_image: int = 1
    negatives_per_image: int = 2
    horizontal_flip_probability: float = 0.50
    vertical_flip_probability: float = 0.25


@dataclass(frozen=True)
class DomainRandomizationConfigV214:
    """Training-only domain randomization.

    The strict novel-background and real holdouts are never transformed for
    their official metrics.  This augmentation is deliberately aimed at
    removing low-frequency/projector-background shortcuts while keeping the
    source hole's local physical residual.
    """

    profile: str = "standard"
    apply_probability: float = 0.90
    background_remix_probability: float = 0.70
    background_mix_min: float = 0.35
    background_mix_max: float = 0.82
    residual_gain_min: float = 0.85
    residual_gain_max: float = 1.20
    brightness_jitter: float = 0.18
    contrast_jitter: float = 0.30
    gamma_min: float = 0.68
    gamma_max: float = 1.48
    noise_sigma_max_255: float = 4.5
    blur_probability: float = 0.22
    blur_sigma_max: float = 1.0
    shadow_probability: float = 0.45
    shadow_strength_max: float = 0.28
    projected_edge_probability: float = 0.30
    projected_edge_count_max: int = 3

    @classmethod
    def from_profile(cls, profile: str) -> "DomainRandomizationConfigV214":
        name = str(profile or "standard").strip().lower()
        if name == "mild":
            return cls(
                profile="mild",
                apply_probability=0.75,
                background_remix_probability=0.45,
                background_mix_min=0.25,
                background_mix_max=0.60,
                brightness_jitter=0.12,
                contrast_jitter=0.20,
                gamma_min=0.78,
                gamma_max=1.30,
                noise_sigma_max_255=3.0,
                shadow_probability=0.25,
                shadow_strength_max=0.18,
                projected_edge_probability=0.15,
            )
        if name == "strong":
            return cls(
                profile="strong",
                apply_probability=0.98,
                background_remix_probability=0.88,
                background_mix_min=0.50,
                background_mix_max=0.95,
                residual_gain_min=0.75,
                residual_gain_max=1.35,
                brightness_jitter=0.24,
                contrast_jitter=0.42,
                gamma_min=0.55,
                gamma_max=1.75,
                noise_sigma_max_255=7.0,
                blur_probability=0.35,
                blur_sigma_max=1.35,
                shadow_probability=0.65,
                shadow_strength_max=0.38,
                projected_edge_probability=0.50,
                projected_edge_count_max=5,
            )
        return cls(profile="standard")


@dataclass
class ExampleBatchV214:
    patches: list[np.ndarray]
    labels: np.ndarray
    offsets_px: np.ndarray
    baseline_scores: np.ndarray
    backgrounds: list[str]
    asset_stems: list[str]
    candidate_distance_px: np.ndarray


def _procedural_background(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mode = int(rng.integers(0, 7))
    low = float(rng.uniform(0.12, 0.88))
    high = float(rng.uniform(max(low + 0.05, 0.18), 0.96))

    if mode == 0:  # flat + gradient
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        direction = np.cos(angle) * (xx / max(1, w - 1) - 0.5) + np.sin(angle) * (yy / max(1, h - 1) - 0.5)
        field = (low + high) * 0.5 + direction * float(rng.uniform(0.12, 0.45))
    elif mode == 1:  # checker-like projected structure
        period = int(rng.integers(6, 18))
        checker = ((xx // period + yy // period) % 2).astype(np.float32)
        field = low + (high - low) * checker
    elif mode == 2:  # grid
        field = np.full((h, w), float(rng.uniform(0.30, 0.85)), dtype=np.float32)
        period = int(rng.integers(7, 20))
        thickness = int(rng.integers(1, 3))
        mask = ((xx.astype(np.int32) % period) < thickness) | ((yy.astype(np.int32) % period) < thickness)
        field[mask] += float(rng.uniform(-0.30, 0.30))
    elif mode == 3:  # smooth multi-scale texture
        small_h = max(2, h // 8)
        small_w = max(2, w // 8)
        noise = rng.random((small_h, small_w), dtype=np.float32)
        field = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        field = low + (high - low) * field
    elif mode == 4:  # stripes / video edges
        angle = float(rng.uniform(0.0, math.pi))
        coord = np.cos(angle) * xx + np.sin(angle) * yy
        period = float(rng.uniform(6.0, 24.0))
        wave = 0.5 + 0.5 * np.sin(coord * (2.0 * math.pi / period) + float(rng.uniform(0, 6.28)))
        field = low + (high - low) * wave
    elif mode == 5:  # bubble-ish smooth circles
        field = np.full((h, w), float(rng.uniform(0.25, 0.75)), dtype=np.float32)
        for _ in range(int(rng.integers(3, 9))):
            cx, cy = float(rng.uniform(0, w)), float(rng.uniform(0, h))
            radius = float(rng.uniform(3, max(5, min(h, w) * 0.25)))
            rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            field += np.exp(-0.5 * (rr / max(radius, 1.0)) ** 2) * float(rng.uniform(-0.30, 0.30))
    else:  # mixed low-frequency photo-like field
        n1 = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 3.0)
        n2 = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 8.0)
        field = float(rng.uniform(0.35, 0.70)) + 0.18 * n1 / max(0.2, float(np.std(n1))) + 0.12 * n2 / max(0.2, float(np.std(n2)))

    return np.clip(field, 0.02, 0.98).astype(np.float32)


def _add_projected_edges(image: np.ndarray, rng: np.random.Generator, count_max: int) -> np.ndarray:
    canvas = image.copy()
    h, w = canvas.shape[:2]
    count = int(rng.integers(1, max(2, int(count_max) + 1)))
    for _ in range(count):
        value = float(rng.uniform(0.05, 0.95))
        thickness = int(rng.integers(1, 4))
        if rng.random() < 0.6:
            x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
            x2, y2 = int(rng.integers(0, w)), int(rng.integers(0, h))
            cv2.line(canvas, (x1, y1), (x2, y2), value, thickness, cv2.LINE_AA)
        else:
            x1 = int(rng.integers(0, max(1, w - 2)))
            y1 = int(rng.integers(0, max(1, h - 2)))
            x2 = int(rng.integers(x1 + 1, w + 1))
            y2 = int(rng.integers(y1 + 1, h + 1))
            cv2.rectangle(canvas, (x1, y1), (min(w - 1, x2), min(h - 1, y2)), value, thickness)
    return canvas


def domain_randomize_patch(
    patch: np.ndarray,
    offset_px: tuple[float, float],
    *,
    rng: np.random.Generator,
    config: DomainRandomizationConfigV214,
    sampling: SamplingConfigV214,
    enabled: bool,
    force: bool = False,
) -> tuple[np.ndarray, tuple[float, float]]:
    result = patch.astype(np.float32) / 255.0
    dx, dy = float(offset_px[0]), float(offset_px[1])

    # Geometric flips are valid physical symmetries and update the auxiliary
    # offset target so localisation remains honest.
    if enabled and rng.random() < float(sampling.horizontal_flip_probability):
        result = np.fliplr(result)
        dx = -dx
    if enabled and rng.random() < float(sampling.vertical_flip_probability):
        result = np.flipud(result)
        dy = -dy

    if not enabled or (not force and rng.random() > float(config.apply_probability)):
        return np.ascontiguousarray(np.clip(result * 255.0, 0, 255).astype(np.uint8)), (dx, dy)

    if force or rng.random() < float(config.background_remix_probability):
        # Preserve compact/local residuals (including the physical/synthetic
        # hole) while replacing much of the slow background field.  This is the
        # main anti-background-shortcut mechanism in V2.14.
        low = cv2.GaussianBlur(result, (0, 0), 4.0)
        residual = result - low
        procedural = _procedural_background(result.shape, rng)
        alpha = float(rng.uniform(config.background_mix_min, config.background_mix_max))
        gain = float(rng.uniform(config.residual_gain_min, config.residual_gain_max))
        result = (1.0 - alpha) * low + alpha * procedural + gain * residual

    if rng.random() < float(config.projected_edge_probability):
        result = _add_projected_edges(result, rng, config.projected_edge_count_max)

    if rng.random() < float(config.shadow_probability):
        h, w = result.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        field = np.cos(angle) * (xx / max(1, w - 1) - 0.5) + np.sin(angle) * (yy / max(1, h - 1) - 0.5)
        strength = float(rng.uniform(-config.shadow_strength_max, config.shadow_strength_max))
        result = result + strength * field

    # Photometric changes happen after background remix so the compact residual
    # is forced to survive exposure/gamma changes as well.
    mean = float(np.mean(result))
    contrast = 1.0 + float(rng.uniform(-config.contrast_jitter, config.contrast_jitter))
    brightness = float(rng.uniform(-config.brightness_jitter, config.brightness_jitter))
    result = (result - mean) * contrast + mean + brightness
    result = np.clip(result, 0.0, 1.0)
    gamma = float(rng.uniform(config.gamma_min, config.gamma_max))
    result = np.power(result, gamma, dtype=np.float32)

    if rng.random() < float(config.blur_probability):
        sigma = float(rng.uniform(0.25, max(0.26, config.blur_sigma_max)))
        result = cv2.GaussianBlur(result, (0, 0), sigma)

    sigma255 = float(rng.uniform(0.0, max(0.0, config.noise_sigma_max_255)))
    if sigma255 > 0.0:
        result += rng.normal(0.0, sigma255 / 255.0, result.shape).astype(np.float32)

    return np.ascontiguousarray(np.clip(result * 255.0, 0, 255).astype(np.uint8)), (dx, dy)


def _make_examples_for_asset(
    asset: HoleAsset,
    *,
    rng: np.random.Generator,
    model_config: HolePatchAIConfigV214,
    sampling: SamplingConfigV214,
    domain: DomainRandomizationConfigV214,
    augment: bool,
    force_domain_randomization: bool = False,
    positive_jitter_override: float | None = None,
    positive_count_override: int | None = None,
    negative_count_override: int | None = None,
) -> ExampleBatchV214:
    image = read_gray(asset)
    patches: list[np.ndarray] = []
    labels: list[float] = []
    offsets: list[tuple[float, float]] = []
    baselines: list[float] = []
    backgrounds: list[str] = []
    stems: list[str] = []
    distances: list[float] = []

    positive_count = int(positive_count_override if positive_count_override is not None else sampling.positives_per_image)
    negative_count = int(negative_count_override if negative_count_override is not None else sampling.negatives_per_image)
    positive_jitter = float(positive_jitter_override if positive_jitter_override is not None else sampling.positive_jitter_px)

    for label, count in ((1, positive_count), (0, negative_count)):
        for _ in range(max(0, count)):
            candidate, hole_minus_candidate = sample_candidate_center(
                image,
                rng=rng,
                label=label,
                crop_size=int(model_config.crop_size),
                positive_jitter_px=positive_jitter,
                negative_min_px=float(sampling.negative_min_px),
                negative_max_px=float(sampling.negative_max_px),
            )
            patch = crop_candidate(image, candidate, int(model_config.crop_size))
            patch, transformed_offset = domain_randomize_patch(
                patch,
                hole_minus_candidate,
                rng=rng,
                config=domain,
                sampling=sampling,
                enabled=augment,
                force=force_domain_randomization,
            )
            patches.append(patch)
            labels.append(float(label))
            offsets.append(transformed_offset if label else (0.0, 0.0))
            baselines.append(center_contrast_score(patch))
            backgrounds.append(asset.background_mode)
            stems.append(asset.stem)
            distance = math.hypot(*hole_minus_candidate)
            distances.append(distance)

    return ExampleBatchV214(
        patches=patches,
        labels=np.asarray(labels, dtype=np.float32),
        offsets_px=np.asarray(offsets, dtype=np.float32).reshape(-1, 2),
        baseline_scores=np.asarray(baselines, dtype=np.float32),
        backgrounds=backgrounds,
        asset_stems=stems,
        candidate_distance_px=np.asarray(distances, dtype=np.float32),
    )


def _iter_training_batches(
    assets: Sequence[HoleAsset],
    *,
    model: HolePatchAIV214,
    sampling: SamplingConfigV214,
    domain: DomainRandomizationConfigV214,
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
                domain=domain,
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
        yield features, np.concatenate(labels)[order], np.concatenate(offsets)[order]


def evaluate_assets_v214(
    model: HolePatchAIV214,
    assets: Sequence[HoleAsset],
    *,
    sampling: SamplingConfigV214,
    domain: DomainRandomizationConfigV214,
    seed: int,
    positives_per_image: int = 1,
    negatives_per_image: int = 2,
    positive_jitter_px: float | None = None,
    domain_stress: bool = False,
) -> EvaluationRows:
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    predicted_offsets: list[np.ndarray] = []
    target_offsets: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
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
            domain=domain,
            augment=bool(domain_stress),
            force_domain_randomization=bool(domain_stress),
            positive_jitter_override=positive_jitter_px,
            positive_count_override=positives_per_image,
            negative_count_override=negatives_per_image,
        )
        if not examples.patches:
            continue
        probability, offset = model.predict_patches(examples.patches)
        probabilities.append(probability)
        labels.append(examples.labels)
        predicted_offsets.append(offset)
        target_offsets.append(examples.offsets_px)
        baselines.append(examples.baseline_scores)
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
        baseline_scores=np.concatenate(baselines),
        backgrounds=backgrounds,
        candidate_distance_px=np.concatenate(distances),
        asset_stems=stems,
    )


def _combine_rows(*rows: EvaluationRows) -> EvaluationRows:
    nonempty = [row for row in rows if len(row.labels)]
    if not nonempty:
        return EvaluationRows(
            probabilities=np.empty((0,), dtype=np.float32), labels=np.empty((0,), dtype=np.float32),
            predicted_offsets_px=np.empty((0, 2), dtype=np.float32), target_offsets_px=np.empty((0, 2), dtype=np.float32),
            baseline_scores=np.empty((0,), dtype=np.float32), backgrounds=[], candidate_distance_px=np.empty((0,), dtype=np.float32), asset_stems=[]
        )
    return EvaluationRows(
        probabilities=np.concatenate([row.probabilities for row in nonempty]),
        labels=np.concatenate([row.labels for row in nonempty]),
        predicted_offsets_px=np.concatenate([row.predicted_offsets_px for row in nonempty]),
        target_offsets_px=np.concatenate([row.target_offsets_px for row in nonempty]),
        baseline_scores=np.concatenate([row.baseline_scores for row in nonempty]),
        backgrounds=sum((row.backgrounds for row in nonempty), []),
        candidate_distance_px=np.concatenate([row.candidate_distance_px for row in nonempty]),
        asset_stems=sum((row.asset_stems for row in nonempty), []),
    )


def _auc_from_summary(summary: dict[str, Any]) -> float:
    value = summary.get("auc")
    return float(value) if value is not None else float(summary.get("f1") or 0.0)


def train_model_v214(
    model: HolePatchAIV214,
    train_assets: Sequence[HoleAsset],
    validation_assets: Sequence[HoleAsset],
    *,
    sampling: SamplingConfigV214,
    domain: DomainRandomizationConfigV214,
    epochs: int,
    batch_assets: int,
    seed: int,
    validation_limit: int | None = None,
) -> tuple[list[dict[str, Any]], float, float, int]:
    validation_assets = iter_assets_limited(validation_assets, validation_limit, seed + 77)
    history: list[dict[str, Any]] = []
    best_state: list[np.ndarray] | None = None
    best_score = -1.0
    best_threshold = 0.5
    best_baseline_threshold = 0.0
    best_epoch = -1

    for epoch in range(1, int(epochs) + 1):
        started = time.perf_counter()
        losses: list[float] = []
        accuracies: list[float] = []
        for features, labels, offsets in _iter_training_batches(
            train_assets,
            model=model,
            sampling=sampling,
            domain=domain,
            batch_assets=batch_assets,
            seed=seed,
            epoch=epoch,
        ):
            metrics = model.train_batch(features, labels, offsets)
            losses.append(metrics.loss)
            accuracies.append(metrics.accuracy)

        clean = evaluate_assets_v214(
            model, validation_assets, sampling=sampling, domain=domain,
            seed=evaluation_seed(seed, "validation"), positives_per_image=1, negatives_per_image=2,
        )
        stress = evaluate_assets_v214(
            model, validation_assets, sampling=sampling, domain=domain,
            seed=seed + 15000, positives_per_image=1, negatives_per_image=2, domain_stress=True,
        )
        selection_rows = _combine_rows(clean, stress)
        threshold, selection_metrics = choose_threshold(selection_rows.labels, selection_rows.probabilities)
        baseline_threshold, _ = choose_baseline_threshold(clean.labels, clean.baseline_scores)
        clean_summary = summarize_evaluation(clean, ai_threshold=threshold, baseline_threshold=baseline_threshold)
        stress_summary = summarize_evaluation(stress, ai_threshold=threshold, baseline_threshold=baseline_threshold)
        clean_auc = _auc_from_summary(clean_summary["ai"])
        stress_auc = _auc_from_summary(stress_summary["ai"])
        # Geometric mean prevents a high clean score from hiding poor domain
        # stress behaviour.  Strict real/novel holdouts are NOT involved.
        selection_score = math.sqrt(max(0.0, clean_auc) * max(0.0, stress_auc))

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_threshold = float(threshold)
            best_baseline_threshold = float(baseline_threshold)
            best_state = [parameter.copy() for parameter in model.parameters()]

        row = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(losses)) if losses else 0.0, 6),
            "train_accuracy": round(float(np.mean(accuracies)) if accuracies else 0.0, 6),
            "clean_validation": clean_summary["ai"],
            "domain_stress_validation": stress_summary["ai"],
            "selection_score": round(float(selection_score), 6),
            "ai_threshold": round(float(threshold), 6),
            "seconds": round(time.perf_counter() - started, 3),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs}: loss={row['train_loss']:.4f} "
            f"clean_auc={clean_summary['ai'].get('auc')} stress_auc={stress_summary['ai'].get('auc')} "
            f"select={row['selection_score']:.4f} time={row['seconds']:.1f}s"
        )

    if best_state is not None:
        for target, source in zip(model.parameters(), best_state):
            target[...] = source
    for row in history:
        row["selected_epoch"] = row["epoch"] == best_epoch
    return history, best_threshold, best_baseline_threshold, best_epoch


def _read_v213_report(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _comparison_to_v213(current: dict[str, Any], v213_report: dict[str, Any] | None) -> dict[str, Any]:
    if not v213_report:
        return {"available": False}
    previous = v213_report.get("evaluations") or {}
    rows: dict[str, Any] = {}
    for name in ("validation", "synthetic_test", "novel_background_holdout", "real_holdout", "off_center_stress"):
        now_ai = ((current.get(name) or {}).get("ai") or {})
        old_ai = ((previous.get(name) or {}).get("ai") or {})
        if not now_ai or not old_ai:
            continue
        now_auc = now_ai.get("auc")
        old_auc = old_ai.get("auc")
        now_recall = now_ai.get("recall")
        old_recall = old_ai.get("recall")
        rows[name] = {
            "v213_auc": old_auc,
            "v214_auc": now_auc,
            "auc_delta": None if now_auc is None or old_auc is None else round(float(now_auc) - float(old_auc), 6),
            "v213_recall": old_recall,
            "v214_recall": now_recall,
            "recall_delta": None if now_recall is None or old_recall is None else round(float(now_recall) - float(old_recall), 6),
        }
    return {"available": True, "source_report": str(v213_report.get("model_path") or "v213_report"), "metrics": rows}


def run_training_experiment_v214(
    *,
    holes_root: Path,
    output_dir: Path,
    model_config: HolePatchAIConfigV214,
    sampling: SamplingConfigV214,
    domain: DomainRandomizationConfigV214,
    holdout_backgrounds: Iterable[str],
    epochs: int,
    batch_assets: int,
    seed: int,
    max_train_assets: int | None = None,
    max_eval_assets: int | None = None,
    v213_report_path: Path | None = Path("content/ai/reports/v213/hole_v213_report.json"),
) -> dict[str, Any]:
    holes_root = Path(holes_root).expanduser().resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assets, archive_summary = discover_hole_assets(holes_root, inspect_images=True)
    split: DatasetSplit = build_dataset_split(assets, holdout_backgrounds=holdout_backgrounds, seed=seed)
    train_assets = iter_assets_limited(split.train, max_train_assets, seed + 1)
    validation_assets = iter_assets_limited(split.validation, max_eval_assets, seed + 2)
    test_assets = iter_assets_limited(split.test, max_eval_assets, seed + 3)
    background_assets = iter_assets_limited(split.background_holdout, max_eval_assets, seed + 4)
    real_assets = tuple(split.real_holdout)

    if len(train_assets) < 10:
        raise RuntimeError(f"Need at least 10 synthetic training assets, found {len(train_assets)}")
    if len(validation_assets) < 2:
        raise RuntimeError(f"Need at least 2 validation assets, found {len(validation_assets)}")

    model = HolePatchAIV214(model_config, seed=seed)
    print("V2.14 BACKGROUND-GENERALISING HOLE-AI")
    print("====================================")
    print(f"Hole archive             : {holes_root}")
    print(f"Synthetic train          : {len(train_assets)}")
    print(f"Synthetic validation     : {len(validation_assets)}")
    print(f"Synthetic test           : {len(test_assets)}")
    print(f"STRICT novel-background  : {len(background_assets)} {list(split.holdout_backgrounds)}")
    print(f"REAL holdout             : {len(real_assets)} (never training/model selection)")
    print(f"Domain profile           : {domain.profile}")
    print("Model selection          : clean validation + procedural domain-stress ONLY")
    print("Strict novel + REAL data : evaluated only after model selection")
    print("Input channels           : local residual, DoG, black-hat, gradient (no absolute brightness channel)")
    print()

    history, ai_threshold, baseline_threshold, best_epoch = train_model_v214(
        model,
        train_assets,
        validation_assets,
        sampling=sampling,
        domain=domain,
        epochs=epochs,
        batch_assets=batch_assets,
        seed=seed,
        validation_limit=max_eval_assets,
    )

    groups: dict[str, tuple[HoleAsset, ...]] = {
        "validation": tuple(validation_assets),
        "synthetic_test": tuple(test_assets),
        "novel_background_holdout": tuple(background_assets),
        "real_holdout": tuple(real_assets),
    }
    evaluations: dict[str, Any] = {}
    for name, group in groups.items():
        if not group:
            evaluations[name] = {"examples": 0, "note": "no assets in this split"}
            continue
        rows = evaluate_assets_v214(
            model,
            group,
            sampling=sampling,
            domain=domain,
            seed=evaluation_seed(seed, name),
            positives_per_image=2 if name == "real_holdout" else 1,
            negatives_per_image=4 if name == "real_holdout" else 2,
        )
        evaluations[name] = summarize_evaluation(rows, ai_threshold=ai_threshold, baseline_threshold=baseline_threshold)

    # Reproduce the V2.13 anti-centre test and add a much stronger procedural
    # domain test.  Neither strict test affects training/model selection.
    stress_assets = tuple(test_assets[: max(1, min(len(test_assets), 1200))]) or tuple(validation_assets[:1200])
    if stress_assets:
        offcenter_rows = evaluate_assets_v214(
            model, stress_assets, sampling=sampling, domain=domain,
            seed=evaluation_seed(seed, "off_center_stress"), positives_per_image=2, negatives_per_image=2,
            positive_jitter_px=min(sampling.negative_min_px - 2.0, sampling.positive_jitter_px + 5.0),
        )
        evaluations["off_center_stress"] = summarize_evaluation(
            offcenter_rows, ai_threshold=ai_threshold, baseline_threshold=baseline_threshold
        )
        domain_rows = evaluate_assets_v214(
            model, stress_assets, sampling=sampling, domain=DomainRandomizationConfigV214.from_profile("strong"),
            seed=seed + 19000, positives_per_image=2, negatives_per_image=2, domain_stress=True,
        )
        evaluations["procedural_domain_stress"] = summarize_evaluation(
            domain_rows, ai_threshold=ai_threshold, baseline_threshold=baseline_threshold
        )

    model_path = output_dir / "hole_patch_ai_v214.npz"
    report_path = output_dir / "hole_v214_report.json"
    model.save(
        model_path,
        metadata={
            "ai_threshold": ai_threshold,
            "baseline_threshold": baseline_threshold,
            "sampling": asdict(sampling),
            "domain_randomization": asdict(domain),
            "holdout_backgrounds": list(split.holdout_backgrounds),
            "session_assignment": split.session_assignment,
            "seed": int(seed),
            "best_epoch": int(best_epoch),
        },
    )

    comparison = _comparison_to_v213(
        evaluations,
        _read_v213_report(v213_report_path) if v213_report_path else None,
    )
    novel_ai = (evaluations.get("novel_background_holdout") or {}).get("ai") or {}
    real_ai = (evaluations.get("real_holdout") or {}).get("ai") or {}
    off_ai = (evaluations.get("off_center_stress") or {}).get("ai") or {}
    novel_auc = float(novel_ai.get("auc") or 0.0)
    real_recall = float(real_ai.get("recall") or 0.0)
    off_auc = float(off_ai.get("auc") or 0.0)
    gate = {
        "novel_background_auc_ge_0_70": novel_auc >= 0.70,
        "real_holdout_recall_ge_0_85": real_recall >= 0.85,
        "off_center_auc_ge_0_78": off_auc >= 0.78,
    }
    gate["background_generalization_ready_for_candidate_shadow"] = bool(all(gate.values()))

    report = {
        "schema_version": "2.14",
        "purpose": "background_generalisation_before_candidate_shadow_integration",
        "v213_observed_baseline": {
            "validation_auc": 0.874595,
            "synthetic_test_auc": 0.837996,
            "novel_background_auc": 0.529693,
            "real_holdout_auc": 0.924489,
            "real_holdout_recall": 0.918919,
            "off_center_auc": 0.800802,
        },
        "archive": archive_summary.to_dict(),
        "split": split.to_dict(),
        "effective_counts": {
            "train": len(train_assets), "validation": len(validation_assets), "test": len(test_assets),
            "background_holdout": len(background_assets), "real_holdout": len(real_assets),
        },
        "model_config": asdict(model_config),
        "sampling_config": asdict(sampling),
        "domain_randomization": asdict(domain),
        "selected_thresholds": {"hole_ai": ai_threshold, "center_contrast_baseline": baseline_threshold},
        "selected_epoch": best_epoch,
        "history": history,
        "evaluations": evaluations,
        "comparison_to_v213": comparison,
        "gate": gate,
        "model_path": str(model_path),
        "important_interpretation": [
            "strict novel-background and REAL holdouts never participate in training or model selection",
            "V2.14 selection uses clean validation plus procedurally transformed domain-stress validation only",
            "source 128x128 centre is still never fed as the semantic candidate location; candidate jitter remains active",
            "this model is offline-only and must not receive live authority before candidate-level full-frame shadow evaluation",
        ],
    }
    temp = report_path.with_suffix(report_path.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(report_path)
    return report
