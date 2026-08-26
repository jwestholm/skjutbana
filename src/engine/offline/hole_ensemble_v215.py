from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.engine.ai.hole_patch_ai_v214 import HolePatchAIV214
from src.engine.ai.hole_patch_ensemble_v215 import (
    HolePatchEnsembleConfigV215,
    HolePatchEnsembleV215,
)
from src.engine.offline.hole_dataset_v213 import (
    DatasetSplit,
    HoleAsset,
    build_dataset_split,
    discover_hole_assets,
    iter_assets_limited,
)
from src.engine.offline.hole_training_v213 import (
    EvaluationRows,
    choose_threshold,
    evaluation_seed,
    threshold_metrics,
)
from src.engine.offline.hole_training_v214 import (
    DomainRandomizationConfigV214,
    SamplingConfigV214,
    evaluate_assets_v214,
)


@dataclass(frozen=True)
class EnsembleSearchConfigV215:
    weight_step: float = 0.025
    minimum_nonholdout_selection_ratio: float = 0.995
    novel_auc_gate: float = 0.70
    real_recall_gate: float = 0.85
    off_center_auc_gate: float = 0.76
    disagreement_warn: float = 0.35


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float(default)
    except Exception:
        return float(default)


def _model_metadata_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    nested = metadata.get("metadata") if isinstance(metadata, dict) else None
    return nested if isinstance(nested, dict) else {}


def model_ai_threshold(metadata: dict[str, Any], default: float = 0.5) -> float:
    return float(np.clip(_safe_float(_model_metadata_payload(metadata).get("ai_threshold"), default), 1e-4, 1.0 - 1e-4))


def model_sampling(metadata: dict[str, Any]) -> SamplingConfigV214:
    payload = _model_metadata_payload(metadata).get("sampling")
    if isinstance(payload, dict):
        allowed = {key: value for key, value in payload.items() if key in SamplingConfigV214.__dataclass_fields__}
        try:
            return SamplingConfigV214(**allowed)
        except Exception:
            pass
    return SamplingConfigV214()


def model_split_provenance(metadata: dict[str, Any]) -> tuple[int | None, dict[str, str]]:
    nested = _model_metadata_payload(metadata)
    split_seed = nested.get("split_seed")
    try:
        split_seed = int(split_seed) if split_seed is not None else None
    except Exception:
        split_seed = None
    assignment = nested.get("session_assignment")
    clean_assignment = {str(k): str(v) for k, v in assignment.items()} if isinstance(assignment, dict) else {}
    return split_seed, clean_assignment


def _same_examples(a: EvaluationRows, b: EvaluationRows) -> None:
    if len(a.labels) != len(b.labels):
        raise RuntimeError("paired evaluation row count mismatch")
    if not np.array_equal(a.labels, b.labels):
        raise RuntimeError("paired evaluation labels mismatch")
    if a.asset_stems != b.asset_stems:
        raise RuntimeError("paired evaluation asset order mismatch")
    if not np.allclose(a.candidate_distance_px, b.candidate_distance_px, atol=1e-5):
        raise RuntimeError("paired evaluation candidate geometry mismatch")


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    pos = int(np.sum(labels == 1))
    neg = int(np.sum(labels == 0))
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    index = 0
    while index < len(scores):
        end = index + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[index]:
            end += 1
        ranks[order[index:end]] = 0.5 * ((index + 1) + end)
        index = end
    sum_pos_ranks = float(np.sum(ranks[labels == 1]))
    return float((sum_pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg))


def _offset_summary(labels: np.ndarray, predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(labels).reshape(-1) >= 0.5
    if not np.any(mask):
        return {"positive_count": 0, "mean_error_px": None, "median_error_px": None, "p95_error_px": None}
    error = np.linalg.norm(np.asarray(predicted)[mask] - np.asarray(target)[mask], axis=1)
    return {
        "positive_count": int(len(error)),
        "mean_error_px": round(float(np.mean(error)), 6),
        "median_error_px": round(float(np.median(error)), 6),
        "p95_error_px": round(float(np.percentile(error, 95)), 6),
    }


def _summary(labels: np.ndarray, scores: np.ndarray, threshold: float, predicted_offsets: np.ndarray | None = None, target_offsets: np.ndarray | None = None) -> dict[str, Any]:
    result = threshold_metrics(labels, scores, threshold)
    if predicted_offsets is not None and target_offsets is not None:
        result["offset_refinement"] = _offset_summary(labels, predicted_offsets, target_offsets)
    return result


def _complementarity(
    labels: np.ndarray,
    standard_scores: np.ndarray,
    mild_scores: np.ndarray,
    *,
    standard_threshold: float,
    mild_threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels).reshape(-1) >= 0.5
    std_hit = np.asarray(standard_scores) >= float(standard_threshold)
    mild_hit = np.asarray(mild_scores) >= float(mild_threshold)
    positive = labels
    negative = ~labels

    def count(mask: np.ndarray) -> int:
        return int(np.sum(mask))

    positive_count = max(1, count(positive))
    negative_count = max(1, count(negative))
    corr = 1.0
    if len(standard_scores) > 1 and float(np.std(standard_scores)) > 1e-9 and float(np.std(mild_scores)) > 1e-9:
        corr = float(np.corrcoef(standard_scores, mild_scores)[0, 1])

    std_only_pos = positive & std_hit & ~mild_hit
    mild_only_pos = positive & mild_hit & ~std_hit
    both_pos = positive & std_hit & mild_hit
    neither_pos = positive & ~std_hit & ~mild_hit

    std_only_fp = negative & std_hit & ~mild_hit
    mild_only_fp = negative & mild_hit & ~std_hit
    both_fp = negative & std_hit & mild_hit
    neither_fp = negative & ~std_hit & ~mild_hit

    return {
        "score_correlation": round(corr, 6),
        "mean_probability_disagreement": round(float(np.mean(np.abs(np.asarray(standard_scores) - np.asarray(mild_scores)))), 6),
        "positive_examples": count(positive),
        "positive_both_hit": count(both_pos),
        "positive_standard_only_hit": count(std_only_pos),
        "positive_mild_only_hit": count(mild_only_pos),
        "positive_neither_hit": count(neither_pos),
        "positive_oracle_either_recall": round((count(both_pos) + count(std_only_pos) + count(mild_only_pos)) / positive_count, 6),
        "positive_complementary_rescue_fraction": round((count(std_only_pos) + count(mild_only_pos)) / positive_count, 6),
        "negative_examples": count(negative),
        "negative_both_false_positive": count(both_fp),
        "negative_standard_only_false_positive": count(std_only_fp),
        "negative_mild_only_false_positive": count(mild_only_fp),
        "negative_both_reject": count(neither_fp),
        "negative_or_false_positive_rate": round((count(both_fp) + count(std_only_fp) + count(mild_only_fp)) / negative_count, 6),
    }


def _fused_offsets(
    standard_probability: np.ndarray,
    mild_probability: np.ndarray,
    standard_offsets: np.ndarray,
    mild_offsets: np.ndarray,
    standard_weight: float,
) -> np.ndarray:
    w = float(np.clip(standard_weight, 0.0, 1.0))
    std_support = w * np.maximum(np.asarray(standard_probability), 1e-4)
    mild_support = (1.0 - w) * np.maximum(np.asarray(mild_probability), 1e-4)
    denom = np.maximum(std_support + mild_support, 1e-6)[:, None]
    return (np.asarray(standard_offsets) * std_support[:, None] + np.asarray(mild_offsets) * mild_support[:, None]) / denom


def _paired_rows(
    standard_model: HolePatchAIV214,
    mild_model: HolePatchAIV214,
    assets: Sequence[HoleAsset],
    *,
    sampling: SamplingConfigV214,
    seed: int,
    positives_per_image: int = 1,
    negatives_per_image: int = 2,
    positive_jitter_px: float | None = None,
    domain_stress: bool = False,
) -> tuple[EvaluationRows, EvaluationRows]:
    # Use one fixed domain generator for paired stress.  The *trained* profiles
    # differ; evaluation data must not differ between the two models.
    domain = DomainRandomizationConfigV214.from_profile("strong")
    kwargs = dict(
        assets=assets,
        sampling=sampling,
        domain=domain,
        seed=int(seed),
        positives_per_image=int(positives_per_image),
        negatives_per_image=int(negatives_per_image),
        positive_jitter_px=positive_jitter_px,
        domain_stress=bool(domain_stress),
    )
    standard_rows = evaluate_assets_v214(standard_model, **kwargs)
    mild_rows = evaluate_assets_v214(mild_model, **kwargs)
    _same_examples(standard_rows, mild_rows)
    return standard_rows, mild_rows


def _combine_two(a: EvaluationRows, b: EvaluationRows) -> EvaluationRows:
    return EvaluationRows(
        probabilities=np.concatenate([a.probabilities, b.probabilities]),
        labels=np.concatenate([a.labels, b.labels]),
        predicted_offsets_px=np.concatenate([a.predicted_offsets_px, b.predicted_offsets_px]),
        target_offsets_px=np.concatenate([a.target_offsets_px, b.target_offsets_px]),
        baseline_scores=np.concatenate([a.baseline_scores, b.baseline_scores]),
        backgrounds=a.backgrounds + b.backgrounds,
        candidate_distance_px=np.concatenate([a.candidate_distance_px, b.candidate_distance_px]),
        asset_stems=a.asset_stems + b.asset_stems,
    )


def _fuse_rows(
    standard_rows: EvaluationRows,
    mild_rows: EvaluationRows,
    *,
    standard_threshold: float,
    mild_threshold: float,
    standard_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    fused = HolePatchEnsembleV215.fuse_probabilities(
        standard_rows.probabilities,
        mild_rows.probabilities,
        standard_threshold=standard_threshold,
        mild_threshold=mild_threshold,
        standard_weight=standard_weight,
    )
    offsets = _fused_offsets(
        standard_rows.probabilities,
        mild_rows.probabilities,
        standard_rows.predicted_offsets_px,
        mild_rows.predicted_offsets_px,
        standard_weight,
    )
    return fused, offsets


def choose_blend_weight_v215(
    clean_standard: EvaluationRows,
    clean_mild: EvaluationRows,
    stress_standard: EvaluationRows,
    stress_mild: EvaluationRows,
    *,
    standard_threshold: float,
    mild_threshold: float,
    search: EnsembleSearchConfigV215,
) -> dict[str, Any]:
    _same_examples(clean_standard, clean_mild)
    _same_examples(stress_standard, stress_mild)

    step = float(np.clip(search.weight_step, 0.005, 0.25))
    weights = np.arange(0.0, 1.0 + step * 0.5, step)
    candidates: list[dict[str, Any]] = []
    for raw_weight in weights:
        weight = float(np.clip(raw_weight, 0.0, 1.0))
        clean_fused, _ = _fuse_rows(
            clean_standard, clean_mild,
            standard_threshold=standard_threshold, mild_threshold=mild_threshold,
            standard_weight=weight,
        )
        stress_fused, _ = _fuse_rows(
            stress_standard, stress_mild,
            standard_threshold=standard_threshold, mild_threshold=mild_threshold,
            standard_weight=weight,
        )
        labels = np.concatenate([clean_standard.labels, stress_standard.labels])
        scores = np.concatenate([clean_fused, stress_fused])
        threshold, combined = choose_threshold(labels, scores)
        clean_auc = _auc(clean_standard.labels, clean_fused) or 0.0
        stress_auc = _auc(stress_standard.labels, stress_fused) or 0.0
        selection = math.sqrt(max(0.0, clean_auc) * max(0.0, stress_auc))
        candidates.append(
            {
                "standard_weight": round(weight, 6),
                "mild_weight": round(1.0 - weight, 6),
                "threshold": round(float(threshold), 6),
                "clean_auc": round(float(clean_auc), 6),
                "stress_auc": round(float(stress_auc), 6),
                "selection_score": round(float(selection), 6),
                "combined_f1": combined.get("f1"),
            }
        )

    candidates.sort(
        key=lambda row: (
            float(row["selection_score"]),
            float(row.get("combined_f1") or 0.0),
            -abs(float(row["standard_weight"]) - 0.5),
        ),
        reverse=True,
    )
    winner = candidates[0]
    pure_best = max(
        (row for row in candidates if float(row["standard_weight"]) in {0.0, 1.0}),
        key=lambda row: float(row["selection_score"]),
    )
    winner = dict(winner)
    winner["best_pure_selection_score"] = pure_best["selection_score"]
    winner["best_pure_standard_weight"] = pure_best["standard_weight"]
    winner["relative_to_best_pure"] = round(
        float(winner["selection_score"]) / max(1e-9, float(pure_best["selection_score"])), 6
    )
    winner["blend_is_nontrivial"] = bool(0.05 < float(winner["standard_weight"]) < 0.95)
    return {"winner": winner, "candidates": candidates}


def _evaluate_group(
    standard_model: HolePatchAIV214,
    mild_model: HolePatchAIV214,
    assets: Sequence[HoleAsset],
    *,
    sampling: SamplingConfigV214,
    seed: int,
    standard_threshold: float,
    mild_threshold: float,
    standard_weight: float,
    fused_threshold: float,
    positives_per_image: int = 1,
    negatives_per_image: int = 2,
    positive_jitter_px: float | None = None,
    domain_stress: bool = False,
) -> dict[str, Any]:
    standard_rows, mild_rows = _paired_rows(
        standard_model, mild_model, assets,
        sampling=sampling, seed=seed,
        positives_per_image=positives_per_image,
        negatives_per_image=negatives_per_image,
        positive_jitter_px=positive_jitter_px,
        domain_stress=domain_stress,
    )
    fused, fused_offsets = _fuse_rows(
        standard_rows, mild_rows,
        standard_threshold=standard_threshold,
        mild_threshold=mild_threshold,
        standard_weight=standard_weight,
    )
    return {
        "examples": int(len(standard_rows.labels)),
        "standard": _summary(standard_rows.labels, standard_rows.probabilities, standard_threshold, standard_rows.predicted_offsets_px, standard_rows.target_offsets_px),
        "mild": _summary(mild_rows.labels, mild_rows.probabilities, mild_threshold, mild_rows.predicted_offsets_px, mild_rows.target_offsets_px),
        "fused": _summary(standard_rows.labels, fused, fused_threshold, fused_offsets, standard_rows.target_offsets_px),
        "complementarity": _complementarity(
            standard_rows.labels,
            standard_rows.probabilities,
            mild_rows.probabilities,
            standard_threshold=standard_threshold,
            mild_threshold=mild_threshold,
        ),
    }


def _per_background(
    standard_model: HolePatchAIV214,
    mild_model: HolePatchAIV214,
    assets: Sequence[HoleAsset],
    **kwargs: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    backgrounds = sorted({asset.background_mode for asset in assets})
    for index, background in enumerate(backgrounds):
        subset = tuple(asset for asset in assets if asset.background_mode == background)
        if not subset:
            continue
        local_kwargs = dict(kwargs)
        local_kwargs["seed"] = int(kwargs["seed"]) + 997 * (index + 1)
        result[background] = _evaluate_group(standard_model, mild_model, subset, **local_kwargs)
    return result


def run_ensemble_experiment_v215(
    *,
    holes_root: Path,
    standard_model_path: Path,
    mild_model_path: Path,
    output_dir: Path,
    holdout_backgrounds: Iterable[str] = ("black", "checker", "gray", "bubbles"),
    seed: int = 21501,
    max_eval_assets: int | None = None,
    search: EnsembleSearchConfigV215 = EnsembleSearchConfigV215(),
) -> dict[str, Any]:
    holes_root = Path(holes_root).expanduser().resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    standard_model, standard_meta = HolePatchAIV214.load(Path(standard_model_path))
    mild_model, mild_meta = HolePatchAIV214.load(Path(mild_model_path))
    if standard_model.config.crop_size != mild_model.config.crop_size:
        raise RuntimeError("V2.14 standard/mild models use different crop sizes")

    standard_threshold = model_ai_threshold(standard_meta)
    mild_threshold = model_ai_threshold(mild_meta)
    sampling = model_sampling(standard_meta)

    standard_split_seed, standard_assignment = model_split_provenance(standard_meta)
    mild_split_seed, mild_assignment = model_split_provenance(mild_meta)
    if standard_split_seed is None or mild_split_seed is None:
        raise RuntimeError(
            "V2.15 requires models with explicit shared split_seed provenance. "
            "Older V2.14 sweep models varied split and model seed together. "
            "Run: python3 -m automation.hole_v215_pair_train"
        )
    if standard_split_seed != mild_split_seed or standard_assignment != mild_assignment:
        raise RuntimeError(
            "V2.15 refuses unpaired mild/standard models: session split provenance differs. "
            "Run: python3 -m automation.hole_v215_pair_train"
        )
    split_seed = int(standard_split_seed)

    assets, archive = discover_hole_assets(holes_root, inspect_images=False)
    split: DatasetSplit = build_dataset_split(assets, holdout_backgrounds=holdout_backgrounds, seed=split_seed)
    if standard_assignment and split.session_assignment != standard_assignment:
        raise RuntimeError(
            "Current hole archive produces a different session assignment than the paired model metadata. "
            "Do not silently evaluate on a changed split; retrain the pair with hole_v215_pair_train."
        )
    validation_assets = iter_assets_limited(split.validation, max_eval_assets, seed + 1)
    test_assets = iter_assets_limited(split.test, max_eval_assets, seed + 2)
    novel_assets = iter_assets_limited(split.background_holdout, max_eval_assets, seed + 3)
    real_assets = tuple(split.real_holdout)

    if len(validation_assets) < 2:
        raise RuntimeError("Need validation assets to choose V2.15 blend without holdout leakage")

    # Model/weight selection may use ONLY clean validation + transformed copies
    # of that same validation split.  Novel backgrounds, real holes and test
    # sessions remain untouched until the blend is frozen.
    clean_std, clean_mild = _paired_rows(
        standard_model, mild_model, validation_assets,
        sampling=sampling, seed=evaluation_seed(seed, "selection_clean"),
        positives_per_image=1, negatives_per_image=2,
    )
    stress_std, stress_mild = _paired_rows(
        standard_model, mild_model, validation_assets,
        sampling=sampling, seed=evaluation_seed(seed, "selection_stress"),
        positives_per_image=1, negatives_per_image=2, domain_stress=True,
    )
    search_result = choose_blend_weight_v215(
        clean_std, clean_mild, stress_std, stress_mild,
        standard_threshold=standard_threshold,
        mild_threshold=mild_threshold,
        search=search,
    )
    winner = search_result["winner"]
    standard_weight = float(winner["standard_weight"])
    fused_threshold = float(winner["threshold"])

    evaluations: dict[str, Any] = {}
    groups = {
        "validation": tuple(validation_assets),
        "synthetic_test": tuple(test_assets),
        "novel_background_holdout": tuple(novel_assets),
        "real_holdout": tuple(real_assets),
    }
    for name, group in groups.items():
        if not group:
            evaluations[name] = {"examples": 0, "note": "no assets"}
            continue
        evaluations[name] = _evaluate_group(
            standard_model, mild_model, group,
            sampling=sampling,
            seed=evaluation_seed(seed, name),
            standard_threshold=standard_threshold,
            mild_threshold=mild_threshold,
            standard_weight=standard_weight,
            fused_threshold=fused_threshold,
            positives_per_image=2 if name == "real_holdout" else 1,
            negatives_per_image=4 if name == "real_holdout" else 2,
        )

    stress_assets = tuple(test_assets[: max(1, min(len(test_assets), 1200))]) or tuple(validation_assets[:1200])
    if stress_assets:
        evaluations["off_center_stress"] = _evaluate_group(
            standard_model, mild_model, stress_assets,
            sampling=sampling,
            seed=evaluation_seed(seed, "off_center_stress"),
            standard_threshold=standard_threshold,
            mild_threshold=mild_threshold,
            standard_weight=standard_weight,
            fused_threshold=fused_threshold,
            positives_per_image=2,
            negatives_per_image=2,
            positive_jitter_px=min(sampling.negative_min_px - 2.0, sampling.positive_jitter_px + 5.0),
        )
        evaluations["procedural_domain_stress"] = _evaluate_group(
            standard_model, mild_model, stress_assets,
            sampling=sampling,
            seed=seed + 19000,
            standard_threshold=standard_threshold,
            mild_threshold=mild_threshold,
            standard_weight=standard_weight,
            fused_threshold=fused_threshold,
            positives_per_image=2,
            negatives_per_image=2,
            domain_stress=True,
        )

    per_background = _per_background(
        standard_model, mild_model, novel_assets,
        sampling=sampling,
        seed=seed + 27000,
        standard_threshold=standard_threshold,
        mild_threshold=mild_threshold,
        standard_weight=standard_weight,
        fused_threshold=fused_threshold,
        positives_per_image=1,
        negatives_per_image=2,
    ) if novel_assets else {}

    novel_fused = ((evaluations.get("novel_background_holdout") or {}).get("fused") or {})
    real_fused = ((evaluations.get("real_holdout") or {}).get("fused") or {})
    off_fused = ((evaluations.get("off_center_stress") or {}).get("fused") or {})
    relative_to_pure = float(winner.get("relative_to_best_pure") or 0.0)
    gate = {
        "nonholdout_selection_not_worse_than_best_pure": relative_to_pure >= float(search.minimum_nonholdout_selection_ratio),
        "novel_background_auc_ge_gate": _safe_float(novel_fused.get("auc")) >= float(search.novel_auc_gate),
        "real_holdout_recall_ge_gate": _safe_float(real_fused.get("recall")) >= float(search.real_recall_gate),
        "off_center_auc_ge_gate": _safe_float(off_fused.get("auc")) >= float(search.off_center_auc_gate),
    }
    gate["ensemble_worth_candidate_shadow"] = bool(all(gate.values()))

    ensemble_config = HolePatchEnsembleConfigV215(
        standard_model_path=str(Path(standard_model_path)),
        mild_model_path=str(Path(mild_model_path)),
        standard_weight=standard_weight,
        fused_threshold=fused_threshold,
        standard_threshold=standard_threshold,
        mild_threshold=mild_threshold,
        disagreement_warn=float(search.disagreement_warn),
    )
    config_path = output_dir / "hole_v215_ensemble.json"
    report_path = output_dir / "hole_v215_report.json"
    config_payload = {
        "schema_version": "2.15",
        "purpose": "candidate_patch_shadow_evidence_only",
        "ensemble_config": asdict(ensemble_config),
        "selection": winner,
        "gate": gate,
        "note": "Generated without strict novel/REAL/test holdout data in blend selection. This file does not grant live authority.",
    }
    config_path.write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "schema_version": "2.15",
        "purpose": "paired_mild_standard_complementarity_and_nonholdout_blend_selection",
        "models": {
            "standard": {"path": str(standard_model_path), "threshold": standard_threshold, "metadata": _model_metadata_payload(standard_meta)},
            "mild": {"path": str(mild_model_path), "threshold": mild_threshold, "metadata": _model_metadata_payload(mild_meta)},
        },
        "paired_split_provenance": {
            "split_seed": split_seed,
            "session_assignment": standard_assignment,
            "verified_equal_between_models": True,
        },
        "archive": archive.to_dict(),
        "split": split.to_dict(),
        "effective_counts": {
            "validation": len(validation_assets),
            "test": len(test_assets),
            "novel_background": len(novel_assets),
            "real": len(real_assets),
        },
        "selection_policy": "standard_weight + fused threshold selected ONLY on paired clean validation + procedural stress validation; endpoints 0 and 1 are included",
        "search": {"config": asdict(search), "winner": winner, "all_weights": search_result["candidates"]},
        "evaluations": evaluations,
        "novel_per_background": per_background,
        "gate": gate,
        "generated_ensemble_config": str(config_path),
        "important_interpretation": [
            "V2.14 sweep suggested complementary behaviour but did not measure paired error overlap; V2.15 does.",
            "A non-trivial blend is kept only if non-holdout selection supports it; strict holdouts never choose the weight.",
            "The OR/either recall is an oracle diagnostic, not an achievable live score by itself.",
            "V2.15 annotator preserves candidate ordering and cannot become authoritative by loading this config.",
            "This patch-level experiment still does not prove full-frame detection or >=95% game accuracy; detector hard-negative capture/replay is the next gate.",
        ],
    }
    temp = report_path.with_suffix(report_path.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(report_path)
    return report


__all__ = [
    "EnsembleSearchConfigV215",
    "choose_blend_weight_v215",
    "model_ai_threshold",
    "model_sampling",
    "model_split_provenance",
    "run_ensemble_experiment_v215",
]
