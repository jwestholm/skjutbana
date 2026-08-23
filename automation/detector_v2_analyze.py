from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PATH = Path("content/ai/detector_v2/shot_diagnostics.jsonl")
DEFAULT_SUMMARY = Path("content/ai/detector_v2/latest_summary.json")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def _pct(count: int, total: int) -> float:
    return round(100.0 * count / total, 3) if total else 0.0


def _found(value: Any, radius: float) -> bool:
    number = _finite(value)
    return number is not None and number <= radius


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except Exception:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def select_records(
    records: list[dict[str, Any]],
    *,
    session_id: str | None = None,
    include_all: bool = False,
) -> tuple[list[dict[str, Any]], str | None]:
    """Select only the newest detector runtime by default."""
    if include_all or not records:
        return records, None

    if session_id:
        selected = [
            row for row in records
            if str(row.get("runtime_session_id", "")) == session_id
        ]
        return selected, session_id

    latest_session = None
    for row in reversed(records):
        value = row.get("runtime_session_id")
        if value:
            latest_session = str(value)
            break

    if latest_session is None:
        return records, None

    return (
        [
            row for row in records
            if str(row.get("runtime_session_id", "")) == latest_session
        ],
        latest_session,
    )


def _nearest(record: dict[str, Any], key: str) -> float | None:
    block = record.get("nearest_candidate_distance_px", {})
    if not isinstance(block, dict):
        return None
    return _finite(block.get(key))


def _evaluation(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("evaluation_funnel", {})
    return value if isinstance(value, dict) else {}


def classify_detector_miss(record: dict[str, Any], match_radius: float) -> str:
    """Classify why the best/ever merged detector did not cover GT."""
    if not isinstance(record.get("ground_truth"), dict):
        return "unlabelled"

    legacy = _nearest(record, "legacy")
    v2_frame = _nearest(record, "v2_frame")
    v2 = _nearest(record, "v2")
    merged = _nearest(record, "merged")

    if _found(merged, match_radius):
        if _found(v2, match_radius):
            if _found(legacy, match_radius):
                if not _found(v2_frame, match_radius):
                    return "found_both_bank_helped"
                return "found_by_both"
            if not _found(v2_frame, match_radius):
                return "recovered_by_candidate_bank"
            return "recovered_by_v2"
        return "legacy_only"

    if _found(v2, match_radius):
        return "v2_lost_in_merge"
    if _found(legacy, match_radius):
        return "legacy_lost_in_merge"

    gt_signal = record.get("gt_signal_max", {})
    if not isinstance(gt_signal, dict):
        gt_signal = {}

    absdiff = _finite(gt_signal.get("absdiff")) or 0.0
    zscore = _finite(gt_signal.get("zscore")) or 0.0
    saliency = _finite(gt_signal.get("saliency")) or 0.0
    margin = _finite(gt_signal.get("saliency_minus_threshold"))

    if absdiff < 1.2 and zscore < 1.05:
        return "weak_or_no_camera_signal"
    if margin is not None and margin >= 0.0:
        return "signal_above_threshold_but_no_candidate"
    if saliency >= 7.0:
        return "strong_gt_signal_but_peak_missing"
    if absdiff >= 4.0 and saliency < 7.0:
        return "saliency_suppressed"
    return "candidate_generation_miss"


def classify_pipeline_loss(record: dict[str, Any], radius: float) -> str:
    """Classify where a candidate is lost between camera frames and AI output."""
    if not isinstance(record.get("ground_truth"), dict):
        return "unlabelled"

    v24 = record.get("v24", {}) if isinstance(record.get("v24"), dict) else {}
    v25 = record.get("v25", {}) if isinstance(record.get("v25"), dict) else {}
    vectors = v25.get("best_vectors", {}) if isinstance(v25.get("best_vectors"), dict) else {}
    v25_final = vectors.get("v25_final") if isinstance(vectors.get("v25_final"), dict) else {}
    v26 = record.get("v26", {}) if isinstance(record.get("v26"), dict) else {}
    ever = (
        _found(_nearest(record, "merged"), radius)
        or _found(v24.get("final_nearest_px"), radius)
        or _found(v25_final.get("distance_px"), radius)
        or _found(v26.get("final_best_nearest_px"), radius)
    )
    evaluation = _evaluation(record)
    if not evaluation:
        return "no_evaluation_telemetry"

    raw = _found(evaluation.get("raw_nearest_px", evaluation.get("raw_closest_dist")), radius)
    filtered = _found(evaluation.get("filter_closest_dist"), radius)
    ranked = _found(evaluation.get("ranked_nearest_px", evaluation.get("ai_topk_closest_dist")), radius)
    selected = _found(evaluation.get("selected_nearest_px", evaluation.get("selected_dist")), radius)

    if not ever:
        return "detector_never_covered_gt"
    if not raw:
        return "candidate_disappeared_before_evaluation"
    if not filtered:
        return "noise_filter_removed_gt"
    if not ranked:
        return "ranking_topk_removed_gt"
    if not selected:
        return "selected_wrong_candidate"
    return "selected_correct"


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    labelled = [
        record for record in records
        if isinstance(record.get("ground_truth"), dict)
    ]

    by_background: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in labelled:
        background = str(record.get("ground_truth", {}).get("background", "unknown"))
        by_background[background].append(record)

    match_radii = [10.0, 20.0, 42.0]
    detector_sources = ("legacy", "v2_frame", "v2", "merged")

    summary: dict[str, Any] = {
        "schema_version": "2.6",
        "records_total": len(records),
        "records_with_ground_truth": len(labelled),
        "records_with_evaluation_funnel": sum(
            1 for record in labelled if bool(_evaluation(record))
        ),
        "runtime_session_ids": sorted({
            str(record.get("runtime_session_id"))
            for record in records
            if record.get("runtime_session_id")
        }),
        "git_commits": sorted({
            str(record.get("git_commit"))
            for record in records
            if record.get("git_commit")
        }),
        "benchmark_seeds": sorted({
            int(record.get("ground_truth", {}).get("benchmark_seed"))
            for record in labelled
            if isinstance(record.get("ground_truth"), dict)
            and record.get("ground_truth", {}).get("benchmark_seed") is not None
        }),
        "overall": {},
        "by_background": {},
    }

    for radius in match_radii:
        key = f"within_{int(radius)}px"
        summary["overall"][key] = {}
        for source in detector_sources:
            count = sum(
                1 for record in labelled
                if _found(_nearest(record, source), radius)
            )
            summary["overall"][key][source] = {
                "count": count,
                "pct": _pct(count, len(labelled)),
            }

    # V2.4 additive paths are stored separately so historical V2/V2.3 fields
    # remain comparable.
    for radius in match_radii:
        key = f"within_{int(radius)}px"
        for source, field in (
            ("v24_final", "final_nearest_px"),
            ("v24_tile", "tile_probe_nearest_px"),
            ("v24_accumulator", "accumulator_nearest_px"),
        ):
            count = sum(
                1
                for record in labelled
                if _found(
                    (record.get("v24", {}) if isinstance(record.get("v24"), dict) else {}).get(field),
                    radius,
                )
            )
            summary["overall"][key][source] = {
                "count": count,
                "pct": _pct(count, len(labelled)),
            }

    # V2.5 keeps V2.4 as a measured baseline and stores additive refined-centre
    # hypotheses separately. Distances live in best_vectors so dx/dy is retained.
    for radius in match_radii:
        key = f"within_{int(radius)}px"
        for source, vector_key in (("v25_refined_tile", "v25_refined_tile"), ("v25_final", "v25_final")):
            count = 0
            for record in labelled:
                v25 = record.get("v25", {}) if isinstance(record.get("v25"), dict) else {}
                vectors = v25.get("best_vectors", {}) if isinstance(v25.get("best_vectors"), dict) else {}
                vector = vectors.get(vector_key)
                if isinstance(vector, dict) and _found(vector.get("distance_px"), radius):
                    count += 1
            summary["overall"][key][source] = {
                "count": count,
                "pct": _pct(count, len(labelled)),
            }

    # V2.6 shot-vault paths.  `vault` is the spatial union accumulated over
    # the pending shot; `final` is what the detector actually emitted after
    # adding the bounded carry set to each current frame.
    for radius in match_radii:
        key = f"within_{int(radius)}px"
        for source, field in (
            ("v26_vault", "vault_best_nearest_px"),
            ("v26_final", "final_best_nearest_px"),
        ):
            count = 0
            for record in labelled:
                v26 = record.get("v26", {}) if isinstance(record.get("v26"), dict) else {}
                if _found(v26.get(field), radius):
                    count += 1
            summary["overall"][key][source] = {
                "count": count,
                "pct": _pct(count, len(labelled)),
            }

    summary["overall"]["detector_classification_42px"] = dict(
        Counter(classify_detector_miss(record, 42.0) for record in labelled)
    )
    summary["overall"]["pipeline_classification_42px"] = dict(
        Counter(classify_pipeline_loss(record, 42.0) for record in labelled)
    )

    eval_rows = [(record, _evaluation(record)) for record in labelled if _evaluation(record)]
    evaluation_summary: dict[str, Any] = {
        "shots": len(eval_rows),
    }
    for name, field in (
        ("raw", "raw_nearest_px"),
        ("filtered", "filter_closest_dist"),
        ("ranked", "ranked_nearest_px"),
        ("selected", "selected_nearest_px"),
    ):
        for radius in match_radii:
            count = sum(
                1
                for _record, ev in eval_rows
                if _found(ev.get(field), radius)
            )
            evaluation_summary[f"{name}_within_{int(radius)}px"] = {
                "count": count,
                "pct": _pct(count, len(eval_rows)),
            }
    # Provenance inside the exact F2 evaluation snapshot. This is the key
    # measurement for the V2.1 candidate bank: did a candidate that existed
    # earlier actually survive until the evaluator looked?
    if eval_rows:
        for name, field in (
            ("raw_v1", "raw_v1_nearest_px"),
            ("raw_v2", "raw_v2_nearest_px"),
            ("raw_v2_carried", "raw_v2_bank_carried_nearest_px"),
            ("raw_v2_confirmed", "raw_v2_bank_confirmed_nearest_px"),
        ):
            for radius in match_radii:
                count = sum(
                    1
                    for _record, ev in eval_rows
                    if _found(ev.get(field), radius)
                )
                evaluation_summary[f"{name}_within_{int(radius)}px"] = {
                    "count": count,
                    "pct": _pct(count, len(eval_rows)),
                }

        carried_counts = [
            int(ev.get("raw_v2_bank_carried_count", 0) or 0)
            for _record, ev in eval_rows
        ]
        evaluation_summary["raw_v2_bank_carried_count"] = {
            "mean": (
                round(sum(carried_counts) / len(carried_counts), 3)
                if carried_counts
                else 0.0
            ),
            "max": max(carried_counts) if carried_counts else 0,
            "shots_with_carried_candidates": sum(
                1 for value in carried_counts if value > 0
            ),
        }


        generic_carried_counts = [
            int(ev.get("raw_candidate_bank_carried_count", 0) or 0)
            for _record, ev in eval_rows
        ]
        evaluation_summary["raw_candidate_bank_carried_count"] = {
            "mean": (
                round(sum(generic_carried_counts) / len(generic_carried_counts), 3)
                if generic_carried_counts
                else 0.0
            ),
            "max": max(generic_carried_counts) if generic_carried_counts else 0,
            "shots_with_carried_candidates": sum(
                1 for value in generic_carried_counts if value > 0
            ),
        }
        for radius in match_radii:
            count = sum(
                1 for _record, ev in eval_rows
                if _found(ev.get("raw_candidate_bank_carried_nearest_px"), radius)
            )
            evaluation_summary[f"raw_candidate_bank_carried_within_{int(radius)}px"] = {
                "count": count,
                "pct": _pct(count, len(eval_rows)),
            }

        # V2.4 provenance in the exact F2 snapshot.
        for name, field in (
            ("raw_v24_tile", "raw_v24_tile_nearest_px"),
            ("raw_v24_accumulator", "raw_v24_accumulator_nearest_px"),
        ):
            for radius in match_radii:
                count = sum(
                    1 for _record, ev in eval_rows if _found(ev.get(field), radius)
                )
                evaluation_summary[f"{name}_within_{int(radius)}px"] = {
                    "count": count,
                    "pct": _pct(count, len(eval_rows)),
                }

        for name, field in (
            ("raw_v24_tile_count", "raw_v24_tile_count"),
            ("raw_v24_accumulator_count", "raw_v24_accumulator_count"),
        ):
            values = [int(ev.get(field, 0) or 0) for _record, ev in eval_rows]
            evaluation_summary[name] = {
                "mean": round(sum(values) / len(values), 3) if values else 0.0,
                "max": max(values) if values else 0,
                "shots_with_candidates": sum(1 for value in values if value > 0),
            }

        # V2.5 refined tile candidates in the exact F2 snapshot.
        for radius in match_radii:
            count = sum(
                1 for _record, ev in eval_rows
                if _found(ev.get("raw_v25_refined_tile_nearest_px"), radius)
            )
            evaluation_summary[f"raw_v25_refined_tile_within_{int(radius)}px"] = {
                "count": count,
                "pct": _pct(count, len(eval_rows)),
            }
        refined_counts = [int(ev.get("raw_v25_refined_tile_count", 0) or 0) for _record, ev in eval_rows]
        evaluation_summary["raw_v25_refined_tile_count"] = {
            "mean": round(sum(refined_counts) / len(refined_counts), 3) if refined_counts else 0.0,
            "max": max(refined_counts) if refined_counts else 0,
            "shots_with_candidates": sum(1 for value in refined_counts if value > 0),
        }

        # V2.6 vault provenance in the exact F2 snapshot.
        for name, field in (
            ("raw_v26_vault_carried", "raw_v26_vault_carried_nearest_px"),
            ("raw_v26_final", "raw_v26_final_nearest_px"),
        ):
            for radius in match_radii:
                count = sum(
                    1 for _record, ev in eval_rows if _found(ev.get(field), radius)
                )
                evaluation_summary[f"{name}_within_{int(radius)}px"] = {
                    "count": count,
                    "pct": _pct(count, len(eval_rows)),
                }

        v26_carried_counts = [
            int(ev.get("raw_v26_vault_carried_count", 0) or 0)
            for _record, ev in eval_rows
        ]
        v26_final_counts = [
            int(ev.get("raw_v26_final_count", 0) or 0)
            for _record, ev in eval_rows
        ]
        evaluation_summary["raw_v26_vault_carried_count"] = {
            "mean": round(sum(v26_carried_counts) / len(v26_carried_counts), 3) if v26_carried_counts else 0.0,
            "median": round(statistics.median(v26_carried_counts), 3) if v26_carried_counts else 0.0,
            "max": max(v26_carried_counts) if v26_carried_counts else 0,
            "shots_with_candidates": sum(1 for value in v26_carried_counts if value > 0),
        }
        evaluation_summary["raw_v26_final_count"] = {
            "mean": round(sum(v26_final_counts) / len(v26_final_counts), 3) if v26_final_counts else 0.0,
            "median": round(statistics.median(v26_final_counts), 3) if v26_final_counts else 0.0,
            "max": max(v26_final_counts) if v26_final_counts else 0,
        }

        vault_summaries = [
            ev.get("v26_vault_summary")
            for _record, ev in eval_rows
            if isinstance(ev.get("v26_vault_summary"), dict)
        ]
        if vault_summaries:
            def _median_int(field: str) -> float | None:
                values = [int(block.get(field, 0) or 0) for block in vault_summaries]
                return round(statistics.median(values), 3) if values else None
            gt_distances = [
                value for block in vault_summaries
                if (value := _finite(block.get("gt_nearest_px"))) is not None
            ]
            evaluation_summary["v26_vault"] = {
                "shots": len(vault_summaries),
                "cells_median": _median_int("cells"),
                "frames_median": _median_int("frames"),
                "frames_with_candidates_median": _median_int("frames_with_candidates"),
                "observations_median": _median_int("observations"),
                "cells_hits_ge_2_median": _median_int("cells_hits_ge_2"),
                "cells_hits_ge_3_median": _median_int("cells_hits_ge_3"),
                "gt_within_10px_pct": _pct(sum(1 for d in gt_distances if d <= 10.0), len(vault_summaries)),
                "gt_within_20px_pct": _pct(sum(1 for d in gt_distances if d <= 20.0), len(vault_summaries)),
                "gt_within_42px_pct": _pct(sum(1 for d in gt_distances if d <= 42.0), len(vault_summaries)),
            }

        # Ranker V5 PRE-TRAIN validation.  These diagnostics are attached by
        # the outer V5 wrapper after the current shot has been evaluated and
        # before that shot is learned, so this is a genuine forward test.
        v5_diags = [
            ev.get("v26_ranker_v5")
            for _record, ev in eval_rows
            if isinstance(ev.get("v26_ranker_v5"), dict)
        ]
        if v5_diags:
            base_ranks_12 = [
                int(value) for block in v5_diags
                if (value := block.get("base_gt_rank_12px")) is not None and int(value) > 0
            ]
            v5_ranks_12 = [
                int(value) for block in v5_diags
                if (value := block.get("v5_gt_rank_12px")) is not None and int(value) > 0
            ]
            paired_12 = []
            for block in v5_diags:
                base_rank = block.get("base_gt_rank_12px")
                v5_rank = block.get("v5_gt_rank_12px")
                if base_rank is not None and v5_rank is not None and int(base_rank) > 0 and int(v5_rank) > 0:
                    paired_12.append((int(base_rank), int(v5_rank)))

            train_blocks = [block.get("training", {}) for block in v5_diags if isinstance(block.get("training"), dict)]
            gate_blocks = [block.get("gate", {}) for block in v5_diags if isinstance(block.get("gate"), dict)]
            override_blocks = [block.get("override", {}) for block in v5_diags if isinstance(block.get("override"), dict)]
            latest_model = next(
                (block.get("model") for block in reversed(v5_diags) if isinstance(block.get("model"), dict)),
                {},
            )
            latest_gate = gate_blocks[-1] if gate_blocks else {}
            evaluation_summary["ranker_v5"] = {
                "shots": len(v5_diags),
                "strict_radius_px": _finite(v5_diags[-1].get("validation_radius_px")) if v5_diags else 12.0,
                "paired_shots_with_gt_12px": len(paired_12),
                "base_gt_rank_median_12px": round(statistics.median([a for a, _b in paired_12]), 3) if paired_12 else None,
                "v5_gt_rank_median_12px": round(statistics.median([b for _a, b in paired_12]), 3) if paired_12 else None,
                "base_top1_pct_12px": _pct(sum(1 for a, _b in paired_12 if a == 1), len(paired_12)),
                "v5_top1_pct_12px": _pct(sum(1 for _a, b in paired_12 if b == 1), len(paired_12)),
                "base_top3_pct_12px": _pct(sum(1 for a, _b in paired_12 if a <= 3), len(paired_12)),
                "v5_top3_pct_12px": _pct(sum(1 for _a, b in paired_12 if b <= 3), len(paired_12)),
                "v5_better_rank_count": sum(1 for a, b in paired_12 if b < a),
                "v5_worse_rank_count": sum(1 for a, b in paired_12 if b > a),
                "same_rank_count": sum(1 for a, b in paired_12 if b == a),
                "trained_shots": sum(1 for block in train_blocks if bool(block.get("trained"))),
                "skipped_no_positive": sum(1 for block in train_blocks if not bool(block.get("trained"))),
                "gate_open_shots": sum(1 for block in gate_blocks if bool(block.get("open"))),
                "override_applied_shots": sum(1 for block in override_blocks if bool(block.get("applied"))),
                "latest_gate": latest_gate,
                "latest_model": latest_model,
            }

        # Base ranker vs V4 shadow on exactly the same surviving candidate pool.
        shadow_rows = []
        for _record, ev in eval_rows:
            block = ev.get("v25_shadow_ranking")
            if not isinstance(block, dict):
                continue
            base_d = _finite(block.get("base_gt_distance_px"))
            shadow_d = _finite(block.get("v4_shadow_gt_distance_px"))
            base_rank = int(block.get("base_gt_rank", 0) or 0)
            shadow_rank = int(block.get("v4_shadow_gt_rank", 0) or 0)
            if base_d is None or shadow_d is None or base_d > 42.0 or shadow_d > 42.0:
                continue
            if base_rank <= 0 or shadow_rank <= 0:
                continue
            shadow_rows.append((base_rank, shadow_rank))
        base_ranks_shadow = [row[0] for row in shadow_rows]
        v4_ranks_shadow = [row[1] for row in shadow_rows]
        evaluation_summary["ranking_v4_shadow"] = {
            "shots": len(shadow_rows),
            "base_gt_rank_median": round(statistics.median(base_ranks_shadow), 3) if base_ranks_shadow else None,
            "v4_shadow_gt_rank_median": round(statistics.median(v4_ranks_shadow), 3) if v4_ranks_shadow else None,
            "base_top1_pct": _pct(sum(1 for value in base_ranks_shadow if value == 1), len(base_ranks_shadow)),
            "v4_shadow_top1_pct": _pct(sum(1 for value in v4_ranks_shadow if value == 1), len(v4_ranks_shadow)),
            "base_top3_pct": _pct(sum(1 for value in base_ranks_shadow if value <= 3), len(base_ranks_shadow)),
            "v4_shadow_top3_pct": _pct(sum(1 for value in v4_ranks_shadow if value <= 3), len(v4_ranks_shadow)),
            "v4_better_rank_count": sum(1 for base, shadow in shadow_rows if shadow < base),
            "v4_worse_rank_count": sum(1 for base, shadow in shadow_rows if shadow > base),
            "same_rank_count": sum(1 for base, shadow in shadow_rows if shadow == base),
        }

        v4_rows = []
        for _record, ev in eval_rows:
            block = ev.get("v24_ranking")
            if not isinstance(block, dict):
                continue
            distance = _finite(block.get("gt_distance_px"))
            rank = int(block.get("gt_rank", 0) or 0)
            if distance is None or distance > 42.0 or rank <= 0:
                continue
            v4_rows.append(block)

        v4_ranks = [int(block.get("gt_rank", 0) or 0) for block in v4_rows]
        patch_delta_keys = [
            "v24_combined_score",
            "v24_patch_prior",
            "ranker_v4_score",
            "v24_patch_core_to_outer",
            "v24_patch_compactness",
            "v24_patch_centeredness",
            "v24_patch_isotropy",
            "v24_patch_bipolar",
            "v24_patch_local_snr",
            "shot_accumulator_hits",
            "shot_accumulator_stability",
        ]
        delta_summary = {}
        for key in patch_delta_keys:
            values = [
                value
                for block in v4_rows
                if isinstance(block.get("selected_minus_gt"), dict)
                and (value := _finite(block["selected_minus_gt"].get(key))) is not None
            ]
            delta_summary[key] = (
                round(statistics.median(values), 5) if values else None
            )

        evaluation_summary["ranking_v4"] = {
            "shots_with_gt_in_pool_42px": len(v4_rows),
            "gt_rank_median": round(statistics.median(v4_ranks), 3) if v4_ranks else None,
            "gt_rank_mean": round(sum(v4_ranks) / len(v4_ranks), 3) if v4_ranks else None,
            "gt_rank_top1_pct": _pct(sum(1 for value in v4_ranks if value == 1), len(v4_ranks)),
            "gt_rank_top3_pct": _pct(sum(1 for value in v4_ranks if value <= 3), len(v4_ranks)),
            "selected_minus_gt_medians": delta_summary,
        }

        # Ranking diagnostics only make sense when a GT candidate actually
        # survived into the ranked top-K list.
        ranking_rows = []
        for _record, ev in eval_rows:
            distance = _finite(ev.get("ranking_gt_candidate_distance_px"))
            gt_candidate = ev.get("ranking_gt_candidate")
            selected_candidate = ev.get("ranking_selected_candidate")
            if (
                distance is not None
                and distance <= 42.0
                and isinstance(gt_candidate, dict)
                and isinstance(selected_candidate, dict)
            ):
                ranking_rows.append((ev, gt_candidate, selected_candidate))

        gt_ranks = [
            int(gt.get("rank", 0) or 0)
            for _ev, gt, _selected in ranking_rows
            if int(gt.get("rank", 0) or 0) > 0
        ]
        margins = [
            value
            for ev, _gt, _selected in ranking_rows
            if (value := _finite(ev.get("ranking_score_margin_selected_minus_gt"))) is not None
        ]
        ai_deltas = []
        heuristic_deltas = []
        for _ev, gt, selected in ranking_rows:
            gt_ai = _finite(gt.get("ai_score"))
            selected_ai = _finite(selected.get("ai_score"))
            gt_h = _finite(gt.get("heuristic_score"))
            selected_h = _finite(selected.get("heuristic_score"))
            if gt_ai is not None and selected_ai is not None:
                ai_deltas.append(selected_ai - gt_ai)
            if gt_h is not None and selected_h is not None:
                heuristic_deltas.append(selected_h - gt_h)

        evaluation_summary["ranking"] = {
            "shots_with_gt_in_ranked_42px": len(ranking_rows),
            "gt_rank_median": round(statistics.median(gt_ranks), 3) if gt_ranks else None,
            "gt_rank_mean": round(sum(gt_ranks) / len(gt_ranks), 3) if gt_ranks else None,
            "gt_rank_top1_pct": _pct(sum(1 for value in gt_ranks if value == 1), len(gt_ranks)),
            "gt_rank_top3_pct": _pct(sum(1 for value in gt_ranks if value <= 3), len(gt_ranks)),
            "selected_minus_gt_combined_margin_median": (
                round(statistics.median(margins), 5) if margins else None
            ),
            "selected_minus_gt_ai_score_median": (
                round(statistics.median(ai_deltas), 5) if ai_deltas else None
            ),
            "selected_minus_gt_heuristic_score_median": (
                round(statistics.median(heuristic_deltas), 5) if heuristic_deltas else None
            ),
        }

    summary["overall"]["evaluation_funnel"] = evaluation_summary

    gt_abs = [
        value
        for record in labelled
        if (value := _finite(record.get("gt_signal_max", {}).get("absdiff"))) is not None
    ]
    gt_z = [
        value
        for record in labelled
        if (value := _finite(record.get("gt_signal_max", {}).get("zscore"))) is not None
    ]
    gt_sal = [
        value
        for record in labelled
        if (value := _finite(record.get("gt_signal_max", {}).get("saliency"))) is not None
    ]
    gt_margin = [
        value
        for record in labelled
        if (value := _finite(record.get("gt_signal_max", {}).get("saliency_minus_threshold"))) is not None
    ]

    summary["overall"]["gt_signal"] = {
        "absdiff_median": round(statistics.median(gt_abs), 3) if gt_abs else None,
        "zscore_median": round(statistics.median(gt_z), 3) if gt_z else None,
        "saliency_median": round(statistics.median(gt_sal), 3) if gt_sal else None,
        "saliency_minus_threshold_median": (
            round(statistics.median(gt_margin), 3) if gt_margin else None
        ),
    }

    # ------------------------------------------------------------------
    # V2.5 localisation / geometry diagnostics.
    # ------------------------------------------------------------------
    probe_rows = []
    for record in labelled:
        v25 = record.get("v25", {}) if isinstance(record.get("v25"), dict) else {}
        probe = v25.get("gt_local_probe")
        gt = record.get("ground_truth", {})
        if not isinstance(probe, dict) or not probe.get("found") or not isinstance(gt, dict):
            continue
        dx, dy, distance = _finite(probe.get("dx")), _finite(probe.get("dy")), _finite(probe.get("distance_px"))
        sx, sy = _finite(gt.get("screen_x")), _finite(gt.get("screen_y"))
        if dx is None or dy is None or distance is None or sx is None or sy is None:
            continue
        probe_rows.append((sx, sy, dx, dy, distance))

    localisation: dict[str, Any] = {"shots": len(probe_rows), "zones": {}}
    if probe_rows:
        dxs = [row[2] for row in probe_rows]
        dys = [row[3] for row in probe_rows]
        distances = [row[4] for row in probe_rows]
        localisation.update({
            "median_dx": round(statistics.median(dxs), 3),
            "median_dy": round(statistics.median(dys), 3),
            "median_distance_px": round(statistics.median(distances), 3),
            "p90_distance_px": round(
                sorted(distances)[
                    min(len(distances)-1, max(0, math.ceil(0.90*len(distances))-1))
                ],
                3,
            ),
            "within_10px_pct": _pct(sum(1 for value in distances if value <= 10.0), len(distances)),
            "within_20px_pct": _pct(sum(1 for value in distances if value <= 20.0), len(distances)),
            "within_42px_pct": _pct(sum(1 for value in distances if value <= 42.0), len(distances)),
        })
        sx_values, sy_values = [row[0] for row in probe_rows], [row[1] for row in probe_rows]
        min_x, max_x, min_y, max_y = min(sx_values), max(sx_values), min(sy_values), max(sy_values)
        def zone_index(value: float, low: float, high: float) -> int:
            if high <= low + 1e-9:
                return 1
            return min(2, max(0, int(3.0 * (value-low) / (high-low+1e-9))))
        x_names, y_names = ("left", "centre", "right"), ("top", "middle", "bottom")
        zones: dict[str, list[tuple[float,float,float]]] = defaultdict(list)
        for sx, sy, dx, dy, distance in probe_rows:
            zone = f"{y_names[zone_index(sy,min_y,max_y)]}_{x_names[zone_index(sx,min_x,max_x)]}"
            zones[zone].append((dx,dy,distance))
        for zone, rows in sorted(zones.items()):
            localisation["zones"][zone] = {
                "shots": len(rows),
                "median_dx": round(statistics.median([r[0] for r in rows]), 3),
                "median_dy": round(statistics.median([r[1] for r in rows]), 3),
                "median_distance_px": round(statistics.median([r[2] for r in rows]), 3),
            }
    summary["overall"]["v25_localisation"] = localisation

    # Did the additive centre-refinement actually move a V2.4 tile hypothesis
    # closer to GT? Compare only shots where both paths produced a vector.
    refinement_deltas = []
    moved_into_10 = 0
    moved_into_20 = 0
    better = worse = same = 0
    for record in labelled:
        v25 = record.get("v25", {}) if isinstance(record.get("v25"), dict) else {}
        vectors = v25.get("best_vectors", {}) if isinstance(v25.get("best_vectors"), dict) else {}
        old = vectors.get("v24_tile")
        new = vectors.get("v25_refined_tile")
        if not isinstance(old, dict) or not isinstance(new, dict):
            continue
        old_d = _finite(old.get("distance_px"))
        new_d = _finite(new.get("distance_px"))
        if old_d is None or new_d is None:
            continue
        delta = old_d - new_d
        refinement_deltas.append(delta)
        if delta > 0.25:
            better += 1
        elif delta < -0.25:
            worse += 1
        else:
            same += 1
        if old_d > 10.0 and new_d <= 10.0:
            moved_into_10 += 1
        if old_d > 20.0 and new_d <= 20.0:
            moved_into_20 += 1
    summary["overall"]["v25_refinement_effect"] = {
        "shots_compared": len(refinement_deltas),
        "median_improvement_px": round(statistics.median(refinement_deltas), 3) if refinement_deltas else None,
        "mean_improvement_px": round(sum(refinement_deltas)/len(refinement_deltas), 3) if refinement_deltas else None,
        "better_count": better,
        "worse_count": worse,
        "same_count": same,
        "moved_into_10px": moved_into_10,
        "moved_into_20px": moved_into_20,
    }

    accumulator_rows = []
    for record in labelled:
        v25 = record.get("v25", {}) if isinstance(record.get("v25"), dict) else {}
        block = v25.get("shadow_accumulator")
        if isinstance(block, dict):
            accumulator_rows.append(block)
    gt_clusters = [row.get("gt_cluster") for row in accumulator_rows if isinstance(row.get("gt_cluster"), dict)]
    summary["overall"]["v25_shadow_accumulator"] = {
        "shots": len(accumulator_rows),
        "shots_with_gt_cluster_42px": len(gt_clusters),
        "gt_cluster_pct": _pct(len(gt_clusters), len(accumulator_rows)),
        "gt_cluster_hits_ge_2": sum(1 for block in gt_clusters if int(block.get("hits",0) or 0) >= 2),
        "gt_cluster_hits_ge_3": sum(1 for block in gt_clusters if int(block.get("hits",0) or 0) >= 3),
        "gt_cluster_hits_ge_4": sum(1 for block in gt_clusters if int(block.get("hits",0) or 0) >= 4),
        "gt_hits_median": round(statistics.median([int(block.get("hits",0) or 0) for block in gt_clusters]),3) if gt_clusters else None,
        "gt_jitter_median_px": round(statistics.median([_finite(block.get("jitter_px")) or 0.0 for block in gt_clusters]),3) if gt_clusters else None,
        "frames_median": round(statistics.median([int(row.get("frames",0) or 0) for row in accumulator_rows]),3) if accumulator_rows else None,
        "clusters_created_median": round(statistics.median([int(row.get("clusters_created",0) or 0) for row in accumulator_rows]),3) if accumulator_rows else None,
    }

    # Latest persisted V4 model summary copied into diagnostics by the V2.4
    # extension.
    model_summaries = [
        record.get("ranker_v4_summary")
        for record in labelled
        if isinstance(record.get("ranker_v4_summary"), dict)
    ]
    if model_summaries:
        summary["ranker_v4_model"] = model_summaries[-1]

    seeds = summary.get("benchmark_seeds", [])
    if seeds:
        expected = len(seeds) * 100
        missing_evaluation = sum(
            1
            for record in labelled
            if str(record.get("benchmark_integrity", "")) == "missing_evaluation"
            or bool(_evaluation(record).get("integrity_placeholder", False))
        )
        summary["benchmark_integrity"] = {
            "expected_diagnostics": expected,
            "actual_diagnostics": len(labelled),
            "missing_diagnostics": max(0, expected - len(labelled)),
            "missing_evaluation_records": missing_evaluation,
            "complete": len(labelled) == expected and missing_evaluation == 0,
        }

    # Explicit counts for the two changes introduced in V2.1.
    bank_recovered = sum(
        1 for record in labelled
        if not _found(_nearest(record, "v2_frame"), 42.0)
        and _found(_nearest(record, "v2"), 42.0)
    )
    v2_lost_merge = sum(
        1 for record in labelled
        if _found(_nearest(record, "v2"), 42.0)
        and not _found(_nearest(record, "merged"), 42.0)
    )
    legacy_lost_merge = sum(
        1 for record in labelled
        if _found(_nearest(record, "legacy"), 42.0)
        and not _found(_nearest(record, "merged"), 42.0)
    )
    summary["overall"]["v22_changes"] = {
        "candidate_bank_recovered_42px": bank_recovered,
        "v2_lost_during_merge_42px": v2_lost_merge,
        "legacy_lost_during_merge_42px": legacy_lost_merge,
    }

    for background, rows in sorted(by_background.items()):
        entry: dict[str, Any] = {"shots": len(rows)}
        for radius in match_radii:
            key = f"within_{int(radius)}px"
            entry[key] = {
                source: _pct(
                    sum(1 for record in rows if _found(_nearest(record, source), radius)),
                    len(rows),
                )
                for source in detector_sources
            }
        entry["pipeline_classification_42px"] = dict(
            Counter(classify_pipeline_loss(record, 42.0) for record in rows)
        )
        summary["by_background"][background] = entry

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print()
    print("=" * 88)
    print("DETECTOR V2.6 ANALYSIS")
    print("=" * 88)
    print(f"Diagnostics: {summary['records_total']}")
    print(f"With synthetic ground truth: {summary['records_with_ground_truth']}")
    print(f"With evaluation funnel: {summary['records_with_evaluation_funnel']}")
    sessions = summary.get("runtime_session_ids", [])
    commits = summary.get("git_commits", [])
    if sessions:
        print(f"Detector runtime session: {', '.join(sessions)}")
    if commits:
        print(f"Git commit(s): {', '.join(commits)}")
    seeds = summary.get("benchmark_seeds", [])
    if seeds:
        if len(seeds) <= 12:
            print(f"Deterministic benchmark seeds: {seeds}")
        else:
            print(f"Deterministic benchmark seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} unique)")
    integrity = summary.get("benchmark_integrity", {})
    if integrity:
        state = "COMPLETE" if integrity.get("complete") else "INCOMPLETE"
        print(
            f"Benchmark integrity: {state}  "
            f"{integrity.get('actual_diagnostics')}/{integrity.get('expected_diagnostics')} diagnostics"
        )
        if integrity.get("missing_evaluation_records", 0):
            print(
                "  labelled diagnostics missing real F2 evaluation: "
                f"{integrity.get('missing_evaluation_records')}"
            )

    model = summary.get("ranker_v4_model", {})
    if model:
        print()
        print("Ranker V4 model:")
        print(f"  positive shots : {model.get('positive_shots')}")
        print(f"  pair updates   : {model.get('pair_updates')}")
        print(f"  last loss      : {model.get('last_loss')}")
        print(f"  model weight   : {model.get('effective_weight')}")
        strongest = model.get("strongest_weights", [])
        if strongest:
            print("  strongest w    : " + ", ".join(f"{k}={float(v):.3f}" for k, v in strongest[:6]))

    overall = summary.get("overall", {})
    for radius in (10, 20, 42):
        block = overall.get(f"within_{radius}px", {})
        print()
        print(f"BEST/EVER candidate recall within {radius}px:")
        for source in (
            "legacy", "v2_frame", "v2", "merged",
            "v24_tile", "v24_accumulator", "v24_final",
            "v25_refined_tile", "v25_final",
            "v26_vault", "v26_final",
        ):
            data = block.get(source, {})
            label = {
                "legacy": "legacy",
                "v2_frame": "v2 frame",
                "v2": "v2 bank",
                "merged": "merged",
                "v24_tile": "v24 tile",
                "v24_accumulator": "v24 accum",
                "v24_final": "v24 final",
                "v25_refined_tile": "v25 refine",
                "v25_final": "v25 final",
                "v26_vault": "v26 vault",
                "v26_final": "v26 final",
            }[source]
            print(
                f"  {label:10s}: {data.get('count', 0):5d} "
                f"({data.get('pct', 0.0):6.2f}%)"
            )

    changes = overall.get("v22_changes", {})
    print()
    print("Legacy V2/V2.3 candidate preservation (42px):")
    print(
        "  recovered by candidate bank : "
        f"{changes.get('candidate_bank_recovered_42px', 0)}"
    )
    print(
        "  V2 candidates lost in merge : "
        f"{changes.get('v2_lost_during_merge_42px', 0)}"
    )
    print(
        "  V1 candidates lost in merge : "
        f"{changes.get('legacy_lost_during_merge_42px', 0)}"
    )

    ev = overall.get("evaluation_funnel", {})
    if ev.get("shots", 0):
        print()
        print("ACTUAL F2 EVALUATION funnel within 42px:")
        for name in ("raw", "filtered", "ranked", "selected"):
            data = ev.get(f"{name}_within_42px", {})
            print(
                f"  {name:10s}: {data.get('count', 0):5d} "
                f"({data.get('pct', 0.0):6.2f}%)"
            )

        print()
        print("Detector provenance in ACTUAL F2 raw snapshot (42px):")
        for name, label in (
            ("raw_v1", "V1 present"),
            ("raw_v2", "V2 present"),
            ("raw_v2_confirmed", "V2 confirmed"),
            ("raw_v2_carried", "V2 carried"),
        ):
            data = ev.get(f"{name}_within_42px", {})
            print(
                f"  {label:14s}: {data.get('count', 0):5d} "
                f"({data.get('pct', 0.0):6.2f}%)"
            )
        carried = ev.get("raw_v2_bank_carried_count", {})
        if isinstance(carried, dict):
            print(
                "  V2 carried candidates / snapshot: "
                f"mean={carried.get('mean', 0.0)} "
                f"max={carried.get('max', 0)} "
                f"shots={carried.get('shots_with_carried_candidates', 0)}"
            )
        generic_carried = ev.get("raw_candidate_bank_carried_count", {})
        if isinstance(generic_carried, dict):
            recovered = ev.get("raw_candidate_bank_carried_within_42px", {})
            print(
                "  hybrid-bank carried / snapshot: "
                f"mean={generic_carried.get('mean', 0.0)} "
                f"max={generic_carried.get('max', 0)} "
                f"shots={generic_carried.get('shots_with_carried_candidates', 0)} "
                f"GT@42px={recovered.get('count', 0)} ({recovered.get('pct', 0.0):.2f}%)"
            )

        print()
        print("V2.4 provenance in ACTUAL F2 raw snapshot (42px):")
        for name, label in (
            ("raw_v24_tile", "tile probe"),
            ("raw_v24_accumulator", "shot accumulator"),
        ):
            data = ev.get(f"{name}_within_42px", {})
            counts = ev.get(f"{name}_count", {})
            print(
                f"  {label:18s}: GT={data.get('count', 0):5d} "
                f"({data.get('pct', 0.0):6.2f}%) "
                f"mean candidates={counts.get('mean', 0.0)} max={counts.get('max', 0)}"
            )

        print()
        print("V2.5 additions in ACTUAL F2 raw snapshot (42px):")
        refined = ev.get("raw_v25_refined_tile_within_42px", {})
        refined_counts = ev.get("raw_v25_refined_tile_count", {})
        print(
            "  refined tile       : "
            f"GT={refined.get('count', 0):5d} ({refined.get('pct', 0.0):6.2f}%) "
            f"mean candidates={refined_counts.get('mean', 0.0)} "
            f"max={refined_counts.get('max', 0)}"
        )

        print()
        print("V2.6 SHOT VAULT in ACTUAL F2 raw snapshot (42px):")
        vault_gt = ev.get("raw_v26_vault_carried_within_42px", {})
        final_gt = ev.get("raw_v26_final_within_42px", {})
        carried_counts = ev.get("raw_v26_vault_carried_count", {})
        final_counts = ev.get("raw_v26_final_count", {})
        print(
            "  carried history    : "
            f"GT={vault_gt.get('count',0):5d} ({vault_gt.get('pct',0.0):6.2f}%) "
            f"mean={carried_counts.get('mean',0.0)} "
            f"median={carried_counts.get('median',0.0)} "
            f"max={carried_counts.get('max',0)}"
        )
        print(
            "  V2.6 final raw     : "
            f"GT={final_gt.get('count',0):5d} ({final_gt.get('pct',0.0):6.2f}%) "
            f"mean candidates={final_counts.get('mean',0.0)} "
            f"median={final_counts.get('median',0.0)} "
            f"max={final_counts.get('max',0)}"
        )
        vault = ev.get("v26_vault", {})
        if isinstance(vault, dict) and vault.get("shots",0):
            print(
                "  vault GT <=10/20/42: "
                f"{vault.get('gt_within_10px_pct')}% / "
                f"{vault.get('gt_within_20px_pct')}% / "
                f"{vault.get('gt_within_42px_pct')}%"
            )
            print(
                "  cells / frames med : "
                f"{vault.get('cells_median')} / {vault.get('frames_median')} "
                f"(observations med={vault.get('observations_median')})"
            )

        ranker_v5 = ev.get("ranker_v5", {})
        if isinstance(ranker_v5, dict) and ranker_v5.get("shots",0):
            print()
            print("RANKER V5 strict PRE-TRAIN validation (actual candidates):")
            print(f"  shots with diagnostic: {ranker_v5.get('shots')}")
            print(f"  strict radius        : {ranker_v5.get('strict_radius_px')} px")
            print(f"  paired GT shots      : {ranker_v5.get('paired_shots_with_gt_12px')}")
            print(
                "  BASE / V5 median rank: "
                f"{ranker_v5.get('base_gt_rank_median_12px')} / "
                f"{ranker_v5.get('v5_gt_rank_median_12px')}"
            )
            print(
                "  BASE top1/top3      : "
                f"{ranker_v5.get('base_top1_pct_12px')}% / "
                f"{ranker_v5.get('base_top3_pct_12px')}%"
            )
            print(
                "  V5 top1/top3        : "
                f"{ranker_v5.get('v5_top1_pct_12px')}% / "
                f"{ranker_v5.get('v5_top3_pct_12px')}%"
            )
            print(
                "  V5 better/worse/same: "
                f"{ranker_v5.get('v5_better_rank_count',0)}/"
                f"{ranker_v5.get('v5_worse_rank_count',0)}/"
                f"{ranker_v5.get('same_rank_count',0)}"
            )
            print(
                "  trained / skipped   : "
                f"{ranker_v5.get('trained_shots',0)} / "
                f"{ranker_v5.get('skipped_no_positive',0)}"
            )
            gate = ranker_v5.get("latest_gate", {})
            if isinstance(gate, dict):
                print(
                    "  latest auto-gate     : "
                    f"open={gate.get('open')} eligible={gate.get('eligible_shots')} "
                    f"BASE={gate.get('base_top1_pct')}% "
                    f"V5={gate.get('v5_top1_pct')}% "
                    f"adv={gate.get('advantage_pp')}pp"
                )
            print(
                "  gate-open / overrides: "
                f"{ranker_v5.get('gate_open_shots',0)} / "
                f"{ranker_v5.get('override_applied_shots',0)}"
            )

        shadow = ev.get("ranking_v4_shadow", {})
        if isinstance(shadow, dict) and shadow.get("shots", 0):
            print()
            print("V2.5 ranking shadow comparison (GT within 42px):")
            print(f"  shots                 : {shadow.get('shots')}")
            print(f"  BASE median rank      : {shadow.get('base_gt_rank_median')}")
            print(f"  V4 shadow median rank : {shadow.get('v4_shadow_gt_rank_median')}")
            print(f"  BASE top-1 / top-3    : {shadow.get('base_top1_pct')}% / {shadow.get('base_top3_pct')}%")
            print(f"  V4 top-1 / top-3      : {shadow.get('v4_shadow_top1_pct')}% / {shadow.get('v4_shadow_top3_pct')}%")
            print(
                "  V4 better/worse/same  : "
                f"{shadow.get('v4_better_rank_count',0)}/"
                f"{shadow.get('v4_worse_rank_count',0)}/"
                f"{shadow.get('same_rank_count',0)}"
            )

        ranking_v4 = ev.get("ranking_v4", {})
        # In V2.5 the separate shadow block above is authoritative. The old
        # v24_ranking block now describes the actual BASE pool and would be
        # misleading if printed as active V4 ranking.
        if (
            not (isinstance(shadow, dict) and shadow.get("shots", 0))
            and isinstance(ranking_v4, dict)
            and ranking_v4.get("shots_with_gt_in_pool_42px", 0)
        ):
            print()
            print("Ranker V4 quality when GT exists in full pool (42px):")
            print(f"  shots                 : {ranking_v4.get('shots_with_gt_in_pool_42px')}")
            print(f"  GT median rank        : {ranking_v4.get('gt_rank_median')}")
            print(f"  GT mean rank          : {ranking_v4.get('gt_rank_mean')}")
            print(f"  GT rank=1             : {ranking_v4.get('gt_rank_top1_pct')}%")
            print(f"  GT rank<=3            : {ranking_v4.get('gt_rank_top3_pct')}%")
            deltas = ranking_v4.get("selected_minus_gt_medians", {})
            for key in (
                "v24_combined_score",
                "v24_patch_prior",
                "ranker_v4_score",
                "v24_patch_core_to_outer",
                "v24_patch_compactness",
                "v24_patch_centeredness",
                "v24_patch_isotropy",
                "v24_patch_local_snr",
            ):
                if key in deltas:
                    print(f"  selected-GT {key:26s}: {deltas.get(key)}")

        ranking = ev.get("ranking", {})
        if isinstance(ranking, dict) and ranking.get("shots_with_gt_in_ranked_42px", 0):
            print()
            print("Ranking quality when GT survived into ranked list (42px):")
            print(f"  shots                 : {ranking.get('shots_with_gt_in_ranked_42px')}")
            print(f"  GT median rank        : {ranking.get('gt_rank_median')}")
            print(f"  GT mean rank          : {ranking.get('gt_rank_mean')}")
            print(f"  GT rank=1             : {ranking.get('gt_rank_top1_pct')}%")
            print(f"  GT rank<=3            : {ranking.get('gt_rank_top3_pct')}%")
            print(
                "  selected-GT score med : "
                f"{ranking.get('selected_minus_gt_combined_margin_median')}"
            )
            print(
                "  selected-GT AI med    : "
                f"{ranking.get('selected_minus_gt_ai_score_median')}"
            )
            print(
                "  selected-GT heuristic : "
                f"{ranking.get('selected_minus_gt_heuristic_score_median')}"
            )

        print()
        print("Where GT was lost (42px):")
        classes = overall.get("pipeline_classification_42px", {})
        for name, count in sorted(classes.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {name:38s} {count:5d}")

    localisation = overall.get("v25_localisation", {})
    if isinstance(localisation, dict) and localisation.get("shots", 0):
        print()
        print("V2.5 GEOMETRY / LOCALISATION probe (benchmark-only):")
        print(f"  shots                 : {localisation.get('shots')}")
        print(f"  median dx / dy        : {localisation.get('median_dx')} / {localisation.get('median_dy')} px")
        print(f"  median distance       : {localisation.get('median_distance_px')} px")
        print(f"  p90 distance          : {localisation.get('p90_distance_px')} px")
        print(
            "  within 10/20/42px    : "
            f"{localisation.get('within_10px_pct')}% / "
            f"{localisation.get('within_20px_pct')}% / "
            f"{localisation.get('within_42px_pct')}%"
        )
        zones = localisation.get("zones", {})
        if isinstance(zones, dict):
            print("  zones (median dx,dy | distance):")
            for zone, data in sorted(zones.items()):
                print(
                    f"    {zone:20s} n={data.get('shots',0):4d} "
                    f"dx={data.get('median_dx'):7.2f} "
                    f"dy={data.get('median_dy'):7.2f} "
                    f"d={data.get('median_distance_px'):7.2f}"
                )

    refinement = overall.get("v25_refinement_effect", {})
    if isinstance(refinement, dict) and refinement.get("shots_compared", 0):
        print()
        print("V2.5 TILE CENTRE refinement effect (BEST/EVER):")
        print(f"  shots compared        : {refinement.get('shots_compared')}")
        print(f"  median improvement    : {refinement.get('median_improvement_px')} px")
        print(f"  mean improvement      : {refinement.get('mean_improvement_px')} px")
        print(
            "  better / worse / same : "
            f"{refinement.get('better_count')} / "
            f"{refinement.get('worse_count')} / "
            f"{refinement.get('same_count')}"
        )
        print(f"  moved into <=10 px    : {refinement.get('moved_into_10px')}")
        print(f"  moved into <=20 px    : {refinement.get('moved_into_20px')}")

    accumulator = overall.get("v25_shadow_accumulator", {})
    if isinstance(accumulator, dict) and accumulator.get("shots", 0):
        print()
        print("V2.5 SHADOW shot accumulator (does NOT alter candidates):")
        print(f"  shots                 : {accumulator.get('shots')}")
        print(
            "  GT cluster <=42px     : "
            f"{accumulator.get('shots_with_gt_cluster_42px')} "
            f"({accumulator.get('gt_cluster_pct')}%)"
        )
        print(
            "  GT hits >=2 / >=3 / >=4: "
            f"{accumulator.get('gt_cluster_hits_ge_2')} / "
            f"{accumulator.get('gt_cluster_hits_ge_3')} / "
            f"{accumulator.get('gt_cluster_hits_ge_4')}"
        )
        print(f"  GT hit-count median   : {accumulator.get('gt_hits_median')}")
        print(f"  GT jitter median      : {accumulator.get('gt_jitter_median_px')} px")
        print(f"  frames / shot median  : {accumulator.get('frames_median')}")
        print(f"  clusters / shot median: {accumulator.get('clusters_created_median')}")

    gt_signal = overall.get("gt_signal", {})
    print()
    print("Median signal around synthetic ground truth:")
    print(f"  absdiff                    : {gt_signal.get('absdiff_median')}")
    print(f"  z-score                    : {gt_signal.get('zscore_median')}")
    print(f"  saliency                   : {gt_signal.get('saliency_median')}")
    print(f"  saliency - frame threshold : {gt_signal.get('saliency_minus_threshold_median')}")

    print()
    print("Detector miss / recovery classification (42px):")
    classes = overall.get("detector_classification_42px", {})
    for name, count in sorted(classes.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {name:38s} {count:5d}")

    by_background = summary.get("by_background", {})
    if by_background:
        print()
        print("By background (merged BEST/EVER recall within 42px):")
        for background, data in by_background.items():
            block = data.get("within_42px", {})
            print(
                f"  {background:16s} shots={data.get('shots', 0):5d} "
                f"legacy={block.get('legacy', 0.0):6.2f}% "
                f"v2frame={block.get('v2_frame', 0.0):6.2f}% "
                f"v2bank={block.get('v2', 0.0):6.2f}% "
                f"merged={block.get('merged', 0.0):6.2f}%"
            )

    print("=" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse machine-readable Detector V2 through V2.6 shot diagnostics."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_PATH),
        help="Path to shot_diagnostics.jsonl",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_SUMMARY),
        help="JSON summary output path",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyse all historical runtime sessions instead of only the latest.",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Analyse one explicit runtime_session_id.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    all_records = load_records(path)
    if not all_records:
        print(f"No Detector V2 diagnostics found in: {path}")
        return

    records, selected_session = select_records(
        all_records,
        session_id=args.session,
        include_all=bool(args.all),
    )
    if not records:
        print(f"No diagnostics matched session: {args.session}")
        return

    if selected_session:
        print(f"Analysing latest detector runtime session: {selected_session}")
    elif args.all:
        print("Analysing ALL detector runtime sessions.")

    summary = summarise(records)
    print_summary(summary)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Machine-readable summary written to: {output}")


if __name__ == "__main__":
    main()
