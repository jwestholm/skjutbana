from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.engine.ai.hole_patch_ensemble_v215 import HolePatchEnsembleV215
from src.engine.offline.candidate_pack_v216 import CandidatePackV216, discover_candidate_packs


SCHEMA_VERSION = "2.16"
DEFAULT_ENSEMBLE_CONFIG = Path("content/ai/reports/v215/hole_v215_ensemble.json")
DEFAULT_V9_MODEL = Path("content/ai/ranker_v9_offline.json")
DEFAULT_REPORT = Path("content/ai/reports/v216/candidate_shadow_report.json")
DEFAULT_FUSION = Path("content/ai/reports/v216/candidate_fusion_v216.json")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else float(default)
    except Exception:
        return float(default)


def _percentile(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return np.full(arr.shape, 0.5, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    result = np.zeros(arr.shape, dtype=np.float64)
    start = 0
    while start < len(arr):
        end = start + 1
        while end < len(arr) and abs(float(arr[order[end]]) - float(arr[order[start]])) <= 1e-12:
            end += 1
        rank = 0.5 * (start + end - 1) / float(len(arr) - 1)
        result[order[start:end]] = rank
        start = end
    return result


def temporal_candidate_score_v216(pre_patch: np.ndarray | None, post_patches: np.ndarray) -> dict[str, float]:
    """Small candidate-centred temporal evidence source.

    The score is deliberately simple and transparent.  It is not a replacement
    for V2.12 full-frame temporal overlays; it lets V2.16 ask whether persistent
    before/after change at *the same real detector candidate* adds independent
    ranking information.
    """

    posts = np.asarray(post_patches)
    if pre_patch is None or posts.ndim != 3 or posts.shape[0] == 0:
        return {"score": 0.0, "center_absdiff": 0.0, "center_darkening": 0.0, "persistence": 0.0, "locality": 0.0}
    pre = np.asarray(pre_patch, dtype=np.float32)
    if pre.ndim != 2 or pre.shape != posts.shape[1:]:
        return {"score": 0.0, "center_absdiff": 0.0, "center_darkening": 0.0, "persistence": 0.0, "locality": 0.0}
    posts_f = posts.astype(np.float32)
    delta = np.abs(posts_f - pre[None, ...])
    dark = np.maximum(pre[None, ...] - posts_f, 0.0)
    persistent_abs = np.median(delta, axis=0)
    persistent_dark = np.median(dark, axis=0)

    h, w = pre.shape
    cy, cx = h // 2, w // 2
    radius = max(2, min(h, w) // 10)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    center_mask = np.zeros((h, w), dtype=bool)
    center_mask[y0:y1, x0:x1] = True
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ring_mask = (rr >= radius * 2.0) & (rr <= radius * 3.6)
    if not np.any(ring_mask):
        ring_mask = ~center_mask

    center_abs = float(np.mean(persistent_abs[center_mask])) / 255.0
    ring_abs = float(np.mean(persistent_abs[ring_mask])) / 255.0
    center_dark = float(np.mean(persistent_dark[center_mask])) / 255.0
    ring_dark = float(np.mean(persistent_dark[ring_mask])) / 255.0
    per_frame_center = np.mean(delta[:, center_mask], axis=1) / 255.0
    persistence = float(np.mean(per_frame_center >= 0.025))
    locality = max(0.0, center_abs - ring_abs)
    dark_locality = max(0.0, center_dark - ring_dark)
    raw = 4.0 * locality + 2.5 * dark_locality + 0.8 * center_abs + 0.35 * persistence
    score = 1.0 - math.exp(-max(0.0, raw))
    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "center_absdiff": center_abs,
        "center_darkening": center_dark,
        "persistence": persistence,
        "locality": locality,
    }


def _load_v9():
    try:
        from src.engine.ai.ranker_v9 import RankerV9ShadowModel

        model = RankerV9ShadowModel(DEFAULT_V9_MODEL)
        return model if bool(getattr(model, "loaded", False)) else None
    except Exception:
        return None


def _ranker_v9_scores(candidates: Sequence[dict[str, Any]]) -> list[float] | None:
    model = _load_v9()
    if model is None:
        return None
    try:
        ranked = model.rank(candidates)
    except Exception:
        return None
    score_by_marker: dict[tuple[int, int], list[float]] = {}
    for candidate in ranked:
        marker = (int(round(_safe_float(candidate.get("camera_x")) * 2)), int(round(_safe_float(candidate.get("camera_y")) * 2)))
        score_by_marker.setdefault(marker, []).append(_safe_float(candidate.get("ranker_v9_score")))
    result: list[float] = []
    for candidate in candidates:
        marker = (int(round(_safe_float(candidate.get("camera_x")) * 2)), int(round(_safe_float(candidate.get("camera_y")) * 2)))
        values = score_by_marker.get(marker)
        result.append(values.pop(0) if values else 0.0)
    return result


@dataclass
class CandidateShotEvidenceV216:
    json_path: str
    session_id: str
    round_id: int
    provisional_split: bool
    candidates: list[dict[str, Any]]
    gt_xy: tuple[float, float] | None
    pool_oracle_20: bool
    pool_oracle_42: bool


def score_pack_v216(pack: CandidatePackV216, ensemble: HolePatchEnsembleV215) -> CandidateShotEvidenceV216:
    candidates = list(pack.candidates)
    n = len(candidates)
    post = np.asarray(pack.post_patches)
    pre = np.asarray(pack.pre_patches)
    if n != post.shape[0]:
        raise RuntimeError(f"candidate/image count mismatch in {pack.json_path}: {n} vs {post.shape[0]}")

    # Hole-AI sees each available post frame and is aggregated robustly.  This
    # prevents a single camera frame from dominating the candidate score.
    std_matrix: list[list[float]] = [[] for _ in range(n)]
    mild_matrix: list[list[float]] = [[] for _ in range(n)]
    fused_matrix: list[list[float]] = [[] for _ in range(n)]
    off_x: list[list[float]] = [[] for _ in range(n)]
    off_y: list[list[float]] = [[] for _ in range(n)]
    for frame_index in range(post.shape[1] if post.ndim == 4 else 0):
        evidence = ensemble.score_patches([post[i, frame_index] for i in range(n)]) if n else []
        for i, item in enumerate(evidence):
            std_matrix[i].append(float(item.standard_probability))
            mild_matrix[i].append(float(item.mild_probability))
            fused_matrix[i].append(float(item.fused_probability))
            off_x[i].append(float(item.fused_offset_px[0]))
            off_y[i].append(float(item.fused_offset_px[1]))

    source_candidates = [dict((row.get("candidate") or {})) for row in candidates]
    for source, row in zip(source_candidates, candidates):
        source.setdefault("camera_x", row.get("camera_x", 0.0))
        source.setdefault("camera_y", row.get("camera_y", 0.0))
    v9_scores = _ranker_v9_scores(source_candidates)
    if v9_scores is not None:
        v9_percentiles = _percentile(v9_scores)
    else:
        v9_percentiles = np.full((n,), 0.5, dtype=np.float64)

    current_ranks = [row.get("current_rank") for row in candidates]
    current_values = []
    for rank in current_ranks:
        if rank is None:
            current_values.append(0.0)
        else:
            current_values.append(1.0 / max(1.0, float(rank)))
    current_percentiles = _percentile(current_values) if n else np.empty((0,), dtype=float)

    for i, row in enumerate(candidates):
        hole_std = float(np.median(std_matrix[i])) if std_matrix[i] else 0.0
        hole_mild = float(np.median(mild_matrix[i])) if mild_matrix[i] else 0.0
        hole_fused = float(np.median(fused_matrix[i])) if fused_matrix[i] else 0.0
        temporal = temporal_candidate_score_v216(
            pre[i] if pre.ndim == 3 and i < len(pre) else None,
            post[i] if post.ndim == 4 and i < len(post) else np.empty((0, 0, 0), dtype=np.uint8),
        )
        dx = float(np.median(off_x[i])) if off_x[i] else 0.0
        dy = float(np.median(off_y[i])) if off_y[i] else 0.0
        row["evidence_v216"] = {
            "hole_standard": hole_std,
            "hole_mild": hole_mild,
            "hole_fused": hole_fused,
            "hole_disagreement": abs(hole_std - hole_mild),
            "hole_offset_dx": dx,
            "hole_offset_dy": dy,
            "refined_camera_x": _safe_float(row.get("camera_x")) + dx,
            "refined_camera_y": _safe_float(row.get("camera_y")) + dy,
            "temporal": temporal,
            "v9_score": None if v9_scores is None else float(v9_scores[i]),
            "v9_percentile": float(v9_percentiles[i]),
            "current_rank_percentile": float(current_percentiles[i]),
        }

    gt = pack.gt_xy
    distances = [row.get("distance_gt_px") for row in candidates if not row.get("capture_forced_gt_nearest")]
    pool20 = any(value is not None and float(value) <= 20.0 for value in distances)
    pool42 = any(value is not None and float(value) <= 42.0 for value in distances)
    return CandidateShotEvidenceV216(
        json_path=str(pack.json_path or ""),
        session_id=str(pack.metadata.get("session_id", "unknown")),
        round_id=int(pack.metadata.get("round_id", 0)),
        provisional_split=False,
        candidates=candidates,
        gt_xy=gt,
        pool_oracle_20=pool20,
        pool_oracle_42=pool42,
    )


def _eligible_rows(shot: CandidateShotEvidenceV216, *, pool: str = "ranked") -> list[dict[str, Any]]:
    rows = [row for row in shot.candidates if not bool(row.get("capture_forced_gt_nearest"))]
    if pool == "ranked":
        rows = [row for row in rows if bool(row.get("in_ranked_pool"))]
    return rows


def _candidate_distance(row: dict[str, Any], *, refined: bool = False, gt_xy: tuple[float, float] | None = None) -> float | None:
    if gt_xy is None:
        value = row.get("distance_gt_px")
        return None if value is None else float(value)
    ev = row.get("evidence_v216") or {}
    if refined:
        x = _safe_float(ev.get("refined_camera_x"), _safe_float(row.get("camera_x")))
        y = _safe_float(ev.get("refined_camera_y"), _safe_float(row.get("camera_y")))
    else:
        x, y = _safe_float(row.get("camera_x")), _safe_float(row.get("camera_y"))
    return float(math.hypot(x - gt_xy[0], y - gt_xy[1]))


def _score_row(row: dict[str, Any], source: str, weights: dict[str, float] | None = None) -> float:
    ev = row.get("evidence_v216") or {}
    if source == "current":
        rank = row.get("current_rank")
        return -9999.0 if rank is None else -float(rank)
    if source == "hole":
        return _safe_float(ev.get("hole_fused"))
    if source == "temporal":
        return _safe_float((ev.get("temporal") or {}).get("score"))
    if source == "v9":
        return _safe_float(ev.get("v9_percentile"), 0.5)
    if source == "fusion":
        weights = dict(weights or {})
        return (
            _safe_float(weights.get("hole")) * _safe_float(ev.get("hole_fused"))
            + _safe_float(weights.get("temporal")) * _safe_float((ev.get("temporal") or {}).get("score"))
            + _safe_float(weights.get("v9")) * _safe_float(ev.get("v9_percentile"), 0.5)
            + _safe_float(weights.get("current")) * _safe_float(ev.get("current_rank_percentile"), 0.5)
        )
    raise KeyError(source)


def _ranking_for_shot(shot: CandidateShotEvidenceV216, source: str, *, pool: str, weights: dict[str, float] | None = None) -> list[dict[str, Any]]:
    rows = list(_eligible_rows(shot, pool=pool))
    return sorted(rows, key=lambda row: (_score_row(row, source, weights), -int(row.get("capture_index", 0))), reverse=True)


def ranking_metrics_v216(
    shots: Sequence[CandidateShotEvidenceV216],
    source: str,
    *,
    pool: str = "ranked",
    weights: dict[str, float] | None = None,
    radius: float = 20.0,
) -> dict[str, Any]:
    top1 = top3 = top5 = 0
    ranks: list[int] = []
    selected_errors: list[float] = []
    refined_errors: list[float] = []
    eligible = 0
    oracle = 0
    for shot in shots:
        if shot.gt_xy is None:
            continue
        eligible += 1
        rows = _eligible_rows(shot, pool=pool)
        if not rows:
            continue
        if any((lambda d: d is not None and d <= radius)(_candidate_distance(row)) for row in rows):
            oracle += 1
        ranked = _ranking_for_shot(shot, source, pool=pool, weights=weights)
        if not ranked:
            continue
        selected_distance = _candidate_distance(ranked[0])
        if selected_distance is not None:
            selected_errors.append(float(selected_distance))
        refined_distance = _candidate_distance(ranked[0], refined=True, gt_xy=shot.gt_xy)
        if refined_distance is not None:
            refined_errors.append(float(refined_distance))
        gt_rank = None
        for index, row in enumerate(ranked, start=1):
            distance = _candidate_distance(row)
            if distance is not None and distance <= radius:
                gt_rank = index
                break
        if gt_rank is not None:
            ranks.append(gt_rank)
            top1 += int(gt_rank <= 1)
            top3 += int(gt_rank <= 3)
            top5 += int(gt_rank <= 5)
    denom = max(1, eligible)
    return {
        "shots": eligible,
        "oracle_recall": round(oracle / denom, 6),
        "top1": round(top1 / denom, 6),
        "top3": round(top3 / denom, 6),
        "top5": round(top5 / denom, 6),
        "median_gt_rank": None if not ranks else float(np.median(ranks)),
        "mean_gt_rank": None if not ranks else round(float(np.mean(ranks)), 6),
        "median_selected_error_px": None if not selected_errors else round(float(np.median(selected_errors)), 6),
        "p95_selected_error_px": None if not selected_errors else round(float(np.percentile(selected_errors, 95)), 6),
        "median_refined_error_px": None if not refined_errors else round(float(np.median(refined_errors)), 6),
    }


def split_shots_v216(shots: Sequence[CandidateShotEvidenceV216]) -> dict[str, list[CandidateShotEvidenceV216]]:
    sessions = sorted({shot.session_id for shot in shots})
    if len(sessions) >= 3:
        # Whole-session split: last session is sacred holdout, previous session
        # confirmation, all earlier sessions development.
        holdout = sessions[-1]
        confirmation = sessions[-2]
        result = {
            "development": [shot for shot in shots if shot.session_id not in {confirmation, holdout}],
            "confirmation": [shot for shot in shots if shot.session_id == confirmation],
            "holdout": [shot for shot in shots if shot.session_id == holdout],
        }
        return result

    # One/two capture sessions are still useful for engineering, but a shot-level
    # split is explicitly provisional and cannot grant authority.
    ordered = sorted(shots, key=lambda shot: (shot.session_id, shot.round_id))
    dev: list[CandidateShotEvidenceV216] = []
    confirmation: list[CandidateShotEvidenceV216] = []
    holdout: list[CandidateShotEvidenceV216] = []
    for index, shot in enumerate(ordered):
        shot.provisional_split = True
        bucket = index % 5
        if bucket == 4:
            holdout.append(shot)
        elif bucket == 3:
            confirmation.append(shot)
        else:
            dev.append(shot)
    return {"development": dev, "confirmation": confirmation, "holdout": holdout}


def choose_fusion_weights_v216(shots: Sequence[CandidateShotEvidenceV216]) -> dict[str, Any]:
    # Coarse, interpretable convex search.  Pure endpoints are included, so the
    # optimizer is allowed to decide that an extra evidence source adds nothing.
    candidates: list[dict[str, Any]] = []
    values = (0.0, 0.25, 0.5, 0.75, 1.0)
    for hole in values:
        for temporal in values:
            for v9 in values:
                for current in values:
                    total = hole + temporal + v9 + current
                    if total <= 0.0:
                        continue
                    weights = {"hole": hole / total, "temporal": temporal / total, "v9": v9 / total, "current": current / total}
                    m20 = ranking_metrics_v216(shots, "fusion", pool="ranked", weights=weights, radius=20.0)
                    m42 = ranking_metrics_v216(shots, "fusion", pool="ranked", weights=weights, radius=42.0)
                    # Top-1 is the primary goal; Top-3 and GT rank break ties.
                    median_rank = m20.get("median_gt_rank") or 9999.0
                    selection = 0.65 * float(m20["top1"]) + 0.20 * float(m20["top3"]) + 0.15 * float(m42["top1"])
                    candidates.append({"weights": weights, "selection": selection, "median_rank20": median_rank, "metrics20": m20, "metrics42": m42})
    candidates.sort(key=lambda row: (float(row["selection"]), -float(row["median_rank20"])), reverse=True)
    return candidates[0] if candidates else {"weights": {"hole": 1.0, "temporal": 0.0, "v9": 0.0, "current": 0.0}, "selection": 0.0}


def benchmark_candidate_packs_v216(
    root: Path,
    *,
    ensemble_config: Path = DEFAULT_ENSEMBLE_CONFIG,
    max_shots: int | None = None,
) -> dict[str, Any]:
    paths = discover_candidate_packs(root)
    if max_shots is not None:
        paths = paths[: max(0, int(max_shots))]
    if not paths:
        raise RuntimeError(f"No V2.16 candidate packs found under {root}")
    ensemble = HolePatchEnsembleV215.load(Path(ensemble_config))
    shots = [score_pack_v216(CandidatePackV216.load(path), ensemble) for path in paths]
    split = split_shots_v216(shots)
    winner = choose_fusion_weights_v216(split["development"])
    weights = dict(winner["weights"])

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "shadow_only": True,
        "candidate_packs": len(shots),
        "sessions": sorted({shot.session_id for shot in shots}),
        "split": {key: len(value) for key, value in split.items()},
        "split_is_provisional": bool(any(shot.provisional_split for shot in shots)),
        "model_sources": {
            "hole_v215": str(ensemble_config),
            "ranker_v9": str(DEFAULT_V9_MODEL),
            "ranker_v9_available": _load_v9() is not None,
        },
        "selected_fusion": {"weights": weights, "development_selection": round(float(winner.get("selection", 0.0)), 6)},
        "results": {},
    }
    for split_name, split_shots in split.items():
        rows: dict[str, Any] = {}
        for pool in ("ranked", "union"):
            pool_key = "ranked_pool" if pool == "ranked" else "raw_plus_ranked_union"
            rows[pool_key] = {}
            for source in ("current", "hole", "temporal", "v9", "fusion"):
                if source == "current" and pool == "union":
                    continue
                rows[pool_key][source] = {
                    "r20": ranking_metrics_v216(split_shots, source, pool=pool, weights=weights, radius=20.0),
                    "r42": ranking_metrics_v216(split_shots, source, pool=pool, weights=weights, radius=42.0),
                }
        report["results"][split_name] = rows

    confirmation = report["results"].get("confirmation", {}).get("ranked_pool", {})
    current = confirmation.get("current", {}).get("r20", {})
    fusion = confirmation.get("fusion", {}).get("r20", {})
    holdout = report["results"].get("holdout", {}).get("ranked_pool", {})
    holdout_current = holdout.get("current", {}).get("r20", {})
    holdout_fusion = holdout.get("fusion", {}).get("r20", {})
    report["gate"] = {
        "confirmation_top1_not_worse": float(fusion.get("top1", 0.0)) >= float(current.get("top1", 0.0)),
        "holdout_top1_not_worse": float(holdout_fusion.get("top1", 0.0)) >= float(holdout_current.get("top1", 0.0)),
        "has_nontrivial_candidate_data": len(shots) >= 50,
        "eligible_for_candidate_shadow_learning": len(shots) >= 50,
        "eligible_for_live_authority": False,
        "reason_live_authority_false": "V2.16 is capture/offline/shadow only; >=3 unseen physical sessions are required before any authority discussion.",
    }
    return report


def save_benchmark_report_v216(report: dict[str, Any], report_path: Path = DEFAULT_REPORT, fusion_path: Path = DEFAULT_FUSION) -> None:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    fusion = {
        "schema_version": SCHEMA_VERSION,
        "shadow_only": True,
        "weights": dict(report.get("selected_fusion", {}).get("weights") or {}),
        "split_is_provisional": bool(report.get("split_is_provisional", True)),
        "gate": dict(report.get("gate") or {}),
        "source_report": str(report_path),
    }
    fusion_path = Path(fusion_path)
    fusion_path.parent.mkdir(parents=True, exist_ok=True)
    fusion_path.write_text(json.dumps(fusion, indent=2, ensure_ascii=False), encoding="utf-8")


def hard_negative_rows_v216(
    shot: CandidateShotEvidenceV216,
    *,
    min_distance_px: float = 55.0,
    max_per_shot: int = 8,
) -> list[dict[str, Any]]:
    rows = []
    for row in _eligible_rows(shot, pool="union"):
        distance = row.get("distance_gt_px")
        if distance is None or float(distance) < float(min_distance_px):
            continue
        ev = row.get("evidence_v216") or {}
        hardness = max(
            _safe_float(ev.get("hole_fused")),
            _safe_float((ev.get("temporal") or {}).get("score")),
            _safe_float(ev.get("v9_percentile"), 0.5),
            _safe_float(ev.get("current_rank_percentile"), 0.5),
        )
        item = dict(row)
        item["hardness_v216"] = float(hardness)
        rows.append(item)
    rows.sort(key=lambda row: (float(row.get("hardness_v216", 0.0)), -float(row.get("distance_gt_px", 0.0))), reverse=True)
    return rows[: max(1, int(max_per_shot))]
