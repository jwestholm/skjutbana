from __future__ import annotations

"""Candidate-level synthetic <-> projector/camera domain-gap profiler for V2.21.

No model is trained for hit authority here.  A tiny domain classifier is used
only diagnostically: if it can trivially tell generated candidate groups from
projector/camera candidate groups, then synthetic-only ranking success should
not be trusted as transfer evidence.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .candidate_ranking_training_v218 import CandidateGroupV218, prepare_groups_v218, split_groups_v218


SCALAR_NAMES = (
    "mean_absdiff",
    "p95_absdiff",
    "center_absdiff",
    "center_darkening",
    "center_minus_ring_absdiff",
    "mean_brightening",
    "mean_darkening",
    "persistence",
)


@dataclass(frozen=True)
class DomainGapConfigV221:
    max_candidates_per_domain: int = 24000
    seed: int = 22101
    folds: int = 5
    classifier_steps: int = 700
    classifier_lr: float = 0.06
    l2: float = 0.002


def _candidate_feature_matrix(group: CandidateGroupV218) -> tuple[np.ndarray, list[str]]:
    emb = np.asarray(group.embedding, dtype=np.float32)
    scal = np.asarray(group.temporal_scalars, dtype=np.float32)
    p = np.asarray(group.base_probability, dtype=np.float32).reshape(-1, 1)
    off_mag = np.linalg.norm(np.asarray(group.base_offsets, dtype=np.float32), axis=1, keepdims=True)
    emb_norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_mean = np.mean(emb, axis=1, keepdims=True)
    emb_std = np.std(emb, axis=1, keepdims=True)
    known = np.asarray(group.known_hole_distance, dtype=np.float32).reshape(-1, 1)
    known = np.where(np.isfinite(known), np.clip(known, 0.0, 250.0) / 250.0, -1.0)
    rank = np.asarray(group.current_rank, dtype=np.float32).reshape(-1, 1)
    rank = np.where(rank > 0, rank / max(1.0, float(group.size)), -1.0)
    cols = [p, off_mag / 48.0, scal, emb_norm / max(1.0, math.sqrt(max(1, emb.shape[1]))), emb_mean, emb_std, known, rank]
    names = ["v217_probability", "v217_offset_magnitude"] + list(SCALAR_NAMES[:scal.shape[1]]) + [
        "embedding_norm", "embedding_mean", "embedding_std", "known_hole_distance_scaled", "current_rank_fraction"
    ]
    return np.concatenate(cols, axis=1).astype(np.float32), names


def _sample_rows(groups: Sequence[CandidateGroupV218], limit: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    mats: list[np.ndarray] = []
    near: list[np.ndarray] = []
    names: list[str] | None = None
    for group in groups:
        mat, n = _candidate_feature_matrix(group)
        mats.append(mat)
        near.append(np.asarray(group.distances <= 20.0, dtype=bool))
        names = n
    if not mats:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=bool), names or []
    x = np.concatenate(mats, axis=0)
    ynear = np.concatenate(near, axis=0)
    if len(x) > max(1, int(limit)):
        rng = np.random.default_rng(int(seed))
        keep = rng.choice(len(x), size=int(limit), replace=False)
        x = x[keep]
        ynear = ynear[keep]
    return x, ynear, names or []


def _quantiles(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"mean": 0.0, "std": 0.0, "q10": 0.0, "q50": 0.0, "q90": 0.0, "q99": 0.0}
    q = np.percentile(v, [10, 50, 90, 99])
    return {
        "mean": float(np.mean(v)), "std": float(np.std(v)),
        "q10": float(q[0]), "q50": float(q[1]), "q90": float(q[2]), "q99": float(q[3]),
    }


def _ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=np.float64)); b = np.sort(np.asarray(b, dtype=np.float64))
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return 0.0
    points = np.unique(np.concatenate([a, b]))
    ca = np.searchsorted(a, points, side="right") / len(a)
    cb = np.searchsorted(b, points, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def _wasserstein_quantile(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return 0.0
    qs = np.linspace(0.0, 100.0, 201)
    return float(np.mean(np.abs(np.percentile(a, qs) - np.percentile(b, qs))))


def _feature_shift(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va, vb = float(np.var(a)), float(np.var(b))
    pooled = math.sqrt(max(1e-10, 0.5 * (va + vb)))
    smd = (ma - mb) / pooled
    return {
        "standardized_mean_difference": float(smd),
        "abs_smd": float(abs(smd)),
        "ks": _ks_stat(a, b),
        "wasserstein": _wasserstein_quantile(a, b),
    }


def _group_summary(group: CandidateGroupV218) -> np.ndarray:
    x, _ = _candidate_feature_matrix(group)
    if not len(x):
        return np.empty((0,), dtype=np.float32)
    parts = [
        np.mean(x, axis=0), np.std(x, axis=0),
        np.percentile(x, 50, axis=0), np.percentile(x, 90, axis=0),
        np.array([math.log1p(group.size)], dtype=np.float32),
    ]
    return np.concatenate(parts).astype(np.float32)


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int32); score = np.asarray(score, dtype=np.float64)
    pos = int(np.sum(y == 1)); neg = int(np.sum(y == 0))
    if not pos or not neg:
        return 0.5
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    # Average tied ranks.
    sorted_score = score[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        if end - start > 1:
            avg = float(np.mean(ranks[order[start:end]]))
            ranks[order[start:end]] = avg
        start = end
    rank_sum = float(np.sum(ranks[y == 1]))
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def _domain_classifier_cv(synth_groups: Sequence[CandidateGroupV218], physical_groups: Sequence[CandidateGroupV218], cfg: DomainGapConfigV221) -> dict[str, Any]:
    sx = [x for g in synth_groups if (x := _group_summary(g)).size]
    px = [x for g in physical_groups if (x := _group_summary(g)).size]
    if len(sx) < 5 or len(px) < 5:
        return {"auc": 0.5, "groups": len(sx) + len(px), "note": "insufficient groups"}
    x = np.stack(sx + px).astype(np.float32)
    y = np.array([0] * len(sx) + [1] * len(px), dtype=np.int32)
    # Deterministic domain-stratified folds by within-domain index.
    fold = np.array([i % max(2, cfg.folds) for i in range(len(sx))] + [i % max(2, cfg.folds) for i in range(len(px))], dtype=np.int32)
    pred = np.zeros(len(y), dtype=np.float64)
    for f in range(max(2, cfg.folds)):
        train = fold != f; test = fold == f
        if not np.any(test):
            continue
        mean = np.mean(x[train], axis=0)
        std = np.maximum(np.std(x[train], axis=0), 1e-4)
        xt = np.clip((x[train] - mean) / std, -6.0, 6.0).astype(np.float64)
        xv = np.clip((x[test] - mean) / std, -6.0, 6.0).astype(np.float64)
        yt = y[train].astype(np.float64)
        w = np.zeros(xt.shape[1], dtype=np.float64); b = 0.0
        for _ in range(int(cfg.classifier_steps)):
            z = np.clip(xt @ w + b, -20.0, 20.0)
            p = 1.0 / (1.0 + np.exp(-z))
            err = p - yt
            gw = (xt.T @ err) / max(1, len(xt)) + float(cfg.l2) * w
            gb = float(np.mean(err))
            w -= float(cfg.classifier_lr) * gw
            b -= float(cfg.classifier_lr) * gb
        pred[test] = xv @ w + b
    auc = _auc(y, pred)
    return {
        "auc": float(auc),
        "groups": int(len(y)),
        "synthetic_groups": len(sx),
        "physical_groups": len(px),
        "interpretation": (
            "severe/trivial domain separation" if auc >= 0.95 else
            "large domain separation" if auc >= 0.85 else
            "moderate domain separation" if auc >= 0.70 else
            "weak domain separation"
        ),
    }


def profile_domain_gap_v221(
    *,
    synthetic_root: Path,
    physical_root: Path,
    v217_model: Path,
    synthetic_cache: Path,
    physical_cache: Path,
    config: DomainGapConfigV221 | None = None,
) -> dict[str, Any]:
    cfg = config or DomainGapConfigV221()
    synth_groups, synth_cache_info = prepare_groups_v218(
        Path(synthetic_root), v217_model_path=Path(v217_model), cache_root=Path(synthetic_cache)
    )
    physical_groups_all, physical_cache_info = prepare_groups_v218(
        Path(physical_root), v217_model_path=Path(v217_model), cache_root=Path(physical_cache)
    )
    # Data policy: only physical development groups may drive V2.21 domain-gap
    # fitting/design decisions. Confirmation/holdout remain protected report-only
    # material and are not fed to the diagnostic classifier below.
    physical_split, physical_split_provisional = split_groups_v218(Path(physical_root), physical_groups_all)
    physical_groups = list(physical_split.get("development") or [])
    if not physical_groups:
        raise RuntimeError("No physical development groups available for domain-gap profiling")
    sx, snear, names = _sample_rows(synth_groups, cfg.max_candidates_per_domain, cfg.seed)
    px, pnear, pnames = _sample_rows(physical_groups, cfg.max_candidates_per_domain, cfg.seed + 1)
    if names != pnames:
        raise RuntimeError("Synthetic/physical feature schema mismatch")
    if not len(sx) or not len(px):
        raise RuntimeError("Need candidate rows in both domains")

    feature_rows: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        shift = _feature_shift(sx[:, i], px[:, i])
        feature_rows.append({
            "feature": name,
            "synthetic": _quantiles(sx[:, i]),
            "physical": _quantiles(px[:, i]),
            **shift,
        })
    feature_rows.sort(key=lambda row: (float(row["ks"]), float(row["abs_smd"])), reverse=True)

    near_feature_rows: list[dict[str, Any]] = []
    if np.any(snear) and np.any(pnear):
        for i, name in enumerate(names):
            shift = _feature_shift(sx[snear, i], px[pnear, i])
            near_feature_rows.append({
                "feature": name,
                "synthetic": _quantiles(sx[snear, i]),
                "physical": _quantiles(px[pnear, i]),
                **shift,
            })
        near_feature_rows.sort(key=lambda row: (float(row["ks"]), float(row["abs_smd"])), reverse=True)

    classifier = _domain_classifier_cv(synth_groups, physical_groups, cfg)
    return {
        "schema_version": "2.21",
        "purpose": "diagnostic_only_no_hit_authority",
        "synthetic_root": str(synthetic_root),
        "physical_root": str(physical_root),
        "v217_model": str(v217_model),
        "cache": {"synthetic": synth_cache_info, "physical": physical_cache_info},
        "physical_split_is_provisional": bool(physical_split_provisional),
        "physical_groups": {
            "all_available": len(physical_groups_all),
            "development_used": len(physical_groups),
            "confirmation_protected": len(physical_split.get("confirmation") or []),
            "holdout_protected": len(physical_split.get("holdout") or []),
        },
        "sampled_candidates": {"synthetic": len(sx), "physical_development": len(px)},
        "near_gt20_candidates": {"synthetic": int(np.sum(snear)), "physical": int(np.sum(pnear))},
        "group_domain_classifier": classifier,
        "feature_shift_all_candidates": feature_rows,
        "feature_shift_near_gt20": near_feature_rows,
        "shortcut_warning": bool(float(classifier.get("auc", 0.5)) >= 0.95),
        "next_action": (
            "Inspect top shifted temporal/embedding features and collect/use full-frame projector-camera evidence before more synthetic-only ranking training."
            if float(classifier.get("auc", 0.5)) >= 0.85 else
            "Domain separation is not trivial; continue direct-proposal/full-frame evaluation."
        ),
    }


def write_domain_gap_report_v221(path: Path, report: dict[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
