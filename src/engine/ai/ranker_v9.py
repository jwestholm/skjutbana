from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from src.engine.ai.ranker_v7 import vectors_for_pool


Candidate = dict[str, Any]
MODEL_PATH = Path("content/ai/ranker_v9_offline.json")

# V2.11 deliberately excludes:
# - baseline_score / rel_baseline_score
# - core_member
# - every reason_* pool-policy feature
# - all rel_* duplicates (the model already converts each raw physical feature
#   to an in-shot percentile before scoring).
PHYSICAL_FEATURE_KEYS = [
    "support_score",
    "signal_score",
    "member_count",
    "compactness",
    "source_diversity",
    "current_fraction",
    "carried_fraction",
    "hits_max",
    "hits_mean",
    "v1_fraction",
    "v2_fraction",
    "tile_fraction",
    "agreement_fraction",
    "patch_prior_max",
    "patch_prior_median",
    "zscore",
    "absdiff",
    "dog",
    "saliency",
    "persistence",
    "not_existed_before",
    "age_good",
    "spread_good",
    "single_member",
    "support_x_current",
    "support_x_diversity",
    "signal_x_patch",
    "tile_x_current",
    "agreement_x_current",
    "fresh_single",
    "low_signal_current",
]

# Prevent a tiny 100-shot dataset from stacking five near-identical views of
# the same underlying quantity into one "strong" rule.
FEATURE_FAMILIES = {
    "support_score": "support",
    "support_x_current": "support",
    "support_x_diversity": "support",

    "signal_score": "signal",
    "zscore": "signal",
    "absdiff": "signal",
    "dog": "signal",
    "saliency": "signal",
    "signal_x_patch": "signal",
    "low_signal_current": "signal",

    "member_count": "history",
    "hits_max": "history",
    "hits_mean": "history",
    "persistence": "history",
    "current_fraction": "history",
    "carried_fraction": "history",
    "age_good": "history",
    "single_member": "history",
    "fresh_single": "history",

    "source_diversity": "sources",
    "v1_fraction": "sources",
    "v2_fraction": "sources",
    "tile_fraction": "sources",
    "agreement_fraction": "sources",
    "tile_x_current": "sources",
    "agreement_x_current": "sources",

    "compactness": "geometry",
    "spread_good": "geometry",

    "patch_prior_max": "patch",
    "patch_prior_median": "patch",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float(default)
    except Exception:
        return float(default)


def _percentile_scores(values: Sequence[float]) -> list[float]:
    count = len(values)
    if count <= 1:
        return [0.5] * count

    order = sorted(range(count), key=lambda index: (float(values[index]), index))
    result = [0.0] * count
    start = 0

    while start < count:
        end = start + 1
        value = float(values[order[start]])
        while end < count and abs(float(values[order[end]]) - value) <= 1e-12:
            end += 1

        average_rank = 0.5 * (start + end - 1)
        percentile = average_rank / float(count - 1)
        for position in range(start, end):
            result[order[position]] = percentile
        start = end

    return result


def physical_feature_rows(candidates: Sequence[Candidate]) -> list[dict[str, float]]:
    """Reuse V2.9's feature extraction but expose physical/signal fields only."""
    rows = vectors_for_pool(candidates)
    return [
        {key: _safe_float(row.get(key)) for key in PHYSICAL_FEATURE_KEYS}
        for row in rows
    ]


def percentile_matrix(
    feature_rows: Sequence[dict[str, Any]],
    feature_keys: Sequence[str],
) -> list[list[float]]:
    if not feature_rows:
        return []

    columns = {
        key: [_safe_float(row.get(key)) for row in feature_rows]
        for key in feature_keys
    }
    percentiles = {
        key: _percentile_scores(values)
        for key, values in columns.items()
    }

    return [
        [percentiles[key][index] for key in feature_keys]
        for index in range(len(feature_rows))
    ]


class RankerV9ShadowModel:
    """V2.11 physical-feature-only monotonic percentile ranker.

    Shadow-only by design. It never changes the authoritative selected hit.
    """

    VERSION = 9

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.feature_keys: list[str] = []
        self.weights: dict[str, float] = {}
        self.loaded = False
        self.metadata: dict[str, Any] = {}
        self._mtime: float | None = None
        self.reload(force=True)

    def reload(self, *, force: bool = False) -> None:
        try:
            mtime = self.model_path.stat().st_mtime
        except Exception:
            mtime = None

        if not force and mtime == self._mtime:
            return

        self._mtime = mtime
        self.loaded = False
        self.feature_keys = []
        self.weights = {}
        self.metadata = {}

        if mtime is None:
            return

        try:
            payload = json.loads(self.model_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            if not bool(payload.get("shadow_only", True)):
                return

            keys = payload.get("feature_keys")
            weights = payload.get("weights")
            if not isinstance(keys, list) or not isinstance(weights, dict):
                return

            clean_keys = [
                str(key)
                for key in keys
                if str(key) in PHYSICAL_FEATURE_KEYS
            ]
            if not clean_keys:
                return

            self.feature_keys = clean_keys
            self.weights = {
                key: _safe_float(weights.get(key))
                for key in clean_keys
            }
            self.metadata = {
                "schema_version": payload.get("schema_version"),
                "model_type": payload.get("model_type"),
                "trained_session": payload.get("trained_session"),
                "trained_shots": payload.get("trained_shots"),
                "development_cv": payload.get("development_cv"),
                "confirmation": payload.get("confirmation"),
                "recommendation": payload.get("recommendation"),
                "shadow_ready": bool(payload.get("shadow_ready", False)),
            }
            self.loaded = True
        except Exception:
            self.loaded = False

    def rank(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        self.reload()
        source = [dict(candidate) for candidate in candidates]
        if not source:
            return []

        feature_rows = physical_feature_rows(source)
        matrix = percentile_matrix(feature_rows, self.feature_keys)

        ranked: list[Candidate] = []
        for candidate, row in zip(source, matrix):
            score = 0.0
            contributions: dict[str, float] = {}

            for key, percentile in zip(self.feature_keys, row):
                centered = 2.0 * (float(percentile) - 0.5)
                contribution = self.weights.get(key, 0.0) * centered
                contributions[key] = float(contribution)
                score += contribution

            item = dict(candidate)
            item["ranker_v9_score"] = float(score)
            item["ranker_v9_contributions"] = contributions
            ranked.append(item)

        # Baseline is only a deterministic tie breaker. It is NOT a model
        # feature and has zero influence when V9 scores differ.
        ranked.sort(
            key=lambda candidate: (
                _safe_float(candidate.get("ranker_v9_score")),
                _safe_float(candidate.get("v27_baseline_score")),
            ),
            reverse=True,
        )
        for rank, candidate in enumerate(ranked, start=1):
            candidate["ranker_v9_rank"] = rank

        return ranked

    def summary(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "loaded": bool(self.loaded),
            "model_path": str(self.model_path),
            "feature_keys": list(self.feature_keys),
            "weights": dict(self.weights),
            "metadata": dict(self.metadata),
        }


__all__ = [
    "FEATURE_FAMILIES",
    "MODEL_PATH",
    "PHYSICAL_FEATURE_KEYS",
    "RankerV9ShadowModel",
    "percentile_matrix",
    "physical_feature_rows",
]
