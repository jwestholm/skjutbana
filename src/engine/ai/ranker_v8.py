from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from src.engine.ai.ranker_v7 import vectors_for_pool


Candidate = dict[str, Any]
MODEL_PATH = Path("content/ai/ranker_v8_offline.json")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float(default)
    except Exception:
        return float(default)


def _sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


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


def percentile_feature_matrix(
    feature_rows: Sequence[dict[str, Any]],
    feature_keys: Sequence[str],
) -> list[list[float]]:
    if not feature_rows:
        return []
    columns: dict[str, list[float]] = {
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


class RankerV8ShadowModel:
    """Monotonic percentile-rank ensemble trained fully offline.

    Important design property: V2.10 is shadow-only. The score is intentionally
    simple and auditable. A feature weight may be positive (HIGH is hole-like)
    or negative (LOW is hole-like). Raw feature magnitudes are converted to
    within-shot percentiles before combination, preventing one projector
    artifact with a huge numeric magnitude from dominating the score.
    """

    VERSION = 8

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.feature_keys: list[str] = []
        self.weights: dict[str, float] = {}
        self.evidence_power = 1.0
        self.baseline_blend = 0.0
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
        if mtime is None:
            return
        try:
            payload = json.loads(self.model_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            keys = payload.get("feature_keys")
            raw_weights = payload.get("weights")
            if not isinstance(keys, list) or not isinstance(raw_weights, dict):
                return
            self.feature_keys = [str(key) for key in keys]
            self.weights = {
                key: _safe_float(raw_weights.get(key))
                for key in self.feature_keys
            }
            self.evidence_power = max(0.25, min(3.0, _safe_float(payload.get("evidence_power"), 1.0)))
            self.baseline_blend = max(0.0, min(0.5, _safe_float(payload.get("baseline_blend"), 0.0)))
            self.metadata = {
                "schema_version": payload.get("schema_version"),
                "model_type": payload.get("model_type"),
                "trained_session": payload.get("trained_session"),
                "trained_shots": payload.get("trained_shots"),
                "development_cv": payload.get("development_cv"),
                "confirmation": payload.get("confirmation"),
                "recommendation": payload.get("recommendation"),
                "shadow_only": bool(payload.get("shadow_only", True)),
            }
            self.loaded = True
        except Exception:
            self.loaded = False

    def _score_feature_rows(self, feature_rows: Sequence[dict[str, Any]]) -> list[float]:
        if not feature_rows:
            return []
        matrix = percentile_feature_matrix(feature_rows, self.feature_keys)
        scores: list[float] = []
        for index, row in enumerate(matrix):
            total = 0.0
            for key, percentile in zip(self.feature_keys, row):
                centered = 2.0 * (float(percentile) - 0.5)
                shaped = math.copysign(abs(centered) ** self.evidence_power, centered)
                total += self.weights.get(key, 0.0) * shaped
            scores.append(total)
        return scores

    def rank(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        self.reload()
        source = [dict(candidate) for candidate in candidates]
        if not source:
            return []
        feature_rows = vectors_for_pool(source)
        model_scores = self._score_feature_rows(feature_rows) if self.loaded else [0.0] * len(source)

        # Baseline is only a small optional stabilizer. It never becomes a hard
        # veto and therefore cannot recreate the V2.8 preference for strong
        # projector artifacts if the optimizer sets baseline_blend to zero.
        baseline_values = [
            _safe_float(candidate.get("v27_baseline_score"))
            for candidate in source
        ]
        baseline_percentiles = _percentile_scores(baseline_values)

        ranked: list[Candidate] = []
        for candidate, raw, baseline_pct in zip(source, model_scores, baseline_percentiles):
            combined = (
                (1.0 - self.baseline_blend) * float(raw)
                + self.baseline_blend * (2.0 * float(baseline_pct) - 1.0)
            )
            item = dict(candidate)
            item["ranker_v8_raw"] = float(raw)
            item["ranker_v8_combined"] = float(combined)
            item["ranker_v8_score"] = float(_sigmoid(combined))
            ranked.append(item)

        ranked.sort(
            key=lambda candidate: (
                _safe_float(candidate.get("ranker_v8_combined")),
                _safe_float(candidate.get("v27_baseline_score")),
            ),
            reverse=True,
        )
        for rank, candidate in enumerate(ranked, start=1):
            candidate["ranker_v8_rank"] = rank
        return ranked

    def summary(self) -> dict[str, Any]:
        strongest = sorted(
            self.weights.items(),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:15]
        return {
            "version": self.VERSION,
            "loaded": bool(self.loaded),
            "model_path": str(self.model_path),
            "feature_count": len(self.feature_keys),
            "evidence_power": self.evidence_power,
            "baseline_blend": self.baseline_blend,
            "metadata": self.metadata,
            "strongest_weights": [
                [key, round(float(value), 6)]
                for key, value in strongest
            ],
        }


__all__ = [
    "MODEL_PATH",
    "RankerV8ShadowModel",
    "percentile_feature_matrix",
]
