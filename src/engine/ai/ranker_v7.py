from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence


Candidate = dict[str, Any]

MODEL_PATH = Path("content/ai/ranker_v7_offline.json")

BASE_FEATURE_KEYS = [
    "baseline_score",
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
    "core_member",
]

POOL_REASON_FEATURE_KEYS = [
    "reason_core",
    "reason_keep_all",
    "reason_core_baseline",
    "reason_support",
    "reason_signal",
    "reason_diversity",
    "reason_vault",
    "reason_spatial",
    "reason_baseline_fill",
]

RELATIVE_SOURCE_KEYS = [
    "baseline_score",
    "support_score",
    "signal_score",
    "member_count",
    "compactness",
    "source_diversity",
    "current_fraction",
    "hits_max",
    "patch_prior_max",
    "zscore",
    "absdiff",
    "dog",
    "saliency",
    "spread_good",
]

RELATIVE_FEATURE_KEYS = [f"rel_{key}" for key in RELATIVE_SOURCE_KEYS]

INTERACTION_FEATURE_KEYS = [
    "support_x_current",
    "support_x_diversity",
    "signal_x_patch",
    "tile_x_current",
    "agreement_x_current",
    "fresh_single",
    "low_signal_current",
]

FEATURE_KEYS = (
    BASE_FEATURE_KEYS
    + POOL_REASON_FEATURE_KEYS
    + RELATIVE_FEATURE_KEYS
    + INTERACTION_FEATURE_KEYS
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float(default)
    except Exception:
        return float(default)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sat(value: Any, scale: float) -> float:
    return math.tanh(max(0.0, _safe_float(value)) / max(1e-6, float(scale)))


def _sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def base_vector(candidate: Candidate) -> dict[str, float]:
    member_count_raw = max(0.0, _safe_float(candidate.get("v27_member_count")))
    source_diversity_raw = max(0.0, _safe_float(candidate.get("v27_source_diversity")))
    current = _clip01(_safe_float(candidate.get("v27_current_fraction")))
    support = _clip01(_safe_float(candidate.get("v27_support_score")))
    signal = _clip01(_safe_float(candidate.get("v27_signal_score")))
    patch = _clip01(_safe_float(candidate.get("v27_patch_prior_max")))
    compactness = _clip01(_safe_float(candidate.get("v27_compactness")))
    agreement = _clip01(_safe_float(candidate.get("v27_agreement_fraction")))
    tile = _clip01(_safe_float(candidate.get("v27_tile_fraction")))
    spread = max(0.0, _safe_float(candidate.get("v27_spread_px")))
    age = max(0.0, _safe_float(candidate.get("v27_age_median_s")))
    existed = _clip01(_safe_float(candidate.get("v27_existed_before_median")))
    reasons = {str(item) for item in (candidate.get("v28_pool_reasons") or [])}

    vector = {
        "baseline_score": _clip01(_safe_float(candidate.get("v27_baseline_score"))),
        "support_score": support,
        "signal_score": signal,
        "member_count": _sat(member_count_raw, 4.0),
        "compactness": compactness,
        "source_diversity": _clip01(source_diversity_raw / 4.0),
        "current_fraction": current,
        "carried_fraction": _clip01(_safe_float(candidate.get("v27_carried_fraction"))),
        "hits_max": _sat(candidate.get("v27_hits_max"), 3.0),
        "hits_mean": _sat(candidate.get("v27_hits_mean"), 2.2),
        "v1_fraction": _clip01(_safe_float(candidate.get("v27_v1_fraction"))),
        "v2_fraction": _clip01(_safe_float(candidate.get("v27_v2_fraction"))),
        "tile_fraction": tile,
        "agreement_fraction": agreement,
        "patch_prior_max": patch,
        "patch_prior_median": _clip01(_safe_float(candidate.get("v27_patch_prior_median"))),
        "zscore": _clip01(_safe_float(candidate.get("v27_zscore_norm"))),
        "absdiff": _clip01(_safe_float(candidate.get("v27_absdiff_norm"))),
        "dog": _clip01(_safe_float(candidate.get("v27_dog_norm"))),
        "saliency": _clip01(_safe_float(candidate.get("v27_saliency_norm"))),
        "persistence": _clip01(_safe_float(candidate.get("v27_persistence_median"), 0.5)),
        "not_existed_before": 1.0 - existed,
        "age_good": math.exp(-age / 1.2),
        "spread_good": math.exp(-spread / 18.0),
        "single_member": 1.0 if member_count_raw <= 1.01 else 0.0,
        "core_member": 1.0 if _safe_float(candidate.get("v28_core_pool")) > 0.5 else 0.0,
        "reason_core": 1.0 if ("core" in reasons or any(r.startswith("core_") for r in reasons)) else 0.0,
        "reason_keep_all": 1.0 if "keep_all" in reasons else 0.0,
        "reason_core_baseline": 1.0 if "core_baseline" in reasons else 0.0,
        "reason_support": 1.0 if "support" in reasons else 0.0,
        "reason_signal": 1.0 if "signal" in reasons else 0.0,
        "reason_diversity": 1.0 if "diversity" in reasons else 0.0,
        "reason_vault": 1.0 if "vault" in reasons else 0.0,
        "reason_spatial": 1.0 if "spatial" in reasons else 0.0,
        "reason_baseline_fill": 1.0 if "baseline_fill" in reasons else 0.0,
    }

    vector.update(
        {
            "support_x_current": support * current,
            "support_x_diversity": support * vector["source_diversity"],
            "signal_x_patch": signal * patch,
            "tile_x_current": tile * current,
            "agreement_x_current": agreement * current,
            "fresh_single": vector["single_member"] * current * vector["not_existed_before"],
            "low_signal_current": (1.0 - signal) * current,
        }
    )

    for key in RELATIVE_FEATURE_KEYS:
        vector[key] = 0.5
    return vector


def _percentile_scores(values: list[float]) -> list[float]:
    """Return stable [0,1] within-shot percentiles; larger value => larger percentile."""
    count = len(values)
    if count <= 1:
        return [0.5] * count
    order = sorted(range(count), key=lambda index: (values[index], index))
    result = [0.0] * count

    # Ties receive their average rank so identical values cannot get arbitrary
    # preference from list order.
    start = 0
    while start < count:
        end = start + 1
        base_value = values[order[start]]
        while end < count and abs(values[order[end]] - base_value) <= 1e-12:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        percentile = average_rank / float(count - 1)
        for position in range(start, end):
            result[order[position]] = percentile
        start = end
    return result


def vectors_for_pool(candidates: Sequence[Candidate]) -> list[dict[str, float]]:
    vectors = [base_vector(candidate) for candidate in candidates]
    if not vectors:
        return vectors
    for source_key in RELATIVE_SOURCE_KEYS:
        values = [_safe_float(vector.get(source_key)) for vector in vectors]
        percentiles = _percentile_scores(values)
        target_key = f"rel_{source_key}"
        for vector, percentile in zip(vectors, percentiles):
            vector[target_key] = float(percentile)
    return vectors


def dense_vector(features: dict[str, Any], feature_keys: Sequence[str] = FEATURE_KEYS) -> list[float]:
    return [_safe_float(features.get(key)) for key in feature_keys]


class RankerV7ShadowModel:
    """Offline-trained, runtime shadow-only linear ranker.

    V2.9 never gives this model authority. It is loaded only so a later camera
    run can compare its ordering against the still-authoritative V2.8 baseline.
    """

    VERSION = 7

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.feature_keys = list(FEATURE_KEYS)
        self.weights = {key: 0.0 for key in self.feature_keys}
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
            raw = payload.get("weights")
            if not isinstance(keys, list) or not isinstance(raw, dict):
                return
            self.feature_keys = [str(key) for key in keys]
            self.weights = {
                key: _safe_float(raw.get(key))
                for key in self.feature_keys
            }
            self.metadata = {
                "schema_version": payload.get("schema_version"),
                "model_type": payload.get("model_type"),
                "trained_session": payload.get("trained_session"),
                "cv": payload.get("cv"),
                "shadow_only": bool(payload.get("shadow_only", True)),
            }
            self.loaded = True
        except Exception:
            self.loaded = False

    def raw_score_features(self, features: dict[str, Any]) -> float:
        return sum(
            self.weights.get(key, 0.0) * _safe_float(features.get(key))
            for key in self.feature_keys
        )

    def rank(
        self,
        candidates: Sequence[Candidate],
    ) -> list[Candidate]:
        self.reload()
        source = [dict(candidate) for candidate in candidates]
        if not source:
            return []
        vectors = vectors_for_pool(source)
        ranked: list[Candidate] = []
        for candidate, vector in zip(source, vectors):
            raw = self.raw_score_features(vector) if self.loaded else 0.0
            item = dict(candidate)
            item["ranker_v7_raw"] = float(raw)
            item["ranker_v7_score"] = float(_sigmoid(raw))
            ranked.append(item)
        ranked.sort(
            key=lambda candidate: (
                _safe_float(candidate.get("ranker_v7_raw")),
                _safe_float(candidate.get("v27_baseline_score")),
            ),
            reverse=True,
        )
        for index, candidate in enumerate(ranked, start=1):
            candidate["ranker_v7_rank"] = index
        return ranked

    def summary(self) -> dict[str, Any]:
        strongest = sorted(
            self.weights.items(),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:12]
        return {
            "version": self.VERSION,
            "loaded": bool(self.loaded),
            "model_path": str(self.model_path),
            "feature_count": len(self.feature_keys),
            "metadata": self.metadata,
            "strongest_weights": [
                [key, round(float(value), 6)]
                for key, value in strongest
            ],
        }


__all__ = [
    "BASE_FEATURE_KEYS",
    "FEATURE_KEYS",
    "POOL_REASON_FEATURE_KEYS",
    "RELATIVE_FEATURE_KEYS",
    "RELATIVE_SOURCE_KEYS",
    "RankerV7ShadowModel",
    "base_vector",
    "dense_vector",
    "vectors_for_pool",
]
