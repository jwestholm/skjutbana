from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

Candidate = dict[str, Any]
Point = tuple[float, float]

CONFIG_PATH = Path("content/ai/ranker_v6_config.json")
MODEL_PATH = Path("content/ai/ranker_v6.json")

FEATURE_KEYS = [
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
]

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "shadow_enabled": True,
    # Two training labels. <=12 px is strong; <=20 px is useful but downweighted.
    "strict_positive_radius_px": 12.0,
    "soft_positive_radius_px": 20.0,
    "soft_positive_weight": 0.42,
    "negative_safe_radius_px": 55.0,
    "hard_negatives": 28,
    "learning_rate": 0.070,
    "l2": 0.0009,
    "epochs_per_shot": 1,
    "save_every_shots": 5,
    # Rolling pre-train validation gate. V6 can only become authoritative after
    # it has beaten the hypothesis baseline on real candidate pools.
    "auto_override_enabled": True,
    "validation_radius_px": 20.0,
    "validation_window": 220,
    "min_validation_shots": 120,
    "min_conditional_top1_pct": 10.0,
    "min_advantage_pp": 5.0,
    "min_top3_advantage_pp": 4.0,
    "max_median_rank_ratio": 0.72,
    "min_score_margin": 0.030,
    "min_top_score": 0.535,
}


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


def _distance(candidate: Candidate, gt_xy: Point) -> float:
    return math.hypot(
        _safe_float(candidate.get("camera_x")) - float(gt_xy[0]),
        _safe_float(candidate.get("camera_y")) - float(gt_xy[1]),
    )


def _median(values: Sequence[float]) -> float | None:
    data = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not data:
        return None
    middle = len(data) // 2
    if len(data) % 2:
        return data[middle]
    return 0.5 * (data[middle - 1] + data[middle])


class RankerV6:
    """Pairwise online ranker for V2.7 *hypotheses*, never raw hotspots.

    V2.6 proved that the correct neighbourhood survives into the filtered pool
    roughly three quarters of the time, but direct ranking of hundreds of raw
    observations placed GT around rank ~179. V6 is deliberately trained only
    after spatial/temporal consolidation has reduced the problem to a compact
    hypothesis set.
    """

    VERSION = 6

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        config_path: Path = CONFIG_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.config = dict(DEFAULT_CONFIG)
        self.weights = {key: 0.0 for key in FEATURE_KEYS}
        # Small priors only. They keep a brand-new model deterministic while the
        # hand-built hypothesis baseline remains authoritative until the gate.
        self.weights["source_diversity"] = 0.08
        self.weights["compactness"] = 0.06
        self.weights["current_fraction"] = 0.04
        self.weights["not_existed_before"] = 0.04
        self.stats: dict[str, Any] = self._fresh_stats()
        self._load_config()
        self.load()

    def _fresh_stats(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "strict_positive_shots": 0,
            "soft_positive_shots": 0,
            "pair_updates": 0,
            "skipped_no_positive": 0,
            "last_loss": None,
            "last_updated": None,
            "validation": [],
            "override_count": 0,
        }

    def _load_config(self) -> None:
        values = dict(DEFAULT_CONFIG)
        try:
            if self.config_path.exists():
                loaded = json.loads(self.config_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    values.update(loaded)
        except Exception:
            pass
        self.config = values

    def load(self) -> None:
        if not self.model_path.exists():
            return
        try:
            payload = json.loads(self.model_path.read_text(encoding="utf-8"))
            raw_weights = payload.get("weights", {})
            if isinstance(raw_weights, dict):
                for key in FEATURE_KEYS:
                    if key in raw_weights:
                        self.weights[key] = _safe_float(raw_weights[key])
            raw_stats = payload.get("stats", {})
            if isinstance(raw_stats, dict):
                self.stats.update(raw_stats)
            if not isinstance(self.stats.get("validation"), list):
                self.stats["validation"] = []
        except Exception:
            pass

    def save(self) -> None:
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": self.VERSION,
                "feature_keys": FEATURE_KEYS,
                "weights": self.weights,
                "stats": self.stats,
            }
            self.model_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def reset(self) -> None:
        self.weights = {key: 0.0 for key in FEATURE_KEYS}
        self.weights["source_diversity"] = 0.08
        self.weights["compactness"] = 0.06
        self.weights["current_fraction"] = 0.04
        self.weights["not_existed_before"] = 0.04
        self.stats = self._fresh_stats()
        self.stats["last_updated"] = time.time()
        self.save()

    def vector(self, candidate: Candidate) -> dict[str, float]:
        return {
            "baseline_score": _clip01(_safe_float(candidate.get("v27_baseline_score"))),
            "support_score": _clip01(_safe_float(candidate.get("v27_support_score"))),
            "signal_score": _clip01(_safe_float(candidate.get("v27_signal_score"))),
            "member_count": _sat(candidate.get("v27_member_count"), 4.0),
            "compactness": _clip01(_safe_float(candidate.get("v27_compactness"))),
            "source_diversity": _clip01(_safe_float(candidate.get("v27_source_diversity")) / 4.0),
            "current_fraction": _clip01(_safe_float(candidate.get("v27_current_fraction"))),
            "carried_fraction": _clip01(_safe_float(candidate.get("v27_carried_fraction"))),
            "hits_max": _sat(candidate.get("v27_hits_max"), 3.0),
            "hits_mean": _sat(candidate.get("v27_hits_mean"), 2.2),
            "v1_fraction": _clip01(_safe_float(candidate.get("v27_v1_fraction"))),
            "v2_fraction": _clip01(_safe_float(candidate.get("v27_v2_fraction"))),
            "tile_fraction": _clip01(_safe_float(candidate.get("v27_tile_fraction"))),
            "agreement_fraction": _clip01(_safe_float(candidate.get("v27_agreement_fraction"))),
            "patch_prior_max": _clip01(_safe_float(candidate.get("v27_patch_prior_max"))),
            "patch_prior_median": _clip01(_safe_float(candidate.get("v27_patch_prior_median"))),
            "zscore": _clip01(_safe_float(candidate.get("v27_zscore_norm"))),
            "absdiff": _clip01(_safe_float(candidate.get("v27_absdiff_norm"))),
            "dog": _clip01(_safe_float(candidate.get("v27_dog_norm"))),
            "saliency": _clip01(_safe_float(candidate.get("v27_saliency_norm"))),
            "persistence": _clip01(_safe_float(candidate.get("v27_persistence_median"), 0.5)),
            "not_existed_before": 1.0 - _clip01(_safe_float(candidate.get("v27_existed_before_median"))),
            "age_good": math.exp(-max(0.0, _safe_float(candidate.get("v27_age_median_s"))) / 1.2),
        }

    def raw_score(self, candidate: Candidate) -> float:
        vector = self.vector(candidate)
        return sum(self.weights[key] * vector[key] for key in FEATURE_KEYS)

    def score(self, candidate: Candidate) -> float:
        return _sigmoid(self.raw_score(candidate))

    def rank(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        ranked: list[Candidate] = []
        for candidate in candidates:
            item = dict(candidate)
            raw = self.raw_score(candidate)
            item["ranker_v6_raw"] = float(raw)
            item["ranker_v6_score"] = float(_sigmoid(raw))
            ranked.append(item)
        ranked.sort(
            key=lambda candidate: (
                _safe_float(candidate.get("ranker_v6_raw")),
                _safe_float(candidate.get("v27_baseline_score")),
            ),
            reverse=True,
        )
        for index, candidate in enumerate(ranked, start=1):
            candidate["ranker_v6_rank"] = index
        return ranked

    @staticmethod
    def _rank_of_gt(pool: Sequence[Candidate], gt_xy: Point, radius: float) -> int | None:
        for index, candidate in enumerate(pool, start=1):
            if _distance(candidate, gt_xy) <= radius:
                return index
        return None

    def record_validation(
        self,
        gt_xy: Point,
        baseline_pool: Sequence[Candidate],
        v6_pool: Sequence[Candidate],
    ) -> dict[str, Any]:
        radius = float(self.config.get("validation_radius_px", 20.0))
        baseline = list(baseline_pool)
        v6 = list(v6_pool)
        nearest = min((_distance(candidate, gt_xy) for candidate in baseline), default=float("inf"))
        eligible = nearest <= radius
        baseline_rank = self._rank_of_gt(baseline, gt_xy, radius)
        v6_rank = self._rank_of_gt(v6, gt_xy, radius)
        scores = [_safe_float(candidate.get("ranker_v6_score"), 0.5) for candidate in v6[:2]]
        margin = scores[0] - scores[1] if len(scores) >= 2 else (scores[0] if scores else 0.0)
        record = {
            "timestamp": time.time(),
            "eligible": bool(eligible),
            "nearest_px": float(nearest) if math.isfinite(nearest) else None,
            "baseline_gt_rank": baseline_rank,
            "v6_gt_rank": v6_rank,
            "baseline_top1_correct": baseline_rank == 1,
            "v6_top1_correct": v6_rank == 1,
            "baseline_top3_correct": baseline_rank is not None and baseline_rank <= 3,
            "v6_top3_correct": v6_rank is not None and v6_rank <= 3,
            "v6_top_score": scores[0] if scores else None,
            "v6_margin": float(margin),
        }
        history = self.stats.setdefault("validation", [])
        history.append(record)
        window = max(50, int(self.config.get("validation_window", 220)))
        if len(history) > window * 2:
            del history[:-window * 2]
        self.stats["last_validation"] = record
        return record

    def gate_status(self) -> dict[str, Any]:
        history = self.stats.get("validation", [])
        if not isinstance(history, list):
            history = []
        window = max(50, int(self.config.get("validation_window", 220)))
        eligible = [
            row for row in history[-window:]
            if isinstance(row, dict) and bool(row.get("eligible"))
        ]
        n = len(eligible)
        baseline_top1 = sum(bool(row.get("baseline_top1_correct")) for row in eligible)
        v6_top1 = sum(bool(row.get("v6_top1_correct")) for row in eligible)
        baseline_top3 = sum(bool(row.get("baseline_top3_correct")) for row in eligible)
        v6_top3 = sum(bool(row.get("v6_top3_correct")) for row in eligible)
        baseline_pct = 100.0 * baseline_top1 / n if n else 0.0
        v6_pct = 100.0 * v6_top1 / n if n else 0.0
        baseline_top3_pct = 100.0 * baseline_top3 / n if n else 0.0
        v6_top3_pct = 100.0 * v6_top3 / n if n else 0.0

        baseline_ranks = [
            int(row["baseline_gt_rank"]) for row in eligible
            if row.get("baseline_gt_rank") is not None
        ]
        v6_ranks = [
            int(row["v6_gt_rank"]) for row in eligible
            if row.get("v6_gt_rank") is not None
        ]
        baseline_median = _median(baseline_ranks)
        v6_median = _median(v6_ranks)
        rank_ratio = (
            float(v6_median) / max(1.0, float(baseline_median))
            if baseline_median is not None and v6_median is not None
            else None
        )

        min_shots = max(20, int(self.config.get("min_validation_shots", 120)))
        min_top1 = float(self.config.get("min_conditional_top1_pct", 10.0))
        min_advantage = float(self.config.get("min_advantage_pp", 5.0))
        min_top3_adv = float(self.config.get("min_top3_advantage_pp", 4.0))
        max_rank_ratio = float(self.config.get("max_median_rank_ratio", 0.72))
        open_gate = (
            bool(self.config.get("auto_override_enabled", True))
            and n >= min_shots
            and v6_pct >= min_top1
            and (v6_pct - baseline_pct) >= min_advantage
            and (v6_top3_pct - baseline_top3_pct) >= min_top3_adv
            and rank_ratio is not None
            and rank_ratio <= max_rank_ratio
        )
        return {
            "open": bool(open_gate),
            "eligible": n,
            "baseline_top1_pct": round(baseline_pct, 3),
            "v6_top1_pct": round(v6_pct, 3),
            "advantage_pp": round(v6_pct - baseline_pct, 3),
            "baseline_top3_pct": round(baseline_top3_pct, 3),
            "v6_top3_pct": round(v6_top3_pct, 3),
            "top3_advantage_pp": round(v6_top3_pct - baseline_top3_pct, 3),
            "baseline_median_rank": baseline_median,
            "v6_median_rank": v6_median,
            "median_rank_ratio": round(rank_ratio, 4) if rank_ratio is not None else None,
        }

    def choose_authoritative(self, v6_pool: Sequence[Candidate]) -> tuple[bool, dict[str, Any]]:
        gate = self.gate_status()
        if not bool(gate.get("open")) or not v6_pool:
            return False, {**gate, "reason": "gate_closed" if not gate.get("open") else "empty"}
        scores = [_safe_float(candidate.get("ranker_v6_score"), 0.5) for candidate in v6_pool[:2]]
        top = scores[0] if scores else 0.0
        margin = top - scores[1] if len(scores) >= 2 else top
        min_score = float(self.config.get("min_top_score", 0.535))
        min_margin = float(self.config.get("min_score_margin", 0.030))
        allowed = top >= min_score and margin >= min_margin
        return allowed, {
            **gate,
            "reason": "confidence_ok" if allowed else "low_current_confidence",
            "top_score": round(top, 6),
            "margin": round(margin, 6),
        }

    def learn_from_ground_truth(
        self,
        gt_xy: Point,
        hypotheses: Sequence[Candidate],
    ) -> dict[str, Any]:
        pool = list(hypotheses)
        if not pool:
            self.stats["skipped_no_positive"] = int(self.stats.get("skipped_no_positive", 0)) + 1
            return {"trained": False, "reason": "empty_pool"}

        strict_radius = float(self.config.get("strict_positive_radius_px", 12.0))
        soft_radius = max(strict_radius, float(self.config.get("soft_positive_radius_px", 20.0)))
        negative_radius = max(soft_radius + 5.0, float(self.config.get("negative_safe_radius_px", 55.0)))

        nearest = min(pool, key=lambda candidate: _distance(candidate, gt_xy))
        nearest_distance = _distance(nearest, gt_xy)
        if nearest_distance <= strict_radius:
            label_kind = "strict"
            label_weight = 1.0
        elif nearest_distance <= soft_radius:
            label_kind = "soft"
            label_weight = max(0.05, min(1.0, float(self.config.get("soft_positive_weight", 0.42))))
        else:
            self.stats["skipped_no_positive"] = int(self.stats.get("skipped_no_positive", 0)) + 1
            return {
                "trained": False,
                "reason": "no_positive_within_soft_radius",
                "nearest_distance_px": float(nearest_distance),
            }

        negatives = [
            candidate for candidate in pool
            if candidate is not nearest and _distance(candidate, gt_xy) >= negative_radius
        ]
        # Hard negatives from both learned rank and deterministic baseline. This
        # prevents the model from learning only against one failure mode.
        negatives.sort(
            key=lambda candidate: (
                max(self.score(candidate), _safe_float(candidate.get("v27_baseline_score"))),
                _safe_float(candidate.get("v27_baseline_score")),
            ),
            reverse=True,
        )
        negatives = negatives[:max(1, int(self.config.get("hard_negatives", 28)))]
        if not negatives:
            return {
                "trained": False,
                "reason": "no_safe_negatives",
                "nearest_distance_px": float(nearest_distance),
                "label_kind": label_kind,
            }

        positive_vector = self.vector(nearest)
        lr = max(1e-5, float(self.config.get("learning_rate", 0.070))) * label_weight
        l2 = max(0.0, min(0.1, float(self.config.get("l2", 0.0009))))
        epochs = max(1, int(self.config.get("epochs_per_shot", 1)))
        losses: list[float] = []
        pair_updates = 0

        for _ in range(epochs):
            for negative in negatives:
                negative_vector = self.vector(negative)
                diff = {
                    key: positive_vector[key] - negative_vector[key]
                    for key in FEATURE_KEYS
                }
                margin = sum(self.weights[key] * diff[key] for key in FEATURE_KEYS)
                probability = _sigmoid(margin)
                losses.append(-math.log(max(1e-9, probability)))
                gradient_scale = (1.0 - probability) * label_weight
                for key in FEATURE_KEYS:
                    weight = self.weights[key] * (1.0 - lr * l2)
                    weight += lr * gradient_scale * diff[key]
                    self.weights[key] = max(-6.0, min(6.0, weight))
                pair_updates += 1

        stat_key = "strict_positive_shots" if label_kind == "strict" else "soft_positive_shots"
        self.stats[stat_key] = int(self.stats.get(stat_key, 0)) + 1
        self.stats["pair_updates"] = int(self.stats.get("pair_updates", 0)) + pair_updates
        self.stats["last_loss"] = sum(losses) / len(losses) if losses else None
        self.stats["last_updated"] = time.time()
        self.stats["last_label_kind"] = label_kind
        self.stats["last_positive_distance_px"] = float(nearest_distance)

        trained_shots = int(self.stats.get("strict_positive_shots", 0)) + int(self.stats.get("soft_positive_shots", 0))
        save_every = max(1, int(self.config.get("save_every_shots", 5)))
        if trained_shots % save_every == 0:
            self.save()

        return {
            "trained": True,
            "label_kind": label_kind,
            "label_weight": float(label_weight),
            "positive_distance_px": float(nearest_distance),
            "pairs": pair_updates,
            "loss": self.stats.get("last_loss"),
        }

    def summary(self) -> dict[str, Any]:
        strongest = sorted(
            self.weights.items(),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:8]
        return {
            "version": self.VERSION,
            "strict_positive_shots": int(self.stats.get("strict_positive_shots", 0)),
            "soft_positive_shots": int(self.stats.get("soft_positive_shots", 0)),
            "pair_updates": int(self.stats.get("pair_updates", 0)),
            "skipped_no_positive": int(self.stats.get("skipped_no_positive", 0)),
            "last_loss": self.stats.get("last_loss"),
            "strongest_weights": [[key, round(value, 5)] for key, value in strongest],
            "gate": self.gate_status(),
        }


__all__ = ["DEFAULT_CONFIG", "FEATURE_KEYS", "RankerV6"]
