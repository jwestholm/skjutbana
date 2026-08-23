from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence


Candidate = dict[str, Any]
Point = tuple[float, float]

PATCH_GRID_SIZE = 5
PATCH_ABS_GRID_KEYS = [f"patch_abs_g{r}{c}" for r in range(PATCH_GRID_SIZE) for c in range(PATCH_GRID_SIZE)]
PATCH_SIGNED_GRID_KEYS = [f"patch_signed_g{r}{c}" for r in range(PATCH_GRID_SIZE) for c in range(PATCH_GRID_SIZE)]
PATCH_Z_GRID_KEYS = [f"patch_z_g{r}{c}" for r in range(PATCH_GRID_SIZE) for c in range(PATCH_GRID_SIZE)]

FEATURE_KEYS = [
    # Detector/source features. Unlike V4, V5 may learn from them because V5's
    # positive label is an ACTUAL candidate <=12 px from synthetic GT.
    "detector_score",
    "detector_v1",
    "detector_v2",
    "detector_agreement",
    "v24_tile_probe",
    "candidate_bank_confirmed",
    "shot_accumulator_hits",
    "shot_accumulator_stability",
    "v26_vault_carried",
    "v26_vault_hits",
    "v26_vault_age",
    # Local temporal / patch descriptors.
    "v2_absdiff",
    "v2_zscore",
    "v2_dog",
    "v2_saliency",
    "patch_core_abs",
    "patch_ring_abs",
    "patch_outer_abs",
    "patch_core_z",
    "patch_ring_z",
    "patch_core_dark",
    "patch_ring_bright",
    "patch_core_to_outer",
    "patch_compactness",
    "patch_centeredness",
    "patch_isotropy",
    "patch_bipolar",
    "patch_local_snr",
    "patch_ringness",
] + PATCH_ABS_GRID_KEYS + PATCH_SIGNED_GRID_KEYS + PATCH_Z_GRID_KEYS

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "shadow_enabled": True,
    "positive_radius_px": 12.0,
    "negative_safe_radius_px": 55.0,
    "hard_negatives": 24,
    "learning_rate": 0.060,
    "l2": 0.0008,
    "epochs_per_shot": 1,
    "save_every_shots": 5,
    "training_log_enabled": True,
    # Conservative auto-promotion. V5 may only move its favourite candidate to
    # rank 1 after previous PRE-TRAIN validation demonstrates a real advantage.
    "auto_override_enabled": True,
    "validation_window": 180,
    "min_validation_shots": 80,
    "min_conditional_top1_pct": 12.0,
    "min_advantage_pp": 6.0,
    "min_score_margin": 0.035,
    "min_top_score": 0.54,
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
    value_f = max(0.0, _safe_float(value))
    return math.tanh(value_f / max(1e-6, float(scale)))


def _sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def _distance(candidate: Candidate, gt_xy: Point) -> float:
    return math.hypot(
        _safe_float(candidate.get("camera_x")) - float(gt_xy[0]),
        _safe_float(candidate.get("camera_y")) - float(gt_xy[1]),
    )


class RankerV5:
    """Strict-candidate supervised pairwise ranker.

    V4 learned from a descriptor sampled exactly at projected GT. V2.5 proved
    that the strongest visible temporal response is often tens of camera pixels
    from that projected point and V4 learned inverted/implausible weights.

    V5 therefore labels ONLY a real detector candidate <=12 px from GT as a
    positive. If no such candidate exists the shot is skipped for training.
    Current-shot prediction is always evaluated before that shot is learned.
    """

    VERSION = 5

    def __init__(
        self,
        model_path: Path = Path("content/ai/ranker_v5.json"),
        config_path: Path = Path("content/ai/ranker_v5_config.json"),
        log_path: Path = Path("content/ai/ranker_v5/training_pairs.jsonl"),
    ) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.log_path = Path(log_path)
        self.config = dict(DEFAULT_CONFIG)
        self.weights = {key: 0.0 for key in FEATURE_KEYS}
        # Tiny priors only; supervised pairs are expected to dominate quickly.
        self.weights["detector_agreement"] = 0.05
        self.weights["v24_tile_probe"] = 0.03
        self.stats: dict[str, Any] = {
            "version": self.VERSION,
            "positive_shots": 0,
            "pair_updates": 0,
            "skipped_no_positive": 0,
            "last_loss": None,
            "last_updated": None,
            "validation": [],
            "override_count": 0,
        }
        self._load_config()
        self.load()

    def _load_config(self) -> None:
        config = dict(DEFAULT_CONFIG)
        try:
            if self.config_path.exists():
                loaded = json.loads(self.config_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    config.update(loaded)
        except Exception:
            pass
        self.config = config

    def load(self) -> None:
        if not self.model_path.exists():
            return
        try:
            payload = json.loads(self.model_path.read_text(encoding="utf-8"))
            raw_weights = payload.get("weights", {})
            if isinstance(raw_weights, dict):
                for key in FEATURE_KEYS:
                    if key in raw_weights:
                        self.weights[key] = _safe_float(raw_weights.get(key))
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
            self.model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def reset(self) -> None:
        self.weights = {key: 0.0 for key in FEATURE_KEYS}
        self.weights["detector_agreement"] = 0.05
        self.weights["v24_tile_probe"] = 0.03
        self.stats = {
            "version": self.VERSION,
            "positive_shots": 0,
            "pair_updates": 0,
            "skipped_no_positive": 0,
            "last_loss": None,
            "last_updated": time.time(),
            "validation": [],
            "override_count": 0,
        }
        self.save()

    def vector(self, candidate: Candidate) -> dict[str, float]:
        result = {
            "detector_score": _sat(candidate.get("score", 0.0), 15.0),
            "detector_v1": _clip01(_safe_float(candidate.get("detector_v1", 0.0))),
            "detector_v2": _clip01(_safe_float(candidate.get("detector_v2", 0.0))),
            "detector_agreement": _clip01(_safe_float(candidate.get("detector_agreement", 0.0))),
            "v24_tile_probe": _clip01(_safe_float(candidate.get("v24_tile_probe", 0.0))),
            "candidate_bank_confirmed": _clip01(max(
                _safe_float(candidate.get("candidate_bank_confirmed", 0.0)),
                _safe_float(candidate.get("v2_bank_confirmed", 0.0)),
            )),
            "shot_accumulator_hits": _sat(candidate.get("shot_accumulator_hits", 0.0), 3.0),
            "shot_accumulator_stability": _clip01(_safe_float(candidate.get("shot_accumulator_stability", 0.0))),
            "v26_vault_carried": _clip01(_safe_float(candidate.get("v26_vault_carried", 0.0))),
            "v26_vault_hits": _sat(candidate.get("v26_vault_hits", 0.0), 3.0),
            "v26_vault_age": _sat(candidate.get("v26_vault_age_s", 0.0), 0.45),
            "v2_absdiff": _sat(candidate.get("v2_absdiff", candidate.get("v24_patch_core_abs", 0.0)), 7.0),
            "v2_zscore": _sat(candidate.get("v2_zscore", candidate.get("v24_patch_core_z", 0.0)), 3.5),
            "v2_dog": _sat(candidate.get("v2_dog", 0.0), 9.0),
            "v2_saliency": _sat(candidate.get("v2_saliency", 0.0), 28.0),
            "patch_core_abs": _sat(candidate.get("v24_patch_core_abs", 0.0), 7.0),
            "patch_ring_abs": _sat(candidate.get("v24_patch_ring_abs", 0.0), 7.0),
            "patch_outer_abs": _sat(candidate.get("v24_patch_outer_abs", 0.0), 7.0),
            "patch_core_z": _sat(candidate.get("v24_patch_core_z", 0.0), 3.5),
            "patch_ring_z": _sat(candidate.get("v24_patch_ring_z", 0.0), 3.5),
            "patch_core_dark": _sat(candidate.get("v24_patch_core_dark", 0.0), 7.0),
            "patch_ring_bright": _sat(candidate.get("v24_patch_ring_bright", 0.0), 7.0),
            "patch_core_to_outer": _clip01(_safe_float(candidate.get("v24_patch_core_to_outer", 0.0))),
            "patch_compactness": _clip01(_safe_float(candidate.get("v24_patch_compactness", 0.0))),
            "patch_centeredness": _clip01(_safe_float(candidate.get("v24_patch_centeredness", 0.0))),
            "patch_isotropy": _clip01(_safe_float(candidate.get("v24_patch_isotropy", 0.0))),
            "patch_bipolar": _clip01(_safe_float(candidate.get("v24_patch_bipolar", 0.0))),
            "patch_local_snr": _clip01(_safe_float(candidate.get("v24_patch_local_snr", 0.0))),
            "patch_ringness": _clip01(_safe_float(candidate.get("v24_patch_ringness", 0.0))),
        }
        for row in range(PATCH_GRID_SIZE):
            for col in range(PATCH_GRID_SIZE):
                suffix = f"g{row}{col}"
                result[f"patch_abs_{suffix}"] = max(0.0, min(1.5, _safe_float(candidate.get(f"v24_patch_abs_{suffix}", 0.0))))
                result[f"patch_signed_{suffix}"] = max(-1.0, min(1.0, _safe_float(candidate.get(f"v24_patch_signed_{suffix}", 0.0))))
                result[f"patch_z_{suffix}"] = max(0.0, min(1.5, _safe_float(candidate.get(f"v24_patch_z_{suffix}", 0.0))))
        return result

    def raw_score(self, candidate: Candidate) -> float:
        vector = self.vector(candidate)
        return sum(self.weights[key] * vector[key] for key in FEATURE_KEYS)

    def score(self, candidate: Candidate) -> float:
        return _sigmoid(self.raw_score(candidate))

    def rank(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        ranked: list[Candidate] = []
        for candidate in candidates:
            item = dict(candidate)
            item["ranker_v5_raw"] = float(self.raw_score(candidate))
            item["ranker_v5_score"] = float(_sigmoid(item["ranker_v5_raw"]))
            ranked.append(item)
        ranked.sort(
            key=lambda candidate: (
                _safe_float(candidate.get("ranker_v5_raw")),
                _safe_float(candidate.get("combined_score", candidate.get("score", 0.0))),
            ),
            reverse=True,
        )
        for index, candidate in enumerate(ranked, start=1):
            candidate["ranker_v5_rank"] = index
        return ranked

    def record_validation(
        self,
        gt_xy: Point,
        base_pool: Sequence[Candidate],
        v5_pool: Sequence[Candidate],
        *,
        match_radius_px: float = 42.0,
    ) -> dict[str, Any]:
        base = list(base_pool)
        v5 = list(v5_pool)
        nearest = min((_distance(candidate, gt_xy) for candidate in base), default=float("inf"))
        eligible = nearest <= float(match_radius_px)

        base_top1_d = _distance(base[0], gt_xy) if base else float("inf")
        v5_top1_d = _distance(v5[0], gt_xy) if v5 else float("inf")
        base_top3 = any(_distance(candidate, gt_xy) <= match_radius_px for candidate in base[:3])
        v5_top3 = any(_distance(candidate, gt_xy) <= match_radius_px for candidate in v5[:3])

        def gt_rank(pool: list[Candidate]) -> int | None:
            for index, candidate in enumerate(pool, start=1):
                if _distance(candidate, gt_xy) <= match_radius_px:
                    return index
            return None

        scores = [_safe_float(candidate.get("ranker_v5_score"), 0.5) for candidate in v5[:2]]
        margin = scores[0] - scores[1] if len(scores) >= 2 else (scores[0] if scores else 0.0)
        record = {
            "timestamp": time.time(),
            "eligible": bool(eligible),
            "nearest_px": float(nearest) if math.isfinite(nearest) else None,
            "base_top1_correct": bool(base_top1_d <= match_radius_px),
            "v5_top1_correct": bool(v5_top1_d <= match_radius_px),
            "base_top3_correct": bool(base_top3),
            "v5_top3_correct": bool(v5_top3),
            "base_gt_rank": gt_rank(base),
            "v5_gt_rank": gt_rank(v5),
            "v5_top_score": scores[0] if scores else None,
            "v5_margin": float(margin),
        }
        history = self.stats.setdefault("validation", [])
        history.append(record)
        window = max(40, int(self.config.get("validation_window", 180)))
        if len(history) > window * 2:
            del history[:-window * 2]
        self.stats["last_validation"] = record
        return record

    def gate_status(self) -> dict[str, Any]:
        history = self.stats.get("validation", [])
        if not isinstance(history, list):
            history = []
        window = max(40, int(self.config.get("validation_window", 180)))
        eligible = [row for row in history[-window:] if isinstance(row, dict) and bool(row.get("eligible"))]
        n = len(eligible)
        base_correct = sum(1 for row in eligible if bool(row.get("base_top1_correct")))
        v5_correct = sum(1 for row in eligible if bool(row.get("v5_top1_correct")))
        base_pct = 100.0 * base_correct / n if n else 0.0
        v5_pct = 100.0 * v5_correct / n if n else 0.0
        advantage = v5_pct - base_pct
        minimum_n = max(10, int(self.config.get("min_validation_shots", 80)))
        min_top1 = _safe_float(self.config.get("min_conditional_top1_pct", 12.0), 12.0)
        min_adv = _safe_float(self.config.get("min_advantage_pp", 6.0), 6.0)
        enabled = bool(self.config.get("auto_override_enabled", True))
        open_gate = bool(enabled and n >= minimum_n and v5_pct >= min_top1 and advantage >= min_adv)
        return {
            "open": open_gate,
            "eligible_shots": n,
            "base_top1_pct": round(base_pct, 3),
            "v5_top1_pct": round(v5_pct, 3),
            "advantage_pp": round(advantage, 3),
            "min_validation_shots": minimum_n,
            "min_conditional_top1_pct": min_top1,
            "min_advantage_pp": min_adv,
        }

    def override_candidate(self, v5_pool: Sequence[Candidate]) -> tuple[Candidate | None, dict[str, Any]]:
        pool = list(v5_pool)
        gate = self.gate_status()
        if not bool(gate.get("open")) or not pool:
            return None, {**gate, "confidence_ok": False, "reason": "gate_closed"}
        top_score = _safe_float(pool[0].get("ranker_v5_score"), 0.5)
        second = _safe_float(pool[1].get("ranker_v5_score"), 0.5) if len(pool) > 1 else 0.0
        margin = top_score - second
        min_margin = _safe_float(self.config.get("min_score_margin", 0.035), 0.035)
        min_top = _safe_float(self.config.get("min_top_score", 0.54), 0.54)
        confidence_ok = top_score >= min_top and margin >= min_margin
        if not confidence_ok:
            return None, {
                **gate,
                "confidence_ok": False,
                "reason": "low_confidence",
                "top_score": top_score,
                "margin": margin,
            }
        return dict(pool[0]), {
            **gate,
            "confidence_ok": True,
            "reason": "validated_override",
            "top_score": top_score,
            "margin": margin,
        }

    def learn_from_ground_truth(self, gt_xy: Point, candidates: Sequence[Candidate]) -> dict[str, Any]:
        pool = list(candidates)
        if not bool(self.config.get("enabled", True)):
            return {"trained": False, "reason": "disabled"}
        if not pool:
            self._skip()
            return {"trained": False, "reason": "no_candidates"}

        positive_radius = max(1.0, _safe_float(self.config.get("positive_radius_px", 12.0), 12.0))
        safe_radius = max(positive_radius + 1.0, _safe_float(self.config.get("negative_safe_radius_px", 55.0), 55.0))
        nearest_index = min(range(len(pool)), key=lambda index: _distance(pool[index], gt_xy))
        nearest_distance = _distance(pool[nearest_index], gt_xy)
        if nearest_distance > positive_radius:
            self._skip()
            return {
                "trained": False,
                "reason": "no_strict_positive",
                "nearest_distance": float(nearest_distance),
            }

        positive = pool[nearest_index]
        pos_vec = self.vector(positive)
        negatives = [
            candidate
            for index, candidate in enumerate(pool)
            if index != nearest_index and _distance(candidate, gt_xy) >= safe_radius
        ]
        negatives.sort(
            key=lambda candidate: (
                _safe_float(candidate.get("combined_score", candidate.get("score", 0.0))),
                self.raw_score(candidate),
            ),
            reverse=True,
        )
        negatives = negatives[: max(1, int(self.config.get("hard_negatives", 24)))]
        if not negatives:
            return {"trained": False, "reason": "no_safe_negatives"}

        lr = max(1e-5, _safe_float(self.config.get("learning_rate", 0.060), 0.060))
        l2 = max(0.0, min(0.05, _safe_float(self.config.get("l2", 0.0008), 0.0008)))
        epochs = max(1, int(self.config.get("epochs_per_shot", 1)))
        losses: list[float] = []
        updates = 0
        for _epoch in range(epochs):
            for negative in negatives:
                neg_vec = self.vector(negative)
                diff = {key: pos_vec[key] - neg_vec[key] for key in FEATURE_KEYS}
                margin = sum(self.weights[key] * diff[key] for key in FEATURE_KEYS)
                probability = _sigmoid(margin)
                losses.append(-math.log(max(1e-9, probability)))
                gradient_scale = 1.0 - probability
                for key in FEATURE_KEYS:
                    weight = self.weights[key]
                    weight *= 1.0 - lr * l2
                    weight += lr * gradient_scale * diff[key]
                    self.weights[key] = max(-6.0, min(6.0, weight))
                updates += 1

        self.stats["positive_shots"] = int(self.stats.get("positive_shots", 0)) + 1
        self.stats["pair_updates"] = int(self.stats.get("pair_updates", 0)) + updates
        self.stats["last_loss"] = sum(losses) / len(losses) if losses else None
        self.stats["last_updated"] = time.time()
        self.stats["last_positive_distance"] = float(nearest_distance)

        if int(self.stats["positive_shots"]) % max(1, int(self.config.get("save_every_shots", 5))) == 0:
            self.save()
        self._append_log(gt_xy, positive, negatives, nearest_distance)
        return {
            "trained": True,
            "positive_shots": int(self.stats["positive_shots"]),
            "pair_updates": int(self.stats["pair_updates"]),
            "updates_this_shot": updates,
            "nearest_distance": float(nearest_distance),
            "loss": self.stats.get("last_loss"),
        }

    def _skip(self) -> None:
        self.stats["skipped_no_positive"] = int(self.stats.get("skipped_no_positive", 0)) + 1

    def _append_log(self, gt_xy: Point, positive: Candidate, negatives: Sequence[Candidate], nearest_distance: float) -> None:
        if not bool(self.config.get("training_log_enabled", True)):
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": "5.0",
                "timestamp": time.time(),
                "gt": [float(gt_xy[0]), float(gt_xy[1])],
                "nearest_distance": float(nearest_distance),
                "positive": {
                    "camera_x": _safe_float(positive.get("camera_x")),
                    "camera_y": _safe_float(positive.get("camera_y")),
                    "vector": self.vector(positive),
                },
                "hard_negatives": [
                    {
                        "camera_x": _safe_float(candidate.get("camera_x")),
                        "camera_y": _safe_float(candidate.get("camera_y")),
                        "vector": self.vector(candidate),
                    }
                    for candidate in negatives[:6]
                ],
            }
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def summary(self) -> dict[str, Any]:
        strongest = sorted(self.weights.items(), key=lambda item: abs(float(item[1])), reverse=True)[:10]
        gate = self.gate_status()
        return {
            "version": self.VERSION,
            "positive_shots": int(self.stats.get("positive_shots", 0)),
            "pair_updates": int(self.stats.get("pair_updates", 0)),
            "skipped_no_positive": int(self.stats.get("skipped_no_positive", 0)),
            "last_loss": self.stats.get("last_loss"),
            "override_count": int(self.stats.get("override_count", 0)),
            "gate": gate,
            "strongest_weights": strongest,
        }
