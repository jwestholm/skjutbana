from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Sequence


Candidate = dict[str, Any]
Point = tuple[float, float]

DEFAULT_PATH = Path("content/ai/ranker_v3.json")

# Fixed, position-independent feature representation.  The transforms below
# deliberately avoid dataset min/max normalization so one historical outlier
# cannot change the meaning of every later candidate.
FEATURE_KEYS = [
    "detector_score",
    "area",
    "radius",
    "circularity",
    "center_change",
    "local_contrast",
    "pre_shot_change",
    "change_value",
    "patch_std",
    "edge_strength",
    "persistence",
    "existed_before",
    "detector_v1",
    "detector_v2",
    "detector_agreement",
    "bank_confirmed",
    "bank_hits",
    "v2_absdiff",
    "v2_zscore",
    "v2_dog",
    "v2_saliency",
    "v2_primary",
    "v2_rescue_temporal",
    "v2_rescue_blob",
    "v2_refine_shift",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float(default)
    except Exception:
        return float(default)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sat_positive(value: Any, scale: float) -> float:
    """Smooth bounded transform for non-negative detector features."""
    v = max(0.0, _safe_float(value))
    scale = max(1e-6, float(scale))
    return math.tanh(v / scale)


def _sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


class PairwiseRankerV3:
    """Tiny online pairwise logistic ranker.

    It learns from *within-shot* comparisons:

        true candidate > hard wrong candidate

    Synthetic F2 training already provides exact camera-space ground truth, so
    no neural-network dependency is needed to learn a useful ordering signal.

    The model is intentionally small, transparent and JSON-persisted.  It does
    not replace the existing AI memory; runtime blends it in only after enough
    labelled pairs have been observed.
    """

    VERSION = 3

    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.weights = {key: 0.0 for key in FEATURE_KEYS}
        self.stats: dict[str, Any] = {
            "version": self.VERSION,
            "pair_updates": 0,
            "positive_shots": 0,
            "skipped_no_positive": 0,
            "last_updated": None,
        }
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            raw = payload.get("weights", {})
            if isinstance(raw, dict):
                for key in FEATURE_KEYS:
                    self.weights[key] = _safe_float(raw.get(key, 0.0))
            stats = payload.get("stats", {})
            if isinstance(stats, dict):
                self.stats.update(stats)
        except Exception:
            # A corrupt experimental ranker must never prevent the game from
            # starting. Keep a clean model and allow new training to rebuild it.
            self.weights = {key: 0.0 for key in FEATURE_KEYS}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "feature_keys": FEATURE_KEYS,
            "weights": self.weights,
            "stats": self.stats,
        }
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def reset(self) -> None:
        self.weights = {key: 0.0 for key in FEATURE_KEYS}
        self.stats = {
            "version": self.VERSION,
            "pair_updates": 0,
            "positive_shots": 0,
            "skipped_no_positive": 0,
            "last_updated": time.time(),
        }
        self.save()

    # ------------------------------------------------------------------
    # Features / scoring
    # ------------------------------------------------------------------

    def vector(self, candidate: Candidate) -> dict[str, float]:
        old_features = candidate.get("features", {})
        if not isinstance(old_features, dict):
            old_features = {}

        def old(name: str, fallback: Any = 0.0) -> Any:
            return old_features.get(name, fallback)

        bank_confirmed = max(
            _safe_float(candidate.get("candidate_bank_confirmed", 0.0)),
            _safe_float(candidate.get("v2_bank_confirmed", 0.0)),
        )
        bank_hits = max(
            _safe_float(candidate.get("candidate_bank_hits", 0.0)),
            _safe_float(candidate.get("v2_bank_hits", 0.0)),
        )

        return {
            "detector_score": _sat_positive(candidate.get("score", old("detector_score")), 18.0),
            "area": _sat_positive(candidate.get("area", old("area")), 120.0),
            "radius": _sat_positive(candidate.get("radius", old("radius")), 14.0),
            "circularity": _clip01(_safe_float(candidate.get("circularity", old("circularity")))),
            "center_change": _sat_positive(
                candidate.get("center_darkening", old("center_change")), 14.0
            ),
            "local_contrast": _sat_positive(
                candidate.get("local_contrast_gain", old("local_contrast")), 14.0
            ),
            "pre_shot_change": _sat_positive(
                candidate.get("pre_shot_change", old("pre_shot_change")), 14.0
            ),
            "change_value": _sat_positive(
                candidate.get("change_value", old("change_value")), 14.0
            ),
            "patch_std": _sat_positive(old("patch_std"), 28.0),
            "edge_strength": _sat_positive(old("edge_strength"), 35.0),
            "persistence": _clip01(_safe_float(candidate.get("persistence", 0.5))),
            "existed_before": _clip01(_safe_float(candidate.get("existed_before", 0.0))),
            "detector_v1": _clip01(_safe_float(candidate.get("detector_v1", 0.0))),
            "detector_v2": _clip01(_safe_float(candidate.get("detector_v2", 0.0))),
            "detector_agreement": _clip01(
                _safe_float(candidate.get("detector_agreement", 0.0))
            ),
            "bank_confirmed": _clip01(bank_confirmed),
            "bank_hits": _sat_positive(bank_hits, 4.0),
            "v2_absdiff": _sat_positive(candidate.get("v2_absdiff", 0.0), 8.0),
            "v2_zscore": _sat_positive(candidate.get("v2_zscore", 0.0), 4.0),
            "v2_dog": _sat_positive(candidate.get("v2_dog", 0.0), 10.0),
            "v2_saliency": _sat_positive(candidate.get("v2_saliency", 0.0), 35.0),
            "v2_primary": _clip01(_safe_float(candidate.get("v2_primary_peak", 0.0))),
            "v2_rescue_temporal": _clip01(
                _safe_float(candidate.get("v2_rescue_temporal", 0.0))
            ),
            "v2_rescue_blob": _clip01(
                _safe_float(candidate.get("v2_rescue_blob", 0.0))
            ),
            # A large refine shift can mean that the coarse peak was actually
            # on a nearby ring/edge. The learner is allowed to discover whether
            # this should be positive or negative.
            "v2_refine_shift": _sat_positive(
                candidate.get("v2_refine_shift_px", 0.0), 5.0
            ),
        }

    def raw_score(self, candidate: Candidate) -> float:
        vector = self.vector(candidate)
        return sum(self.weights[key] * vector[key] for key in FEATURE_KEYS)

    def score(self, candidate: Candidate) -> float:
        return _sigmoid(self.raw_score(candidate))

    def effective_weight(
        self,
        *,
        min_positive_shots: int = 20,
        full_weight_shots: int = 120,
        max_weight: float = 0.72,
    ) -> float:
        positives = int(self.stats.get("positive_shots", 0) or 0)
        if positives < max(0, int(min_positive_shots)):
            return 0.0
        full = max(min_positive_shots + 1, int(full_weight_shots))
        progress = (positives - min_positive_shots) / float(full - min_positive_shots)
        return max(0.0, min(float(max_weight), float(max_weight) * _clip01(progress)))

    # ------------------------------------------------------------------
    # Online pairwise training
    # ------------------------------------------------------------------

    def learn_from_ground_truth(
        self,
        click_camera_xy: Point,
        candidates: Sequence[Candidate],
        *,
        positive_radius_px: float = 42.0,
        negative_safe_radius_px: float = 70.0,
        hard_negative_count: int = 12,
        learning_rate: float = 0.075,
        l2: float = 0.0008,
    ) -> dict[str, Any]:
        if not candidates:
            self.stats["skipped_no_positive"] = int(
                self.stats.get("skipped_no_positive", 0) or 0
            ) + 1
            return {"trained": False, "reason": "no_candidates"}

        gt_x, gt_y = float(click_camera_xy[0]), float(click_camera_xy[1])
        nearest_index = -1
        nearest_distance = float("inf")

        for index, candidate in enumerate(candidates):
            distance = math.hypot(
                _safe_float(candidate.get("camera_x", 0.0)) - gt_x,
                _safe_float(candidate.get("camera_y", 0.0)) - gt_y,
            )
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = index

        if nearest_index < 0 or nearest_distance > float(positive_radius_px):
            self.stats["skipped_no_positive"] = int(
                self.stats.get("skipped_no_positive", 0) or 0
            ) + 1
            return {
                "trained": False,
                "reason": "gt_candidate_missing",
                "nearest_distance": float(nearest_distance),
            }

        positive = candidates[nearest_index]
        pos_vec = self.vector(positive)

        # Hard negatives are the candidates the current system most strongly
        # preferred while still being safely outside the GT acceptance region.
        negatives: list[Candidate] = []
        for index, candidate in enumerate(candidates):
            if index == nearest_index:
                continue
            distance = math.hypot(
                _safe_float(candidate.get("camera_x", 0.0)) - gt_x,
                _safe_float(candidate.get("camera_y", 0.0)) - gt_y,
            )
            if distance <= float(negative_safe_radius_px):
                continue
            negatives.append(candidate)

        negatives.sort(
            key=lambda candidate: (
                _safe_float(candidate.get("combined_score", 0.0)),
                _safe_float(candidate.get("score", 0.0)),
            ),
            reverse=True,
        )
        negatives = negatives[: max(1, int(hard_negative_count))]

        if not negatives:
            return {
                "trained": False,
                "reason": "no_safe_negatives",
                "nearest_distance": float(nearest_distance),
            }

        lr = max(1e-5, float(learning_rate))
        decay = max(0.0, min(0.1, float(l2)))
        pair_updates = 0
        losses = []

        for negative in negatives:
            neg_vec = self.vector(negative)
            diff = {
                key: pos_vec[key] - neg_vec[key]
                for key in FEATURE_KEYS
            }
            margin = sum(self.weights[key] * diff[key] for key in FEATURE_KEYS)
            probability = _sigmoid(margin)
            # -log(sigmoid(margin))
            losses.append(-math.log(max(1e-9, probability)))
            gradient_scale = 1.0 - probability

            for key in FEATURE_KEYS:
                weight = self.weights[key]
                weight *= 1.0 - lr * decay
                weight += lr * gradient_scale * diff[key]
                self.weights[key] = max(-5.0, min(5.0, weight))

            pair_updates += 1

        self.stats["pair_updates"] = int(self.stats.get("pair_updates", 0) or 0) + pair_updates
        self.stats["positive_shots"] = int(
            self.stats.get("positive_shots", 0) or 0
        ) + 1
        self.stats["last_updated"] = time.time()
        self.stats["last_loss"] = (
            sum(losses) / len(losses) if losses else None
        )
        self.stats["last_positive_distance"] = float(nearest_distance)

        self.save()

        return {
            "trained": True,
            "pairs": pair_updates,
            "nearest_distance": float(nearest_distance),
            "loss": self.stats.get("last_loss"),
            "positive_shots": self.stats["positive_shots"],
            "pair_updates": self.stats["pair_updates"],
        }

    def summary(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "positive_shots": int(self.stats.get("positive_shots", 0) or 0),
            "pair_updates": int(self.stats.get("pair_updates", 0) or 0),
            "skipped_no_positive": int(
                self.stats.get("skipped_no_positive", 0) or 0
            ),
            "last_loss": self.stats.get("last_loss"),
            "last_updated": self.stats.get("last_updated"),
            "weights": dict(self.weights),
        }
