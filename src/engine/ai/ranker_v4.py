from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence


Candidate = dict[str, Any]
Point = tuple[float, float]

DEFAULT_MODEL_PATH = Path("content/ai/ranker_v4.json")
DEFAULT_CONFIG_PATH = Path("content/ai/ranker_v4_config.json")
DEFAULT_LOG_PATH = Path("content/ai/ranker_v4/training_pairs.jsonl")


PATCH_GRID_SIZE = 5
PATCH_ABS_GRID_KEYS = [
    f"patch_abs_g{row}{col}"
    for row in range(PATCH_GRID_SIZE)
    for col in range(PATCH_GRID_SIZE)
]
PATCH_SIGNED_GRID_KEYS = [
    f"patch_signed_g{row}{col}"
    for row in range(PATCH_GRID_SIZE)
    for col in range(PATCH_GRID_SIZE)
]
PATCH_Z_GRID_KEYS = [
    f"patch_z_g{row}{col}"
    for row in range(PATCH_GRID_SIZE)
    for col in range(PATCH_GRID_SIZE)
]

FEATURE_KEYS = [
    # Scalar local temporal / shape descriptors.
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
    "min_positive_shots": 20,
    "full_weight_shots": 180,
    "max_model_weight": 0.82,
    "initial_patch_weight": 0.55,
    "full_patch_weight": 0.22,
    "hard_negatives": 18,
    "learning_rate": 0.085,
    "l2": 0.0007,
    "epochs_per_shot": 2,
    # Fallback candidate-positive radius is deliberately much tighter than the
    # 42 px evaluation tolerance. Loose 42 px positives taught V3 from nearby
    # artifacts. Normally V4 instead learns from the exact synthetic GT patch.
    "positive_radius_px": 16.0,
    "negative_safe_radius_px": 58.0,
    "ground_truth_patch_enabled": True,
    "ground_truth_patch_min_absdiff": 1.2,
    "ground_truth_patch_min_local_snr": 0.10,
    "save_every_shots": 5,
    "training_log_enabled": True,
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


class RankerV4:
    """Small online pairwise ranker focused on *local hole shape*.

    V3 mostly saw scalar detector strength. Measurements showed that the true
    candidate survived into the ranked list in ~40-45% of synthetic rounds yet
    had a median rank around 78. V4 therefore adds descriptors of the local
    temporal patch: compactness, isotropy, centredness, dark-core/bright-ring
    behaviour and local SNR.

    Learning is pairwise and shot-local:

        ground-truth candidate > strongest safe wrong candidates

    The model is intentionally tiny and JSON persisted. No sklearn/PyTorch is
    required, and failures never disable the base ranking path.
    """

    VERSION = 4

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        config_path: Path = DEFAULT_CONFIG_PATH,
        log_path: Path = DEFAULT_LOG_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.log_path = Path(log_path)
        self.config = dict(DEFAULT_CONFIG)
        self.weights = self._initial_weights()
        self.stats: dict[str, Any] = {
            "version": self.VERSION,
            "positive_shots": 0,
            "pair_updates": 0,
            "skipped_no_positive": 0,
            "last_loss": None,
            "last_updated": None,
        }
        self._load_config()
        self.load()

    @staticmethod
    def _initial_weights() -> dict[str, float]:
        # Small priors only. Pairwise learning is expected to correct these.
        weights = {key: 0.0 for key in FEATURE_KEYS}
        weights.update(
            {
                "patch_core_to_outer": 0.35,
                "patch_compactness": 0.25,
                "patch_centeredness": 0.28,
                "patch_isotropy": 0.20,
                "patch_bipolar": 0.20,
                "patch_local_snr": 0.22,
                "v2_absdiff": 0.06,
                "v2_zscore": 0.08,
            }
        )
        return weights

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
                        self.weights[key] = _safe_float(raw_weights[key])
            stats = payload.get("stats", {})
            if isinstance(stats, dict):
                self.stats.update(stats)
        except Exception:
            # Keep the initialized model. Experimental persistence must never
            # prevent the game from starting.
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
        self.weights = self._initial_weights()
        self.stats = {
            "version": self.VERSION,
            "positive_shots": 0,
            "pair_updates": 0,
            "skipped_no_positive": 0,
            "last_loss": None,
            "last_updated": time.time(),
        }
        self.save()

    def vector(self, candidate: Candidate) -> dict[str, float]:
        result = {
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
                result[f"patch_abs_{suffix}"] = max(
                    0.0,
                    min(1.5, _safe_float(candidate.get(f"v24_patch_abs_{suffix}", 0.0))),
                )
                result[f"patch_signed_{suffix}"] = max(
                    -1.0,
                    min(1.0, _safe_float(candidate.get(f"v24_patch_signed_{suffix}", 0.0))),
                )
                result[f"patch_z_{suffix}"] = max(
                    0.0,
                    min(1.5, _safe_float(candidate.get(f"v24_patch_z_{suffix}", 0.0))),
                )
        return result

    def raw_score(self, candidate: Candidate) -> float:
        vector = self.vector(candidate)
        return sum(self.weights[key] * vector[key] for key in FEATURE_KEYS)

    def score(self, candidate: Candidate) -> float:
        return _sigmoid(self.raw_score(candidate))

    def patch_prior(self, candidate: Candidate) -> float:
        """Hand-crafted image-shape prior used before enough V4 labels exist."""
        compactness = _clip01(_safe_float(candidate.get("v24_patch_compactness", 0.0)))
        centered = _clip01(_safe_float(candidate.get("v24_patch_centeredness", 0.0)))
        isotropy = _clip01(_safe_float(candidate.get("v24_patch_isotropy", 0.0)))
        bipolar = _clip01(_safe_float(candidate.get("v24_patch_bipolar", 0.0)))
        snr = _clip01(_safe_float(candidate.get("v24_patch_local_snr", 0.0)))
        core_outer = _clip01(_safe_float(candidate.get("v24_patch_core_to_outer", 0.0)))
        agreement = _clip01(_safe_float(candidate.get("detector_agreement", 0.0)))
        stable = _clip01(_safe_float(candidate.get("shot_accumulator_stability", 0.0)))

        # A hole can be ragged or ring-shaped, so no single morphology feature
        # is allowed to dominate. The prior mostly asks whether a compact local
        # temporal event is centred on the candidate and stands above its local
        # surroundings.
        prior = (
            0.22 * snr
            + 0.18 * centered
            + 0.16 * compactness
            + 0.14 * core_outer
            + 0.11 * isotropy
            + 0.09 * bipolar
            + 0.06 * agreement
            + 0.04 * stable
        )
        return _clip01(prior)

    def effective_weight(self) -> float:
        positives = int(self.stats.get("positive_shots", 0) or 0)
        minimum = max(0, int(self.config.get("min_positive_shots", 20)))
        full = max(minimum + 1, int(self.config.get("full_weight_shots", 180)))
        maximum = _clip01(_safe_float(self.config.get("max_model_weight", 0.82), 0.82))
        if positives < minimum:
            return 0.0
        progress = _clip01((positives - minimum) / float(full - minimum))
        return maximum * progress

    def learn_from_ground_truth(
        self,
        gt_xy: Point,
        candidates: Sequence[Candidate],
        *,
        positive_override: Candidate | None = None,
    ) -> dict[str, Any]:
        if not bool(self.config.get("enabled", True)):
            return {"trained": False, "reason": "disabled"}
        if not candidates:
            self._skip()
            return {"trained": False, "reason": "no_candidates"}

        gt_x, gt_y = float(gt_xy[0]), float(gt_xy[1])
        positive_radius = max(
            1.0,
            _safe_float(self.config.get("positive_radius_px", 16.0), 16.0),
        )
        safe_radius = max(
            positive_radius + 1.0,
            _safe_float(self.config.get("negative_safe_radius_px", 58.0), 58.0),
        )

        nearest_idx = -1
        nearest_distance = float("inf")
        for index, candidate in enumerate(candidates):
            distance = math.hypot(
                _safe_float(candidate.get("camera_x", 0.0)) - gt_x,
                _safe_float(candidate.get("camera_y", 0.0)) - gt_y,
            )
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_idx = index

        # Synthetic GT lets us sample the temporal descriptor exactly at the
        # true projected point AFTER this shot has already been ranked. This is
        # training-only: it cannot leak GT into the current result. It also
        # avoids V3's noisy label assumption that an arbitrary candidate within
        # the loose 42 px evaluation radius must be a true hole candidate.
        positive: Candidate | None = None
        positive_source = "candidate"
        if (
            bool(self.config.get("ground_truth_patch_enabled", True))
            and isinstance(positive_override, dict)
        ):
            min_abs = _safe_float(
                self.config.get("ground_truth_patch_min_absdiff", 1.2), 1.2
            )
            min_snr = _safe_float(
                self.config.get("ground_truth_patch_min_local_snr", 0.10), 0.10
            )
            override_abs = _safe_float(
                positive_override.get(
                    "v24_patch_core_abs",
                    positive_override.get("v2_absdiff", 0.0),
                )
            )
            override_snr = _safe_float(
                positive_override.get("v24_patch_local_snr", 0.0)
            )
            if override_abs >= min_abs or override_snr >= min_snr:
                positive = dict(positive_override)
                positive_source = "ground_truth_patch"

        if positive is None:
            if nearest_idx < 0 or nearest_distance > positive_radius:
                self._skip()
                return {
                    "trained": False,
                    "reason": "gt_candidate_missing",
                    "nearest_distance": nearest_distance,
                }
            positive = dict(candidates[nearest_idx])

        pos_vector = self.vector(positive)

        negatives: list[Candidate] = []
        for index, candidate in enumerate(candidates):
            if positive_source == "candidate" and index == nearest_idx:
                continue
            distance = math.hypot(
                _safe_float(candidate.get("camera_x", 0.0)) - gt_x,
                _safe_float(candidate.get("camera_y", 0.0)) - gt_y,
            )
            if distance <= safe_radius:
                continue
            negatives.append(candidate)

        # Mine the mistakes the current final ranker actually likes, plus a few
        # high patch-prior artifacts. This is stronger supervision than learning
        # from arbitrary easy background negatives.
        negatives.sort(
            key=lambda candidate: (
                _safe_float(candidate.get("v24_combined_score", candidate.get("combined_score", 0.0))),
                self.patch_prior(candidate),
            ),
            reverse=True,
        )
        negatives = negatives[: max(1, int(self.config.get("hard_negatives", 18)))]
        if not negatives:
            return {"trained": False, "reason": "no_safe_negatives"}

        learning_rate = max(1e-5, _safe_float(self.config.get("learning_rate", 0.085), 0.085))
        l2 = max(0.0, min(0.05, _safe_float(self.config.get("l2", 0.0007), 0.0007)))
        epochs = max(1, int(self.config.get("epochs_per_shot", 2)))

        losses: list[float] = []
        updates = 0
        for _epoch in range(epochs):
            for negative in negatives:
                neg_vector = self.vector(negative)
                diff = {key: pos_vector[key] - neg_vector[key] for key in FEATURE_KEYS}
                margin = sum(self.weights[key] * diff[key] for key in FEATURE_KEYS)
                probability = _sigmoid(margin)
                losses.append(-math.log(max(1e-9, probability)))
                gradient_scale = 1.0 - probability

                for key in FEATURE_KEYS:
                    weight = self.weights[key]
                    weight *= 1.0 - learning_rate * l2
                    weight += learning_rate * gradient_scale * diff[key]
                    self.weights[key] = max(-6.0, min(6.0, weight))
                updates += 1

        self.stats["positive_shots"] = int(self.stats.get("positive_shots", 0) or 0) + 1
        self.stats["pair_updates"] = int(self.stats.get("pair_updates", 0) or 0) + updates
        self.stats["last_loss"] = sum(losses) / len(losses) if losses else None
        self.stats["last_updated"] = time.time()
        self.stats["last_positive_distance"] = nearest_distance
        self.stats["last_positive_source"] = positive_source

        save_every = max(1, int(self.config.get("save_every_shots", 5)))
        if int(self.stats["positive_shots"]) % save_every == 0:
            self.save()
        self._append_training_log(
            gt_xy,
            positive,
            negatives,
            nearest_distance,
            positive_source=positive_source,
        )

        return {
            "trained": True,
            "positive_shots": int(self.stats["positive_shots"]),
            "pair_updates": int(self.stats["pair_updates"]),
            "updates_this_shot": updates,
            "nearest_distance": nearest_distance,
            "positive_source": positive_source,
            "loss": self.stats.get("last_loss"),
        }

    def _skip(self) -> None:
        self.stats["skipped_no_positive"] = int(
            self.stats.get("skipped_no_positive", 0) or 0
        ) + 1

    def _append_training_log(
        self,
        gt_xy: Point,
        positive: Candidate,
        negatives: Sequence[Candidate],
        nearest_distance: float,
        *,
        positive_source: str = "candidate",
    ) -> None:
        if not bool(self.config.get("training_log_enabled", True)):
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "schema_version": "4.0",
                "timestamp": time.time(),
                "gt": [float(gt_xy[0]), float(gt_xy[1])],
                "nearest_distance": float(nearest_distance),
                "positive_source": positive_source,
                "positive": {
                    "camera_x": _safe_float(positive.get("camera_x", 0.0)),
                    "camera_y": _safe_float(positive.get("camera_y", 0.0)),
                    "patch_prior": self.patch_prior(positive),
                    "vector": self.vector(positive),
                },
                "hard_negatives": [
                    {
                        "camera_x": _safe_float(candidate.get("camera_x", 0.0)),
                        "camera_y": _safe_float(candidate.get("camera_y", 0.0)),
                        "patch_prior": self.patch_prior(candidate),
                        "vector": self.vector(candidate),
                    }
                    for candidate in negatives[:6]
                ],
            }
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def summary(self) -> dict[str, Any]:
        strongest = sorted(
            self.weights.items(),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:10]
        return {
            "version": self.VERSION,
            "positive_shots": int(self.stats.get("positive_shots", 0) or 0),
            "pair_updates": int(self.stats.get("pair_updates", 0) or 0),
            "skipped_no_positive": int(self.stats.get("skipped_no_positive", 0) or 0),
            "last_loss": self.stats.get("last_loss"),
            "last_positive_source": self.stats.get("last_positive_source"),
            "effective_weight": self.effective_weight(),
            "strongest_weights": strongest,
        }
