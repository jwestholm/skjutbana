from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "2.23.2"

# Deliberately physical / observational features only.  Do not add fields that
# encode GT distance, current rank policy, reason_* bookkeeping or model output.
FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "detector_score": ("detector_score", "score"),
    "area": ("area",),
    "radius": ("radius",),
    "circularity": ("circularity",),
    "center_change": ("center_change", "center_darkening"),
    "local_contrast": ("local_contrast", "local_contrast_gain"),
    "pre_shot_change": ("pre_shot_change",),
    "change_value": ("change_value",),
    "darkening": ("darkening", "v2_darkening", "darkening_value"),
    "dog_value": ("dog_value", "v2_dog", "v2_dog_value"),
    "zscore": ("zscore", "v2_zscore", "persistent_zscore"),
    "persistence": ("persistence", "persistent", "persist_score"),
    "existed_before": ("existed_before",),
    "temporal_hits": ("temporal_hits", "hits"),
    "same_frame_support": ("same_frame_support", "v2226_same_frame_support", "member_count"),
    "support_score": ("support_score", "v2226_support_score", "v2226_support_score_max"),
    "v2_rescue_temporal": ("v2_rescue_temporal", "rescue_temporal"),
    "patch_mean": ("patch_mean",),
    "patch_std": ("patch_std",),
    "edge_strength": ("edge_strength",),
    "x_norm": ("x_norm",),
    "y_norm": ("y_norm",),
    # V2.23.2 offline dense-proposal evidence. These are still GT-free physical/
    # relational features computed from PRE->POST maps; missing values remain 0.
    "dense_score": ("dense_score",),
    "dense_source_support": ("dense_source_support",),
    "dense_map_percentile_max": ("dense_map_percentile_max",),
    "dense_map_percentile_top3": ("dense_map_percentile_top3",),
    "dense_map_percentile_mean": ("dense_map_percentile_mean",),
    "dense_current_distance_clip100": ("dense_current_distance_clip100",),
    "dense_current_distance_exp24": ("dense_current_distance_exp24",),
    "dense_current_within20": ("dense_current_within20",),
    "dense_current_within42": ("dense_current_within42",),
    "dense_local_distance_clip100": ("dense_local_distance_clip100",),
    "dense_local_distance_exp24": ("dense_local_distance_exp24",),
    "dense_local_within20": ("dense_local_within20",),
    "dense_local_within42": ("dense_local_within42",),
    "dense_percentile_support": ("dense_percentile_support",),
}
FEATURE_NAMES: tuple[str, ...] = tuple(FEATURE_ALIASES)

FORBIDDEN_FEATURE_PREFIXES = (
    "reason_", "rel_", "gt_", "target_", "label", "capture_forced",
)
FORBIDDEN_FEATURE_NAMES = {
    "rank", "baseline_rank", "combined_score", "ai_score", "distance_px",
    "core_member", "baseline_score", "is_positive", "correct",
}


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _nested_sources(candidate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = [candidate]
    for key in ("features", "detector_features", "physical_features", "original_features"):
        value = candidate.get(key)
        if isinstance(value, Mapping):
            out.append(value)
    return out


def extract_physical_features(
    candidate: Mapping[str, Any],
    *,
    frame_shape: tuple[int, int] | None = None,
) -> dict[str, float]:
    """Normalize a candidate into the stable V2.23 feature contract.

    Missing features are represented as 0.0.  Policy/GT/model fields never enter
    the returned mapping. x/y normalization may be derived from frame shape.
    """
    sources = _nested_sources(candidate)
    features: dict[str, float] = {}
    for canonical, aliases in FEATURE_ALIASES.items():
        found = None
        for src in sources:
            for alias in aliases:
                if alias in src:
                    found = src.get(alias)
                    break
            if found is not None:
                break
        features[canonical] = _finite_float(found, 0.0)

    if frame_shape is not None:
        h, w = frame_shape
        x = _finite_float(candidate.get("camera_x", candidate.get("x", 0.0)))
        y = _finite_float(candidate.get("camera_y", candidate.get("y", 0.0)))
        if features["x_norm"] == 0.0 and w > 0:
            features["x_norm"] = x / float(w)
        if features["y_norm"] == 0.0 and h > 0:
            features["y_norm"] = y / float(h)

    # Explicit booleans should be exactly 0/1 where possible.
    features["existed_before"] = 1.0 if bool(features["existed_before"]) else 0.0
    features["v2_rescue_temporal"] = 1.0 if bool(features["v2_rescue_temporal"]) else 0.0
    return features


def candidate_is_storage_forced(candidate: Mapping[str, Any]) -> bool:
    return bool(
        candidate.get("capture_forced_gt_nearest", False)
        or candidate.get("forced_gt", False)
        or candidate.get("diagnostic_forced_gt", False)
    )


@dataclass
class CandidateTrainingRow:
    candidate_id: str
    camera_x: float
    camera_y: float
    features: dict[str, float]
    baseline_rank: int | None = None
    baseline_score: float | None = None
    source: str = "live"
    provenance: list[str] = field(default_factory=list)
    gt_distance_px: float | None = None
    relevance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateTrainingRow":
        return cls(
            candidate_id=str(data.get("candidate_id", "")),
            camera_x=_finite_float(data.get("camera_x")),
            camera_y=_finite_float(data.get("camera_y")),
            features={k: _finite_float(v) for k, v in dict(data.get("features", {})).items()},
            baseline_rank=(int(data["baseline_rank"]) if data.get("baseline_rank") is not None else None),
            baseline_score=(_finite_float(data.get("baseline_score")) if data.get("baseline_score") is not None else None),
            source=str(data.get("source", "live")),
            provenance=[str(x) for x in data.get("provenance", [])],
            gt_distance_px=(_finite_float(data.get("gt_distance_px")) if data.get("gt_distance_px") is not None else None),
            relevance=(_finite_float(data.get("relevance")) if data.get("relevance") is not None else None),
        )


@dataclass
class ShotTrainingRecord:
    session_id: str
    shot_id: str
    source_kind: str
    timestamp: float
    gt_camera_x: float
    gt_camera_y: float
    candidates: list[CandidateTrainingRow]
    gt_screen_x: float | None = None
    gt_screen_y: float | None = None
    background: str = "unknown"
    split_hint: str | None = None
    sampling_mode: str | None = None
    challenge_tags: list[str] = field(default_factory=list)
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def finalize_labels(self, *, sigma_px: float = 8.0) -> None:
        for row in self.candidates:
            d = math.hypot(row.camera_x - self.gt_camera_x, row.camera_y - self.gt_camera_y)
            row.gt_distance_px = float(d)
            row.relevance = float(math.exp(-(d * d) / (2.0 * sigma_px * sigma_px))) if d <= 42.0 else 0.0

    @property
    def oracle20(self) -> bool:
        return any((r.gt_distance_px if r.gt_distance_px is not None else 1e9) <= 20.0 for r in self.candidates)

    @property
    def nearest_distance_px(self) -> float:
        vals = [r.gt_distance_px for r in self.candidates if r.gt_distance_px is not None]
        return min(vals) if vals else float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "shot_id": self.shot_id,
            "source_kind": self.source_kind,
            "timestamp": self.timestamp,
            "gt_camera_x": self.gt_camera_x,
            "gt_camera_y": self.gt_camera_y,
            "gt_screen_x": self.gt_screen_x,
            "gt_screen_y": self.gt_screen_y,
            "background": self.background,
            "split_hint": self.split_hint,
            "sampling_mode": self.sampling_mode,
            "challenge_tags": list(self.challenge_tags),
            "source_path": self.source_path,
            "metadata": dict(self.metadata),
            "oracle20": self.oracle20,
            "nearest_distance_px": None if not math.isfinite(self.nearest_distance_px) else self.nearest_distance_px,
            "candidates": [r.to_dict() for r in self.candidates],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShotTrainingRecord":
        record = cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            session_id=str(data.get("session_id", "unknown")),
            shot_id=str(data.get("shot_id", "unknown")),
            source_kind=str(data.get("source_kind", "unknown")),
            timestamp=_finite_float(data.get("timestamp")),
            gt_camera_x=_finite_float(data.get("gt_camera_x")),
            gt_camera_y=_finite_float(data.get("gt_camera_y")),
            gt_screen_x=(_finite_float(data.get("gt_screen_x")) if data.get("gt_screen_x") is not None else None),
            gt_screen_y=(_finite_float(data.get("gt_screen_y")) if data.get("gt_screen_y") is not None else None),
            background=str(data.get("background", "unknown")),
            split_hint=(str(data.get("split_hint")) if data.get("split_hint") else None),
            sampling_mode=(str(data.get("sampling_mode")) if data.get("sampling_mode") else None),
            challenge_tags=[str(x) for x in data.get("challenge_tags", [])],
            source_path=(str(data.get("source_path")) if data.get("source_path") else None),
            metadata=dict(data.get("metadata", {})),
            candidates=[CandidateTrainingRow.from_dict(x) for x in data.get("candidates", []) if isinstance(x, Mapping)],
        )
        return record


def candidate_rows_from_pool(
    candidates: Sequence[Mapping[str, Any]],
    *,
    gt_camera_xy: tuple[float, float],
    frame_shape: tuple[int, int] | None = None,
    dedupe_radius_px: float = 1.25,
) -> list[CandidateTrainingRow]:
    """Convert actual candidates to rows without ever forcing GT into the pool."""
    rows: list[CandidateTrainingRow] = []
    gx, gy = gt_camera_xy
    for idx, cand in enumerate(candidates):
        if not isinstance(cand, Mapping) or candidate_is_storage_forced(cand):
            continue
        x = _finite_float(cand.get("camera_x", cand.get("x", 0.0)))
        y = _finite_float(cand.get("camera_y", cand.get("y", 0.0)))
        # Dedupe only exact/near-identical hypotheses; never use GT for retention.
        duplicate = None
        for old in rows:
            if math.hypot(old.camera_x - x, old.camera_y - y) <= dedupe_radius_px:
                duplicate = old
                break
        rank_value = cand.get("rank")
        try:
            baseline_rank = int(rank_value) if rank_value is not None else None
        except Exception:
            baseline_rank = None
        score_val = cand.get("combined_score", cand.get("score"))
        baseline_score = _finite_float(score_val) if score_val is not None else None
        provenance: list[str] = []
        for key in ("source", "source_name", "provenance"):
            value = cand.get(key)
            if isinstance(value, str) and value:
                provenance.append(value)
            elif isinstance(value, (list, tuple)):
                provenance.extend(str(v) for v in value)
        row = CandidateTrainingRow(
            candidate_id=str(cand.get("candidate_id", cand.get("id", idx))),
            camera_x=x,
            camera_y=y,
            features=extract_physical_features(cand, frame_shape=frame_shape),
            baseline_rank=baseline_rank,
            baseline_score=baseline_score,
            source=str(cand.get("source", "live")),
            provenance=sorted(set(provenance)),
        )
        d = math.hypot(x - gx, y - gy)
        row.gt_distance_px = d
        row.relevance = math.exp(-(d * d) / (2.0 * 8.0 * 8.0)) if d <= 42.0 else 0.0
        if duplicate is None:
            rows.append(row)
        else:
            # Preserve the better original baseline rank/score but merge provenance.
            if row.baseline_rank is not None and (duplicate.baseline_rank is None or row.baseline_rank < duplicate.baseline_rank):
                duplicate.baseline_rank = row.baseline_rank
            if row.baseline_score is not None and (duplicate.baseline_score is None or row.baseline_score > duplicate.baseline_score):
                duplicate.baseline_score = row.baseline_score
            duplicate.provenance = sorted(set(duplicate.provenance + row.provenance))
    return rows
