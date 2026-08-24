from __future__ import annotations

import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.engine.ai.ranker_v7 import FEATURE_KEYS, vectors_for_pool


DATA_ROOT = Path("content/ai/ranking_v29")
SESSION_ROOT = DATA_ROOT / "sessions"
JSONL_PATH = DATA_ROOT / "ranking_dataset.jsonl"
STATUS_PATH = DATA_ROOT / "status.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float(default)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _distance(candidate: dict[str, Any], gt_xy: tuple[float, float]) -> float:
    return math.hypot(
        _safe_float(candidate.get("camera_x")) - float(gt_xy[0]),
        _safe_float(candidate.get("camera_y")) - float(gt_xy[1]),
    )


def _marker(candidate: dict[str, Any]) -> str:
    return (
        f"{_safe_float(candidate.get('camera_x')):.4f},"
        f"{_safe_float(candidate.get('camera_y')):.4f}"
    )


def _rank_map(pool: Sequence[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, candidate in enumerate(pool, start=1):
        result.setdefault(_marker(candidate), index)
    return result


def _membership(pool: Sequence[dict[str, Any]]) -> set[str]:
    return {_marker(candidate) for candidate in pool}


def _jsonable_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def _raw_numeric_features(candidate: dict[str, Any]) -> dict[str, Any]:
    """Keep compact numeric/raw hypothesis evidence for later feature research."""
    allowed_prefixes = ("v27_", "v28_", "v26_", "v24_", "v2_")
    result: dict[str, Any] = {}
    for key, value in candidate.items():
        if key in {"camera_x", "camera_y", "score", "area", "radius", "circularity"}:
            scalar = _jsonable_scalar(value)
            if scalar is not None:
                result[key] = scalar
            continue
        if not str(key).startswith(allowed_prefixes):
            continue
        if isinstance(value, list):
            # Pool reasons are valuable categorical evidence.
            if key == "v28_pool_reasons":
                result[key] = [str(item) for item in value]
            continue
        scalar = _jsonable_scalar(value)
        if scalar is not None:
            result[str(key)] = scalar
    return result


def _nearest_rank(pool: Sequence[dict[str, Any]], gt_xy: tuple[float, float], radius: float) -> int | None:
    for index, candidate in enumerate(pool, start=1):
        if _distance(candidate, gt_xy) <= radius:
            return index
    return None


class RankingDatasetWriter:
    """Atomic per-shot ranking dataset writer.

    V2.9 treats these per-shot files as authoritative. JSONL is only a mirror.
    """

    SCHEMA_VERSION = "2.9"

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = (
            str(session_id)
            if session_id
            else time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        )
        self.session_dir = SESSION_ROOT / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.sequence = self._existing_sequence()
        self.jsonl_rows = 0
        self.last_error: str | None = None
        self._write_session_meta()

    def _existing_sequence(self) -> int:
        highest = 0
        for path in self.session_dir.glob("shot_*.json"):
            try:
                highest = max(highest, int(path.stem.split("_")[-1]))
            except Exception:
                pass
        return highest

    def _write_session_meta(self) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "session_id": self.session_id,
            "created_at": time.time(),
            "feature_keys": list(FEATURE_KEYS),
        }
        path = self.session_dir / "session.json"
        if not path.exists():
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def write_shot(
        self,
        *,
        gt_xy: tuple[float, float],
        all_hypotheses: Sequence[dict[str, Any]],
        hypothesis_pool: Sequence[dict[str, Any]],
        core_pool: Sequence[dict[str, Any]],
        baseline_pool: Sequence[dict[str, Any]],
        recall_baseline_pool: Sequence[dict[str, Any]],
        v6_pool: Sequence[dict[str, Any]],
        actual_pool: Sequence[dict[str, Any]],
        filtered_input: Sequence[dict[str, Any]] | None = None,
        v7_shadow_pool: Sequence[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.sequence += 1
        sequence = self.sequence
        gt = (float(gt_xy[0]), float(gt_xy[1]))

        all_source = [dict(candidate) for candidate in all_hypotheses]

        # Pool copies contain V2.8 selection reasons that are not necessarily
        # attached to the source hypothesis object. Merge those categorical
        # annotations back by stable camera-space marker before feature extraction.
        pool_annotations = {
            _marker(candidate): {
                "v28_pool_reasons": list(candidate.get("v28_pool_reasons") or []),
                "v28_core_pool": _safe_float(candidate.get("v28_core_pool")),
            }
            for candidate in hypothesis_pool
        }
        for candidate in all_source:
            annotation = pool_annotations.get(_marker(candidate))
            if annotation:
                candidate.update(annotation)

        # Runtime V7 ranks the V2.8 hypothesis_pool, so relative within-shot
        # features for pool members MUST be computed against that same pool.
        # Non-pool hypotheses keep all-cluster-relative features for diagnostics.
        all_feature_vectors = vectors_for_pool(all_source)
        all_feature_map = {
            _marker(candidate): vector
            for candidate, vector in zip(all_source, all_feature_vectors)
        }

        pool_source = [dict(candidate) for candidate in hypothesis_pool]
        pool_feature_vectors = vectors_for_pool(pool_source)
        pool_feature_map = {
            _marker(candidate): vector
            for candidate, vector in zip(pool_source, pool_feature_vectors)
        }

        core_members = _membership(core_pool)
        pool_members = _membership(hypothesis_pool)
        baseline_ranks = _rank_map(baseline_pool)
        recall_ranks = _rank_map(recall_baseline_pool)
        v6_ranks = _rank_map(v6_pool)
        actual_ranks = _rank_map(actual_pool)
        v7_ranks = _rank_map(v7_shadow_pool or [])

        candidate_rows: list[dict[str, Any]] = []
        for candidate in all_source:
            marker = _marker(candidate)
            features = (
                pool_feature_map.get(marker)
                or all_feature_map.get(marker)
                or {}
            )
            distance = _distance(candidate, gt)
            row = {
                "id": marker,
                "camera_x": _safe_float(candidate.get("camera_x")),
                "camera_y": _safe_float(candidate.get("camera_y")),
                "distance_gt_px": float(distance),
                "labels": {
                    "within_10": bool(distance <= 10.0),
                    "within_12": bool(distance <= 12.0),
                    "within_20": bool(distance <= 20.0),
                    "within_42": bool(distance <= 42.0),
                },
                "membership": {
                    "core": marker in core_members,
                    "hypothesis_pool": marker in pool_members,
                },
                "ranks": {
                    "baseline": baseline_ranks.get(marker),
                    "recall_baseline": recall_ranks.get(marker),
                    "v6": v6_ranks.get(marker),
                    "actual": actual_ranks.get(marker),
                    "v7_shadow": v7_ranks.get(marker),
                },
                "features": {
                    key: float(features.get(key, 0.0))
                    for key in FEATURE_KEYS
                },
                "raw": _raw_numeric_features(candidate),
            }
            candidate_rows.append(row)

        pool_candidates = [
            row for row in candidate_rows
            if bool(row.get("membership", {}).get("hypothesis_pool"))
        ]
        nearest_distance = min(
            (float(row["distance_gt_px"]) for row in pool_candidates),
            default=None,
        )

        selected_distance = (
            _distance(actual_pool[0], gt)
            if actual_pool
            else None
        )
        v7_selected_distance = (
            _distance(v7_shadow_pool[0], gt)
            if v7_shadow_pool
            else None
        )

        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "session_id": self.session_id,
            "sequence": sequence,
            "captured_at": time.time(),
            "ground_truth": {
                "camera_x": gt[0],
                "camera_y": gt[1],
            },
            "metadata": dict(metadata or {}),
            "counts": {
                "filtered_input": len(filtered_input or []),
                "all_hypotheses": len(all_source),
                "hypothesis_pool": len(hypothesis_pool),
                "core_pool": len(core_pool),
                "baseline_pool": len(baseline_pool),
                "recall_baseline_pool": len(recall_baseline_pool),
                "v6_pool": len(v6_pool),
                "actual_pool": len(actual_pool),
                "v7_shadow_pool": len(v7_shadow_pool or []),
            },
            "oracle": {
                "pool_nearest_px": nearest_distance,
                "pool_within_10": bool(nearest_distance is not None and nearest_distance <= 10.0),
                "pool_within_20": bool(nearest_distance is not None and nearest_distance <= 20.0),
                "pool_within_42": bool(nearest_distance is not None and nearest_distance <= 42.0),
            },
            "baseline": {
                "rank_10": _nearest_rank(baseline_pool, gt, 10.0),
                "rank_20": _nearest_rank(baseline_pool, gt, 20.0),
                "rank_42": _nearest_rank(baseline_pool, gt, 42.0),
            },
            "recall_baseline": {
                "rank_10": _nearest_rank(recall_baseline_pool, gt, 10.0),
                "rank_20": _nearest_rank(recall_baseline_pool, gt, 20.0),
                "rank_42": _nearest_rank(recall_baseline_pool, gt, 42.0),
            },
            "v6_shadow": {
                "rank_10": _nearest_rank(v6_pool, gt, 10.0),
                "rank_20": _nearest_rank(v6_pool, gt, 20.0),
                "rank_42": _nearest_rank(v6_pool, gt, 42.0),
            },
            "v7_shadow": {
                "loaded": bool(v7_shadow_pool),
                "rank_10": _nearest_rank(v7_shadow_pool or [], gt, 10.0),
                "rank_20": _nearest_rank(v7_shadow_pool or [], gt, 20.0),
                "rank_42": _nearest_rank(v7_shadow_pool or [], gt, 42.0),
                "selected_distance_px": (
                    float(v7_selected_distance)
                    if v7_selected_distance is not None
                    else None
                ),
            },
            "actual": {
                "selected_distance_px": (
                    float(selected_distance)
                    if selected_distance is not None
                    else None
                ),
                "selected_within_10": bool(selected_distance is not None and selected_distance <= 10.0),
                "selected_within_20": bool(selected_distance is not None and selected_distance <= 20.0),
                "selected_within_42": bool(selected_distance is not None and selected_distance <= 42.0),
            },
            "candidates": candidate_rows,
        }

        encoded = json.dumps(payload, ensure_ascii=False)
        final_path = self.session_dir / f"shot_{sequence:06d}.json"
        temp_path = self.session_dir / f".shot_{sequence:06d}.{os.getpid()}.tmp"
        temp_path.write_text(encoded + "\n", encoding="utf-8")
        os.replace(temp_path, final_path)

        try:
            DATA_ROOT.mkdir(parents=True, exist_ok=True)
            with JSONL_PATH.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.jsonl_rows += 1
        except Exception as exc:
            self.last_error = repr(exc)

        return payload


def latest_session_dir(session: str | None = None) -> Path | None:
    if session:
        candidate = SESSION_ROOT / str(session)
        return candidate if candidate.is_dir() else None
    if not SESSION_ROOT.is_dir():
        return None
    folders = [path for path in SESSION_ROOT.iterdir() if path.is_dir()]
    if not folders:
        return None
    return max(folders, key=lambda path: path.stat().st_mtime)


def load_session(session: str | None = None) -> tuple[list[dict[str, Any]], str | None]:
    folder = latest_session_dir(session)
    if folder is None:
        return [], None
    rows: list[dict[str, Any]] = []
    for path in sorted(folder.glob("shot_*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(value, dict) and isinstance(value.get("candidates"), list):
            rows.append(value)
    return rows, folder.name


__all__ = [
    "DATA_ROOT",
    "JSONL_PATH",
    "RankingDatasetWriter",
    "SESSION_ROOT",
    "STATUS_PATH",
    "latest_session_dir",
    "load_session",
]
