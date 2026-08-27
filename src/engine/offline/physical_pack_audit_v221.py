from __future__ import annotations

"""V2.21 audit of candidate-pack evidence availability.

The first V2.21 decision is deliberately boring and explicit: determine what
we can honestly evaluate from the already captured physical/projector-camera
packs before inventing a full-frame proposal experiment.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .candidate_pack_v216 import CandidatePackV216, discover_candidate_packs
from .new_hole_training_v217 import _shot_split_keys_v217


RADII = (5.0, 10.0, 20.0, 42.0)


@dataclass(frozen=True)
class PhysicalPackAuditConfigV221:
    root: Path = Path("content/ai/candidate_shadow_v216")


def _distance_from_row(row: dict[str, Any], gt: tuple[float, float]) -> float:
    value = row.get("distance_gt_px")
    try:
        if value is not None and math.isfinite(float(value)):
            return float(value)
    except Exception:
        pass
    candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
    try:
        x = float(row.get("camera_x", candidate.get("camera_x", 0.0)))
        y = float(row.get("camera_y", candidate.get("camera_y", 0.0)))
        return float(math.hypot(x - gt[0], y - gt[1]))
    except Exception:
        return float("inf")


def _eligible_rows(pack: CandidatePackV216) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in pack.candidates:
        if bool(row.get("capture_forced_gt_nearest")):
            continue
        if not (bool(row.get("in_ranked_pool")) or bool(row.get("in_raw_pool"))):
            continue
        rows.append(row)
    return rows


def audit_physical_packs_v221(root: Path = Path("content/ai/candidate_shadow_v216")) -> dict[str, Any]:
    root = Path(root)
    paths = discover_candidate_packs(root)
    if not paths:
        raise RuntimeError(f"No candidate packs under {root}")

    split_keys, split_provisional = _shot_split_keys_v217(root)
    split_lookup: dict[tuple[str, int], str] = {}
    for name, keys in split_keys.items():
        for key in keys:
            split_lookup[key] = name

    rows: list[dict[str, Any]] = []
    sessions: set[str] = set()
    total_npz_bytes = 0
    for path in paths:
        pack = CandidatePackV216.load(path)
        session_id = str(pack.metadata.get("session_id", "unknown"))
        round_id = int(pack.metadata.get("round_id", 0))
        sessions.add(session_id)
        gt = pack.gt_xy
        eligible = _eligible_rows(pack)
        nearest = float("inf")
        if gt is not None and eligible:
            nearest = min(_distance_from_row(row, gt) for row in eligible)
        full_post_count = 0
        if isinstance(pack.full_post_frames, np.ndarray):
            full_post_count = 1 if pack.full_post_frames.ndim == 2 else int(len(pack.full_post_frames))
        try:
            total_npz_bytes += int(Path(pack.npz_path).stat().st_size) if pack.npz_path else 0
        except Exception:
            pass
        rows.append({
            "json_path": str(path),
            "session_id": session_id,
            "round_id": round_id,
            "split": split_lookup.get((session_id, round_id), "unknown"),
            "candidates": len(eligible),
            "gt_available": gt is not None,
            "gt_patch_available": pack.gt_pre_patch is not None and bool(pack.gt_post_patches.size),
            "recent_pre_patches_available": isinstance(pack.recent_pre_patches, np.ndarray) and pack.recent_pre_patches.ndim == 3,
            "full_pre_frame": isinstance(pack.full_pre_frame, np.ndarray) and pack.full_pre_frame.ndim == 2,
            "full_recent_pre_frame": isinstance(pack.full_recent_pre_frame, np.ndarray) and pack.full_recent_pre_frame.ndim == 2,
            "full_post_frames": full_post_count,
            "can_direct_full_frame": bool(
                (isinstance(pack.full_recent_pre_frame, np.ndarray) or isinstance(pack.full_pre_frame, np.ndarray))
                and full_post_count > 0
            ),
            "nearest_candidate_distance_px": None if not math.isfinite(nearest) else float(nearest),
            "frame_shapes": pack.metadata.get("frame_shapes"),
            "full_frames_saved_metadata": bool(pack.metadata.get("full_frames_saved", False)),
        })

    def count(key: str) -> int:
        return int(sum(bool(row[key]) for row in rows))

    oracle = {
        str(int(radius)): float(sum(
            row["nearest_candidate_distance_px"] is not None and float(row["nearest_candidate_distance_px"]) <= radius
            for row in rows
        ) / max(1, len(rows)))
        for radius in RADII
    }
    split_summary: dict[str, Any] = {}
    for split in ("development", "confirmation", "holdout"):
        subset = [row for row in rows if row["split"] == split]
        split_summary[split] = {
            "shots": len(subset),
            "full_frame_ready": sum(bool(row["can_direct_full_frame"]) for row in subset),
            "oracle": {
                str(int(radius)): float(sum(
                    row["nearest_candidate_distance_px"] is not None and float(row["nearest_candidate_distance_px"]) <= radius
                    for row in subset
                ) / max(1, len(subset))) if subset else 0.0
                for radius in RADII
            },
        }

    full_ready = count("can_direct_full_frame")
    return {
        "schema_version": "2.21",
        "root": str(root),
        "packs": len(rows),
        "sessions": sorted(sessions),
        "split_is_provisional": bool(split_provisional),
        "availability": {
            "gt_patch_shots": count("gt_patch_available"),
            "recent_pre_patch_shots": count("recent_pre_patches_available"),
            "full_reference_pre_shots": count("full_pre_frame"),
            "full_recent_pre_shots": count("full_recent_pre_frame"),
            "full_post_shots": int(sum(int(row["full_post_frames"]) > 0 for row in rows)),
            "full_frame_direct_ready_shots": full_ready,
        },
        "candidate_oracle_all": oracle,
        "splits": split_summary,
        "npz_storage_mib": round(total_npz_bytes / (1024.0 * 1024.0), 2),
        "can_benchmark_direct_proposals_now": bool(full_ready > 0),
        "next_capture_requirement": None if full_ready > 0 else (
            "Existing packs do not contain honest full-frame recent-PRE+POST evidence. "
            "Enable V2.21 full-frame shadow capture and collect a new projector/camera automation session."
        ),
        "rows": rows,
        "semantic_note": (
            "Forced GT-nearest diagnostic rows are excluded from oracle. shot_diag overlays are not used. "
            "This audit never changes detector authority or candidate ordering."
        ),
    }


def write_physical_pack_audit_v221(path: Path, report: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
