from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import ShotTrainingRecord, candidate_rows_from_pool

NATIVE_ROOT = Path("content/ai/training_v223/sessions")
LEGACY_ROOTS = (
    Path("content/ai/candidate_shadow_v216/sessions"),
    Path("content/ai/candidate_synthetic_v220/sessions"),
    Path("content/ai/candidate_synthetic_v220_validation/sessions"),
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _first_mapping(*values: Any) -> Mapping[str, Any]:
    for value in values:
        if isinstance(value, Mapping):
            return value
    return {}


def _get_xy(container: Mapping[str, Any], prefix: str) -> tuple[float, float] | None:
    direct = container.get(prefix)
    if isinstance(direct, (list, tuple)) and len(direct) >= 2:
        try:
            return float(direct[0]), float(direct[1])
        except Exception:
            pass
    for px, py in ((f"{prefix}_x", f"{prefix}_y"), ("gt_camera_x", "gt_camera_y")):
        if px in container and py in container:
            try:
                return float(container[px]), float(container[py])
            except Exception:
                pass
    return None


def _normalize_split(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    aliases = {
        "train": "development", "dev": "development", "development": "development",
        "validation": "validation", "val": "validation", "confirmation": "validation", "confirm": "validation",
        "holdout": "holdout", "test": "holdout", "protected": "holdout",
    }
    return aliases.get(s)


def discover_native_records(root: Path = NATIVE_ROOT) -> list[ShotTrainingRecord]:
    records: list[ShotTrainingRecord] = []
    if not root.exists():
        return records
    for path in sorted(root.glob("*/shot_*.json")):
        raw = _read_json(path)
        if raw is None:
            continue
        try:
            record = ShotTrainingRecord.from_dict(raw)
            record.source_path = str(path)
            records.append(record)
        except Exception:
            continue
    return records


def _legacy_candidate_list(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("candidates", "candidate_rows", "ranked_candidates", "rows"):
        value = raw.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, Mapping)]
    metadata = raw.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("candidates", "candidate_rows", "ranked_candidates"):
            value = metadata.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, Mapping)]
    return []


def load_legacy_candidate_record(path: Path, root: Path) -> ShotTrainingRecord | None:
    raw = _read_json(path)
    if raw is None:
        return None
    meta = _first_mapping(raw.get("metadata"), raw.get("shot"), raw)
    gt = (
        _get_xy(meta, "gt_camera_xy")
        or _get_xy(raw, "gt_camera_xy")
        or _get_xy(meta, "gt_camera")
        or _get_xy(raw, "gt_camera")
    )
    if gt is None:
        return None
    candidates = _legacy_candidate_list(raw)
    if not candidates:
        return None
    frame_shape = None
    shape = meta.get("frame_shape", raw.get("frame_shape"))
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        try:
            frame_shape = (int(shape[0]), int(shape[1]))
        except Exception:
            frame_shape = None
    rows = candidate_rows_from_pool(candidates, gt_camera_xy=gt, frame_shape=frame_shape)
    if not rows:
        return None
    session_id = str(meta.get("session_id") or path.parent.name)
    source_kind = "synthetic_generated" if "synthetic" in str(root) else "legacy_candidate_pack"
    split_hint = _normalize_split(meta.get("split") or meta.get("dataset_split") or raw.get("split"))
    screen = _get_xy(meta, "gt_screen_xy") or _get_xy(raw, "gt_screen_xy")
    challenge_tags = meta.get("challenge_tags", raw.get("challenge_tags", []))
    if not isinstance(challenge_tags, list):
        challenge_tags = []
    try:
        ts = float(meta.get("timestamp", raw.get("timestamp", 0.0)) or 0.0)
    except Exception:
        ts = 0.0
    record = ShotTrainingRecord(
        session_id=session_id,
        shot_id=str(meta.get("shot_id", raw.get("shot_id", path.stem))),
        source_kind=source_kind,
        timestamp=ts,
        gt_camera_x=gt[0], gt_camera_y=gt[1],
        gt_screen_x=(screen[0] if screen else None), gt_screen_y=(screen[1] if screen else None),
        background=str(meta.get("background_mode", meta.get("background", raw.get("background", "unknown")))),
        split_hint=split_hint,
        sampling_mode=(str(meta.get("sampling_mode")) if meta.get("sampling_mode") else None),
        challenge_tags=[str(x) for x in challenge_tags],
        source_path=str(path),
        candidates=rows,
        metadata={"legacy_schema": raw.get("schema_version", meta.get("schema_version")), "legacy_root": str(root)},
    )
    return record


def discover_legacy_records(roots: Iterable[Path] = LEGACY_ROOTS) -> tuple[list[ShotTrainingRecord], dict[str, Any]]:
    records: list[ShotTrainingRecord] = []
    report: dict[str, Any] = {"roots": {}, "loaded": 0, "skipped": 0}
    for root in roots:
        stats = {"json_files": 0, "loaded": 0, "skipped": 0}
        if root.exists():
            for path in sorted(root.glob("**/shot_*.json")):
                stats["json_files"] += 1
                record = load_legacy_candidate_record(path, root)
                if record is None:
                    stats["skipped"] += 1
                else:
                    records.append(record)
                    stats["loaded"] += 1
        report["roots"][str(root)] = stats
        report["loaded"] += stats["loaded"]
        report["skipped"] += stats["skipped"]
    return records, report


@dataclass
class DatasetSplitV223:
    development: list[ShotTrainingRecord] = field(default_factory=list)
    validation: list[ShotTrainingRecord] = field(default_factory=list)
    holdout: list[ShotTrainingRecord] = field(default_factory=list)
    provisional: bool = True
    notes: list[str] = field(default_factory=list)


@dataclass
class DatasetV223:
    records: list[ShotTrainingRecord]
    legacy_report: dict[str, Any] = field(default_factory=dict)

    @property
    def session_ids(self) -> list[str]:
        return sorted(set(r.session_id for r in self.records))

    def summary(self) -> dict[str, Any]:
        by_source: dict[str, int] = {}
        oracle = 0
        candidates = 0
        for r in self.records:
            by_source[r.source_kind] = by_source.get(r.source_kind, 0) + 1
            oracle += int(r.oracle20)
            candidates += len(r.candidates)
        return {
            "shots": len(self.records),
            "sessions": len(self.session_ids),
            "candidates": candidates,
            "oracle20": oracle,
            "oracle20_rate": (oracle / len(self.records) if self.records else 0.0),
            "by_source": by_source,
        }

    def split(self) -> DatasetSplitV223:
        out = DatasetSplitV223()
        unknown: list[ShotTrainingRecord] = []
        explicit_seen = False
        for r in self.records:
            split = _normalize_split(r.split_hint)
            if split == "development":
                out.development.append(r); explicit_seen = True
            elif split == "validation":
                out.validation.append(r); explicit_seen = True
            elif split == "holdout":
                out.holdout.append(r); explicit_seen = True
            else:
                unknown.append(r)

        # Unknown records are assigned by whole session whenever at least three
        # sessions exist. Protected explicit holdout records are never recycled.
        unknown_sessions = sorted(set(r.session_id for r in unknown))
        if len(unknown_sessions) >= 3:
            buckets: dict[str, str] = {}
            for sid in unknown_sessions:
                value = int(hashlib.sha256(sid.encode("utf-8")).hexdigest()[:8], 16) % 100
                buckets[sid] = "holdout" if value < 10 else ("validation" if value < 30 else "development")

            # Hash thresholds can be unlucky for a small number of sessions. If
            # either dev/validation would be empty, choose whole-session fallback
            # buckets by hash order. Mark provisional because this assignment was
            # created for engineering rather than from a frozen split manifest.
            assigned_dev = [sid for sid, b in buckets.items() if b == "development"]
            assigned_val = [sid for sid, b in buckets.items() if b == "validation"]
            if not assigned_dev or not assigned_val:
                ordered = sorted(unknown_sessions, key=lambda sid: hashlib.sha256(sid.encode("utf-8")).hexdigest())
                val_sid = ordered[0]
                hold_sid = ordered[1] if len(ordered) >= 4 else None
                buckets = {sid: "development" for sid in ordered}
                buckets[val_sid] = "validation"
                if hold_sid is not None:
                    buckets[hold_sid] = "holdout"
                out.provisional = True
                out.notes.append("Whole-session fallback split created because stable hash thresholds lacked dev/validation; freeze a split manifest before authority work.")
            else:
                # Explicit legacy split hints may themselves be provisional; stay
                # conservative if any were present.
                out.provisional = bool(explicit_seen)
                if explicit_seen:
                    out.notes.append("Legacy explicit split hints are preserved; authority remains conservative until session provenance is audited.")

            for r in unknown:
                getattr(out, buckets[r.session_id]).append(r)
        else:
            # With <3 unknown sessions we cannot honestly create independent
            # dev/validation/holdout by session. Keep any explicit validation/
            # holdout intact and only split the non-protected unknown pool by shot.
            out.provisional = True
            out.notes.append("Fewer than three unsplit sessions: engineering validation is provisional; no live-authority claim is allowed.")
            if out.validation:
                out.development.extend(unknown)
            else:
                for r in unknown:
                    key = f"{r.session_id}:{r.shot_id}:{r.timestamp}"
                    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
                    (out.validation if value < 20 else out.development).append(r)

        # A data set can consist entirely of explicit development records. Create
        # a provisional validation only from DEVELOPMENT records; never touch the
        # protected holdout.
        if not out.validation and out.development:
            original_dev = list(out.development)
            out.development = []
            for r in original_dev:
                key = f"provisional:{r.session_id}:{r.shot_id}:{r.timestamp}"
                value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
                (out.validation if value < 20 else out.development).append(r)
            if not out.validation and out.development:
                out.validation.append(out.development.pop(-1))
            out.provisional = True
            out.notes.append("Validation was derived from development records only; protected holdout remained untouched.")

        if not out.development and out.validation:
            # Keep one validation record, move surplus non-holdout validation to
            # development only for provisional engineering. Never move holdout.
            if len(out.validation) > 1:
                out.development.extend(out.validation[1:])
                out.validation = out.validation[:1]
                out.provisional = True
                out.notes.append("Development was missing; provisional development was formed from non-protected validation records.")
        return out


def compile_dataset(*, include_legacy: bool = True) -> DatasetV223:
    native = discover_native_records()
    legacy: list[ShotTrainingRecord] = []
    report: dict[str, Any] = {}
    if include_legacy:
        legacy, report = discover_legacy_records()
    # De-duplicate by source path/session+shot, preferring native V2.23 records.
    records: list[ShotTrainingRecord] = []
    keys: set[tuple[str, str, str]] = set()
    for r in native + legacy:
        key = (r.session_id, r.shot_id, r.source_path or r.source_kind)
        if key in keys:
            continue
        keys.add(key)
        records.append(r)
    return DatasetV223(records=records, legacy_report=report)
