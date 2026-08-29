from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import CandidateTrainingRow, ShotTrainingRecord, candidate_rows_from_pool

NATIVE_ROOT = Path("content/ai/training_v223/sessions")
LEGACY_CACHE_ROOT = Path("content/ai/training_v223/cache/legacy_v2231")
PROPOSAL_V2232_ROOT = Path("content/ai/training_v223/proposals_v2232")
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
    if isinstance(direct, Mapping):
        for xk, yk in (("camera_x", "camera_y"), ("x", "y")):
            if xk in direct and yk in direct:
                try:
                    return float(direct[xk]), float(direct[yk])
                except Exception:
                    pass
    if isinstance(direct, (list, tuple)) and len(direct) >= 2:
        try:
            return float(direct[0]), float(direct[1])
        except Exception:
            pass
    pairs = (
        (f"{prefix}_x", f"{prefix}_y"),
        ("gt_camera_x", "gt_camera_y"),
        ("camera_x", "camera_y") if prefix in {"gt_camera", "gt_camera_xy"} else ("__none__", "__none__"),
    )
    for px, py in pairs:
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


def _oracle(record: ShotTrainingRecord, radius: float) -> bool:
    return any(
        row.gt_distance_px is not None and float(row.gt_distance_px) <= float(radius)
        for row in record.candidates
    )


def _proposal_v2232_lookup(root: Path = PROPOSAL_V2232_ROOT) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    if not root.exists():
        return out
    for path in sorted(root.glob("*/shot_*.json")):
        raw = _read_json(path)
        if not raw or not isinstance(raw.get("candidates"), list):
            continue
        key = (str(raw.get("session_id", path.parent.name)), str(raw.get("shot_id", "")))
        out[key] = raw
    return out


def _merge_proposal_rows(record: ShotTrainingRecord, sidecar: Mapping[str, Any]) -> None:
    candidates = [x for x in sidecar.get("candidates", []) if isinstance(x, Mapping)]
    if not candidates:
        return
    frame_shape = None
    try:
        shape = record.metadata.get("frame_shape")
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            frame_shape = (int(shape[0]), int(shape[1]))
    except Exception:
        frame_shape = None
    new_rows = candidate_rows_from_pool(
        candidates,
        gt_camera_xy=(record.gt_camera_x, record.gt_camera_y),
        frame_shape=frame_shape,
        dedupe_radius_px=1.25,
    )
    merged = list(record.candidates)
    for row in new_rows:
        found = None
        for old in merged:
            if math.hypot(old.camera_x - row.camera_x, old.camera_y - row.camera_y) <= 1.25:
                found = old
                break
        if found is None:
            merged.append(row)
        else:
            # Preserve native/baseline fields while enriching with offline physical evidence.
            found.features.update(row.features)
            found.provenance = sorted(set(found.provenance + row.provenance + ["v2232_offline_proposal"]))
            if found.baseline_score is None and row.baseline_score is not None:
                found.baseline_score = row.baseline_score
    record.candidates = merged
    record.finalize_labels()
    record.metadata["v2232_proposal_expanded"] = True
    record.metadata["v2232_proposal_counts"] = dict(sidecar.get("counts", {}))
    record.metadata["v2232_proposal_nearest"] = dict(sidecar.get("nearest", {}))


def discover_native_records(root: Path = NATIVE_ROOT) -> list[ShotTrainingRecord]:
    records: list[ShotTrainingRecord] = []
    if not root.exists():
        return records
    proposals = _proposal_v2232_lookup()
    for path in sorted(root.glob("*/shot_*.json")):
        raw = _read_json(path)
        if raw is None:
            continue
        try:
            record = ShotTrainingRecord.from_dict(raw)
            record.source_path = str(path)
            sidecar = proposals.get((record.session_id, record.shot_id))
            if sidecar is not None:
                _merge_proposal_rows(record, sidecar)
            records.append(record)
        except Exception:
            continue
    return records


def _frame_shape_from_pack(pack: Any, meta: Mapping[str, Any]) -> tuple[int, int] | None:
    for attr in ("full_recent_pre_frame", "full_pre_frame"):
        frame = getattr(pack, attr, None)
        try:
            if frame is not None and getattr(frame, "ndim", 0) >= 2:
                return int(frame.shape[0]), int(frame.shape[1])
        except Exception:
            pass
    posts = getattr(pack, "full_post_frames", None)
    try:
        if posts is not None and getattr(posts, "ndim", 0) >= 3 and len(posts):
            shape = tuple(int(x) for x in posts.shape)
            if len(shape) == 3:      # N,H,W
                return shape[1], shape[2]
            if len(shape) >= 4:      # N,H,W,C
                return shape[1], shape[2]
    except Exception:
        pass
    for key in ("frame_shape", "camera_shape", "image_shape"):
        shape = meta.get(key)
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            try:
                return int(shape[0]), int(shape[1])
            except Exception:
                pass
    return None



def _legacy_cache_path(path: Path) -> Path:
    try:
        st = path.stat()
        sibling = path.with_suffix(".npz")
        if sibling.exists():
            nst = sibling.stat()
            npz_sig = f"{nst.st_size}|{nst.st_mtime_ns}"
        else:
            npz_sig = "no_npz"
        token = f"2.23.1b|{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{npz_sig}"
    except Exception:
        token = f"2.23.1b|{path}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]
    return LEGACY_CACHE_ROOT / f"{digest}.json"


def _load_cached_legacy(path: Path) -> ShotTrainingRecord | None:
    cache = _legacy_cache_path(path)
    raw = _read_json(cache)
    if raw is None:
        return None
    try:
        rec = ShotTrainingRecord.from_dict(raw["record"] if isinstance(raw.get("record"), Mapping) else raw)
        rec.source_path = str(path)
        rec.metadata["legacy_cache_hit"] = True
        return rec
    except Exception:
        return None


def _save_cached_legacy(path: Path, record: ShotTrainingRecord) -> None:
    try:
        cache = _legacy_cache_path(path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": "2.23.1", "source_path": str(path), "record": record.to_dict()}
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp.replace(cache)
    except Exception:
        pass

def _official_pack_loader(path: Path) -> tuple[Any | None, str | None]:
    """Load V2.16/V2.20 JSON+NPZ through the repository's canonical loader.

    V2.23.0 incorrectly expected candidates to live directly in shot JSON. The
    real pack is split across JSON + sibling NPZ and the canonical contract is
    CandidatePackV216.load(path).
    """
    try:
        mod = importlib.import_module("src.engine.offline.candidate_pack_v216")
        cls = getattr(mod, "CandidatePackV216")
        loader = getattr(cls, "load")
        return loader(Path(path)), None
    except Exception as exc:
        return None, f"official_loader:{type(exc).__name__}:{exc}"


def _v217_split_lookup(base_root: Path) -> dict[tuple[str, int], str]:
    """Reuse V2.17's split contract for physical V2.16 packs when available."""
    try:
        mod = importlib.import_module("src.engine.offline.new_hole_training_v217")
        fn = getattr(mod, "_shot_split_keys_v217")
        split_keys, _provisional = fn(Path(base_root))
        return {
            (str(session_id), int(round_id)): _normalize_split(name) or str(name)
            for name, keys in dict(split_keys).items()
            for session_id, round_id in keys
        }
    except Exception:
        return {}


def _legacy_source_kind(root: Path) -> str:
    text = str(root)
    if "candidate_synthetic_v220_validation" in text:
        return "v220_synthetic_validation"
    if "candidate_synthetic_v220" in text:
        return "v220_synthetic"
    return "v216_projected_candidate_pack"


def _legacy_forced_split(root: Path, *, session_id: str = "", round_id: int = 0) -> str | None:
    text = str(root)
    if "candidate_synthetic_v220_validation" in text:
        return "holdout"
    if "candidate_synthetic_v220" in text:
        # V2.20 engineering worlds are independent seeded scenarios. Keep the
        # separately named validation root protected, while making a stable 80/20
        # DEVELOPMENT/engineering-validation split inside the training corpus.
        # This is explicitly provisional and is never an authority split.
        token = f"v220-engineering:{session_id}:{round_id}"
        value = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % 100
        return "validation" if value < 20 else "development"
    return None


def load_legacy_candidate_record(
    path: Path,
    root: Path,
    *,
    split_lookup: Mapping[tuple[str, int], str] | None = None,
) -> tuple[ShotTrainingRecord | None, str]:
    cached = _load_cached_legacy(path)
    if cached is not None:
        return cached, "cache"
    pack, error = _official_pack_loader(path)
    if pack is None:
        return None, error or "official_loader_failed"

    meta = getattr(pack, "metadata", {})
    if not isinstance(meta, Mapping):
        meta = {}
    gt_value = getattr(pack, "gt_xy", None)
    gt: tuple[float, float] | None = None
    if isinstance(gt_value, (list, tuple)) and len(gt_value) >= 2:
        try:
            gt = float(gt_value[0]), float(gt_value[1])
        except Exception:
            gt = None
    if gt is None:
        gt = _get_xy(meta, "gt_camera_xy") or _get_xy(meta, "gt_camera")
    if gt is None:
        return None, "missing_gt"

    candidates = getattr(pack, "candidates", None)
    if not isinstance(candidates, list) or not candidates:
        return None, "missing_candidates"
    candidates = [c for c in candidates if isinstance(c, Mapping)]
    if not candidates:
        return None, "no_mapping_candidates"

    frame_shape = _frame_shape_from_pack(pack, meta)
    rows = candidate_rows_from_pool(candidates, gt_camera_xy=gt, frame_shape=frame_shape)
    if not rows:
        return None, "no_trainable_candidates"

    session_id = str(meta.get("session_id") or path.parent.name)
    try:
        round_id = int(meta.get("round_id", meta.get("shot_id", 0)) or 0)
    except Exception:
        round_id = 0
    split_hint = _legacy_forced_split(root, session_id=session_id, round_id=round_id)
    if split_hint is None:
        split_hint = _normalize_split(meta.get("split") or meta.get("dataset_split"))
    if split_hint is None and split_lookup:
        split_hint = _normalize_split(split_lookup.get((session_id, round_id)))

    screen = (
        _get_xy(meta, "gt_screen_xy")
        or _get_xy(meta, "gt_screen")
    )
    challenge_tags = meta.get("challenge_tags", [])
    if not isinstance(challenge_tags, list):
        challenge_tags = []
    try:
        ts = float(meta.get("timestamp", meta.get("captured_at", 0.0)) or 0.0)
    except Exception:
        ts = 0.0

    record = ShotTrainingRecord(
        session_id=session_id,
        shot_id=str(meta.get("shot_id", round_id or path.stem)),
        source_kind=_legacy_source_kind(root),
        timestamp=ts,
        gt_camera_x=gt[0], gt_camera_y=gt[1],
        gt_screen_x=(screen[0] if screen else None), gt_screen_y=(screen[1] if screen else None),
        background=str(meta.get("background_mode", meta.get("background", meta.get("media_category", "unknown")))),
        split_hint=split_hint,
        sampling_mode=(str(meta.get("sampling_mode")) if meta.get("sampling_mode") else None),
        challenge_tags=[str(x) for x in challenge_tags],
        source_path=str(path),
        candidates=rows,
        metadata={
            "legacy_schema": meta.get("schema_version", "2.16"),
            "legacy_root": str(root),
            "legacy_loader": "CandidatePackV216.load",
            "legacy_round_id": round_id,
            "frame_shape": list(frame_shape) if frame_shape else None,
            "pack_has_recent_pre": getattr(pack, "recent_pre_patches", None) is not None,
            "pack_has_full_frames": any(getattr(pack, name, None) is not None for name in ("full_pre_frame", "full_recent_pre_frame", "full_post_frames")),
        },
    )
    _save_cached_legacy(path, record)
    return record, "loaded"


def discover_legacy_records(roots: Iterable[Path] = LEGACY_ROOTS) -> tuple[list[ShotTrainingRecord], dict[str, Any]]:
    records: list[ShotTrainingRecord] = []
    report: dict[str, Any] = {
        "loader": "CandidatePackV216.load",
        "roots": {}, "loaded": 0, "skipped": 0,
        "skip_reasons": {},
    }
    for root in roots:
        stats: dict[str, Any] = {
            "json_files": 0, "loaded": 0, "skipped": 0,
            "skip_reasons": {}, "oracle20": 0, "oracle42": 0,
            "candidates": 0,
        }
        base_root = root.parent if root.name == "sessions" else root
        split_lookup = _v217_split_lookup(base_root) if "candidate_shadow_v216" in str(root) else {}
        if root.exists():
            for path in sorted(root.glob("**/shot_*.json")):
                stats["json_files"] += 1
                record, reason = load_legacy_candidate_record(path, root, split_lookup=split_lookup)
                if record is None:
                    stats["skipped"] += 1
                    stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
                    report["skip_reasons"][reason] = report["skip_reasons"].get(reason, 0) + 1
                else:
                    records.append(record)
                    stats["loaded"] += 1
                    if reason == "cache":
                        stats["cache_hits"] = int(stats.get("cache_hits", 0)) + 1
                    stats["candidates"] += len(record.candidates)
                    stats["oracle20"] += int(_oracle(record, 20.0))
                    stats["oracle42"] += int(_oracle(record, 42.0))
        report["roots"][str(root)] = stats
        report["loaded"] += int(stats["loaded"])
        report["skipped"] += int(stats["skipped"])
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
        candidates = 0
        oracle_counts = {5: 0, 10: 0, 20: 0, 42: 0}
        per_shot_candidates: list[int] = []
        for r in self.records:
            by_source[r.source_kind] = by_source.get(r.source_kind, 0) + 1
            candidates += len(r.candidates)
            per_shot_candidates.append(len(r.candidates))
            for radius in oracle_counts:
                oracle_counts[radius] += int(_oracle(r, float(radius)))
        n = len(self.records)
        sorted_counts = sorted(per_shot_candidates)
        median_candidates = (
            float(sorted_counts[n // 2]) if n and n % 2 == 1
            else float((sorted_counts[n // 2 - 1] + sorted_counts[n // 2]) / 2.0) if n
            else 0.0
        )
        result = {
            "shots": n,
            "sessions": len(self.session_ids),
            "candidates": candidates,
            "mean_candidates_per_shot": (candidates / n if n else 0.0),
            "median_candidates_per_shot": median_candidates,
            "by_source": by_source,
        }
        for radius, count in oracle_counts.items():
            result[f"oracle{radius}"] = count
            result[f"oracle{radius}_rate"] = (count / n if n else 0.0)
        return result

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

        unknown_sessions = sorted(set(r.session_id for r in unknown))
        if len(unknown_sessions) >= 3:
            buckets: dict[str, str] = {}
            for sid in unknown_sessions:
                value = int(hashlib.sha256(sid.encode("utf-8")).hexdigest()[:8], 16) % 100
                buckets[sid] = "holdout" if value < 10 else ("validation" if value < 30 else "development")
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
                out.provisional = bool(explicit_seen)
                if explicit_seen:
                    out.notes.append("Legacy explicit split hints are preserved; authority remains conservative until session provenance is audited.")
            for r in unknown:
                getattr(out, buckets[r.session_id]).append(r)
        else:
            out.provisional = True
            out.notes.append("Fewer than three unsplit sessions: engineering validation is provisional; no live-authority claim is allowed.")
            if out.validation:
                out.development.extend(unknown)
            else:
                for r in unknown:
                    key = f"{r.session_id}:{r.shot_id}:{r.timestamp}"
                    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
                    (out.validation if value < 20 else out.development).append(r)

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

        if not out.development and out.validation and len(out.validation) > 1:
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
    records: list[ShotTrainingRecord] = []
    keys: set[tuple[str, str, str]] = set()
    for r in native + legacy:
        key = (r.session_id, r.shot_id, r.source_path or r.source_kind)
        if key in keys:
            continue
        keys.add(key)
        records.append(r)
    return DatasetV223(records=records, legacy_report=report)
