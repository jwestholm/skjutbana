from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.engine.ai.hole_patch_ensemble_v215 import extract_candidate_patch


DEFAULT_CONFIG_PATH = Path("content/ai/candidate_shadow_v216.json")
DEFAULT_DATA_ROOT = Path("content/ai/candidate_shadow_v216")
SCHEMA_VERSION = "2.16"


@dataclass(frozen=True)
class CandidateCaptureConfigV216:
    enabled: bool = True
    data_root: str = str(DEFAULT_DATA_ROOT)
    patch_size: int = 64
    max_post_frames: int = 3
    max_candidates: int = 384
    include_raw_extras: bool = True
    save_gt_patches: bool = True
    save_full_frames: bool = False
    full_frame_post_count: int = 1
    compress: bool = True
    radii_px: tuple[float, ...] = (10.0, 20.0, 42.0)

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "CandidateCaptureConfigV216":
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return cls()
            allowed = {
                key: value
                for key, value in payload.items()
                if key in cls.__dataclass_fields__
            }
            if "radii_px" in allowed:
                allowed["radii_px"] = tuple(float(v) for v in allowed["radii_px"])
            return cls(**allowed)
        except Exception:
            return cls()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float(default)
    except Exception:
        return float(default)


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Keep useful candidate provenance without serialising image objects.

    Candidate dicts have gradually accumulated nested diagnostics.  V2.16 keeps
    numeric/string/list/dict provenance but deliberately drops ndarray/surface
    objects and deeply nested runtime state.
    """

    if depth > 5:
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item(), depth=depth + 1)
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            safe = _json_safe(item, depth=depth + 1)
            if safe is not None:
                result[str(key)] = safe
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value[:256]:
            safe = _json_safe(item, depth=depth + 1)
            if safe is not None:
                result.append(safe)
        return result
    return None


def _candidate_xy(candidate: dict[str, Any]) -> tuple[float, float]:
    return _safe_float(candidate.get("camera_x")), _safe_float(candidate.get("camera_y"))


def _marker(candidate: dict[str, Any]) -> tuple[int, int]:
    x, y = _candidate_xy(candidate)
    # Half-pixel quantisation is stable across the copying/ranking paths while
    # still keeping genuinely different hypotheses separate in practice.
    return int(round(x * 2.0)), int(round(y * 2.0))


def _distance(candidate: dict[str, Any], gt_xy: tuple[float, float]) -> float:
    x, y = _candidate_xy(candidate)
    return float(math.hypot(x - float(gt_xy[0]), y - float(gt_xy[1])))


def _normalise_frame(frame: np.ndarray | None) -> np.ndarray | None:
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim == 3:
        try:
            import cv2

            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        except Exception:
            arr = np.mean(arr[..., :3], axis=2)
    if arr.ndim != 2 or not arr.size:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _prepare_post_frames(
    post_frames: Sequence[Any] | None,
    post_gray: np.ndarray | None,
    max_frames: int,
) -> tuple[list[np.ndarray], list[float]]:
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    for item in list(post_frames or []):
        if len(frames) >= max(1, int(max_frames)):
            break
        frame = item
        ts = 0.0
        if isinstance(item, (tuple, list)) and item:
            frame = item[0]
            if len(item) > 1:
                ts = _safe_float(item[1])
        clean = _normalise_frame(frame)
        if clean is None:
            continue
        frames.append(clean)
        timestamps.append(float(ts))
    if not frames:
        clean = _normalise_frame(post_gray)
        if clean is not None:
            frames.append(clean)
            timestamps.append(0.0)
    return frames, timestamps


def _merge_candidates(
    raw_candidates: Sequence[dict[str, Any]],
    ranked_candidates: Sequence[dict[str, Any]],
    *,
    max_candidates: int,
    include_raw_extras: bool,
    gt_xy: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    raw = [dict(c) for c in raw_candidates if isinstance(c, dict)]
    ranked = [dict(c) for c in ranked_candidates if isinstance(c, dict)]
    raw_by_marker: dict[tuple[int, int], list[int]] = {}
    for index, candidate in enumerate(raw):
        raw_by_marker.setdefault(_marker(candidate), []).append(index)

    rows: list[dict[str, Any]] = []
    used_raw: set[int] = set()
    for rank_index, candidate in enumerate(ranked, start=1):
        marker = _marker(candidate)
        raw_index = None
        for candidate_index in raw_by_marker.get(marker, []):
            if candidate_index not in used_raw:
                raw_index = candidate_index
                used_raw.add(candidate_index)
                break
        rows.append(
            {
                "candidate": candidate,
                "raw_index": raw_index,
                "current_rank": int(candidate.get("rank", rank_index) or rank_index),
                "in_ranked_pool": True,
                "in_raw_pool": raw_index is not None,
                "capture_forced_gt_nearest": False,
            }
        )

    if include_raw_extras:
        for raw_index, candidate in enumerate(raw):
            if raw_index in used_raw:
                continue
            rows.append(
                {
                    "candidate": candidate,
                    "raw_index": raw_index,
                    "current_rank": None,
                    "in_ranked_pool": False,
                    "in_raw_pool": True,
                    "capture_forced_gt_nearest": False,
                }
            )

    limit = max(1, int(max_candidates))
    selected = rows[:limit]

    # Storage limits must not make diagnostics claim the detector had no GT
    # candidate.  If the nearest candidate is beyond the capture cap, retain it
    # as an explicit diagnostic-only row.  Benchmarks can exclude this row from
    # the authoritative/current-pool metric using the provenance flag.
    if gt_xy is not None and rows:
        nearest = min(rows, key=lambda row: _distance(row["candidate"], gt_xy))
        if nearest not in selected:
            nearest_copy = dict(nearest)
            nearest_copy["capture_forced_gt_nearest"] = True
            selected.append(nearest_copy)
    return selected


@dataclass
class CandidatePackV216:
    metadata: dict[str, Any]
    candidates: list[dict[str, Any]]
    pre_patches: np.ndarray
    post_patches: np.ndarray
    gt_pre_patch: np.ndarray | None
    gt_post_patches: np.ndarray
    post_timestamps: np.ndarray
    # V2.17 extension: immediate recent pre-shot camera patches for temporal
    # NEW-hole learning. Older V2.16 packs load these as None.
    recent_pre_patches: np.ndarray | None = None
    gt_recent_pre_patch: np.ndarray | None = None
    recent_pre_timestamp: float | None = None
    full_pre_frame: np.ndarray | None = None
    full_recent_pre_frame: np.ndarray | None = None
    full_post_frames: np.ndarray | None = None
    json_path: Path | None = None
    npz_path: Path | None = None

    @property
    def gt_xy(self) -> tuple[float, float] | None:
        gt = self.metadata.get("ground_truth")
        if not isinstance(gt, dict):
            return None
        if gt.get("camera_x") is None or gt.get("camera_y") is None:
            return None
        return _safe_float(gt.get("camera_x")), _safe_float(gt.get("camera_y"))

    @classmethod
    def load(cls, json_path: Path) -> "CandidatePackV216":
        json_path = Path(json_path)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        npz_name = str(meta.get("array_file") or json_path.with_suffix(".npz").name)
        npz_path = json_path.parent / npz_name
        with np.load(npz_path, allow_pickle=False) as data:
            pre = np.array(data["pre_patches"], copy=True)
            post = np.array(data["post_patches"], copy=True)
            gt_pre = np.array(data["gt_pre_patch"], copy=True) if "gt_pre_patch" in data else None
            gt_post = np.array(data["gt_post_patches"], copy=True) if "gt_post_patches" in data else np.empty((0, 0, 0), dtype=np.uint8)
            timestamps = np.array(data["post_timestamps"], copy=True) if "post_timestamps" in data else np.empty((0,), dtype=np.float64)
            recent_pre = np.array(data["recent_pre_patches"], copy=True) if "recent_pre_patches" in data else None
            gt_recent_pre = np.array(data["gt_recent_pre_patch"], copy=True) if "gt_recent_pre_patch" in data else None
            full_pre = np.array(data["full_pre_frame"], copy=True) if "full_pre_frame" in data else None
            full_recent_pre = np.array(data["full_recent_pre_frame"], copy=True) if "full_recent_pre_frame" in data else None
            full_post = np.array(data["full_post_frames"], copy=True) if "full_post_frames" in data else None
        recent_pre_timestamp = meta.get("recent_pre_timestamp")
        try:
            recent_pre_timestamp = None if recent_pre_timestamp is None else float(recent_pre_timestamp)
        except Exception:
            recent_pre_timestamp = None
        return cls(
            metadata=meta,
            candidates=list(meta.get("candidates") or []),
            pre_patches=pre,
            post_patches=post,
            gt_pre_patch=gt_pre,
            gt_post_patches=gt_post,
            post_timestamps=timestamps,
            recent_pre_patches=recent_pre,
            gt_recent_pre_patch=gt_recent_pre,
            recent_pre_timestamp=recent_pre_timestamp,
            full_pre_frame=full_pre,
            full_recent_pre_frame=full_recent_pre,
            full_post_frames=full_post,
            json_path=json_path,
            npz_path=npz_path,
        )


class CandidateShadowRecorderV216:
    """Capture real detector candidate patches during automation F2 runs.

    This recorder never changes candidate order, detector configuration or hit
    authority.  Its only output is an offline dataset for later candidate-level
    experiments.
    """

    VERSION = SCHEMA_VERSION

    def __init__(
        self,
        config: CandidateCaptureConfigV216 | None = None,
        *,
        background: str = "unknown",
        benchmark_seed: int | None = None,
        sampling_mode: str = "unknown",
        session_id: str | None = None,
    ) -> None:
        self.config = config or CandidateCaptureConfigV216.load()
        self.enabled = bool(self.config.enabled)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        seed_text = "noseed" if benchmark_seed is None else f"seed{int(benchmark_seed)}"
        bg_text = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(background))
        self.session_id = session_id or f"{stamp}_{seed_text}_{bg_text or 'unknown'}"
        self.root = Path(self.config.data_root) / "sessions" / self.session_id
        self.background = str(background)
        self.benchmark_seed = benchmark_seed
        self.sampling_mode = str(sampling_mode)
        self.shots_saved = 0
        self.capture_errors = 0
        self.started_at = time.time()
        self._finalized = False
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            self._write_session_manifest(final=False)

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": bool(self.enabled),
            "session_id": self.session_id,
            "root": str(self.root),
            "shots_saved": int(self.shots_saved),
            "capture_errors": int(self.capture_errors),
            "patch_size": int(self.config.patch_size),
            "max_post_frames": int(self.config.max_post_frames),
            "max_candidates": int(self.config.max_candidates),
            "shadow_only": True,
        }

    def _write_session_manifest(self, *, final: bool) -> None:
        if not self.enabled:
            return
        payload = self.summary()
        payload.update(
            {
                "background": self.background,
                "benchmark_seed": self.benchmark_seed,
                "sampling_mode": self.sampling_mode,
                "started_at": float(self.started_at),
                "finalized": bool(final),
                "finalized_at": time.time() if final else None,
                "capture_config": asdict(self.config),
            }
        )
        path = self.root / "session.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)

    def finalize(self) -> dict[str, Any]:
        if self.enabled and not self._finalized:
            self._finalized = True
            self._write_session_manifest(final=True)
        return self.summary()

    def capture_shot(
        self,
        *,
        round_id: int,
        raw_candidates: Sequence[dict[str, Any]],
        ranked_candidates: Sequence[dict[str, Any]],
        pre_gray: np.ndarray | None,
        recent_pre_gray: np.ndarray | None = None,
        recent_pre_timestamp: float | None = None,
        post_gray: np.ndarray | None = None,
        post_frames: Sequence[Any] | None = None,
        gt_camera_xy: tuple[float, float] | None,
        gt_screen_xy: tuple[float, float] | None = None,
        match_radius_px: float = 42.0,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"saved": False, "reason": "disabled"}
        try:
            pre = _normalise_frame(pre_gray)
            recent_pre = _normalise_frame(recent_pre_gray)
            posts, post_timestamps = _prepare_post_frames(
                post_frames,
                post_gray,
                max_frames=int(self.config.max_post_frames),
            )
            if not posts:
                return {"saved": False, "reason": "no_post_frame"}

            rows = _merge_candidates(
                raw_candidates,
                ranked_candidates,
                max_candidates=int(self.config.max_candidates),
                include_raw_extras=bool(self.config.include_raw_extras),
                gt_xy=gt_camera_xy,
            )
            size = int(self.config.patch_size)
            candidate_xy = [_candidate_xy(row["candidate"]) for row in rows]

            if pre is None:
                pre_patches = np.empty((0, size, size), dtype=np.uint8)
            else:
                pre_patches = np.stack(
                    [extract_candidate_patch(pre, xy, size) for xy in candidate_xy], axis=0
                ) if rows else np.empty((0, size, size), dtype=np.uint8)

            recent_pre_patches: np.ndarray | None = None
            if recent_pre is not None:
                recent_pre_patches = np.stack(
                    [extract_candidate_patch(recent_pre, xy, size) for xy in candidate_xy], axis=0
                ) if rows else np.empty((0, size, size), dtype=np.uint8)

            post_patches = np.stack(
                [
                    np.stack([extract_candidate_patch(frame, xy, size) for frame in posts], axis=0)
                    for xy in candidate_xy
                ],
                axis=0,
            ) if rows else np.empty((0, len(posts), size, size), dtype=np.uint8)

            gt_pre: np.ndarray | None = None
            gt_recent_pre: np.ndarray | None = None
            gt_posts = np.empty((0, size, size), dtype=np.uint8)
            if bool(self.config.save_gt_patches) and gt_camera_xy is not None:
                if pre is not None:
                    gt_pre = extract_candidate_patch(pre, gt_camera_xy, size)
                if recent_pre is not None:
                    gt_recent_pre = extract_candidate_patch(recent_pre, gt_camera_xy, size)
                gt_posts = np.stack(
                    [extract_candidate_patch(frame, gt_camera_xy, size) for frame in posts], axis=0
                )

            candidate_payload: list[dict[str, Any]] = []
            radii = tuple(float(v) for v in self.config.radii_px)
            for capture_index, row in enumerate(rows):
                candidate = dict(row["candidate"])
                x, y = candidate_xy[capture_index]
                distance_gt = _distance(candidate, gt_camera_xy) if gt_camera_xy is not None else None
                candidate_payload.append(
                    {
                        "capture_index": int(capture_index),
                        "camera_x": float(x),
                        "camera_y": float(y),
                        "raw_index": row.get("raw_index"),
                        "current_rank": row.get("current_rank"),
                        "in_ranked_pool": bool(row.get("in_ranked_pool")),
                        "in_raw_pool": bool(row.get("in_raw_pool")),
                        "capture_forced_gt_nearest": bool(row.get("capture_forced_gt_nearest")),
                        "distance_gt_px": None if distance_gt is None else float(distance_gt),
                        "labels": {
                            f"within_{int(radius) if radius.is_integer() else radius}": bool(distance_gt is not None and distance_gt <= radius)
                            for radius in radii
                        },
                        "candidate": _json_safe(candidate) or {},
                    }
                )

            base = f"shot_{int(round_id):06d}"
            npz_path = self.root / f"{base}.npz"
            json_path = self.root / f"{base}.json"
            arrays: dict[str, Any] = {
                "pre_patches": pre_patches,
                "post_patches": post_patches,
                "gt_post_patches": gt_posts,
                "post_timestamps": np.asarray(post_timestamps, dtype=np.float64),
                "candidate_xy": np.asarray(candidate_xy, dtype=np.float32).reshape(-1, 2),
            }
            if gt_pre is not None:
                arrays["gt_pre_patch"] = gt_pre
            if recent_pre_patches is not None:
                arrays["recent_pre_patches"] = recent_pre_patches
            if gt_recent_pre is not None:
                arrays["gt_recent_pre_patch"] = gt_recent_pre
            if bool(self.config.save_full_frames):
                if pre is not None:
                    arrays["full_pre_frame"] = pre
                if recent_pre is not None:
                    arrays["full_recent_pre_frame"] = recent_pre
                full_count = max(1, min(len(posts), int(self.config.full_frame_post_count)))
                arrays["full_post_frames"] = np.stack(posts[-full_count:], axis=0)
            if bool(self.config.compress):
                np.savez_compressed(npz_path, **arrays)
            else:
                np.savez(npz_path, **arrays)

            gt_payload = None
            if gt_camera_xy is not None:
                gt_payload = {
                    "camera_x": float(gt_camera_xy[0]),
                    "camera_y": float(gt_camera_xy[1]),
                    "screen_x": None if gt_screen_xy is None else float(gt_screen_xy[0]),
                    "screen_y": None if gt_screen_xy is None else float(gt_screen_xy[1]),
                }
            payload = {
                "schema_version": SCHEMA_VERSION,
                "capture_type": "candidate_shadow_pack",
                "shadow_only": True,
                "session_id": self.session_id,
                "round_id": int(round_id),
                "captured_at": time.time(),
                "background": self.background,
                "benchmark_seed": self.benchmark_seed,
                "sampling_mode": self.sampling_mode,
                "match_radius_px": float(match_radius_px),
                "ground_truth": gt_payload,
                "full_frames_saved": bool(self.config.save_full_frames),
                "recent_pre_available": bool(recent_pre is not None),
                "recent_pre_timestamp": None if recent_pre_timestamp is None else float(recent_pre_timestamp),
                "capture_extensions": ["v217_recent_pre"] if recent_pre is not None else [],
                "counts": {
                    "raw_candidates": int(len(raw_candidates)),
                    "ranked_candidates": int(len(ranked_candidates)),
                    "captured_candidates": int(len(candidate_payload)),
                    "post_frames": int(len(posts)),
                    "forced_gt_nearest": int(sum(1 for row in candidate_payload if row["capture_forced_gt_nearest"])),
                },
                "frame_shapes": {
                    "pre": None if pre is None else [int(pre.shape[0]), int(pre.shape[1])],
                    "recent_pre": None if recent_pre is None else [int(recent_pre.shape[0]), int(recent_pre.shape[1])],
                    "post": [int(posts[0].shape[0]), int(posts[0].shape[1])],
                },
                "array_file": npz_path.name,
                "candidates": candidate_payload,
                "extra": _json_safe(extra_metadata or {}) or {},
            }
            temp = json_path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            temp.replace(json_path)
            with (self.root / "index.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "round_id": int(round_id),
                    "json_file": json_path.name,
                    "array_file": npz_path.name,
                    "captured_candidates": len(candidate_payload),
                    "ground_truth": gt_payload,
                }, ensure_ascii=False) + "\n")
            self.shots_saved += 1
            self._write_session_manifest(final=False)
            return {
                "saved": True,
                "json_path": str(json_path),
                "npz_path": str(npz_path),
                "candidate_count": len(candidate_payload),
                "post_frame_count": len(posts),
            }
        except Exception as exc:
            self.capture_errors += 1
            self._write_session_manifest(final=False)
            return {"saved": False, "reason": "exception", "error": str(exc)}


def discover_candidate_packs(root: Path = DEFAULT_DATA_ROOT) -> list[Path]:
    root = Path(root)
    return sorted(path for path in root.glob("sessions/*/shot_*.json") if path.is_file())
