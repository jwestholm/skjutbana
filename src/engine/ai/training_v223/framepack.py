from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

FRAMEPACK_ROOT = Path("content/ai/training_v223/framepacks")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _gray_u8(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.ndim != 2 or arr.size == 0:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _compact_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in candidates:
        try:
            x = float(c.get("camera_x", c.get("x", 0.0)))
            y = float(c.get("camera_y", c.get("y", 0.0)))
        except Exception:
            continue
        item: dict[str, Any] = {"camera_x": x, "camera_y": y}
        for key in ("score", "combined_score", "rank", "area", "radius", "source", "source_name"):
            if key in c:
                value = c.get(key)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    item[key] = value
        out.append(item)
    return out


def save_scene_framepack(
    scene: Any,
    *,
    session_id: str,
    shot_id: str | int,
    sequence: int,
    gt_camera_xy: tuple[float, float],
    gt_screen_xy: tuple[float, float] | None,
    current_candidates: Sequence[Mapping[str, Any]],
    source_kind: str,
    background: str,
    sampling_mode: str,
) -> Path | None:
    """Persist the physical PRE/POST evidence needed for offline proposal work.

    The file is GT-labelled only in metadata. PRE/POST pixels are copied from the
    completed AIRuntime shot context. No GT coordinate participates in proposal
    generation later.
    """
    runtime = getattr(scene, "runtime", None)
    if runtime is None:
        return None
    pre = _gray_u8(getattr(runtime, "pre_shot_gray", None))
    if pre is None:
        return None

    posts: list[np.ndarray] = []
    post_ts: list[float] = []
    for item in list(getattr(runtime, "_post_shot_frames", []) or [])[:3]:
        try:
            gray, ts = item
        except Exception:
            continue
        arr = _gray_u8(gray)
        if arr is None or arr.shape != pre.shape:
            continue
        posts.append(arr)
        try:
            post_ts.append(float(ts))
        except Exception:
            post_ts.append(0.0)
    if not posts:
        post = _gray_u8(getattr(runtime, "post_shot_gray", None))
        if post is not None and post.shape == pre.shape:
            posts = [post]
            post_ts = [float(getattr(runtime, "_latest_frame_ts", 0.0) or 0.0)]
    if not posts:
        return None

    directory = FRAMEPACK_ROOT / str(session_id)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"shot_{int(sequence):06d}"
    npz_path = directory / f"{stem}.npz"
    json_path = directory / f"{stem}.json"
    stack = np.stack(posts, axis=0)
    tmp_npz = directory / f".{stem}.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        tmp_npz,
        pre_gray=pre,
        post_frames=stack,
        post_timestamps=np.asarray(post_ts, dtype=np.float64),
    )
    os.replace(tmp_npz, npz_path)
    meta = {
        "schema_version": "2.23.2-framepack-1",
        "session_id": str(session_id),
        "shot_id": str(shot_id),
        "sequence": int(sequence),
        "created_at": time.time(),
        "source_kind": str(source_kind),
        "background": str(background),
        "sampling_mode": str(sampling_mode),
        "gt_camera_xy": [float(gt_camera_xy[0]), float(gt_camera_xy[1])],
        "gt_screen_xy": ([float(gt_screen_xy[0]), float(gt_screen_xy[1])] if gt_screen_xy else None),
        "frame_shape": [int(pre.shape[0]), int(pre.shape[1])],
        "pre_timestamp": float(getattr(runtime, "_pre_shot_ts", 0.0) or 0.0),
        "shot_timestamp": float(getattr(runtime, "_shot_ts", 0.0) or 0.0),
        "post_frame_count": int(len(posts)),
        "current_candidates": _compact_candidates(current_candidates),
        "authority": "offline_shadow_only",
        "gt_used_for_proposal_generation": False,
    }
    _atomic_json(json_path, meta)
    return json_path


def discover_framepacks(root: Path = FRAMEPACK_ROOT) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.glob("*/shot_*.json") if p.with_suffix(".npz").exists())


def load_framepack(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
        pre = np.asarray(data["pre_gray"], dtype=np.uint8)
        posts = np.asarray(data["post_frames"], dtype=np.uint8)
        ts = np.asarray(data["post_timestamps"], dtype=np.float64)
    if pre.ndim != 2 or posts.ndim != 3 or posts.shape[1:] != pre.shape:
        raise ValueError(f"Invalid framepack shapes: pre={pre.shape} posts={posts.shape}")
    return meta, pre, posts, ts
