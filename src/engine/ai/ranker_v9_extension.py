from __future__ import annotations

import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from src.engine.ai.ranker_v9 import RankerV9ShadowModel


_INSTALLED = False
_RUNTIME_SESSION_ID = (
    time.strftime("%Y%m%d_%H%M%S")
    + "_"
    + uuid.uuid4().hex[:8]
)

ROOT = Path("content/ai/ranking_v211")
STATUS_PATH = ROOT / "status.json"
SHADOW_ROOT = ROOT / "shadow_sessions"

_METRICS: dict[str, Any] = {
    "schema_version": "2.11",
    "installed": False,
    "pid": os.getpid(),
    "runtime_session_id": _RUNTIME_SESSION_ID,
    "installed_at": None,
    "updated_at": None,
    "install_source": None,
    "install_sources": [],
    "rank_with_funnel_calls": 0,
    "labelled_calls": 0,
    "shadow_rows": 0,
    "model_loaded": False,
    "shadow_ready": False,
    "last_error": None,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float(default)
    except Exception:
        return float(default)


def _remember_source(source: str) -> None:
    value = str(source or "unknown")
    sources = _METRICS.setdefault("install_sources", [])
    if isinstance(sources, list) and value not in sources:
        sources.append(value)
    if value != "unknown" and not _METRICS.get("install_source"):
        _METRICS["install_source"] = value
    elif not _METRICS.get("install_source"):
        _METRICS["install_source"] = value


def _write_status(error: str | None = None) -> None:
    try:
        ROOT.mkdir(parents=True, exist_ok=True)
        _METRICS["updated_at"] = time.time()
        if error is not None:
            _METRICS["last_error"] = str(error)
        temp = STATUS_PATH.with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(
            json.dumps(_METRICS, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temp, STATUS_PATH)
    except Exception:
        pass


def _distance(
    candidate: dict[str, Any],
    gt: tuple[float, float],
) -> float:
    return math.hypot(
        _safe_float(candidate.get("camera_x")) - float(gt[0]),
        _safe_float(candidate.get("camera_y")) - float(gt[1]),
    )


def _rank_for_radius(
    candidates: Sequence[dict[str, Any]],
    gt: tuple[float, float],
    radius: float,
) -> int | None:
    for rank, candidate in enumerate(candidates, start=1):
        if _distance(candidate, gt) <= float(radius):
            return rank
    return None


def _get_model(runtime: Any) -> RankerV9ShadowModel:
    model = getattr(runtime, "_ranker_v9_shadow", None)
    if isinstance(model, RankerV9ShadowModel):
        return model
    model = RankerV9ShadowModel()
    runtime._ranker_v9_shadow = model
    return model


def _write_shadow_row(
    runtime: Any,
    gt: tuple[float, float],
    model: RankerV9ShadowModel,
) -> None:
    pool = [
        dict(candidate)
        for candidate in getattr(runtime, "_v28_hypothesis_pool", []) or []
    ]
    actual = [
        dict(candidate)
        for candidate in getattr(runtime, "_v28_actual_pool", []) or []
    ]

    if not pool:
        return

    shadow = model.rank(pool) if model.loaded else []
    sequence = int(_METRICS.get("shadow_rows", 0) or 0) + 1

    session_dir = SHADOW_ROOT / _RUNTIME_SESSION_ID
    session_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "schema_version": "2.11",
        "runtime_session_id": _RUNTIME_SESSION_ID,
        "sequence": sequence,
        "captured_at": time.time(),
        "ground_truth": {
            "camera_x": float(gt[0]),
            "camera_y": float(gt[1]),
        },
        "pool_count": len(pool),
        "actual": {
            "rank_10": _rank_for_radius(actual, gt, 10.0),
            "rank_20": _rank_for_radius(actual, gt, 20.0),
            "rank_42": _rank_for_radius(actual, gt, 42.0),
            "selected_distance_px": (
                _distance(actual[0], gt)
                if actual
                else None
            ),
        },
        "v9_shadow": {
            "loaded": bool(model.loaded),
            "shadow_ready": bool(
                model.metadata.get("shadow_ready", False)
                if model.loaded
                else False
            ),
            "rank_10": (
                _rank_for_radius(shadow, gt, 10.0)
                if shadow
                else None
            ),
            "rank_20": (
                _rank_for_radius(shadow, gt, 20.0)
                if shadow
                else None
            ),
            "rank_42": (
                _rank_for_radius(shadow, gt, 42.0)
                if shadow
                else None
            ),
            "selected_distance_px": (
                _distance(shadow[0], gt)
                if shadow
                else None
            ),
            "model": model.summary(),
        },
    }

    final_path = session_dir / f"shot_{sequence:06d}.json"
    temp_path = session_dir / f".shot_{sequence:06d}.{os.getpid()}.tmp"
    temp_path.write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, final_path)

    _METRICS["shadow_rows"] = sequence
    _METRICS["shadow_session"] = _RUNTIME_SESSION_ID
    _METRICS["shadow_path"] = str(session_dir)


def install_ranker_v9_extension(source: str = "unknown") -> None:
    """Install V2.11 V9 SHADOW comparison. Never changes actual selection."""
    global _INSTALLED

    _remember_source(source)
    if _INSTALLED:
        _write_status()
        return

    from src.engine.ai.runtime import AIRuntime

    if bool(getattr(AIRuntime, "_ranker_v9_extension_installed", False)):
        _INSTALLED = True
        _METRICS["installed"] = True
        _METRICS["installed_at"] = (
            _METRICS.get("installed_at")
            or time.time()
        )
        _write_status()
        return

    original_rank_with_funnel = AIRuntime.rank_with_funnel

    def rank_with_funnel_wrapped(
        self: Any,
        raw_hotspots: Sequence[dict[str, Any]],
        gt_xy: tuple[float, float] | None = None,
        limit: int | None = None,
        match_radius_px: float | None = None,
    ) -> Any:
        _METRICS["rank_with_funnel_calls"] = (
            int(_METRICS.get("rank_with_funnel_calls", 0) or 0)
            + 1
        )

        result = original_rank_with_funnel(
            self,
            raw_hotspots,
            gt_xy=gt_xy,
            limit=limit,
            match_radius_px=match_radius_px,
        )

        model = _get_model(self)
        model.reload()
        _METRICS["model_loaded"] = bool(model.loaded)
        _METRICS["model_path"] = str(model.model_path)
        _METRICS["shadow_ready"] = bool(
            model.metadata.get("shadow_ready", False)
            if model.loaded
            else False
        )

        if gt_xy is not None:
            _METRICS["labelled_calls"] = (
                int(_METRICS.get("labelled_calls", 0) or 0)
                + 1
            )
            try:
                _write_shadow_row(
                    self,
                    (float(gt_xy[0]), float(gt_xy[1])),
                    model,
                )
                _METRICS["last_error"] = None
            except Exception as exc:
                _METRICS["last_error"] = repr(exc)

        _write_status()
        return result

    AIRuntime.rank_with_funnel = rank_with_funnel_wrapped
    AIRuntime._ranker_v9_extension_installed = True

    _INSTALLED = True
    _METRICS["installed"] = True
    _METRICS["installed_at"] = time.time()
    _write_status()

    print(
        "[RANKER-V9] V2.11 physical/listwise SHADOW integration installed "
        f"(pid={os.getpid()} session={_RUNTIME_SESSION_ID} "
        f"source={_METRICS.get('install_source')})"
    )


__all__ = ["install_ranker_v9_extension"]
