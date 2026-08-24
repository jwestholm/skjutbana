from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from src.engine.ai.ranker_v7 import RankerV7ShadowModel
from src.engine.ai.ranking_dataset_v29 import RankingDatasetWriter, STATUS_PATH


_INSTALLED = False
_RUNTIME_SESSION_ID = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
_METRICS: dict[str, Any] = {
    "schema_version": "2.9",
    "installed": False,
    "pid": os.getpid(),
    "runtime_session_id": _RUNTIME_SESSION_ID,
    "installed_at": None,
    "updated_at": None,
    "install_source": None,
    "install_sources": [],
    "rank_with_funnel_calls": 0,
    "labelled_calls": 0,
    "dataset_rows": 0,
    "last_call_ts": None,
    "last_error": None,
}


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
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
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


def _metadata_from_detector() -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    try:
        from src.engine.camera.hit_scanner import HitScanner, hit_scanner

        engine = getattr(HitScanner, "_candidate_generator_v2_engine", None)
        detector_gt = getattr(hit_scanner, "_detector_v2_ground_truth", None)
        if isinstance(detector_gt, dict):
            metadata["shot_id"] = detector_gt.get("shot_id")
            if detector_gt.get("benchmark_seed") is not None:
                metadata["benchmark_seed"] = detector_gt.get("benchmark_seed")
            if detector_gt.get("background") is not None:
                metadata["background"] = detector_gt.get("background")
            if detector_gt.get("screen_x") is not None:
                metadata["screen_x"] = detector_gt.get("screen_x")
            if detector_gt.get("screen_y") is not None:
                metadata["screen_y"] = detector_gt.get("screen_y")

            shot_id = int(detector_gt.get("shot_id", 0) or 0)
            record = getattr(engine, "_diagnostics", {}).get(shot_id)
            if isinstance(record, dict):
                if record.get("runtime_session_id"):
                    metadata["detector_runtime_session_id"] = str(
                        record.get("runtime_session_id")
                    )
                if record.get("git_commit"):
                    metadata["git_commit"] = str(record.get("git_commit"))
                ground_truth = record.get("ground_truth")
                if isinstance(ground_truth, dict):
                    if ground_truth.get("benchmark_seed") is not None:
                        metadata["benchmark_seed"] = ground_truth.get("benchmark_seed")
                    if ground_truth.get("background") is not None:
                        metadata["background"] = ground_truth.get("background")
    except Exception:
        pass
    return metadata


def _get_writer(runtime: Any) -> RankingDatasetWriter:
    writer = getattr(runtime, "_v29_dataset_writer", None)
    if isinstance(writer, RankingDatasetWriter):
        return writer
    writer = RankingDatasetWriter(_RUNTIME_SESSION_ID)
    runtime._v29_dataset_writer = writer
    return writer


def _get_shadow_model(runtime: Any) -> RankerV7ShadowModel:
    model = getattr(runtime, "_ranker_v7_shadow", None)
    if isinstance(model, RankerV7ShadowModel):
        return model
    model = RankerV7ShadowModel()
    runtime._ranker_v7_shadow = model
    return model


def install_ranker_v7_extension(source: str = "unknown") -> None:
    """Install V2.9 ranking-dataset capture and V7 shadow evaluation.

    This extension is deliberately observational:
    - it does not replace V2.8 hypothesis construction;
    - it does not change the V2.8 pool;
    - it does not change the actual candidate selected by the game;
    - any offline V7 model is shadow-ranked only.
    """

    global _INSTALLED
    _remember_source(source)

    if _INSTALLED:
        _write_status()
        return

    from src.engine.ai.runtime import AIRuntime

    _METRICS["v28_integration_present"] = bool(
        getattr(AIRuntime, "_ranker_v6_extension_installed", False)
    )

    if bool(getattr(AIRuntime, "_ranker_v7_extension_installed", False)):
        _INSTALLED = True
        _METRICS["installed"] = True
        _METRICS["installed_at"] = _METRICS.get("installed_at") or time.time()
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
        _METRICS["rank_with_funnel_calls"] = int(
            _METRICS.get("rank_with_funnel_calls", 0) or 0
        ) + 1
        _METRICS["last_call_ts"] = time.time()

        # Let V2.8 do EVERYTHING first, including the still-authoritative
        # baseline selection. V2.9 only observes the fully constructed pools.
        result = original_rank_with_funnel(
            self,
            raw_hotspots,
            gt_xy=gt_xy,
            limit=limit,
            match_radius_px=match_radius_px,
        )

        if gt_xy is None:
            _write_status()
            return result

        _METRICS["labelled_calls"] = int(
            _METRICS.get("labelled_calls", 0) or 0
        ) + 1

        try:
            all_hypotheses = [
                dict(candidate)
                for candidate in getattr(self, "_v28_all_hypotheses", []) or []
            ]
            hypothesis_pool = [
                dict(candidate)
                for candidate in getattr(self, "_v28_hypothesis_pool", []) or []
            ]
            core_pool = [
                dict(candidate)
                for candidate in getattr(self, "_v28_core_pool", []) or []
            ]
            baseline_pool = [
                dict(candidate)
                for candidate in getattr(self, "_v28_baseline_pool", []) or []
            ]
            recall_baseline_pool = [
                dict(candidate)
                for candidate in getattr(self, "_v28_recall_baseline_pool", []) or []
            ]
            v6_pool = [
                dict(candidate)
                for candidate in getattr(self, "_v28_v6_shadow_pool", []) or []
            ]
            actual_pool = [
                dict(candidate)
                for candidate in getattr(self, "_v28_actual_pool", []) or []
            ]
            filtered_input = [
                dict(candidate)
                for candidate in getattr(self, "_v27_input_candidates", []) or []
            ]

            if not all_hypotheses and filtered_input:
                # V2.9 requires the V2.8 hypothesis integration, but a row with
                # empty hypotheses is still useful because it makes integration
                # failure obvious and keeps dataset sequence integrity.
                _METRICS["last_error"] = (
                    "V2.8 hypothesis pool missing after rank_with_funnel"
                )

            shadow_model = _get_shadow_model(self)
            shadow_model.reload()
            v7_shadow_pool = (
                shadow_model.rank(hypothesis_pool)
                if shadow_model.loaded
                else []
            )

            writer = _get_writer(self)
            payload = writer.write_shot(
                gt_xy=(float(gt_xy[0]), float(gt_xy[1])),
                all_hypotheses=all_hypotheses,
                hypothesis_pool=hypothesis_pool,
                core_pool=core_pool,
                baseline_pool=baseline_pool,
                recall_baseline_pool=recall_baseline_pool,
                v6_pool=v6_pool,
                actual_pool=actual_pool,
                filtered_input=filtered_input,
                v7_shadow_pool=v7_shadow_pool,
                metadata={
                    **_metadata_from_detector(),
                    "v29_runtime_session_id": _RUNTIME_SESSION_ID,
                    "v7_model": shadow_model.summary(),
                },
            )

            _METRICS["dataset_rows"] = int(
                _METRICS.get("dataset_rows", 0) or 0
            ) + 1
            _METRICS["dataset_session"] = writer.session_id
            _METRICS["dataset_path"] = str(writer.session_dir)
            _METRICS["last_sequence"] = payload.get("sequence")
            _METRICS["v7_shadow_loaded"] = bool(shadow_model.loaded)
            _METRICS["v7_model_path"] = str(shadow_model.model_path)
            if writer.last_error:
                _METRICS["last_error"] = writer.last_error
            elif all_hypotheses or not filtered_input:
                _METRICS["last_error"] = None
        except Exception as exc:
            _METRICS["last_error"] = repr(exc)

        _write_status()
        return result

    AIRuntime.rank_with_funnel = rank_with_funnel_wrapped
    AIRuntime._ranker_v7_extension_installed = True

    _INSTALLED = True
    _METRICS["installed"] = True
    _METRICS["installed_at"] = time.time()
    _write_status()
    print(
        "[RANKER-V7] V2.9 offline-dataset + shadow-ranker installed "
        f"(pid={os.getpid()} session={_RUNTIME_SESSION_ID} "
        f"source={_METRICS.get('install_source')})"
    )


__all__ = ["install_ranker_v7_extension"]
