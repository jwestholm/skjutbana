from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

from src.engine.ai.ranker_v5 import RankerV5


_INSTALLED = False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float(default)
    except Exception:
        return float(default)


def _candidate_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        round(_safe_float(candidate.get("camera_x")), 3),
        round(_safe_float(candidate.get("camera_y")), 3),
        round(_safe_float(candidate.get("score")), 4),
        round(_safe_float(candidate.get("timestamp")), 5),
    )


def _get_model(runtime: Any) -> RankerV5:
    model = getattr(runtime, "_ranker_v5", None)
    if isinstance(model, RankerV5):
        return model

    storage_dir = getattr(runtime, "storage_dir", None)
    root = Path(storage_dir) if storage_dir is not None else Path("content/ai")
    model = RankerV5(
        model_path=root / "ranker_v5.json",
        config_path=root / "ranker_v5_config.json",
        log_path=root / "ranker_v5" / "training_pairs.jsonl",
    )
    runtime._ranker_v5 = model
    _install_reset_bridge(runtime, model)
    return model


def _install_reset_bridge(runtime: Any, model: RankerV5) -> None:
    memory = getattr(runtime, "memory", None)
    if memory is None or bool(getattr(memory, "_ranker_v5_reset_bridge", False)):
        return
    original_reset = memory.reset

    def reset_wrapped(*args: Any, **kwargs: Any) -> Any:
        result = original_reset(*args, **kwargs)
        try:
            model.reset()
        except Exception:
            pass
        return result

    memory.reset = reset_wrapped
    memory._ranker_v5_reset_bridge = True


def _rank_of_gt(pool: Sequence[dict[str, Any]], gt_xy: tuple[float, float], radius: float) -> int | None:
    for index, candidate in enumerate(pool, start=1):
        distance = math.hypot(
            _safe_float(candidate.get("camera_x")) - float(gt_xy[0]),
            _safe_float(candidate.get("camera_y")) - float(gt_xy[1]),
        )
        if distance <= radius:
            return index
    return None


def install_ranker_v5_extension() -> None:
    """Install V5 after the V4 shadow wrapper.

    V5 sees the untouched/base ordering returned by V4 shadow mode. It trains
    only on generated candidates <=12 px from synthetic GT. It remains shadow
    unless its PRE-TRAIN rolling validation is clearly better than the base;
    only then may it move one high-confidence candidate to rank 1.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from src.engine.ai.runtime import AIRuntime

    if bool(getattr(AIRuntime, "_ranker_v5_extension_installed", False)):
        _INSTALLED = True
        return

    original_rank_candidates = AIRuntime.rank_candidates
    original_rank_with_funnel = AIRuntime.rank_with_funnel

    def rank_candidates_wrapped(
        self: Any,
        candidates: Sequence[dict[str, Any]],
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            self._v26_base_rank_pool = []
            self._v26_v5_shadow_pool = []
            self._v26_v5_override_status = {"applied": False, "reason": "no_candidates"}
            return []

        requested_limit = int(limit or getattr(self, "settings", {}).get("top_k", 10))
        full_limit = max(requested_limit, len(candidates))
        try:
            base_pool = original_rank_candidates(self, candidates, limit=full_limit)
        except TypeError:
            base_pool = original_rank_candidates(self, candidates, full_limit)
        base_pool = [dict(candidate) for candidate in base_pool]
        if not base_pool:
            self._v26_base_rank_pool = []
            self._v26_v5_shadow_pool = []
            return []

        model = _get_model(self)
        if not bool(model.config.get("enabled", True)):
            self._v26_base_rank_pool = base_pool
            self._v26_v5_shadow_pool = []
            return base_pool[:requested_limit]

        v5_pool = model.rank(base_pool)
        shadow_by_key = {_candidate_key(candidate): candidate for candidate in v5_pool}
        annotated_base: list[dict[str, Any]] = []
        for index, candidate in enumerate(base_pool, start=1):
            item = dict(candidate)
            shadow = shadow_by_key.get(_candidate_key(candidate))
            if shadow is not None:
                item["v26_v5_shadow_rank"] = int(shadow.get("ranker_v5_rank", 0) or 0)
                item["ranker_v5_score"] = _safe_float(shadow.get("ranker_v5_score"), 0.5)
                item["ranker_v5_raw"] = _safe_float(shadow.get("ranker_v5_raw"), 0.0)
            item["rank"] = index
            annotated_base.append(item)

        actual_pool = [dict(candidate) for candidate in annotated_base]
        override_candidate, gate = model.override_candidate(v5_pool)
        applied = False
        override_key = None
        if isinstance(override_candidate, dict):
            override_key = _candidate_key(override_candidate)
            override_index = next(
                (index for index, candidate in enumerate(actual_pool) if _candidate_key(candidate) == override_key),
                None,
            )
            if override_index is not None and override_index > 0:
                chosen = actual_pool.pop(override_index)
                chosen["v26_v5_override"] = 1.0
                actual_pool.insert(0, chosen)
                applied = True
                model.stats["override_count"] = int(model.stats.get("override_count", 0)) + 1
            elif override_index == 0:
                actual_pool[0]["v26_v5_override"] = 1.0
                applied = True

        for index, candidate in enumerate(actual_pool, start=1):
            candidate["rank"] = index
            candidate["ranking_version"] = "2.6-base+v5-gate"

        self._v26_base_rank_pool = [dict(candidate) for candidate in annotated_base]
        self._v26_v5_shadow_pool = [dict(candidate) for candidate in v5_pool]
        self._v26_v5_override_status = {
            **dict(gate),
            "applied": bool(applied),
            "override_key": list(override_key) if override_key is not None else None,
        }
        return actual_pool[:requested_limit]

    def rank_with_funnel_wrapped(
        self: Any,
        raw_hotspots: Sequence[dict[str, Any]],
        gt_xy: tuple[float, float] | None = None,
        limit: int | None = None,
        match_radius_px: float | None = None,
    ) -> Any:
        # Synthetic labelled runs need the complete candidate pool. V2.5 lost
        # 57 GT candidates only because the training UI truncated the sorted
        # list at candidate_limit. Keeping the full list is diagnostic/training
        # correctness, not GT leakage: no coordinate is used to choose a point.
        effective_limit = limit
        if gt_xy is not None:
            effective_limit = max(int(limit or 0), len(raw_hotspots))

        result = original_rank_with_funnel(
            self,
            raw_hotspots,
            gt_xy=gt_xy,
            limit=effective_limit,
            match_radius_px=match_radius_px,
        )

        if gt_xy is None:
            return result

        model = _get_model(self)
        base_pool = [dict(candidate) for candidate in getattr(self, "_v26_base_rank_pool", []) or []]
        v5_pool = [dict(candidate) for candidate in getattr(self, "_v26_v5_shadow_pool", []) or []]
        radius = float(match_radius_px or getattr(self, "settings", {}).get("click_match_radius_px", 42.0))
        validation_radius = float(model.config.get("positive_radius_px", 12.0))

        # Evaluate this shot before learning from it.  The auto-gate is based on
        # the same strict radius used for positive labels, not the loose 42 px
        # detector-recall radius.  A model that merely points to the right
        # neighbourhood must never earn authority to override the base ranker.
        validation = model.record_validation(
            (float(gt_xy[0]), float(gt_xy[1])),
            base_pool,
            v5_pool,
            match_radius_px=validation_radius,
        )
        gate_before_training = dict(model.gate_status())
        override_status = dict(getattr(self, "_v26_v5_override_status", {}) or {})

        benchmark_mode = bool(getattr(self, "settings", {}).get("benchmark_mode", False))
        if not benchmark_mode:
            train_result = model.learn_from_ground_truth(
                (float(gt_xy[0]), float(gt_xy[1])),
                base_pool,
            )
        else:
            train_result = {"trained": False, "reason": "benchmark_mode"}

        # Persist rolling validation periodically; supervised updates already
        # save on their own cadence. Avoid 1000 full JSON writes per benchmark.
        try:
            validation_count = len(model.stats.get("validation", []) or [])
            if validation_count and validation_count % 10 == 0:
                model.save()
        except Exception:
            pass

        diagnostic = {
            "validation": validation,
            "gate": gate_before_training,
            "override": override_status,
            "training": train_result,
            "model": model.summary(),
            "validation_radius_px": validation_radius,
            "base_pool_count": len(base_pool),
            "v5_pool_count": len(v5_pool),
            "base_gt_rank_42px": _rank_of_gt(base_pool, gt_xy, radius),
            "v5_gt_rank_42px": _rank_of_gt(v5_pool, gt_xy, radius),
            "base_gt_rank_12px": _rank_of_gt(base_pool, gt_xy, 12.0),
            "v5_gt_rank_12px": _rank_of_gt(v5_pool, gt_xy, 12.0),
        }
        self._v26_last_v5_diagnostic = diagnostic

        # Attach the diagnostic directly to the current detector record.  The
        # candidate-generator funnel hook may execute before this outer wrapper
        # returns, so relying on it to read runtime state can otherwise record
        # the previous shot's V5 diagnostic.
        try:
            from src.engine.camera.hit_scanner import HitScanner, hit_scanner

            engine = getattr(HitScanner, "_candidate_generator_v2_engine", None)
            gt = getattr(hit_scanner, "_detector_v2_ground_truth", None)
            shot_id = int(gt.get("shot_id", 0)) if isinstance(gt, dict) else 0
            record = getattr(engine, "_diagnostics", {}).get(shot_id)
            if isinstance(record, dict):
                record.setdefault("evaluation_funnel", {})["v26_ranker_v5"] = diagnostic
        except Exception:
            pass

        return result

    AIRuntime.rank_candidates = rank_candidates_wrapped
    AIRuntime.rank_with_funnel = rank_with_funnel_wrapped
    AIRuntime._ranker_v5_extension_installed = True
    _INSTALLED = True
    print("[RANKER-V5] strict-candidate supervised ranker installed (validated auto-gate)")


__all__ = ["install_ranker_v5_extension"]
