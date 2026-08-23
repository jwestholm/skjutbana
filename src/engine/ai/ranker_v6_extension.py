from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

from src.engine.ai.hypothesis_v27 import HypothesisBuilderV27
from src.engine.ai.ranker_v6 import RankerV6

_INSTALLED = False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float(default)
    except Exception:
        return float(default)


def _distance(candidate: dict[str, Any], gt_xy: tuple[float, float]) -> float:
    return math.hypot(
        _safe_float(candidate.get("camera_x")) - float(gt_xy[0]),
        _safe_float(candidate.get("camera_y")) - float(gt_xy[1]),
    )


def _nearest(pool: Sequence[dict[str, Any]], gt_xy: tuple[float, float]) -> float | None:
    if not pool:
        return None
    return float(min(_distance(candidate, gt_xy) for candidate in pool))


def _rank(pool: Sequence[dict[str, Any]], gt_xy: tuple[float, float], radius: float) -> int | None:
    for index, candidate in enumerate(pool, start=1):
        if _distance(candidate, gt_xy) <= radius:
            return index
    return None


def _coverage(pool: Sequence[dict[str, Any]], gt_xy: tuple[float, float]) -> dict[str, Any]:
    nearest = _nearest(pool, gt_xy)
    return {
        "nearest_px": nearest,
        "within_10": bool(nearest is not None and nearest <= 10.0),
        "within_12": bool(nearest is not None and nearest <= 12.0),
        "within_20": bool(nearest is not None and nearest <= 20.0),
        "within_42": bool(nearest is not None and nearest <= 42.0),
    }


def _get_builder(runtime: Any) -> HypothesisBuilderV27:
    builder = getattr(runtime, "_v27_hypothesis_builder", None)
    if isinstance(builder, HypothesisBuilderV27):
        return builder
    builder = HypothesisBuilderV27()
    runtime._v27_hypothesis_builder = builder
    return builder


def _install_reset_bridge(runtime: Any, model: RankerV6) -> None:
    memory = getattr(runtime, "memory", None)
    if memory is None or bool(getattr(memory, "_ranker_v6_reset_bridge", False)):
        return
    register = getattr(memory, "register_reset_callback", None)
    if callable(register):
        register(model.reset)
        memory._ranker_v6_reset_bridge = True
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
    memory._ranker_v6_reset_bridge = True


def _get_model(runtime: Any) -> RankerV6:
    model = getattr(runtime, "_ranker_v6", None)
    if isinstance(model, RankerV6):
        return model
    storage_dir = getattr(runtime, "storage_dir", None)
    root = Path(storage_dir) if storage_dir is not None else Path("content/ai")
    model = RankerV6(
        model_path=root / "ranker_v6.json",
        config_path=root / "ranker_v6_config.json",
    )
    runtime._ranker_v6 = model
    _install_reset_bridge(runtime, model)
    return model


def _annotate_actual(
    actual: Sequence[dict[str, Any]],
    *,
    authoritative_v6: bool,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, candidate in enumerate(actual, start=1):
        item = dict(candidate)
        item["rank"] = index
        item["ranking_version"] = (
            "2.7-hypothesis-v6" if authoritative_v6 else "2.7-hypothesis-baseline"
        )
        item["v27_v6_authoritative"] = 1.0 if authoritative_v6 else 0.0
        if authoritative_v6:
            # Real emission reads ai_score as confidence. Only expose V6's
            # confidence when its validation gate has actually granted authority.
            item["ai_score"] = _safe_float(item.get("ranker_v6_score"), 0.5)
            item["combined_score"] = _safe_float(item.get("ranker_v6_score"), 0.5)
        else:
            item["combined_score"] = _safe_float(item.get("v27_baseline_score"), 0.0)
        result.append(item)
    return result


def install_ranker_v6_extension() -> None:
    """Install V2.7 hypothesis consolidation and the validated V6 ranker.

    The V2.6 detector/vault is intentionally untouched. This wrapper replaces
    *ranking* of filtered candidates with:

        filtered observations -> V2.7 hypotheses -> spatial pool -> ranking

    No GT coordinate is used in hypothesis construction or current-shot ranking.
    Synthetic GT is used only after that shot was already ranked, for validation
    and subsequent online training.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    from src.engine.ai.runtime import AIRuntime

    if bool(getattr(AIRuntime, "_ranker_v6_extension_installed", False)):
        _INSTALLED = True
        return

    original_rank_with_funnel = AIRuntime.rank_with_funnel

    def rank_candidates_wrapped(
        self: Any,
        candidates: Sequence[dict[str, Any]],
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        requested_limit = max(1, int(limit or getattr(self, "settings", {}).get("top_k", 10)))
        source = [dict(candidate) for candidate in candidates]
        self._v27_input_candidates = source
        if not source:
            self._v27_all_hypotheses = []
            self._v27_hypothesis_pool = []
            self._v27_baseline_pool = []
            self._v27_v6_shadow_pool = []
            self._v27_actual_pool = []
            self._v27_hypothesis_stats = {
                "input_count": 0,
                "cluster_count": 0,
                "pool_count": 0,
                "reduction_ratio": 0.0,
            }
            return []

        builder = _get_builder(self)
        all_hypotheses, pool, stats = builder.build(source)
        baseline = sorted(
            (dict(candidate) for candidate in pool),
            key=lambda candidate: _safe_float(candidate.get("v27_baseline_score")),
            reverse=True,
        )

        model = _get_model(self)
        if bool(model.config.get("enabled", True)):
            v6_pool = model.rank(pool)
        else:
            v6_pool = [dict(candidate) for candidate in baseline]

        authoritative, authority_status = model.choose_authoritative(v6_pool)
        actual_full = v6_pool if authoritative else baseline
        actual_full = _annotate_actual(actual_full, authoritative_v6=authoritative)

        self._v27_all_hypotheses = [dict(candidate) for candidate in all_hypotheses]
        self._v27_hypothesis_pool = [dict(candidate) for candidate in pool]
        self._v27_baseline_pool = [dict(candidate) for candidate in baseline]
        self._v27_v6_shadow_pool = [dict(candidate) for candidate in v6_pool]
        self._v27_actual_pool = [dict(candidate) for candidate in actual_full]
        self._v27_hypothesis_stats = dict(stats)
        self._v27_authority_status = dict(authority_status)

        return actual_full[:requested_limit]

    def rank_with_funnel_wrapped(
        self: Any,
        raw_hotspots: Sequence[dict[str, Any]],
        gt_xy: tuple[float, float] | None = None,
        limit: int | None = None,
        match_radius_px: float | None = None,
    ) -> Any:
        # A labelled synthetic run must keep the complete V2.7 hypothesis pool.
        # The pool is already capped (~120), so this is both cheap and necessary
        # to distinguish clustering loss from ranker loss.
        effective_limit = limit
        if gt_xy is not None:
            try:
                hypothesis_limit = int(_get_builder(self).config.snapshot().get("max_hypotheses", 120))
            except Exception:
                hypothesis_limit = 120
            effective_limit = max(int(limit or 0), hypothesis_limit)

        result = original_rank_with_funnel(
            self,
            raw_hotspots,
            gt_xy=gt_xy,
            limit=effective_limit,
            match_radius_px=match_radius_px,
        )

        if gt_xy is None:
            return result

        gt = (float(gt_xy[0]), float(gt_xy[1]))
        input_candidates = [dict(c) for c in getattr(self, "_v27_input_candidates", []) or []]
        all_hypotheses = [dict(c) for c in getattr(self, "_v27_all_hypotheses", []) or []]
        hypothesis_pool = [dict(c) for c in getattr(self, "_v27_hypothesis_pool", []) or []]
        baseline_pool = [dict(c) for c in getattr(self, "_v27_baseline_pool", []) or []]
        v6_pool = [dict(c) for c in getattr(self, "_v27_v6_shadow_pool", []) or []]
        actual_pool = [dict(c) for c in getattr(self, "_v27_actual_pool", []) or []]
        stats = dict(getattr(self, "_v27_hypothesis_stats", {}) or {})
        authority = dict(getattr(self, "_v27_authority_status", {}) or {})

        model = _get_model(self)
        # PRE-TRAIN validation: current shot cannot improve its own gate result.
        validation = model.record_validation(gt, baseline_pool, v6_pool)
        gate_before_training = dict(model.gate_status())
        benchmark_mode = bool(getattr(self, "settings", {}).get("benchmark_mode", False))
        if not benchmark_mode:
            training = model.learn_from_ground_truth(gt, hypothesis_pool)
        else:
            training = {"trained": False, "reason": "benchmark_mode"}

        try:
            validation_count = len(model.stats.get("validation", []) or [])
            if validation_count and validation_count % 10 == 0:
                model.save()
        except Exception:
            pass

        diagnostic = {
            "schema_version": "2.7",
            "stats": stats,
            "coverage": {
                "filtered_input": _coverage(input_candidates, gt),
                "all_hypotheses": _coverage(all_hypotheses, gt),
                "hypothesis_pool": _coverage(hypothesis_pool, gt),
            },
            "oracle": {
                "input_nearest_px": _nearest(input_candidates, gt),
                "cluster_nearest_px": _nearest(all_hypotheses, gt),
                "pool_nearest_px": _nearest(hypothesis_pool, gt),
            },
            "ranks": {
                "baseline_10": _rank(baseline_pool, gt, 10.0),
                "baseline_20": _rank(baseline_pool, gt, 20.0),
                "baseline_42": _rank(baseline_pool, gt, 42.0),
                "v6_10": _rank(v6_pool, gt, 10.0),
                "v6_20": _rank(v6_pool, gt, 20.0),
                "v6_42": _rank(v6_pool, gt, 42.0),
                "actual_10": _rank(actual_pool, gt, 10.0),
                "actual_20": _rank(actual_pool, gt, 20.0),
                "actual_42": _rank(actual_pool, gt, 42.0),
            },
            "selected": {
                "distance_px": _distance(actual_pool[0], gt) if actual_pool else None,
                "within_10": bool(actual_pool and _distance(actual_pool[0], gt) <= 10.0),
                "within_20": bool(actual_pool and _distance(actual_pool[0], gt) <= 20.0),
                "within_42": bool(actual_pool and _distance(actual_pool[0], gt) <= 42.0),
            },
            "pool_counts": {
                "filtered_input": len(input_candidates),
                "all_hypotheses": len(all_hypotheses),
                "hypothesis_pool": len(hypothesis_pool),
                "baseline_pool": len(baseline_pool),
                "v6_pool": len(v6_pool),
            },
            "validation": validation,
            "gate_before_training": gate_before_training,
            "authority_for_current_shot": authority,
            "training": training,
            "model": model.summary(),
        }
        self._v27_last_diagnostic = diagnostic

        # Attach to the current detector record before it is flushed to JSONL.
        try:
            from src.engine.camera.hit_scanner import HitScanner, hit_scanner

            engine = getattr(HitScanner, "_candidate_generator_v2_engine", None)
            ground_truth = getattr(hit_scanner, "_detector_v2_ground_truth", None)
            shot_id = int(ground_truth.get("shot_id", 0)) if isinstance(ground_truth, dict) else 0
            record = getattr(engine, "_diagnostics", {}).get(shot_id)
            if isinstance(record, dict):
                record.setdefault("evaluation_funnel", {})["v27_hypotheses"] = diagnostic
        except Exception:
            pass

        return result

    AIRuntime.rank_candidates = rank_candidates_wrapped
    AIRuntime.rank_with_funnel = rank_with_funnel_wrapped
    AIRuntime._ranker_v6_extension_installed = True
    _INSTALLED = True
    print("[RANKER-V6] V2.7 hypothesis clustering + validated ranker installed")


__all__ = ["install_ranker_v6_extension"]
