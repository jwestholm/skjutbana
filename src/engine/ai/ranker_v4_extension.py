from __future__ import annotations

import math
from typing import Any, Sequence

from src.engine.ai.ranker_v4 import RankerV4


_INSTALLED = False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float(default)
    except Exception:
        return float(default)


def _percentile_ranks(values: list[float]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [1.0]
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    denominator = float(max(1, len(values) - 1))
    for position, index in enumerate(order):
        result[index] = float(position) / denominator
    return result


def _get_model(runtime: Any) -> RankerV4:
    model = getattr(runtime, "_ranker_v4", None)
    if isinstance(model, RankerV4):
        return model

    storage_dir = getattr(runtime, "storage_dir", None)
    if storage_dir is not None:
        try:
            from pathlib import Path

            root = Path(storage_dir)
            model = RankerV4(
                model_path=root / "ranker_v4.json",
                config_path=root / "ranker_v4_config.json",
                log_path=root / "ranker_v4" / "training_pairs.jsonl",
            )
        except Exception:
            model = RankerV4()
    else:
        model = RankerV4()

    runtime._ranker_v4 = model
    _install_memory_reset_bridge(runtime, model)
    return model


def _install_memory_reset_bridge(runtime: Any, model: RankerV4) -> None:
    memory = getattr(runtime, "memory", None)
    if memory is None or bool(getattr(memory, "_ranker_v4_reset_bridge", False)):
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
    memory._ranker_v4_reset_bridge = True


def install_ranker_v4_extension() -> None:
    """Patch AIRuntime additively without replacing runtime.py.

    This is intentionally compatible with both the tested local V2.3 runtime
    and the older V2.2 runtime: the previous ranker still enriches candidates,
    then V4 reranks the complete surviving candidate pool using local patch
    shape and an online pairwise model.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    from src.engine.ai.runtime import AIRuntime

    if bool(getattr(AIRuntime, "_ranker_v4_extension_installed", False)):
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
            self._v24_last_rank_pool = []
            return []

        requested_limit = int(limit or getattr(self, "settings", {}).get("top_k", 10))
        # Ask the existing ranker to enrich/sort the WHOLE list instead of only
        # the visible Top-K. This is essential because V2.3 measurements showed
        # the GT candidate around median rank ~78.
        full_limit = max(requested_limit, len(candidates))
        try:
            base_ranked = original_rank_candidates(self, candidates, limit=full_limit)
        except TypeError:
            base_ranked = original_rank_candidates(self, candidates, full_limit)

        if not base_ranked:
            self._v24_last_rank_pool = []
            return []

        model = _get_model(self)
        if not bool(model.config.get("enabled", True)):
            self._v24_last_rank_pool = [dict(candidate) for candidate in base_ranked]
            return self._v24_last_rank_pool[:requested_limit]

        base_values = [
            _safe_float(candidate.get("combined_score", candidate.get("score", 0.0)))
            for candidate in base_ranked
        ]
        patch_values = [model.patch_prior(candidate) for candidate in base_ranked]
        learned_values = [model.raw_score(candidate) for candidate in base_ranked]

        base_ranks = _percentile_ranks(base_values)
        patch_ranks = _percentile_ranks(patch_values)
        learned_ranks = _percentile_ranks(learned_values)

        model_weight = model.effective_weight()
        initial_patch = max(
            0.0,
            min(0.90, _safe_float(model.config.get("initial_patch_weight", 0.55), 0.55)),
        )
        full_patch = max(
            0.0,
            min(0.70, _safe_float(model.config.get("full_patch_weight", 0.22), 0.22)),
        )
        max_model = max(
            0.0,
            min(0.95, _safe_float(model.config.get("max_model_weight", 0.82), 0.82)),
        )
        learned_progress = 0.0 if max_model <= 1e-9 else min(1.0, model_weight / max_model)
        patch_weight = (
            initial_patch * (1.0 - learned_progress)
            + full_patch * learned_progress
        )
        base_weight = max(0.05, 1.0 - model_weight - patch_weight)
        total = base_weight + patch_weight + model_weight
        base_weight /= total
        patch_weight /= total
        learned_weight = model_weight / total

        reranked: list[dict[str, Any]] = []
        for index, candidate in enumerate(base_ranked):
            enriched = dict(candidate)
            v4_score = (
                base_weight * base_ranks[index]
                + patch_weight * patch_ranks[index]
                + learned_weight * learned_ranks[index]
            )
            enriched["v24_base_rank_score"] = float(base_ranks[index])
            enriched["v24_patch_prior"] = float(patch_values[index])
            enriched["v24_patch_rank_score"] = float(patch_ranks[index])
            enriched["ranker_v4_raw"] = float(learned_values[index])
            enriched["ranker_v4_score"] = float(model.score(candidate))
            enriched["ranker_v4_rank_score"] = float(learned_ranks[index])
            enriched["ranker_v4_weight"] = float(learned_weight)
            enriched["v24_patch_weight"] = float(patch_weight)
            enriched["v24_base_weight"] = float(base_weight)
            enriched["v24_combined_score"] = float(v4_score)
            enriched["combined_score"] = float(v4_score)
            enriched["ranking_version"] = "2.4"
            reranked.append(enriched)

        reranked.sort(
            key=lambda candidate: _safe_float(candidate.get("v24_combined_score", 0.0)),
            reverse=True,
        )
        for index, candidate in enumerate(reranked, start=1):
            candidate["rank"] = index

        self._v24_last_rank_pool = reranked
        return reranked[:requested_limit]

    def rank_with_funnel_wrapped(
        self: Any,
        raw_hotspots: Sequence[dict[str, Any]],
        gt_xy: tuple[float, float] | None = None,
        limit: int | None = None,
        match_radius_px: float | None = None,
    ) -> Any:
        result = original_rank_with_funnel(
            self,
            raw_hotspots,
            gt_xy=gt_xy,
            limit=limit,
            match_radius_px=match_radius_px,
        )

        # Train only AFTER this shot has been ranked. That prevents ground-truth
        # leakage into the result being measured for the same round.
        if gt_xy is not None:
            try:
                benchmark_mode = bool(getattr(self, "settings", {}).get("benchmark_mode", False))
                if not benchmark_mode:
                    model = _get_model(self)
                    pool = list(getattr(self, "_v24_last_rank_pool", []) or [])
                    positive_override = None
                    try:
                        from src.engine.camera.hit_scanner import HitScanner
                        from src.engine.camera.detector_v24_extension import (
                            build_ground_truth_patch_candidate,
                        )

                        detector_engine = getattr(
                            HitScanner,
                            "_candidate_generator_v2_engine",
                            None,
                        )
                        if detector_engine is not None:
                            positive_override = build_ground_truth_patch_candidate(
                                detector_engine,
                                (float(gt_xy[0]), float(gt_xy[1])),
                            )
                    except Exception:
                        positive_override = None

                    train_result = model.learn_from_ground_truth(
                        (float(gt_xy[0]), float(gt_xy[1])),
                        pool,
                        positive_override=positive_override,
                    )
                    self._v24_last_train_result = train_result
            except Exception as exc:
                self._v24_last_train_result = {
                    "trained": False,
                    "reason": f"error:{exc}",
                }

        return result

    AIRuntime.rank_candidates = rank_candidates_wrapped
    AIRuntime.rank_with_funnel = rank_with_funnel_wrapped
    AIRuntime._ranker_v4_extension_installed = True
    _INSTALLED = True

    print("[RANKER-V4] Patch-descriptor pairwise ranker installed")
