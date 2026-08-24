from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Sequence


CONFIG_PATH = Path("content/ai/hypothesis_v28_config.json")

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    # Micro-clustering: candidates closer than this are treated as alternate
    # observations of the same physical hypothesis, unless doing so would make
    # the cluster implausibly wide.
    "merge_radius_px": 16.0,
    "max_cluster_diameter_px": 30.0,
    "max_members_per_cluster": 64,
    # The raw V2.6 pool can contain 500-700 candidates. V2.7 keeps spatial
    # coverage while reducing that to a rankable pool.
    "core_hypotheses": 120,
    "max_hypotheses": 220,
    "macro_cell_px": 240.0,
    "macro_bucket_depth": 4,
    "overflow_baseline_reserve": 56,
    "overflow_support_reserve": 32,
    "overflow_signal_reserve": 32,
    "overflow_diversity_reserve": 24,
    "overflow_vault_reserve": 20,
    # Local score ingredients. These are intentionally modest priors, not a
    # replacement detector. The online ranker may learn a different ordering.
    "carried_penalty": 0.08,
    "age_penalty": 0.05,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float(default)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sat(value: Any, scale: float) -> float:
    v = max(0.0, _safe_float(value))
    return math.tanh(v / max(1e-6, float(scale)))


def _median(values: Iterable[float], default: float = 0.0) -> float:
    data = sorted(float(v) for v in values)
    if not data:
        return float(default)
    middle = len(data) // 2
    if len(data) % 2:
        return data[middle]
    return 0.5 * (data[middle - 1] + data[middle])


def _candidate_xy(candidate: dict[str, Any]) -> tuple[float, float]:
    return (
        _safe_float(candidate.get("camera_x")),
        _safe_float(candidate.get("camera_y")),
    )


def _candidate_patch_prior(candidate: dict[str, Any]) -> float:
    if "v24_patch_prior" in candidate:
        return _clip01(_safe_float(candidate.get("v24_patch_prior")))
    # Dependency-free approximation of the V2.4 patch prior. It is deliberately
    # weak because the physical benchmark showed patch descriptors are useful
    # as evidence but dangerous as a dominant global ranker.
    compact = _clip01(_safe_float(candidate.get("v24_patch_compactness")))
    centred = _clip01(_safe_float(candidate.get("v24_patch_centeredness")))
    isotropy = _clip01(_safe_float(candidate.get("v24_patch_isotropy")))
    local_snr = _clip01(_safe_float(candidate.get("v24_patch_local_snr")))
    return 0.30 * compact + 0.25 * centred + 0.20 * isotropy + 0.25 * local_snr


def _candidate_evidence_weight(candidate: dict[str, Any]) -> float:
    v1 = 1.0 if _safe_float(candidate.get("detector_v1")) > 0.5 else 0.0
    v2 = 1.0 if _safe_float(candidate.get("detector_v2")) > 0.5 else 0.0
    tile = 1.0 if _safe_float(candidate.get("v24_tile_probe")) > 0.5 else 0.0
    agreement = 1.0 if _safe_float(candidate.get("detector_agreement")) > 0.5 else 0.0
    hits = _sat(candidate.get("v26_vault_hits", 1.0), 2.5)
    patch = _candidate_patch_prior(candidate)
    current = 0.0 if _safe_float(candidate.get("v26_vault_carried")) > 0.5 else 1.0
    return (
        1.0
        + 0.24 * v1
        + 0.24 * v2
        + 0.18 * tile
        + 0.30 * agreement
        + 0.18 * hits
        + 0.12 * patch
        + 0.08 * current
    )


def _weighted_geometric_median(
    points: Sequence[tuple[float, float, float]],
    *,
    iterations: int = 12,
) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    total = sum(max(1e-6, p[2]) for p in points)
    x = sum(p[0] * max(1e-6, p[2]) for p in points) / total
    y = sum(p[1] * max(1e-6, p[2]) for p in points) / total
    for _ in range(max(1, iterations)):
        num_x = num_y = den = 0.0
        snapped = None
        for px, py, weight in points:
            distance = math.hypot(x - px, y - py)
            if distance < 1e-5:
                snapped = (px, py)
                break
            local_weight = max(1e-6, weight) / distance
            num_x += local_weight * px
            num_y += local_weight * py
            den += local_weight
        if snapped is not None:
            return snapped
        if den <= 1e-9:
            break
        new_x = num_x / den
        new_y = num_y / den
        if math.hypot(new_x - x, new_y - y) < 0.02:
            x, y = new_x, new_y
            break
        x, y = new_x, new_y
    return float(x), float(y)


class HypothesisV28Config:
    def __init__(self, path: Path = CONFIG_PATH) -> None:
        self.path = Path(path)
        self.values = dict(DEFAULT_CONFIG)
        self.reload()

    def reload(self) -> None:
        values = dict(DEFAULT_CONFIG)
        try:
            if self.path.exists():
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    values.update(loaded)
        except Exception:
            pass
        self.values = values

    def snapshot(self) -> dict[str, Any]:
        self.reload()
        return dict(self.values)


class HypothesisBuilderV28:
    """Reduce filtered detector candidates to spatial/temporal shot hypotheses.

    No ground-truth coordinate enters this class. It can therefore be used in
    normal gameplay and in labelled synthetic benchmarks without leakage.
    """

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self.config = HypothesisV28Config(config_path)

    def build(
        self,
        candidates: Sequence[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        """Build micro-clusters, a conservative V2.7-style core pool and a
        larger V2.8 recall pool.

        The 120-item core preserves the previously measured baseline ordering.
        The recall pool is deliberately much larger (220 by default) so correct
        clusters are available to V6 training/shadow ranking instead of being
        discarded by spatial pooling.
        """
        cfg = self.config.snapshot()
        source = [dict(candidate) for candidate in candidates]
        if not source or not bool(cfg.get("enabled", True)):
            stats = {
                "input_count": len(source),
                "cluster_count": len(source),
                "core_pool_count": len(source),
                "pool_count": len(source),
                "reduction_ratio": 1.0,
                "pool_mode": "disabled_or_empty",
            }
            return source, source, source, stats

        clusters = self._cluster(source, cfg)
        hypotheses = [self._summarise(cluster, cfg) for cluster in clusters]
        core_pool = self._legacy_core_pool(hypotheses, cfg)
        core_markers = {self._marker(h) for h in core_pool}
        recall_pool = self._recall_pool(hypotheses, core_markers, cfg)

        for item in hypotheses:
            marker = self._marker(item)
            item["v28_core_pool"] = 1.0 if marker in core_markers else 0.0
        # Re-annotate copied pool members after marking the source hypotheses.
        core_pool = [self._copy_with_core(h, core_markers) for h in core_pool]
        recall_pool = [self._copy_with_core(h, core_markers) for h in recall_pool]

        reason_counts: dict[str, int] = defaultdict(int)
        for item in recall_pool:
            for reason in item.get("v28_pool_reasons", []) or []:
                reason_counts[str(reason)] += 1

        stats = {
            "schema_version": "2.8",
            "input_count": len(source),
            "cluster_count": len(hypotheses),
            "core_pool_count": len(core_pool),
            "pool_count": len(recall_pool),
            "pool_limit": max(12, _safe_int(cfg.get("max_hypotheses", 220), 220)),
            "pool_dropped": max(0, len(hypotheses) - len(recall_pool)),
            "pool_mode": "keep_all" if len(hypotheses) <= max(12, _safe_int(cfg.get("max_hypotheses", 220), 220)) else "recall_overflow",
            "pool_reason_counts": dict(reason_counts),
            "reduction_ratio": round(len(recall_pool) / max(1, len(source)), 5),
            "member_count_median": _median(
                (_safe_float(h.get("v27_member_count")) for h in hypotheses), 0.0
            ),
            "spread_median_px": round(
                _median((_safe_float(h.get("v27_spread_px")) for h in hypotheses), 0.0),
                3,
            ),
        }
        return hypotheses, recall_pool, core_pool, stats

    @staticmethod
    def _marker(hypothesis: dict[str, Any]) -> tuple[float, float]:
        x, y = _candidate_xy(hypothesis)
        return (round(x, 4), round(y, 4))

    def _copy_with_core(
        self,
        hypothesis: dict[str, Any],
        core_markers: set[tuple[float, float]],
    ) -> dict[str, Any]:
        item = dict(hypothesis)
        item["v28_core_pool"] = 1.0 if self._marker(hypothesis) in core_markers else 0.0
        return item

    def _cluster(
        self,
        candidates: list[dict[str, Any]],
        cfg: dict[str, Any],
    ) -> list[list[dict[str, Any]]]:
        radius = max(4.0, _safe_float(cfg.get("merge_radius_px", 16.0), 16.0))
        max_diameter = max(
            radius,
            _safe_float(cfg.get("max_cluster_diameter_px", 30.0), 30.0),
        )
        max_members = max(2, _safe_int(cfg.get("max_members_per_cluster", 64), 64))

        # Spatial hash avoids the O(N^2) scan that becomes expensive with the
        # 500-700 observation pools produced by V2.6 Shot Vault. Cluster centres
        # only need to compare with neighbouring hash cells.
        cell_size = max(radius, 6.0)
        ordered = sorted(candidates, key=_candidate_evidence_weight, reverse=True)
        clusters: list[dict[str, Any]] = []
        buckets: dict[tuple[int, int], set[int]] = defaultdict(set)

        def bucket_for(x: float, y: float) -> tuple[int, int]:
            return (int(math.floor(x / cell_size)), int(math.floor(y / cell_size)))

        for candidate in ordered:
            cx, cy = _candidate_xy(candidate)
            bx, by = bucket_for(cx, cy)
            possible: set[int] = set()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    possible.update(buckets.get((bx + dx, by + dy), set()))

            best_index = -1
            best_distance = float("inf")
            for index in possible:
                cluster = clusters[index]
                if len(cluster["members"]) >= max_members:
                    continue
                distance = math.hypot(cx - cluster["cx"], cy - cluster["cy"])
                if distance > radius or distance >= best_distance:
                    continue
                min_x = min(cluster["min_x"], cx)
                max_x = max(cluster["max_x"], cx)
                min_y = min(cluster["min_y"], cy)
                max_y = max(cluster["max_y"], cy)
                if math.hypot(max_x - min_x, max_y - min_y) > max_diameter:
                    continue
                best_index = index
                best_distance = distance

            if best_index < 0:
                index = len(clusters)
                clusters.append(
                    {
                        "members": [candidate],
                        "cx": cx,
                        "cy": cy,
                        "weight": _candidate_evidence_weight(candidate),
                        "min_x": cx,
                        "max_x": cx,
                        "min_y": cy,
                        "max_y": cy,
                        "bucket": (bx, by),
                    }
                )
                buckets[(bx, by)].add(index)
                continue

            cluster = clusters[best_index]
            old_bucket = cluster.get("bucket")
            weight = _candidate_evidence_weight(candidate)
            old_weight = max(1e-6, _safe_float(cluster.get("weight"), 1.0))
            total = old_weight + weight
            cluster["cx"] = (cluster["cx"] * old_weight + cx * weight) / total
            cluster["cy"] = (cluster["cy"] * old_weight + cy * weight) / total
            cluster["weight"] = total
            cluster["min_x"] = min(cluster["min_x"], cx)
            cluster["max_x"] = max(cluster["max_x"], cx)
            cluster["min_y"] = min(cluster["min_y"], cy)
            cluster["max_y"] = max(cluster["max_y"], cy)
            cluster["members"].append(candidate)

            new_bucket = bucket_for(cluster["cx"], cluster["cy"])
            if new_bucket != old_bucket:
                if old_bucket in buckets:
                    buckets[old_bucket].discard(best_index)
                buckets[new_bucket].add(best_index)
                cluster["bucket"] = new_bucket

        return [list(cluster["members"]) for cluster in clusters]

    def _summarise(
        self,
        members: list[dict[str, Any]],
        cfg: dict[str, Any],
    ) -> dict[str, Any]:
        points = [
            (*_candidate_xy(candidate), _candidate_evidence_weight(candidate))
            for candidate in members
        ]
        center_x, center_y = _weighted_geometric_median(points)

        # Representative supplies ordinary candidate fields used elsewhere.
        representative = min(
            members,
            key=lambda candidate: (
                math.hypot(
                    _candidate_xy(candidate)[0] - center_x,
                    _candidate_xy(candidate)[1] - center_y,
                ),
                -_candidate_evidence_weight(candidate),
            ),
        )
        result = dict(representative)
        result["camera_x"] = float(center_x)
        result["camera_y"] = float(center_y)
        result["v27_hypothesis"] = 1.0

        distances = [
            math.hypot(x - center_x, y - center_y)
            for x, y, _weight in points
        ]
        weights = [p[2] for p in points]
        total_weight = max(1e-6, sum(weights))
        spread = math.sqrt(
            sum(weight * distance * distance for weight, distance in zip(weights, distances))
            / total_weight
        )

        count = len(members)
        v1_count = sum(1 for c in members if _safe_float(c.get("detector_v1")) > 0.5)
        v2_count = sum(1 for c in members if _safe_float(c.get("detector_v2")) > 0.5)
        tile_count = sum(1 for c in members if _safe_float(c.get("v24_tile_probe")) > 0.5)
        agreement_count = sum(1 for c in members if _safe_float(c.get("detector_agreement")) > 0.5)
        carried_count = sum(1 for c in members if _safe_float(c.get("v26_vault_carried")) > 0.5)
        source_diversity = sum(
            1
            for value in (v1_count, v2_count, tile_count, agreement_count)
            if value > 0
        )

        scores = [_safe_float(c.get("score")) for c in members]
        patches = [_candidate_patch_prior(c) for c in members]
        hits = [max(1.0, _safe_float(c.get("v26_vault_hits", 1.0))) for c in members]
        ages = [max(0.0, _safe_float(c.get("v26_vault_age_s", 0.0))) for c in members]
        persistence = [_clip01(_safe_float(c.get("persistence", 0.5))) for c in members]
        existed = [_clip01(_safe_float(c.get("existed_before", 0.0))) for c in members]

        def max_feature(key: str, scale: float) -> float:
            return _sat(max((_safe_float(c.get(key)) for c in members), default=0.0), scale)

        member_norm = _sat(count, 4.0)
        hits_norm = _sat(max(hits, default=1.0), 3.0)
        source_norm = source_diversity / 4.0
        current_fraction = 1.0 - carried_count / max(1, count)
        carried_fraction = carried_count / max(1, count)
        compactness = 1.0 / (1.0 + spread / 10.0)
        score_norm = _sat(max(scores, default=0.0), 15.0)
        patch_max = max(patches, default=0.0)
        patch_med = _median(patches, 0.0)
        zscore = max_feature("v2_zscore", 4.0)
        absdiff = max_feature("v2_absdiff", 8.0)
        dog = max_feature("v2_dog", 10.0)
        saliency = max_feature("v2_saliency", 35.0)
        persistence_med = _median(persistence, 0.5)
        existed_med = _median(existed, 0.0)
        age_med = _median(ages, 0.0)
        age_good = math.exp(-age_med / 1.2)

        support_score = (
            0.27 * source_norm
            + 0.22 * hits_norm
            + 0.16 * member_norm
            + 0.18 * current_fraction
            + 0.17 * compactness
        )
        signal_score = (
            0.18 * score_norm
            + 0.17 * patch_max
            + 0.10 * patch_med
            + 0.17 * zscore
            + 0.14 * absdiff
            + 0.10 * dog
            + 0.08 * saliency
            + 0.06 * persistence_med
        )
        baseline = (
            0.50 * support_score
            + 0.42 * signal_score
            + 0.08 * age_good
            - _safe_float(cfg.get("carried_penalty", 0.08), 0.08) * carried_fraction
            - _safe_float(cfg.get("age_penalty", 0.05), 0.05) * (1.0 - age_good)
            - 0.08 * existed_med
        )

        result.update(
            {
                "v27_member_count": float(count),
                "v27_spread_px": float(spread),
                "v27_source_diversity": float(source_diversity),
                "v27_v1_fraction": v1_count / max(1, count),
                "v27_v2_fraction": v2_count / max(1, count),
                "v27_tile_fraction": tile_count / max(1, count),
                "v27_agreement_fraction": agreement_count / max(1, count),
                "v27_current_fraction": float(current_fraction),
                "v27_carried_fraction": float(carried_fraction),
                "v27_hits_max": float(max(hits, default=1.0)),
                "v27_hits_mean": float(sum(hits) / max(1, len(hits))),
                "v27_patch_prior_max": float(patch_max),
                "v27_patch_prior_median": float(patch_med),
                "v27_score_max": float(max(scores, default=0.0)),
                "v27_score_median": float(_median(scores, 0.0)),
                "v27_persistence_median": float(persistence_med),
                "v27_existed_before_median": float(existed_med),
                "v27_age_median_s": float(age_med),
                "v27_support_score": float(support_score),
                "v27_signal_score": float(signal_score),
                "v27_baseline_score": float(baseline),
                "v27_compactness": float(compactness),
                "v27_zscore_norm": float(zscore),
                "v27_absdiff_norm": float(absdiff),
                "v27_dog_norm": float(dog),
                "v27_saliency_norm": float(saliency),
            }
        )
        return result

    def _legacy_core_pool(
        self,
        hypotheses: list[dict[str, Any]],
        cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """The measured V2.7 120-item pool, kept as a conservative core.

        V2.8 does not use this core as an oracle cutoff. It is only used to keep
        the proven V2.7 baseline candidates ahead of newly rescued overflow
        hypotheses until V6 has earned authority through the validation gate.
        """
        limit = max(12, _safe_int(cfg.get("core_hypotheses", 120), 120))
        return self._diverse_pool(hypotheses, limit, cfg, prefix="core")

    def _recall_pool(
        self,
        hypotheses: list[dict[str, Any]],
        core_markers: set[tuple[float, float]],
        cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        limit = max(24, _safe_int(cfg.get("max_hypotheses", 220), 220))
        ordered = sorted(
            (dict(h) for h in hypotheses),
            key=lambda h: _safe_float(h.get("v27_baseline_score")),
            reverse=True,
        )
        if len(ordered) <= limit:
            for item in ordered:
                item["v28_pool_reasons"] = [
                    "core" if self._marker(item) in core_markers else "keep_all"
                ]
            return ordered

        selected: list[dict[str, Any]] = []
        selected_markers: set[tuple[float, float]] = set()
        reason_map: dict[tuple[float, float], list[str]] = defaultdict(list)

        def add(candidate: dict[str, Any], reason: str) -> bool:
            marker = self._marker(candidate)
            if marker in selected_markers:
                if reason not in reason_map[marker]:
                    reason_map[marker].append(reason)
                return False
            if len(selected) >= limit:
                return False
            selected_markers.add(marker)
            reason_map[marker].append(reason)
            selected.append(dict(candidate))
            return True

        def take(order: Sequence[dict[str, Any]], count: int, reason: str) -> None:
            added = 0
            for item in order:
                before = len(selected)
                add(item, reason)
                if len(selected) > before:
                    added += 1
                if added >= max(0, int(count)) or len(selected) >= limit:
                    break

        # Preserve a substantial subset of the previous core first.
        core_order = [h for h in ordered if self._marker(h) in core_markers]
        take(core_order, _safe_int(cfg.get("overflow_baseline_reserve", 56), 56), "core_baseline")

        support_order = sorted(
            hypotheses,
            key=lambda h: (
                _safe_float(h.get("v27_support_score")),
                _safe_float(h.get("v27_baseline_score")),
            ),
            reverse=True,
        )
        signal_order = sorted(
            hypotheses,
            key=lambda h: (
                _safe_float(h.get("v27_signal_score")),
                _safe_float(h.get("v27_baseline_score")),
            ),
            reverse=True,
        )
        diversity_order = sorted(
            hypotheses,
            key=lambda h: (
                _safe_float(h.get("v27_source_diversity")),
                _safe_float(h.get("v27_current_fraction")),
                _safe_float(h.get("v27_hits_max")),
                _safe_float(h.get("v27_baseline_score")),
            ),
            reverse=True,
        )
        vault_order = sorted(
            hypotheses,
            key=lambda h: (
                _safe_float(h.get("v27_hits_max")),
                1.0 - _safe_float(h.get("v27_carried_fraction")),
                _safe_float(h.get("v27_support_score")),
                _safe_float(h.get("v27_baseline_score")),
            ),
            reverse=True,
        )
        take(support_order, _safe_int(cfg.get("overflow_support_reserve", 32), 32), "support")
        take(signal_order, _safe_int(cfg.get("overflow_signal_reserve", 32), 32), "signal")
        take(diversity_order, _safe_int(cfg.get("overflow_diversity_reserve", 24), 24), "diversity")
        take(vault_order, _safe_int(cfg.get("overflow_vault_reserve", 20), 20), "vault")

        # Spatial local winners make sure a strong global artifact cannot consume
        # the complete recall reserve. Use the same adaptive camera-scale logic
        # as V2.7, but fill local winners only after evidence-head reserves.
        macro_px = max(50.0, _safe_float(cfg.get("macro_cell_px", 240.0), 240.0))
        xs = [_candidate_xy(h)[0] for h in hypotheses]
        ys = [_candidate_xy(h)[1] for h in hypotheses]
        if xs and ys:
            span_x = max(xs) - min(xs) + 1.0
            span_y = max(ys) - min(ys) + 1.0
            estimated = max(1.0, math.ceil(span_x / macro_px) * math.ceil(span_y / macro_px))
            target_cells = max(10.0, float(limit) / 4.0)
            if estimated > target_cells:
                macro_px *= math.sqrt(estimated / target_cells)

        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for hypothesis in hypotheses:
            x, y = _candidate_xy(hypothesis)
            grouped[(int(math.floor(x / macro_px)), int(math.floor(y / macro_px)))].append(hypothesis)

        local_orders: list[list[dict[str, Any]]] = []
        for values in grouped.values():
            local_orders.append(sorted(
                values,
                key=lambda h: max(
                    _safe_float(h.get("v27_baseline_score")),
                    _safe_float(h.get("v27_support_score")),
                    _safe_float(h.get("v27_signal_score")),
                ),
                reverse=True,
            ))
        depth = 0
        while len(selected) < limit and any(depth < len(order) for order in local_orders):
            for order in local_orders:
                if depth < len(order) and len(selected) < limit:
                    add(order[depth], "spatial")
            depth += 1

        # Final fill: global baseline order. This makes the overflow selector
        # deterministic and ensures the strongest remaining evidence is kept.
        for item in ordered:
            if len(selected) >= limit:
                break
            add(item, "baseline_fill")

        for item in selected:
            item["v28_pool_reasons"] = list(reason_map.get(self._marker(item), []))
        return selected

    def _diverse_pool(
        self,
        hypotheses: list[dict[str, Any]],
        limit: int,
        cfg: dict[str, Any],
        *,
        prefix: str,
    ) -> list[dict[str, Any]]:
        if len(hypotheses) <= limit:
            result = sorted(
                (dict(h) for h in hypotheses),
                key=lambda h: _safe_float(h.get("v27_baseline_score")),
                reverse=True,
            )
            for item in result:
                item["v28_pool_reasons"] = [f"{prefix}_all"]
            return result

        macro_px = max(50.0, _safe_float(cfg.get("macro_cell_px", 240.0), 240.0))
        depth = max(1, _safe_int(cfg.get("macro_bucket_depth", 4), 4))
        xs = [_candidate_xy(h)[0] for h in hypotheses]
        ys = [_candidate_xy(h)[1] for h in hypotheses]
        if xs and ys:
            span_x = max(xs) - min(xs) + 1.0
            span_y = max(ys) - min(ys) + 1.0
            estimated = max(1.0, math.ceil(span_x / macro_px) * math.ceil(span_y / macro_px))
            target_cells = max(8.0, float(limit) / 3.0)
            if estimated > target_cells:
                macro_px *= math.sqrt(estimated / target_cells)

        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for hypothesis in hypotheses:
            x, y = _candidate_xy(hypothesis)
            grouped[(int(math.floor(x / macro_px)), int(math.floor(y / macro_px)))].append(hypothesis)

        buckets: dict[tuple[int, int], deque[dict[str, Any]]] = {}
        for key, values in grouped.items():
            orders = [
                sorted(values, key=lambda h: _safe_float(h.get("v27_baseline_score")), reverse=True),
                sorted(values, key=lambda h: _safe_float(h.get("v27_support_score")), reverse=True),
                sorted(values, key=lambda h: _safe_float(h.get("v27_signal_score")), reverse=True),
            ]
            local: list[dict[str, Any]] = []
            seen: set[int] = set()
            for rank in range(depth):
                for order in orders:
                    if rank < len(order):
                        item = order[rank]
                        marker = id(item)
                        if marker not in seen:
                            seen.add(marker)
                            local.append(item)
            for item in orders[0]:
                marker = id(item)
                if marker not in seen:
                    seen.add(marker)
                    local.append(item)
            buckets[key] = deque(local)

        selected: list[dict[str, Any]] = []
        keys = sorted(
            buckets,
            key=lambda key: max(
                (_safe_float(item.get("v27_baseline_score")) for item in buckets[key]),
                default=0.0,
            ),
            reverse=True,
        )
        while keys and len(selected) < limit:
            next_keys: list[tuple[int, int]] = []
            for key in keys:
                bucket = buckets[key]
                if bucket and len(selected) < limit:
                    item = dict(bucket.popleft())
                    item["v28_pool_reasons"] = [prefix]
                    selected.append(item)
                if bucket:
                    next_keys.append(key)
            keys = next_keys
        return selected


__all__ = [
    "DEFAULT_CONFIG",
    "HypothesisBuilderV28",
    "HypothesisV28Config",
]
