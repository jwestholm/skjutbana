from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Sequence


CONFIG_PATH = Path("content/ai/hypothesis_v27_config.json")

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
    "max_hypotheses": 120,
    "macro_cell_px": 240.0,
    "macro_bucket_depth": 4,
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


class HypothesisV27Config:
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


class HypothesisBuilderV27:
    """Reduce filtered detector candidates to spatial/temporal shot hypotheses.

    No ground-truth coordinate enters this class. It can therefore be used in
    normal gameplay and in labelled synthetic benchmarks without leakage.
    """

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self.config = HypothesisV27Config(config_path)

    def build(
        self,
        candidates: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        cfg = self.config.snapshot()
        source = [dict(candidate) for candidate in candidates]
        if not source or not bool(cfg.get("enabled", True)):
            return source, source, {
                "input_count": len(source),
                "cluster_count": len(source),
                "pool_count": len(source),
                "reduction_ratio": 1.0,
            }

        clusters = self._cluster(source, cfg)
        hypotheses = [self._summarise(cluster, cfg) for cluster in clusters]
        pool = self._spatial_pool(hypotheses, cfg)
        stats = {
            "input_count": len(source),
            "cluster_count": len(hypotheses),
            "pool_count": len(pool),
            "reduction_ratio": round(len(pool) / max(1, len(source)), 5),
            "member_count_median": _median(
                (_safe_float(h.get("v27_member_count")) for h in hypotheses), 0.0
            ),
            "spread_median_px": round(
                _median((_safe_float(h.get("v27_spread_px")) for h in hypotheses), 0.0),
                3,
            ),
        }
        return hypotheses, pool, stats

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

    def _spatial_pool(
        self,
        hypotheses: list[dict[str, Any]],
        cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        limit = max(12, _safe_int(cfg.get("max_hypotheses", 120), 120))
        if len(hypotheses) <= limit:
            return sorted(
                (dict(h) for h in hypotheses),
                key=lambda h: _safe_float(h.get("v27_baseline_score")),
                reverse=True,
            )

        macro_px = max(50.0, _safe_float(cfg.get("macro_cell_px", 240.0), 240.0))
        depth = max(1, _safe_int(cfg.get("macro_bucket_depth", 4), 4))

        # Adapt to the actual camera coordinate span. On a 4K camera, a fixed
        # 240 px grid can contain more macro cells than the whole 120-item pool;
        # a naive round-robin would then systematically starve one edge of the
        # frame. Grow macro cells until roughly 1/3 of the pool is occupied by
        # spatial regions, leaving several hypothesis slots per region.
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
            # Diversity within each macro cell: take winners from three different
            # evidence heads before filling from the baseline ordering.
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
        # If an unusually sparse/huge frame still creates more buckets than the
        # target, choose the first-round regions by their best local evidence,
        # never by x/y sort order.
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
                    selected.append(dict(bucket.popleft()))
                if bucket:
                    next_keys.append(key)
            keys = next_keys

        return selected


__all__ = [
    "DEFAULT_CONFIG",
    "HypothesisBuilderV27",
    "HypothesisV27Config",
]
