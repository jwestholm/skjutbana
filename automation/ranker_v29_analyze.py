from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from src.engine.ai.ranker_v7 import FEATURE_KEYS
from src.engine.ai.ranking_dataset_v29 import DATA_ROOT, load_session


SUMMARY_PATH = DATA_ROOT / "latest_v29_summary.json"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def _pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def _median(values: Sequence[float]) -> float | None:
    data = [float(value) for value in values if _finite(value) is not None]
    return statistics.median(data) if data else None


def _mean(values: Sequence[float]) -> float | None:
    data = [float(value) for value in values if _finite(value) is not None]
    return statistics.fmean(data) if data else None


def _rank(row: dict[str, Any], key: str, radius: int) -> int | None:
    block = row.get(key)
    if not isinstance(block, dict):
        return None
    value = block.get(f"rank_{radius}")
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _pool_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = row.get("candidates")
    if not isinstance(candidates, list):
        return []
    return [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and bool(candidate.get("membership", {}).get("hypothesis_pool"))
    ]


def _nearest_candidate(
    candidates: Sequence[dict[str, Any]],
    radius: float,
) -> dict[str, Any] | None:
    eligible = []
    for candidate in candidates:
        distance = _finite(candidate.get("distance_gt_px"))
        if distance is not None and distance <= float(radius):
            eligible.append((distance, candidate))
    if not eligible:
        return None
    return min(eligible, key=lambda item: item[0])[1]


def _hard_negative(
    candidates: Sequence[dict[str, Any]],
    *,
    negative_radius: float = 55.0,
) -> dict[str, Any] | None:
    eligible = []
    for candidate in candidates:
        distance = _finite(candidate.get("distance_gt_px"))
        if distance is None or distance < negative_radius:
            continue
        rank = candidate.get("ranks", {}).get("baseline")
        try:
            rank_value = int(rank) if rank is not None else 10**9
        except Exception:
            rank_value = 10**9
        baseline = _finite(candidate.get("features", {}).get("baseline_score")) or 0.0
        eligible.append((rank_value, -baseline, candidate))
    if not eligible:
        return None
    return min(eligible, key=lambda item: (item[0], item[1]))[2]


def _rank_metrics(rows: Sequence[dict[str, Any]], key: str, radius: int) -> dict[str, Any]:
    ranks = [
        rank
        for row in rows
        if (rank := _rank(row, key, radius)) is not None
    ]
    return {
        "covered": len(ranks),
        "top1_pct": round(_pct(sum(rank == 1 for rank in ranks), len(ranks)), 3),
        "top3_pct": round(_pct(sum(rank <= 3 for rank in ranks), len(ranks)), 3),
        "top5_pct": round(_pct(sum(rank <= 5 for rank in ranks), len(ranks)), 3),
        "median_rank": _median(ranks),
        "mean_rank": round(_mean(ranks), 3) if ranks else None,
        "mrr": round(_mean([1.0 / rank for rank in ranks]), 5) if ranks else None,
    }


def _feature_discrimination(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: dict[str, list[tuple[float, float]]] = {
        key: [] for key in FEATURE_KEYS
    }

    used_shots = 0
    for row in rows:
        pool = _pool_candidates(row)
        positive = _nearest_candidate(pool, 42.0)
        negative = _hard_negative(pool)
        if positive is None or negative is None:
            continue
        used_shots += 1
        pos_features = positive.get("features", {})
        neg_features = negative.get("features", {})
        for key in FEATURE_KEYS:
            pos = _finite(pos_features.get(key))
            neg = _finite(neg_features.get(key))
            if pos is not None and neg is not None:
                pairs[key].append((pos, neg))

    result: list[dict[str, Any]] = []
    for key, values in pairs.items():
        if not values:
            continue
        pos_values = [item[0] for item in values]
        neg_values = [item[1] for item in values]
        higher_wins = sum(pos > neg for pos, neg in values) / len(values)
        lower_wins = sum(pos < neg for pos, neg in values) / len(values)
        ties = 1.0 - higher_wins - lower_wins
        direction = "HIGHER_GT" if higher_wins >= lower_wins else "LOWER_GT"
        win_rate = max(higher_wins, lower_wins)
        diffs = [pos - neg for pos, neg in values]

        result.append(
            {
                "feature": key,
                "shots": len(values),
                "positive_median": round(_median(pos_values) or 0.0, 5),
                "negative_median": round(_median(neg_values) or 0.0, 5),
                "median_pos_minus_neg": round(_median(diffs) or 0.0, 5),
                "direction": direction,
                "direction_win_pct": round(100.0 * win_rate, 2),
                "tie_pct": round(100.0 * ties, 2),
                "strength": round((win_rate - 0.5) * math.sqrt(len(values)), 5),
            }
        )

    result.sort(
        key=lambda item: (
            item["strength"],
            item["direction_win_pct"],
        ),
        reverse=True,
    )
    return result


def summarise(rows: list[dict[str, Any]], session: str | None) -> dict[str, Any]:
    total = len(rows)
    sequences = [
        int(row.get("sequence"))
        for row in rows
        if row.get("sequence") is not None
    ]
    unique_sequences = len(set(sequences))

    oracle = {}
    for radius in (10, 20, 42):
        count = sum(
            bool(row.get("oracle", {}).get(f"pool_within_{radius}"))
            for row in rows
        )
        oracle[str(radius)] = {
            "count": count,
            "pct": round(_pct(count, total), 3),
        }

    selected = {}
    for radius in (10, 20, 42):
        count = sum(
            bool(row.get("actual", {}).get(f"selected_within_{radius}"))
            for row in rows
        )
        selected[str(radius)] = {
            "count": count,
            "pct": round(_pct(count, total), 3),
        }

    pool_counts = [
        int(row.get("counts", {}).get("hypothesis_pool", 0) or 0)
        for row in rows
    ]
    cluster_counts = [
        int(row.get("counts", {}).get("all_hypotheses", 0) or 0)
        for row in rows
    ]

    rankers = {
        "baseline": {
            str(radius): _rank_metrics(rows, "baseline", radius)
            for radius in (10, 20, 42)
        },
        "recall_baseline": {
            str(radius): _rank_metrics(rows, "recall_baseline", radius)
            for radius in (10, 20, 42)
        },
        "v6_shadow": {
            str(radius): _rank_metrics(rows, "v6_shadow", radius)
            for radius in (10, 20, 42)
        },
        "v7_shadow": {
            str(radius): _rank_metrics(rows, "v7_shadow", radius)
            for radius in (10, 20, 42)
        },
    }

    seeds = sorted(
        {
            int(seed)
            for row in rows
            if (seed := row.get("metadata", {}).get("benchmark_seed")) is not None
        }
    )
    backgrounds = Counter(
        str(value)
        for row in rows
        if (value := row.get("metadata", {}).get("background")) is not None
    )

    feature_discrimination = _feature_discrimination(rows)

    return {
        "schema_version": "2.9",
        "session": session,
        "shots": total,
        "sequence_integrity": {
            "unique": unique_sequences,
            "min": min(sequences) if sequences else None,
            "max": max(sequences) if sequences else None,
            "complete": bool(
                sequences
                and unique_sequences == total
                and min(sequences) == 1
                and max(sequences) == total
            ),
        },
        "deterministic_seeds": seeds,
        "backgrounds": dict(backgrounds),
        "pool": {
            "all_hypotheses_median": _median(cluster_counts),
            "hypothesis_pool_median": _median(pool_counts),
            "all_hypotheses_mean": round(_mean(cluster_counts), 3) if cluster_counts else None,
            "hypothesis_pool_mean": round(_mean(pool_counts), 3) if pool_counts else None,
        },
        "oracle_recall": oracle,
        "selected": selected,
        "rankers": rankers,
        "feature_discrimination_shots": max(
            (item["shots"] for item in feature_discrimination),
            default=0,
        ),
        "feature_discrimination": feature_discrimination,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("=" * 78)
    print("RANKING DATASET V2.9 ANALYSIS")
    print("=" * 78)
    print(f"Session: {summary.get('session')}")
    print(f"Shots: {summary.get('shots')}")
    integrity = summary.get("sequence_integrity", {})
    print(
        "Dataset integrity: "
        f"{integrity.get('unique')}/{summary.get('shots')} unique "
        f"complete={integrity.get('complete')}"
    )
    seeds = summary.get("deterministic_seeds", [])
    if seeds:
        print(f"Deterministic seeds: {seeds}")
    print()

    pool = summary.get("pool", {})
    print("POOL SIZE:")
    print(f"  micro-clusters median : {pool.get('all_hypotheses_median')}")
    print(f"  recall-pool median    : {pool.get('hypothesis_pool_median')}")
    print()

    print("ORACLE RECALL / ACTUAL SELECTED:")
    for radius in (10, 20, 42):
        oracle = summary["oracle_recall"][str(radius)]
        selected = summary["selected"][str(radius)]
        print(
            f"  <= {radius:2d}px : oracle={oracle['count']:3d} "
            f"({oracle['pct']:6.2f}%) | "
            f"selected={selected['count']:3d} ({selected['pct']:6.2f}%)"
        )
    print()

    print("RANKING QUALITY WHEN GT EXISTS:")
    for radius in (20, 42):
        print(f"  <= {radius}px")
        for name in ("baseline", "recall_baseline", "v6_shadow", "v7_shadow"):
            metrics = summary["rankers"][name][str(radius)]
            if metrics["covered"] <= 0:
                continue
            print(
                f"    {name:16s} covered={metrics['covered']:3d} "
                f"median={str(metrics['median_rank']):>6s} "
                f"top1={metrics['top1_pct']:6.2f}% "
                f"top3={metrics['top3_pct']:6.2f}% "
                f"MRR={metrics['mrr']}"
            )
    print()

    print(
        "FEATURE DISCRIMINATION: nearest GT hypothesis vs baseline hard negative"
    )
    print(
        f"  paired shots: {summary.get('feature_discrimination_shots', 0)}"
    )
    for item in summary.get("feature_discrimination", [])[:15]:
        arrow = "GT HIGH" if item["direction"] == "HIGHER_GT" else "GT LOW "
        print(
            f"  {item['feature']:24s} {arrow} "
            f"win={item['direction_win_pct']:5.1f}% "
            f"GTmed={item['positive_median']:7.4f} "
            f"NEGmed={item['negative_median']:7.4f}"
        )

    print()
    print(f"Summary file: {SUMMARY_PATH}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse a captured V2.9 ranking dataset"
    )
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--output", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()

    rows, session = load_session(args.session)
    if not rows:
        print("No V2.9 ranking dataset found.")
        print("Run one labelled F2 training session after installing V2.9.")
        raise SystemExit(1)

    summary = summarise(rows, session)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
