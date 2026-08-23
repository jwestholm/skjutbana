from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any, Sequence

DEFAULT_PATH = Path("content/ai/detector_v2/shot_diagnostics.jsonl")
DEFAULT_SUMMARY = Path("content/ai/detector_v2/latest_v27_summary.json")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def _pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def _median(values: Sequence[float]) -> float | None:
    data = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.median(data) if data else None


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except Exception:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def select_latest_v27(
    records: list[dict[str, Any]],
    session_id: str | None,
) -> tuple[list[dict[str, Any]], str | None]:
    candidates = [
        row for row in records
        if isinstance(row.get("evaluation_funnel"), dict)
        and isinstance(row.get("evaluation_funnel", {}).get("v27_hypotheses"), dict)
    ]
    if session_id:
        return [
            row for row in candidates
            if str(row.get("runtime_session_id", "")) == session_id
        ], session_id
    latest = None
    for row in reversed(candidates):
        if row.get("runtime_session_id"):
            latest = str(row.get("runtime_session_id"))
            break
    if latest is None:
        return candidates, None
    return [row for row in candidates if str(row.get("runtime_session_id", "")) == latest], latest


def _v27(row: dict[str, Any]) -> dict[str, Any]:
    funnel = row.get("evaluation_funnel", {})
    if not isinstance(funnel, dict):
        return {}
    value = funnel.get("v27_hypotheses", {})
    return value if isinstance(value, dict) else {}


def _coverage_flag(block: dict[str, Any], name: str, radius: int) -> bool:
    coverage = block.get("coverage", {})
    if not isinstance(coverage, dict):
        return False
    source = coverage.get(name, {})
    return bool(isinstance(source, dict) and source.get(f"within_{radius}"))


def _rank_value(block: dict[str, Any], name: str, radius: int) -> int | None:
    ranks = block.get("ranks", {})
    value = ranks.get(f"{name}_{radius}") if isinstance(ranks, dict) else None
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _selected_flag(block: dict[str, Any], radius: int) -> bool:
    selected = block.get("selected", {})
    return bool(isinstance(selected, dict) and selected.get(f"within_{radius}"))


def summarise(records: list[dict[str, Any]], session_id: str | None) -> dict[str, Any]:
    blocks = [_v27(row) for row in records]
    blocks = [block for block in blocks if block]
    total = len(blocks)

    counts = []
    clusters = []
    pools = []
    reductions = []
    for block in blocks:
        stats = block.get("stats", {})
        if not isinstance(stats, dict):
            stats = {}
        counts.append(int(stats.get("input_count", 0) or 0))
        clusters.append(int(stats.get("cluster_count", 0) or 0))
        pools.append(int(stats.get("pool_count", 0) or 0))
        ratio = _finite(stats.get("reduction_ratio"))
        if ratio is not None:
            reductions.append(ratio)

    oracle: dict[str, Any] = {}
    for radius in (10, 20, 42):
        row = {}
        for name in ("filtered_input", "all_hypotheses", "hypothesis_pool"):
            count = sum(_coverage_flag(block, name, radius) for block in blocks)
            row[name] = {"count": count, "pct": _pct(count, total)}
        oracle[str(radius)] = row

    ranking: dict[str, Any] = {}
    for radius in (10, 20, 42):
        item: dict[str, Any] = {}
        for name in ("baseline", "v6", "actual"):
            rank_values = [
                value for value in (_rank_value(block, name, radius) for block in blocks)
                if value is not None
            ]
            item[name] = {
                "covered": len(rank_values),
                "median_rank": _median(rank_values),
                "top1": sum(value == 1 for value in rank_values),
                "top3": sum(value <= 3 for value in rank_values),
            }
        ranking[str(radius)] = item

    selected = {}
    for radius in (10, 20, 42):
        count = sum(_selected_flag(block, radius) for block in blocks)
        selected[str(radius)] = {"count": count, "pct": _pct(count, total)}

    # Directly quantify where V2.7 loses the detector oracle.
    loss = Counter()
    for block in blocks:
        input42 = _coverage_flag(block, "filtered_input", 42)
        cluster42 = _coverage_flag(block, "all_hypotheses", 42)
        pool42 = _coverage_flag(block, "hypothesis_pool", 42)
        selected42 = _selected_flag(block, 42)
        if not input42:
            loss["filtered_input_no_gt"] += 1
        elif not cluster42:
            loss["clustering_lost_gt"] += 1
        elif not pool42:
            loss["spatial_pool_lost_gt"] += 1
        elif not selected42:
            loss["ranking_selected_wrong"] += 1
        else:
            loss["selected_correct"] += 1

    gate_open = 0
    authority_used = 0
    trained_strict = 0
    trained_soft = 0
    skipped = Counter()
    for block in blocks:
        gate = block.get("gate_before_training", {})
        authority = block.get("authority_for_current_shot", {})
        training = block.get("training", {})
        if isinstance(gate, dict) and bool(gate.get("open")):
            gate_open += 1
        if isinstance(authority, dict) and authority.get("reason") == "confidence_ok":
            authority_used += 1
        if isinstance(training, dict):
            if bool(training.get("trained")):
                if training.get("label_kind") == "strict":
                    trained_strict += 1
                elif training.get("label_kind") == "soft":
                    trained_soft += 1
            else:
                skipped[str(training.get("reason", "unknown"))] += 1

    last_model = blocks[-1].get("model", {}) if blocks else {}
    commits = sorted({str(row.get("git_commit")) for row in records if row.get("git_commit")})
    seeds = sorted({
        int(row.get("ground_truth", {}).get("benchmark_seed"))
        for row in records
        if isinstance(row.get("ground_truth"), dict)
        and row.get("ground_truth", {}).get("benchmark_seed") is not None
    })

    return {
        "schema_version": "2.7",
        "runtime_session_id": session_id,
        "shots": total,
        "git_commits": commits,
        "benchmark_seeds": seeds,
        "pool_reduction": {
            "input_median": _median(counts),
            "cluster_median": _median(clusters),
            "pool_median": _median(pools),
            "reduction_ratio_median": _median(reductions),
        },
        "oracle": oracle,
        "ranking": ranking,
        "selected": selected,
        "loss_42px": dict(loss),
        "training": {
            "strict": trained_strict,
            "soft": trained_soft,
            "skipped": dict(skipped),
            "gate_open_shots": gate_open,
            "v6_authority_shots": authority_used,
        },
        "last_model": last_model,
    }


def print_summary(summary: dict[str, Any]) -> None:
    total = int(summary.get("shots", 0) or 0)
    print("=" * 76)
    print("DETECTOR / HYPOTHESIS V2.7 ANALYSIS")
    print("=" * 76)
    print(f"Shots with V2.7 telemetry: {total}")
    print(f"Runtime session: {summary.get('runtime_session_id')}")
    if summary.get("git_commits"):
        print("Git commit(s): " + ", ".join(summary["git_commits"]))
    seeds = summary.get("benchmark_seeds", [])
    if seeds:
        print(f"Deterministic seeds: {seeds}")

    reduction = summary.get("pool_reduction", {})
    print("\nPOOL REDUCTION (median per shot):")
    print(f"  filtered observations : {_fmt(reduction.get('input_median'), 1)}")
    print(f"  micro-clusters        : {_fmt(reduction.get('cluster_median'), 1)}")
    print(f"  final hypotheses      : {_fmt(reduction.get('pool_median'), 1)}")
    ratio = reduction.get("reduction_ratio_median")
    print(f"  kept ratio            : {_fmt(100.0 * ratio if ratio is not None else None, 1)}%")

    print("\nORACLE RECALL — does the correct neighbourhood survive consolidation?")
    print(" radius | filtered input | all clusters | final hypothesis pool")
    for radius in (10, 20, 42):
        row = summary.get("oracle", {}).get(str(radius), {})
        parts = []
        for name in ("filtered_input", "all_hypotheses", "hypothesis_pool"):
            value = row.get(name, {})
            parts.append(f"{value.get('count', 0):4d} ({value.get('pct', 0.0):6.2f}%)")
        print(f" <= {radius:2d}px | {parts[0]} | {parts[1]} | {parts[2]}")

    print("\nRANKING when GT exists at each radius:")
    for radius in (20, 42):
        row = summary.get("ranking", {}).get(str(radius), {})
        print(f"  <= {radius}px")
        for name, label in (("baseline", "BASE"), ("v6", "V6 shadow"), ("actual", "ACTUAL")):
            value = row.get(name, {})
            covered = int(value.get("covered", 0) or 0)
            top1 = int(value.get("top1", 0) or 0)
            top3 = int(value.get("top3", 0) or 0)
            print(
                f"    {label:9s}: covered={covered:4d} median-rank={_fmt(value.get('median_rank'), 1):>6s} "
                f"top1={_pct(top1, covered):6.2f}% top3={_pct(top3, covered):6.2f}%"
            )

    print("\nFINAL SELECTED:")
    for radius in (10, 20, 42):
        value = summary.get("selected", {}).get(str(radius), {})
        print(f"  <= {radius:2d}px: {value.get('count', 0):4d}/{total} = {value.get('pct', 0.0):6.2f}%")

    print("\nWHERE THE 42px ORACLE WAS LOST:")
    for key, value in summary.get("loss_42px", {}).items():
        print(f"  {key:28s}: {value}")

    training = summary.get("training", {})
    print("\nRANKER V6:")
    print(f"  trained strict / soft : {training.get('strict', 0)} / {training.get('soft', 0)}")
    print(f"  gate open shots       : {training.get('gate_open_shots', 0)}")
    print(f"  V6 authority shots    : {training.get('v6_authority_shots', 0)}")
    if training.get("skipped"):
        print(f"  skipped               : {training.get('skipped')}")
    model = summary.get("last_model", {})
    if isinstance(model, dict):
        print(f"  pair updates          : {model.get('pair_updates', 0)}")
        print(f"  last loss             : {model.get('last_loss')}")
        print(f"  gate now              : {model.get('gate')}")
        strongest = model.get("strongest_weights", [])
        if strongest:
            print(f"  strongest weights     : {strongest}")
    print("=" * 76)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze V2.7 hypothesis/ranker telemetry")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    records = load_records(args.path)
    selected, session = select_latest_v27(records, args.session)
    if not selected:
        print("No V2.7 diagnostics found. Run a new training session after installing V2.7.")
        raise SystemExit(1)

    summary = summarise(selected, session)
    print_summary(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Machine-readable V2.7 summary written to: {args.output}")


if __name__ == "__main__":
    main()
