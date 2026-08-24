from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

SESSION_ROOT = Path("content/ai/detector_v28/sessions")
JSONL_PATH = Path("content/ai/detector_v28/shot_diagnostics.jsonl")
SUMMARY_PATH = Path("content/ai/detector_v28/latest_v28_summary.json")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def _pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def _median(values: Sequence[float]) -> float | None:
    data = [float(v) for v in values if _finite(v) is not None]
    return statistics.median(data) if data else None


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _latest_session_dir(session: str | None = None) -> Path | None:
    if session:
        candidate = SESSION_ROOT / session
        return candidate if candidate.is_dir() else None
    if not SESSION_ROOT.is_dir():
        return None
    dirs = [p for p in SESSION_ROOT.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def load_records(session: str | None = None) -> tuple[list[dict[str, Any]], str | None, str]:
    folder = _latest_session_dir(session)
    if folder is not None:
        rows = []
        for path in sorted(folder.glob("shot_*.json")):
            row = _read_json(path)
            if isinstance(row, dict) and isinstance(row.get("v28_hypotheses"), dict):
                rows.append(row)
        if rows:
            return rows, folder.name, "atomic per-shot files"

    rows = []
    if JSONL_PATH.exists():
        with JSONL_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict) and isinstance(row.get("v28_hypotheses"), dict):
                    rows.append(row)
    if not rows:
        return [], None, "none"
    latest = session
    if latest is None:
        for row in reversed(rows):
            if row.get("runtime_session_id"):
                latest = str(row["runtime_session_id"])
                break
    if latest:
        rows = [row for row in rows if str(row.get("runtime_session_id", "")) == latest]
    return rows, latest, "JSONL fallback"


def _block(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("v28_hypotheses")
    return value if isinstance(value, dict) else {}


def _coverage(block: dict[str, Any], name: str, radius: int) -> bool:
    coverage = block.get("coverage", {})
    source = coverage.get(name, {}) if isinstance(coverage, dict) else {}
    return bool(isinstance(source, dict) and source.get(f"within_{radius}"))


def _rank(block: dict[str, Any], name: str, radius: int) -> int | None:
    ranks = block.get("ranks", {})
    value = ranks.get(f"{name}_{radius}") if isinstance(ranks, dict) else None
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _selected(block: dict[str, Any], radius: int) -> bool:
    selected = block.get("selected", {})
    return bool(isinstance(selected, dict) and selected.get(f"within_{radius}"))


def summarise(records: list[dict[str, Any]], session_id: str | None, source: str) -> dict[str, Any]:
    blocks = [_block(row) for row in records]
    blocks = [block for block in blocks if block]
    total = len(blocks)

    stat_rows = [block.get("stats", {}) if isinstance(block.get("stats"), dict) else {} for block in blocks]
    reduction = {
        "input_median": _median([int(row.get("input_count", 0) or 0) for row in stat_rows]),
        "cluster_median": _median([int(row.get("cluster_count", 0) or 0) for row in stat_rows]),
        "core_median": _median([int(row.get("core_pool_count", 0) or 0) for row in stat_rows]),
        "pool_median": _median([int(row.get("pool_count", 0) or 0) for row in stat_rows]),
        "dropped_median": _median([int(row.get("pool_dropped", 0) or 0) for row in stat_rows]),
    }
    pool_modes = Counter(str(row.get("pool_mode", "unknown")) for row in stat_rows)

    oracle: dict[str, Any] = {}
    for radius in (10, 20, 42):
        item = {}
        for name in ("filtered_input", "all_hypotheses", "core_pool", "hypothesis_pool"):
            count = sum(_coverage(block, name, radius) for block in blocks)
            item[name] = {"count": count, "pct": _pct(count, total)}
        oracle[str(radius)] = item

    ranking: dict[str, Any] = {}
    for radius in (10, 20, 42):
        item = {}
        for name in ("baseline", "recall_baseline", "v6", "actual"):
            values = [v for v in (_rank(block, name, radius) for block in blocks) if v is not None]
            item[name] = {
                "covered": len(values),
                "median_rank": _median(values),
                "top1": sum(v == 1 for v in values),
                "top3": sum(v <= 3 for v in values),
            }
        ranking[str(radius)] = item

    selected = {}
    for radius in (10, 20, 42):
        count = sum(_selected(block, radius) for block in blocks)
        selected[str(radius)] = {"count": count, "pct": _pct(count, total)}

    loss = Counter()
    for block in blocks:
        if not _coverage(block, "filtered_input", 42):
            loss["filtered_input_no_gt"] += 1
        elif not _coverage(block, "all_hypotheses", 42):
            loss["clustering_lost_gt"] += 1
        elif not _coverage(block, "hypothesis_pool", 42):
            loss["recall_pool_lost_gt"] += 1
        elif not _selected(block, 42):
            loss["ranking_selected_wrong"] += 1
        else:
            loss["selected_correct"] += 1

    core_lost = sum(
        _coverage(block, "all_hypotheses", 42) and not _coverage(block, "core_pool", 42)
        for block in blocks
    )

    training = Counter()
    skipped = Counter()
    gate_open = authority = 0
    for block in blocks:
        train = block.get("training", {})
        if isinstance(train, dict):
            if train.get("trained"):
                training[str(train.get("label_kind", "unknown"))] += 1
            else:
                skipped[str(train.get("reason", "unknown"))] += 1
        gate = block.get("gate_before_training", {})
        auth = block.get("authority_for_current_shot", {})
        gate_open += int(isinstance(gate, dict) and bool(gate.get("open")))
        authority += int(isinstance(auth, dict) and auth.get("reason") == "confidence_ok")

    last_model = blocks[-1].get("model", {}) if blocks else {}
    commits = sorted({str(row.get("git_commit")) for row in records if row.get("git_commit")})
    seeds = sorted({
        int(row.get("ground_truth", {}).get("benchmark_seed"))
        for row in records
        if isinstance(row.get("ground_truth"), dict)
        and row.get("ground_truth", {}).get("benchmark_seed") is not None
    })

    return {
        "schema_version": "2.8",
        "runtime_session_id": session_id,
        "telemetry_source": source,
        "shots": total,
        "sequences": [int(row.get("sequence", 0) or 0) for row in records],
        "git_commits": commits,
        "benchmark_seeds": seeds,
        "pool_reduction": reduction,
        "pool_modes": dict(pool_modes),
        "oracle": oracle,
        "core_pool_42px_losses": core_lost,
        "ranking": ranking,
        "selected": selected,
        "loss_42px": dict(loss),
        "training": {
            "strict": training.get("strict", 0),
            "soft": training.get("soft", 0),
            "skipped": dict(skipped),
            "gate_open_shots": gate_open,
            "v6_authority_shots": authority,
        },
        "last_model": last_model,
    }


def print_summary(summary: dict[str, Any]) -> None:
    total = int(summary.get("shots", 0) or 0)
    print("=" * 80)
    print("DETECTOR / HYPOTHESIS V2.8 ANALYSIS")
    print("=" * 80)
    print(f"Shots with V2.8 telemetry: {total}")
    print(f"Runtime session: {summary.get('runtime_session_id')}")
    print(f"Telemetry source: {summary.get('telemetry_source')}")
    seq = summary.get("sequences", [])
    if seq:
        unique = len(set(seq))
        print(f"Diagnostic integrity: {unique}/{max(seq)} unique sequences")
    if summary.get("benchmark_seeds"):
        print(f"Deterministic seeds: {summary.get('benchmark_seeds')}")

    r = summary.get("pool_reduction", {})
    print("\nPOOL REDUCTION (median per shot):")
    print(f"  filtered observations : {_fmt(r.get('input_median'), 1)}")
    print(f"  micro-clusters        : {_fmt(r.get('cluster_median'), 1)}")
    print(f"  V2.7 core             : {_fmt(r.get('core_median'), 1)}")
    print(f"  V2.8 recall pool      : {_fmt(r.get('pool_median'), 1)}")
    print(f"  clusters dropped      : {_fmt(r.get('dropped_median'), 1)}")
    print(f"  pool modes            : {summary.get('pool_modes', {})}")

    print("\nORACLE RECALL:")
    print(" radius | filtered | clusters | old core | V2.8 recall pool")
    for radius in (10, 20, 42):
        row = summary.get("oracle", {}).get(str(radius), {})
        values = []
        for name in ("filtered_input", "all_hypotheses", "core_pool", "hypothesis_pool"):
            v = row.get(name, {})
            values.append(f"{v.get('count',0):3d} ({v.get('pct',0.0):5.1f}%)")
        print(f" <= {radius:2d}px | " + " | ".join(values))
    print(f"  GT lost by old 120-core but available as cluster: {summary.get('core_pool_42px_losses', 0)}")

    print("\nRANKING WHEN GT EXISTS:")
    for radius in (20, 42):
        print(f"  <= {radius}px")
        row = summary.get("ranking", {}).get(str(radius), {})
        for name, label in (
            ("baseline", "CORE-FIRST"),
            ("recall_baseline", "FULL BASE"),
            ("v6", "V6 shadow"),
            ("actual", "ACTUAL"),
        ):
            v = row.get(name, {})
            covered = int(v.get("covered", 0) or 0)
            top1 = int(v.get("top1", 0) or 0)
            top3 = int(v.get("top3", 0) or 0)
            print(
                f"    {label:10s}: covered={covered:3d} median={_fmt(v.get('median_rank'),1):>6s} "
                f"top1={_pct(top1,covered):5.1f}% top3={_pct(top3,covered):5.1f}%"
            )

    print("\nFINAL SELECTED:")
    for radius in (10, 20, 42):
        v = summary.get("selected", {}).get(str(radius), {})
        print(f"  <= {radius:2d}px: {v.get('count',0):3d}/{total} = {v.get('pct',0.0):5.1f}%")

    print("\nWHERE 42px ORACLE WAS LOST:")
    for key, value in summary.get("loss_42px", {}).items():
        print(f"  {key:28s}: {value}")

    t = summary.get("training", {})
    print("\nRANKER V6 / V2.8 FRESH MODEL:")
    print(f"  trained strict / soft : {t.get('strict',0)} / {t.get('soft',0)}")
    print(f"  gate open shots       : {t.get('gate_open_shots',0)}")
    print(f"  V6 authority shots    : {t.get('v6_authority_shots',0)}")
    if t.get("skipped"):
        print(f"  skipped               : {t.get('skipped')}")
    model = summary.get("last_model", {})
    if isinstance(model, dict):
        print(f"  pair updates          : {model.get('pair_updates',0)}")
        print(f"  last loss             : {model.get('last_loss')}")
        print(f"  gate now              : {model.get('gate')}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze V2.8 recall-pool/ranker telemetry")
    parser.add_argument("--session", default=None)
    parser.add_argument("--output", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()

    records, session, source = load_records(args.session)
    if not records:
        print("No V2.8 diagnostics found.")
        print("Run: python3 -m automation.detector_v28_verify")
        raise SystemExit(1)
    summary = summarise(records, session, source)
    print_summary(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Machine-readable V2.8 summary: {args.output}")


if __name__ == "__main__":
    main()
