from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PATH = Path("content/ai/detector_v2/shot_diagnostics.jsonl")
DEFAULT_SUMMARY = Path("content/ai/detector_v2/latest_summary.json")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def _pct(count: int, total: int) -> float:
    return round(100.0 * count / total, 3) if total else 0.0


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if not path.exists():
        return records

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
                records.append(value)

    return records


def select_records(
    records: list[dict[str, Any]],
    *,
    session_id: str | None = None,
    include_all: bool = False,
) -> tuple[list[dict[str, Any]], str | None]:
    """Select one detector runtime session by default.

    The diagnostics file is append-only. Without this filter, an A/B run after
    a config/code change would silently mix old and new detector versions.
    """
    if include_all or not records:
        return records, None

    if session_id:
        selected = [
            row for row in records
            if str(row.get("runtime_session_id", "")) == session_id
        ]
        return selected, session_id

    latest_session = None
    for row in reversed(records):
        value = row.get("runtime_session_id")
        if value:
            latest_session = str(value)
            break

    if latest_session is None:
        # Backward compatibility with diagnostics created before session IDs.
        return records, None

    return (
        [
            row for row in records
            if str(row.get("runtime_session_id", "")) == latest_session
        ],
        latest_session,
    )


def classify(record: dict[str, Any], match_radius: float) -> str:
    gt = record.get("ground_truth")
    if not isinstance(gt, dict):
        return "unlabelled"

    nearest = record.get("nearest_candidate_distance_px", {})
    if not isinstance(nearest, dict):
        nearest = {}

    legacy = _finite(nearest.get("legacy"))
    v2 = _finite(nearest.get("v2"))
    merged = _finite(nearest.get("merged"))

    if merged is not None and merged <= match_radius:
        if v2 is not None and v2 <= match_radius:
            if legacy is not None and legacy <= match_radius:
                return "found_by_both"
            return "recovered_by_v2"
        return "legacy_only"

    gt_signal = record.get("gt_signal_max", {})
    if not isinstance(gt_signal, dict):
        gt_signal = {}

    absdiff = _finite(gt_signal.get("absdiff")) or 0.0
    zscore = _finite(gt_signal.get("zscore")) or 0.0
    saliency = _finite(gt_signal.get("saliency")) or 0.0

    if absdiff < 1.2 and zscore < 1.05:
        return "weak_or_no_camera_signal"

    if saliency >= 7.5:
        return "strong_gt_signal_but_peak_missing"

    if absdiff >= 4.0 and saliency < 7.5:
        return "saliency_suppressed"

    return "candidate_generation_miss"


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    labelled = [
        record
        for record in records
        if isinstance(record.get("ground_truth"), dict)
    ]

    by_background: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in labelled:
        background = str(record.get("ground_truth", {}).get("background", "unknown"))
        by_background[background].append(record)

    def dist_found(record: dict[str, Any], key: str, radius: float) -> bool:
        nearest = record.get("nearest_candidate_distance_px", {})
        if not isinstance(nearest, dict):
            return False
        value = _finite(nearest.get(key))
        return value is not None and value <= radius

    match_radii = [10.0, 20.0, 42.0]

    summary: dict[str, Any] = {
        "schema_version": "2.0",
        "records_total": len(records),
        "records_with_ground_truth": len(labelled),
        "runtime_session_ids": sorted({
            str(record.get("runtime_session_id"))
            for record in records
            if record.get("runtime_session_id")
        }),
        "git_commits": sorted({
            str(record.get("git_commit"))
            for record in records
            if record.get("git_commit")
        }),
        "overall": {},
        "by_background": {},
    }

    for radius in match_radii:
        key = f"within_{int(radius)}px"
        summary["overall"][key] = {
            source: {
                "count": sum(1 for r in labelled if dist_found(r, source, radius)),
                "pct": _pct(
                    sum(1 for r in labelled if dist_found(r, source, radius)),
                    len(labelled),
                ),
            }
            for source in ("legacy", "v2", "merged")
        }

    classifications = Counter(classify(record, 42.0) for record in labelled)
    summary["overall"]["classification"] = dict(classifications)

    gt_abs = [
        value
        for record in labelled
        if (value := _finite(record.get("gt_signal_max", {}).get("absdiff"))) is not None
    ]
    gt_z = [
        value
        for record in labelled
        if (value := _finite(record.get("gt_signal_max", {}).get("zscore"))) is not None
    ]
    gt_sal = [
        value
        for record in labelled
        if (value := _finite(record.get("gt_signal_max", {}).get("saliency"))) is not None
    ]

    summary["overall"]["gt_signal"] = {
        "absdiff_median": round(statistics.median(gt_abs), 3) if gt_abs else None,
        "zscore_median": round(statistics.median(gt_z), 3) if gt_z else None,
        "saliency_median": round(statistics.median(gt_sal), 3) if gt_sal else None,
    }

    for background, rows in sorted(by_background.items()):
        entry: dict[str, Any] = {"shots": len(rows)}
        for radius in match_radii:
            key = f"within_{int(radius)}px"
            entry[key] = {
                source: _pct(
                    sum(1 for r in rows if dist_found(r, source, radius)),
                    len(rows),
                )
                for source in ("legacy", "v2", "merged")
            }
        entry["classification"] = dict(
            Counter(classify(record, 42.0) for record in rows)
        )
        summary["by_background"][background] = entry

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print("DETECTOR V2 ANALYSIS")
    print("=" * 78)
    print(f"Diagnostics: {summary['records_total']}")
    print(f"With synthetic ground truth: {summary['records_with_ground_truth']}")
    sessions = summary.get("runtime_session_ids", [])
    commits = summary.get("git_commits", [])
    if sessions:
        print(f"Detector runtime session: {', '.join(sessions)}")
    if commits:
        print(f"Git commit(s): {', '.join(commits)}")

    overall = summary.get("overall", {})

    for radius in (10, 20, 42):
        block = overall.get(f"within_{radius}px", {})
        print()
        print(f"Candidate recall within {radius}px:")
        for source in ("legacy", "v2", "merged"):
            data = block.get(source, {})
            print(
                f"  {source:7s}: "
                f"{data.get('count', 0):5d} "
                f"({data.get('pct', 0.0):6.2f}%)"
            )

    gt_signal = overall.get("gt_signal", {})
    print()
    print("Median signal exactly around synthetic ground truth:")
    print(f"  absdiff : {gt_signal.get('absdiff_median')}")
    print(f"  z-score : {gt_signal.get('zscore_median')}")
    print(f"  saliency: {gt_signal.get('saliency_median')}")

    print()
    print("Failure / recovery classification (42px):")
    classes = overall.get("classification", {})
    for name, count in sorted(classes.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {name:32s} {count:5d}")

    by_background = summary.get("by_background", {})
    if by_background:
        print()
        print("By background (merged candidate recall within 42px):")
        for background, data in by_background.items():
            merged = data.get("within_42px", {}).get("merged", 0.0)
            legacy = data.get("within_42px", {}).get("legacy", 0.0)
            v2 = data.get("within_42px", {}).get("v2", 0.0)
            print(
                f"  {background:16s} shots={data.get('shots', 0):5d} "
                f"legacy={legacy:6.2f}% v2={v2:6.2f}% merged={merged:6.2f}%"
            )

    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse machine-readable Detector V2 shot diagnostics."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_PATH),
        help="Path to shot_diagnostics.jsonl",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_SUMMARY),
        help="JSON summary output path",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyse all historical Detector V2 runtime sessions instead of only the latest.",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Analyse one explicit runtime_session_id.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    all_records = load_records(path)

    if not all_records:
        print(f"No Detector V2 diagnostics found in: {path}")
        return

    records, selected_session = select_records(
        all_records,
        session_id=args.session,
        include_all=bool(args.all),
    )

    if not records:
        print(f"No diagnostics matched session: {args.session}")
        return

    if selected_session:
        print(f"Analysing latest detector runtime session: {selected_session}")
    elif args.all:
        print("Analysing ALL detector runtime sessions.")

    summary = summarise(records)
    print_summary(summary)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Machine-readable summary written to: {output}")


if __name__ == "__main__":
    main()
