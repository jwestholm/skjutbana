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


def _found(value: Any, radius: float) -> bool:
    number = _finite(value)
    return number is not None and number <= radius


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
    """Select only the newest detector runtime by default."""
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
        return records, None

    return (
        [
            row for row in records
            if str(row.get("runtime_session_id", "")) == latest_session
        ],
        latest_session,
    )


def _nearest(record: dict[str, Any], key: str) -> float | None:
    block = record.get("nearest_candidate_distance_px", {})
    if not isinstance(block, dict):
        return None
    return _finite(block.get(key))


def _evaluation(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("evaluation_funnel", {})
    return value if isinstance(value, dict) else {}


def classify_detector_miss(record: dict[str, Any], match_radius: float) -> str:
    """Classify why the best/ever merged detector did not cover GT."""
    if not isinstance(record.get("ground_truth"), dict):
        return "unlabelled"

    legacy = _nearest(record, "legacy")
    v2_frame = _nearest(record, "v2_frame")
    v2 = _nearest(record, "v2")
    merged = _nearest(record, "merged")

    if _found(merged, match_radius):
        if _found(v2, match_radius):
            if _found(legacy, match_radius):
                if not _found(v2_frame, match_radius):
                    return "found_both_bank_helped"
                return "found_by_both"
            if not _found(v2_frame, match_radius):
                return "recovered_by_candidate_bank"
            return "recovered_by_v2"
        return "legacy_only"

    if _found(v2, match_radius):
        return "v2_lost_in_merge"
    if _found(legacy, match_radius):
        return "legacy_lost_in_merge"

    gt_signal = record.get("gt_signal_max", {})
    if not isinstance(gt_signal, dict):
        gt_signal = {}

    absdiff = _finite(gt_signal.get("absdiff")) or 0.0
    zscore = _finite(gt_signal.get("zscore")) or 0.0
    saliency = _finite(gt_signal.get("saliency")) or 0.0
    margin = _finite(gt_signal.get("saliency_minus_threshold"))

    if absdiff < 1.2 and zscore < 1.05:
        return "weak_or_no_camera_signal"
    if margin is not None and margin >= 0.0:
        return "signal_above_threshold_but_no_candidate"
    if saliency >= 7.0:
        return "strong_gt_signal_but_peak_missing"
    if absdiff >= 4.0 and saliency < 7.0:
        return "saliency_suppressed"
    return "candidate_generation_miss"


def classify_pipeline_loss(record: dict[str, Any], radius: float) -> str:
    """Classify where a candidate is lost between camera frames and AI output."""
    if not isinstance(record.get("ground_truth"), dict):
        return "unlabelled"

    ever = _found(_nearest(record, "merged"), radius)
    evaluation = _evaluation(record)
    if not evaluation:
        return "no_evaluation_telemetry"

    raw = _found(evaluation.get("raw_nearest_px", evaluation.get("raw_closest_dist")), radius)
    filtered = _found(evaluation.get("filter_closest_dist"), radius)
    ranked = _found(evaluation.get("ranked_nearest_px", evaluation.get("ai_topk_closest_dist")), radius)
    selected = _found(evaluation.get("selected_nearest_px", evaluation.get("selected_dist")), radius)

    if not ever:
        return "detector_never_covered_gt"
    if not raw:
        return "candidate_disappeared_before_evaluation"
    if not filtered:
        return "noise_filter_removed_gt"
    if not ranked:
        return "ranking_topk_removed_gt"
    if not selected:
        return "selected_wrong_candidate"
    return "selected_correct"


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    labelled = [
        record for record in records
        if isinstance(record.get("ground_truth"), dict)
    ]

    by_background: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in labelled:
        background = str(record.get("ground_truth", {}).get("background", "unknown"))
        by_background[background].append(record)

    match_radii = [10.0, 20.0, 42.0]
    detector_sources = ("legacy", "v2_frame", "v2", "merged")

    summary: dict[str, Any] = {
        "schema_version": "2.2",
        "records_total": len(records),
        "records_with_ground_truth": len(labelled),
        "records_with_evaluation_funnel": sum(
            1 for record in labelled if bool(_evaluation(record))
        ),
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
        summary["overall"][key] = {}
        for source in detector_sources:
            count = sum(
                1 for record in labelled
                if _found(_nearest(record, source), radius)
            )
            summary["overall"][key][source] = {
                "count": count,
                "pct": _pct(count, len(labelled)),
            }

    summary["overall"]["detector_classification_42px"] = dict(
        Counter(classify_detector_miss(record, 42.0) for record in labelled)
    )
    summary["overall"]["pipeline_classification_42px"] = dict(
        Counter(classify_pipeline_loss(record, 42.0) for record in labelled)
    )

    eval_rows = [(record, _evaluation(record)) for record in labelled if _evaluation(record)]
    evaluation_summary: dict[str, Any] = {
        "shots": len(eval_rows),
    }
    for name, field in (
        ("raw", "raw_nearest_px"),
        ("filtered", "filter_closest_dist"),
        ("ranked", "ranked_nearest_px"),
        ("selected", "selected_nearest_px"),
    ):
        for radius in match_radii:
            count = sum(
                1
                for _record, ev in eval_rows
                if _found(ev.get(field), radius)
            )
            evaluation_summary[f"{name}_within_{int(radius)}px"] = {
                "count": count,
                "pct": _pct(count, len(eval_rows)),
            }
    # Provenance inside the exact F2 evaluation snapshot. This is the key
    # measurement for the V2.1 candidate bank: did a candidate that existed
    # earlier actually survive until the evaluator looked?
    if eval_rows:
        for name, field in (
            ("raw_v1", "raw_v1_nearest_px"),
            ("raw_v2", "raw_v2_nearest_px"),
            ("raw_v2_carried", "raw_v2_bank_carried_nearest_px"),
            ("raw_v2_confirmed", "raw_v2_bank_confirmed_nearest_px"),
        ):
            for radius in match_radii:
                count = sum(
                    1
                    for _record, ev in eval_rows
                    if _found(ev.get(field), radius)
                )
                evaluation_summary[f"{name}_within_{int(radius)}px"] = {
                    "count": count,
                    "pct": _pct(count, len(eval_rows)),
                }

        carried_counts = [
            int(ev.get("raw_v2_bank_carried_count", 0) or 0)
            for _record, ev in eval_rows
        ]
        evaluation_summary["raw_v2_bank_carried_count"] = {
            "mean": (
                round(sum(carried_counts) / len(carried_counts), 3)
                if carried_counts
                else 0.0
            ),
            "max": max(carried_counts) if carried_counts else 0,
            "shots_with_carried_candidates": sum(
                1 for value in carried_counts if value > 0
            ),
        }


        generic_carried_counts = [
            int(ev.get("raw_candidate_bank_carried_count", 0) or 0)
            for _record, ev in eval_rows
        ]
        evaluation_summary["raw_candidate_bank_carried_count"] = {
            "mean": (
                round(sum(generic_carried_counts) / len(generic_carried_counts), 3)
                if generic_carried_counts
                else 0.0
            ),
            "max": max(generic_carried_counts) if generic_carried_counts else 0,
            "shots_with_carried_candidates": sum(
                1 for value in generic_carried_counts if value > 0
            ),
        }
        for radius in match_radii:
            count = sum(
                1 for _record, ev in eval_rows
                if _found(ev.get("raw_candidate_bank_carried_nearest_px"), radius)
            )
            evaluation_summary[f"raw_candidate_bank_carried_within_{int(radius)}px"] = {
                "count": count,
                "pct": _pct(count, len(eval_rows)),
            }

        # Ranking diagnostics only make sense when a GT candidate actually
        # survived into the ranked top-K list.
        ranking_rows = []
        for _record, ev in eval_rows:
            distance = _finite(ev.get("ranking_gt_candidate_distance_px"))
            gt_candidate = ev.get("ranking_gt_candidate")
            selected_candidate = ev.get("ranking_selected_candidate")
            if (
                distance is not None
                and distance <= 42.0
                and isinstance(gt_candidate, dict)
                and isinstance(selected_candidate, dict)
            ):
                ranking_rows.append((ev, gt_candidate, selected_candidate))

        gt_ranks = [
            int(gt.get("rank", 0) or 0)
            for _ev, gt, _selected in ranking_rows
            if int(gt.get("rank", 0) or 0) > 0
        ]
        margins = [
            value
            for ev, _gt, _selected in ranking_rows
            if (value := _finite(ev.get("ranking_score_margin_selected_minus_gt"))) is not None
        ]
        ai_deltas = []
        heuristic_deltas = []
        for _ev, gt, selected in ranking_rows:
            gt_ai = _finite(gt.get("ai_score"))
            selected_ai = _finite(selected.get("ai_score"))
            gt_h = _finite(gt.get("heuristic_score"))
            selected_h = _finite(selected.get("heuristic_score"))
            if gt_ai is not None and selected_ai is not None:
                ai_deltas.append(selected_ai - gt_ai)
            if gt_h is not None and selected_h is not None:
                heuristic_deltas.append(selected_h - gt_h)

        evaluation_summary["ranking"] = {
            "shots_with_gt_in_ranked_42px": len(ranking_rows),
            "gt_rank_median": round(statistics.median(gt_ranks), 3) if gt_ranks else None,
            "gt_rank_mean": round(sum(gt_ranks) / len(gt_ranks), 3) if gt_ranks else None,
            "gt_rank_top1_pct": _pct(sum(1 for value in gt_ranks if value == 1), len(gt_ranks)),
            "gt_rank_top3_pct": _pct(sum(1 for value in gt_ranks if value <= 3), len(gt_ranks)),
            "selected_minus_gt_combined_margin_median": (
                round(statistics.median(margins), 5) if margins else None
            ),
            "selected_minus_gt_ai_score_median": (
                round(statistics.median(ai_deltas), 5) if ai_deltas else None
            ),
            "selected_minus_gt_heuristic_score_median": (
                round(statistics.median(heuristic_deltas), 5) if heuristic_deltas else None
            ),
        }

    summary["overall"]["evaluation_funnel"] = evaluation_summary

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
    gt_margin = [
        value
        for record in labelled
        if (value := _finite(record.get("gt_signal_max", {}).get("saliency_minus_threshold"))) is not None
    ]

    summary["overall"]["gt_signal"] = {
        "absdiff_median": round(statistics.median(gt_abs), 3) if gt_abs else None,
        "zscore_median": round(statistics.median(gt_z), 3) if gt_z else None,
        "saliency_median": round(statistics.median(gt_sal), 3) if gt_sal else None,
        "saliency_minus_threshold_median": (
            round(statistics.median(gt_margin), 3) if gt_margin else None
        ),
    }

    # Explicit counts for the two changes introduced in V2.1.
    bank_recovered = sum(
        1 for record in labelled
        if not _found(_nearest(record, "v2_frame"), 42.0)
        and _found(_nearest(record, "v2"), 42.0)
    )
    v2_lost_merge = sum(
        1 for record in labelled
        if _found(_nearest(record, "v2"), 42.0)
        and not _found(_nearest(record, "merged"), 42.0)
    )
    legacy_lost_merge = sum(
        1 for record in labelled
        if _found(_nearest(record, "legacy"), 42.0)
        and not _found(_nearest(record, "merged"), 42.0)
    )
    summary["overall"]["v22_changes"] = {
        "candidate_bank_recovered_42px": bank_recovered,
        "v2_lost_during_merge_42px": v2_lost_merge,
        "legacy_lost_during_merge_42px": legacy_lost_merge,
    }

    for background, rows in sorted(by_background.items()):
        entry: dict[str, Any] = {"shots": len(rows)}
        for radius in match_radii:
            key = f"within_{int(radius)}px"
            entry[key] = {
                source: _pct(
                    sum(1 for record in rows if _found(_nearest(record, source), radius)),
                    len(rows),
                )
                for source in detector_sources
            }
        entry["pipeline_classification_42px"] = dict(
            Counter(classify_pipeline_loss(record, 42.0) for record in rows)
        )
        summary["by_background"][background] = entry

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print()
    print("=" * 88)
    print("DETECTOR V2.2 ANALYSIS")
    print("=" * 88)
    print(f"Diagnostics: {summary['records_total']}")
    print(f"With synthetic ground truth: {summary['records_with_ground_truth']}")
    print(f"With evaluation funnel: {summary['records_with_evaluation_funnel']}")
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
        print(f"BEST/EVER candidate recall within {radius}px:")
        for source in ("legacy", "v2_frame", "v2", "merged"):
            data = block.get(source, {})
            label = {
                "legacy": "legacy",
                "v2_frame": "v2 frame",
                "v2": "v2 bank",
                "merged": "merged",
            }[source]
            print(
                f"  {label:10s}: {data.get('count', 0):5d} "
                f"({data.get('pct', 0.0):6.2f}%)"
            )

    changes = overall.get("v22_changes", {})
    print()
    print("V2.2 candidate preservation (42px):")
    print(
        "  recovered by candidate bank : "
        f"{changes.get('candidate_bank_recovered_42px', 0)}"
    )
    print(
        "  V2 candidates lost in merge : "
        f"{changes.get('v2_lost_during_merge_42px', 0)}"
    )
    print(
        "  V1 candidates lost in merge : "
        f"{changes.get('legacy_lost_during_merge_42px', 0)}"
    )

    ev = overall.get("evaluation_funnel", {})
    if ev.get("shots", 0):
        print()
        print("ACTUAL F2 EVALUATION funnel within 42px:")
        for name in ("raw", "filtered", "ranked", "selected"):
            data = ev.get(f"{name}_within_42px", {})
            print(
                f"  {name:10s}: {data.get('count', 0):5d} "
                f"({data.get('pct', 0.0):6.2f}%)"
            )

        print()
        print("Detector provenance in ACTUAL F2 raw snapshot (42px):")
        for name, label in (
            ("raw_v1", "V1 present"),
            ("raw_v2", "V2 present"),
            ("raw_v2_confirmed", "V2 confirmed"),
            ("raw_v2_carried", "V2 carried"),
        ):
            data = ev.get(f"{name}_within_42px", {})
            print(
                f"  {label:14s}: {data.get('count', 0):5d} "
                f"({data.get('pct', 0.0):6.2f}%)"
            )
        carried = ev.get("raw_v2_bank_carried_count", {})
        if isinstance(carried, dict):
            print(
                "  V2 carried candidates / snapshot: "
                f"mean={carried.get('mean', 0.0)} "
                f"max={carried.get('max', 0)} "
                f"shots={carried.get('shots_with_carried_candidates', 0)}"
            )
        generic_carried = ev.get("raw_candidate_bank_carried_count", {})
        if isinstance(generic_carried, dict):
            recovered = ev.get("raw_candidate_bank_carried_within_42px", {})
            print(
                "  hybrid-bank carried / snapshot: "
                f"mean={generic_carried.get('mean', 0.0)} "
                f"max={generic_carried.get('max', 0)} "
                f"shots={generic_carried.get('shots_with_carried_candidates', 0)} "
                f"GT@42px={recovered.get('count', 0)} ({recovered.get('pct', 0.0):.2f}%)"
            )

        ranking = ev.get("ranking", {})
        if isinstance(ranking, dict) and ranking.get("shots_with_gt_in_ranked_42px", 0):
            print()
            print("Ranking quality when GT survived into ranked list (42px):")
            print(f"  shots                 : {ranking.get('shots_with_gt_in_ranked_42px')}")
            print(f"  GT median rank        : {ranking.get('gt_rank_median')}")
            print(f"  GT mean rank          : {ranking.get('gt_rank_mean')}")
            print(f"  GT rank=1             : {ranking.get('gt_rank_top1_pct')}%")
            print(f"  GT rank<=3            : {ranking.get('gt_rank_top3_pct')}%")
            print(
                "  selected-GT score med : "
                f"{ranking.get('selected_minus_gt_combined_margin_median')}"
            )
            print(
                "  selected-GT AI med    : "
                f"{ranking.get('selected_minus_gt_ai_score_median')}"
            )
            print(
                "  selected-GT heuristic : "
                f"{ranking.get('selected_minus_gt_heuristic_score_median')}"
            )

        print()
        print("Where GT was lost (42px):")
        classes = overall.get("pipeline_classification_42px", {})
        for name, count in sorted(classes.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {name:38s} {count:5d}")

    gt_signal = overall.get("gt_signal", {})
    print()
    print("Median signal around synthetic ground truth:")
    print(f"  absdiff                    : {gt_signal.get('absdiff_median')}")
    print(f"  z-score                    : {gt_signal.get('zscore_median')}")
    print(f"  saliency                   : {gt_signal.get('saliency_median')}")
    print(f"  saliency - frame threshold : {gt_signal.get('saliency_minus_threshold_median')}")

    print()
    print("Detector miss / recovery classification (42px):")
    classes = overall.get("detector_classification_42px", {})
    for name, count in sorted(classes.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {name:38s} {count:5d}")

    by_background = summary.get("by_background", {})
    if by_background:
        print()
        print("By background (merged BEST/EVER recall within 42px):")
        for background, data in by_background.items():
            block = data.get("within_42px", {})
            print(
                f"  {background:16s} shots={data.get('shots', 0):5d} "
                f"legacy={block.get('legacy', 0.0):6.2f}% "
                f"v2frame={block.get('v2_frame', 0.0):6.2f}% "
                f"v2bank={block.get('v2', 0.0):6.2f}% "
                f"merged={block.get('merged', 0.0):6.2f}%"
            )

    print("=" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse machine-readable Detector V2/V2.1/V2.2 shot diagnostics."
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
        help="Analyse all historical runtime sessions instead of only the latest.",
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
