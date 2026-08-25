from __future__ import annotations

import argparse
from pathlib import Path

from src.engine.offline.replay import ReplaySettings, benchmark_cases, write_benchmark
from src.engine.offline.shot_case import read_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "V2.12 hardware-free replay: run the CURRENT LIVE HitScanner V1->V2 "
            "hybrid and/or the new direct-image temporal physical evidence source "
            "on saved shots."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest from offline_archive_inspect")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Archive root used to resolve relative image paths",
    )
    parser.add_argument("--max-shots", type=int, default=None, help="Replay at most N shots")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("content/ai/offline/v212/replay_benchmark.json"),
        help="Benchmark JSON output",
    )
    parser.add_argument(
        "--no-detector",
        "--no-v2",
        dest="no_detector",
        action="store_true",
        help="Skip current live V1->V2 detector; direct-image overlay only (--no-v2 is retained as an alias)",
    )
    parser.add_argument("--no-overlay", action="store_true", help="Skip V2.12 overlay; current detector only")
    parser.add_argument("--candidate-limit", type=int, default=500)
    parser.add_argument("--union-limit", type=int, default=700)
    parser.add_argument("--merge-radius", type=float, default=5.0)
    parser.add_argument(
        "--include-candidates",
        action="store_true",
        help="Store every candidate in output JSON (large files on big archives)",
    )
    return parser


def _format_recall(summary: dict) -> None:
    recall = summary.get("recall_percent") or {}
    if not recall:
        print("No labelled shots; candidate counts are available but recall cannot be calculated.")
        return
    print("\nRECALL (candidate exists near GT)")
    print("---------------------------------")
    radii = ["5", "10", "20", "42"]
    print(f"{'SOURCE':30s}" + "".join(f" <= {radius:>2s}px" for radius in radii))
    preferred = [
        "current_detector",
        "current_detector_v1",
        "current_detector_v2",
        "current_detector_agreement",
        "overlay_temporal_consensus",
        "overlay_persistent_zscore",
        "overlay_local_contrast",
        "overlay_darkening",
        "overlay_absdiff",
        "overlay",
        "union",
    ]
    for source in [name for name in preferred if name in recall]:
        values = recall.get(source, {})
        cells = []
        for radius in radii:
            value = values.get(radius)
            cells.append(f" {value:7.2f}%" if isinstance(value, (int, float)) else "      n/a")
        print(f"{source:30s}" + "".join(cells))


def main() -> int:
    args = build_parser().parse_args()
    if args.no_detector and args.no_overlay:
        print("ERROR: --no-detector and --no-overlay together leave nothing to replay.")
        return 2
    cases = read_manifest(args.manifest)
    if args.max_shots is not None:
        cases = cases[: max(0, int(args.max_shots))]
    if not cases:
        print("ERROR: manifest contains no shots.")
        return 2

    settings = ReplaySettings(
        use_current_detector=not args.no_detector,
        use_overlay=not args.no_overlay,
        candidate_limit=max(1, args.candidate_limit),
        union_limit=max(1, args.union_limit),
        merge_radius_px=max(0.5, args.merge_radius),
    )
    payload = benchmark_cases(
        cases,
        root=args.root.expanduser().resolve() if args.root else None,
        settings=settings,
        include_candidates=args.include_candidates,
    )
    write_benchmark(args.out, payload)

    summary = payload["summary"]
    print("V2.12 OFFLINE DETECTOR REPLAY")
    print("=============================")
    print(f"Shots requested : {len(cases)}")
    print(f"Shots replayed  : {summary.get('shots_total', 0)}")
    print(f"Labelled        : {summary.get('shots_labelled', 0)}")
    print(f"Errors          : {len(payload.get('errors') or [])}")
    _format_recall(summary)
    complement = summary.get("complementarity") or {}
    if not args.no_detector and not args.no_overlay:
        rescues = complement.get("overlay_rescues_current_detector") or {}
        print("\nSOURCE COMPLEMENTARITY")
        print("----------------------")
        print("Shots where V2.12 direct-image overlay finds GT while CURRENT LIVE detector does not:")
        for radius in ("5.0", "10.0", "20.0", "42.0"):
            if radius in rescues:
                print(f"  <= {radius:>4s}px : {rescues[radius]}")
        neither = complement.get("neither") or {}
        if "20.0" in neither:
            print(f"Neither source <=20px : {neither['20.0']}")
    print(f"\nResult          : {args.out}")
    if payload.get("errors"):
        print("\nFirst replay errors:")
        for row in payload["errors"][:10]:
            print(f"  {row['shot_id']}: {row['error']}")
        print("The run continues past bad archive entries; errors are preserved in JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
