from __future__ import annotations

import argparse
from pathlib import Path

from src.engine.offline.candidate_pack_v216 import DEFAULT_DATA_ROOT
from src.engine.offline.candidate_shadow_analysis_v216 import (
    DEFAULT_ENSEMBLE_CONFIG,
    DEFAULT_REPORT,
    benchmark_candidate_packs_v216,
    save_benchmark_report_v216,
)


def _pct(value):
    return "n/a" if value is None else f"{100.0*float(value):.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="V2.16 candidate-level shadow benchmark")
    parser.add_argument("--root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--ensemble", default=str(DEFAULT_ENSEMBLE_CONFIG))
    parser.add_argument("--max-shots", type=int, default=None)
    parser.add_argument("--out", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    report = benchmark_candidate_packs_v216(Path(args.root), ensemble_config=Path(args.ensemble), max_shots=args.max_shots)
    save_benchmark_report_v216(report, Path(args.out))

    print("V2.16 CANDIDATE-LEVEL SHADOW BENCHMARK")
    print("======================================")
    print(f"Shot packs          : {report['candidate_packs']}")
    print(f"Sessions            : {len(report['sessions'])} {report['sessions']}")
    print(f"Split               : {report['split']}")
    print(f"Provisional split   : {report['split_is_provisional']}")
    print(f"V9 available        : {report['model_sources']['ranker_v9_available']}")
    print(f"Fusion weights      : {report['selected_fusion']['weights']}")
    print()
    for split_name in ("development", "confirmation", "holdout"):
        pool = report["results"][split_name]["ranked_pool"]
        print(f"--- {split_name.upper()} / ranked pool / <=20 px ---")
        for source in ("current", "v9", "hole", "temporal", "fusion"):
            row = pool[source]["r20"]
            print(
                f"{source:10} top1={_pct(row['top1']):>8} top3={_pct(row['top3']):>8} "
                f"oracle={_pct(row['oracle_recall']):>8} median_rank={row['median_gt_rank']}"
            )
        union = report["results"][split_name]["raw_plus_ranked_union"]
        print(f"--- {split_name.upper()} / raw+ranked union / <=20 px ---")
        for source in ("hole", "temporal", "v9", "fusion"):
            row = union[source]["r20"]
            print(
                f"{source:10} top1={_pct(row['top1']):>8} top3={_pct(row['top3']):>8} "
                f"oracle={_pct(row['oracle_recall']):>8} median_rank={row['median_gt_rank']}"
            )
        print()

    print("GATE")
    for key, value in report["gate"].items():
        print(f"  {key}: {value}")
    print(f"\nReport: {Path(args.out)}")
    print("V2.16 remains SHADOW/OFFLINE ONLY. No candidate order or live hit authority is changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
