from __future__ import annotations

import argparse
from pathlib import Path

from src.engine.offline.direct_proposal_benchmark_v221 import benchmark_direct_proposals_v221, write_direct_benchmark_v221
from src.engine.offline.direct_proposal_v221 import DirectProposalConfigV221


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark V2.21 direct full-frame proposals against candidate packs")
    p.add_argument("--root", type=Path, default=Path("content/ai/candidate_shadow_v216"))
    p.add_argument("--config", type=Path, default=Path("content/ai/direct_v221.json"))
    p.add_argument("--out", type=Path, default=Path("content/ai/reports/v221/direct_proposal_v221_benchmark.json"))
    a = p.parse_args()
    report = benchmark_direct_proposals_v221(a.root, config=DirectProposalConfigV221.from_file(a.config))
    write_direct_benchmark_v221(a.out, report)
    print("V2.21 DIRECT FULL-FRAME PROPOSAL BENCHMARK")
    print("==========================================")
    print(f"Packs discovered      : {report['packs_discovered']}")
    print(f"Packs benchmarked     : {report['packs_benchmarked']}")
    print(f"Missing full frames   : {report['packs_missing_full_frames']}")
    if report["can_measure_physical_direct_recall"]:
        all_row = report["all"]
        print(f"\nALL FULL-FRAME SHOTS={all_row['shots']}")
        for src in ("current", "direct", "union"):
            print(
                f"  {src:8s} oracle5={all_row[src]['oracle']['5']:.4f} "
                f"oracle10={all_row[src]['oracle']['10']:.4f} "
                f"oracle20={all_row[src]['oracle']['20']:.4f} "
                f"oracle42={all_row[src]['oracle']['42']:.4f} "
                f"mean_candidates={all_row[src]['mean_candidates']:.1f}"
            )
        print(f"  rescued@20={all_row['rescued_current_miss']['20']}")
        print(f"  runtime_ms_mean={all_row['runtime_ms_mean']:.1f}")
        print(f"  provisional_split={report['split_is_provisional']}")
        for split in ("development", "confirmation", "holdout"):
            row = report["splits"][split]
            print(f"\n{split.upper()} shots={row['shots']}")
            for src in ("current", "direct", "union"):
                print(f"  {src:8s} oracle20={row[src]['oracle']['20']:.4f} oracle42={row[src]['oracle']['42']:.4f} mean_candidates={row[src]['mean_candidates']:.1f}")
            print(f"  rescued@20={row['rescued_current_miss']['20']}")
        print("\nGate:", report["gate"])
    else:
        print("\nNo honest full-frame PRE+POST packs were available, so no fake direct-proposal score was produced.")
    print("NEXT:", report["next_requirement"])
    print(f"Report: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
