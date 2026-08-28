from __future__ import annotations

import argparse
from pathlib import Path

from src.engine.offline.direct_proposal_v221 import DirectProposalConfigV221
from src.engine.offline.fullframe_benchmark_v2212 import benchmark_fullframe_v2212, write_fullframe_report_v2212
from src.engine.offline.temporal_local_v2212 import LocalTemporalConfigV2212


def _fmt(summary, source: str) -> str:
    row = summary[source]
    o = row["oracle"]
    return (
        f"{source:18s} o5={o['5']:.4f} o10={o['10']:.4f} "
        f"o20={o['20']:.4f} o42={o['42']:.4f} mean_n={row['mean_candidates']:.1f}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="V2.21.2 full-frame direct + local-temporal physical benchmark")
    p.add_argument("--root", type=Path, default=Path("content/ai/candidate_shadow_v216"))
    p.add_argument("--direct-config", type=Path, default=Path("content/ai/direct_v221.json"))
    p.add_argument("--out", type=Path, default=Path("content/ai/reports/v2212/fullframe_benchmark_v2212.json"))
    p.add_argument("--debug-dir", type=Path, default=Path("content/ai/reports/v2212/debug"))
    p.add_argument("--debug-limit", type=int, default=12)
    p.add_argument("--local-radius", type=int, default=48)
    args = p.parse_args()

    local_cfg = LocalTemporalConfigV2212(search_radius_px=max(1, int(args.local_radius)))
    report = benchmark_fullframe_v2212(
        args.root,
        direct_config=DirectProposalConfigV221.from_file(args.direct_config),
        local_config=local_cfg,
        debug_dir=args.debug_dir if args.debug_limit > 0 else None,
        debug_limit=max(0, args.debug_limit),
    )
    write_fullframe_report_v2212(args.out, report)

    print("V2.21.2 PHYSICAL FULL-FRAME DIAGNOSTIC")
    print("=======================================")
    print(f"Packs discovered    : {report['packs_discovered']}")
    print(f"Full-frame packs    : {report['packs_benchmarked']}")
    print(f"Missing full frames : {report['packs_missing_full_frames']}")
    print(f"Split provisional   : {report['split_is_provisional']}")

    for title, summary in [("ALL", report["all"])] + [(s.upper(), report["splits"][s]) for s in ("development", "confirmation", "holdout")]:
        print(f"\n{title} shots={summary['shots']}")
        for source in ("current", "direct", "local", "current_plus_local", "all_union"):
            print("  " + _fmt(summary, source))
        print(f"  rescued@20: {summary['rescued_at_20']}")
        off = summary.get("current_offset_within42", {})
        if off.get("count", 0):
            print(
                "  current<=42 offset: "
                f"n={off['count']} median_dx={off['median_dx']:.1f} median_dy={off['median_dy']:.1f} "
                f"median_dist={off['median_distance']:.1f} mad_dx={off['mad_dx']:.1f} mad_dy={off['mad_dy']:.1f}"
            )
        reg = summary.get("registration", {})
        if reg:
            print(
                "  registration: "
                f"applied={reg.get('applied_fraction', 0.0):.2f} "
                f"median_dx={reg.get('median_dx', 0.0):.2f} median_dy={reg.get('median_dy', 0.0):.2f} "
                f"median_response={reg.get('median_response', 0.0):.4f}"
            )

    print("\nGT percentile in temporal maps (diagnostic only; higher means GT itself is salient):")
    ranked = sorted(report["map_gt_summary"].items(), key=lambda kv: kv[1]["gt_percentile_median_development"], reverse=True)
    for name, row in ranked:
        print(f"  {name:24s} dev_median={row['gt_percentile_median_development']:.2f}% all_median={row['gt_percentile_median_all']:.2f}%")

    print("\nGate:", report["gate"])
    print(f"Debug images: {args.debug_dir} ({report['debug_written']} written)")
    print(f"Report      : {args.out}")
    print("NEXT: Use DEVELOPMENT diagnostics to choose whether V2.21.3 should fix registration/map evidence, local refinement, or calibration bias. Do not train on confirmation/holdout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
