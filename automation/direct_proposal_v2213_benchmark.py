from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.engine.offline.fullframe_benchmark_v2213 import benchmark_fullframe_v2213


def _fmt(summary: dict, source: str) -> str:
    item = summary[source]
    o = item["oracle"]
    return (
        f"{source:<22} o5={o['5']:.4f} o10={o['10']:.4f} o20={o['20']:.4f} o42={o['42']:.4f} "
        f"median={item['median_nearest_px']:.1f}px mean_n={item['mean_candidates']:.1f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="V2.21.3 DEVELOPMENT-tuned temporal-consensus benchmark")
    parser.add_argument("--root", default="content/ai/candidate_shadow_v216")
    parser.add_argument("--out", default="content/ai/reports/v2213/fullframe_benchmark_v2213.json")
    parser.add_argument("--debug-dir", default="content/ai/reports/v2213/debug")
    parser.add_argument("--debug-limit", type=int, default=12)
    args = parser.parse_args()

    report = benchmark_fullframe_v2213(
        Path(args.root),
        debug_dir=Path(args.debug_dir),
        debug_limit=max(0, int(args.debug_limit)),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("V2.21.3 TEMPORAL CONSENSUS / TARGET-MASK BENCHMARK")
    print("====================================================")
    print(f"Packs discovered    : {report['packs_discovered']}")
    print(f"Full-frame packs    : {report['packs_benchmarked']}")
    print(f"Missing full frames : {report['packs_missing_full_frames']}")
    print(f"Split provisional   : {report['split_is_provisional']}")
    print()
    print("DEVELOPMENT-ONLY profile sweep:")
    for name, item in report["profile_sweep"].items():
        s = item["development"]
        o = s["oracle"]
        marker = "  <-- SELECTED" if name == report["selected_profile"] else ""
        print(
            f"  {name:<18} shots={s['shots']:>2} o10={o['10']:.4f} o20={o['20']:.4f} o42={o['42']:.4f} "
            f"median={s['median_distance']:.1f}px mean_n={s['mean_candidates']:.1f}{marker}"
        )
    print()
    for split_name in ("all", "development", "confirmation", "holdout"):
        summary = report["all"] if split_name == "all" else report["splits"][split_name]
        print(f"{split_name.upper()} shots={summary['shots']}")
        for source in ("current", "v2212_union", "current_plus_consensus", "masked_direct", "final_union"):
            print("  " + _fmt(summary, source))
        print(f"  rescued@20={summary['rescued_at_20']}")
        print(f"  target-mask mean image fraction={summary['mask_fraction_mean']:.3f}")
        print()
    print(f"Gate        : {report['gate']}")
    print(f"Debug images: {args.debug_dir} ({report['debug_written']} written)")
    print(f"Report      : {out}")
    print("NEXT: If final_union improves on DEVELOPMENT and survives protected splits, keep anchored consensus; otherwise move to learned physical-domain dense evidence. No live authority.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
