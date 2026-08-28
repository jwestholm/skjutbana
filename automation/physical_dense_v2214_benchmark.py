from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.engine.offline.fullframe_benchmark_v2214 import benchmark_physical_dense_v2214
from src.engine.offline.physical_dense_v2214 import DEFAULT_MODEL_PATH


def _source_line(name: str, item: dict) -> str:
    oracle = item["oracle"]
    return (
        f"  {name:<30} o5={oracle['5']:.4f} o10={oracle['10']:.4f} "
        f"o20={oracle['20']:.4f} o42={oracle['42']:.4f} "
        f"median={item['median_nearest_px']:.1f}px mean_n={item['mean_candidates']:.1f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen V2.21.4 physical dense ranker benchmark")
    parser.add_argument("--root", default="content/ai/candidate_shadow_v216")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--config", default="content/ai/physical_dense_v2214.json")
    parser.add_argument("--report", default="content/ai/reports/v2214/fullframe_benchmark_v2214.json")
    parser.add_argument("--debug-dir", default="content/ai/reports/v2214/debug")
    parser.add_argument("--debug-limit", type=int, default=12)
    args = parser.parse_args()

    report = benchmark_physical_dense_v2214(
        Path(args.root),
        model_path=Path(args.model),
        config_path=Path(args.config),
        debug_dir=Path(args.debug_dir),
        debug_limit=args.debug_limit,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    frozen = int(report["frozen_top_k"])
    print("V2.21.4 LEARNED PHYSICAL-DOMAIN DENSE BENCHMARK")
    print("================================================")
    print(f"Packs discovered    : {report['packs_discovered']}")
    print(f"Full-frame packs    : {report['packs_benchmarked']}")
    print(f"Missing full frames : {report['packs_missing_full_frames']}")
    print(f"Split provisional   : {report['split_is_provisional']}")
    print(f"Frozen top-K        : {frozen}")

    for split_name, summary in [("ALL", report["all"])] + [(name.upper(), report["splits"][name]) for name in ("development", "confirmation", "holdout")]:
        print()
        print(f"{split_name} shots={summary['shots']}")
        for source in ("current", "v2212_union", "dense_pool"):
            print(_source_line(source, summary[source]))
        for k in report["top_k_values"]:
            print(_source_line(f"learned_{k}", summary[f"learned_{k}"]))
        print(_source_line(f"v2212_plus_learned_{frozen}", summary[f"v2212_plus_learned_{frozen}"]))
        rank = summary["learned_gt_rank20"]
        print(
            f"  learned GT rank@20: n={rank['shots_with_pool_candidate20']} "
            f"median={rank['median_rank']:.1f} p90={rank['p90_rank']:.1f}"
        )
        print(f"  rescued@20={summary['rescued_at_20']}")
        print(f"  target-mask={summary['target_mask']}")

    print()
    print(f"Gate        : {report['gate']}")
    print(f"Debug images: {args.debug_dir} ({report['debug_written']} written)")
    print(f"Report      : {report_path}")
    print(
        "NEXT: If dense_pool oracle is high but learned top-K is weak, improve learned ranking/add more independent physical DEVELOPMENT data. "
        "If dense_pool itself is weak, improve broad proposal generation first. No live authority."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
