from __future__ import annotations

import argparse
from pathlib import Path

from src.engine.offline.media_bank_v219 import read_media_manifest
from src.engine.offline.scenario_benchmark_v219 import ScenarioBenchmarkConfigV219, ScenarioBenchmarkV219, write_benchmark
from src.engine.offline.scenario_generator_v219 import OfflineScenarioGeneratorV219, ScenarioProfileV219


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the current live V1/V2 detector offline on deterministic V2.19 scenarios")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=9000001)
    parser.add_argument("--split", choices=["train", "validation", "holdout"], default="validation")
    parser.add_argument("--manifest", default="content/ai/media_bank_v219/media_manifest.jsonl")
    parser.add_argument("--config", default="content/ai/offline_v219.json")
    parser.add_argument("--out", default="content/ai/reports/v219/scenario_benchmark.json")
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()

    assets = read_media_manifest(Path(args.manifest))
    generator = OfflineScenarioGeneratorV219(profile=ScenarioProfileV219.from_file(Path(args.config)), media_assets=assets, repo_root=Path("."))
    bench = ScenarioBenchmarkV219(
        generator=generator,
        config=ScenarioBenchmarkConfigV219(
            seeds=args.seeds,
            first_seed=args.first_seed,
            split=args.split,
            use_overlay=not args.no_overlay,
        ),
    )
    report = bench.run()
    write_benchmark(Path(args.out), report)
    summary = report["summary"]
    print("\nV2.19 SCENARIO BENCHMARK")
    print("========================")
    print(f"Shots : {summary['shots']}")
    for source in ("current", "overlay", "union"):
        values = summary["recall"][source]
        print(f"{source:8} recall <=20: {100*values['20']:.1f}% | <=42: {100*values['42']:.1f}%")
    print(f"Mean current candidates: {summary['candidate_count_mean']:.1f}")
    print(f"Mean detector time     : {summary['detector_ms_mean']:.1f} ms/scenario")
    print("\nPer-category and challenge-tag breakdowns are in the JSON report.")
    print("This synthetic/media benchmark can guide training; unseen physical sessions remain the authority.")
    print(f"Report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
