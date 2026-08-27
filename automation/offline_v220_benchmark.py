from __future__ import annotations

import argparse
from pathlib import Path

from src.engine.offline.media_bank_v219 import read_media_manifest
from src.engine.offline.scenario_benchmark_v220 import ScenarioBenchmarkConfigV220, ScenarioBenchmarkV220, write_benchmark
from src.engine.offline.scenario_generator_v220 import OfflineScenarioGeneratorV220, ScenarioProfileV220


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V2.20 synthetic/media benchmark")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--split", choices=["train", "validation", "holdout"], default="validation")
    parser.add_argument("--manifest", default="content/ai/media_bank_v219/media_manifest.jsonl")
    parser.add_argument("--config", default="content/ai/offline_v220.json")
    parser.add_argument("--out", default="content/ai/reports/v220/new_hole_v220_benchmark.json")
    parser.add_argument("--candidate-limit", type=int, default=500)
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()

    assets = read_media_manifest(Path(args.manifest))
    generator = OfflineScenarioGeneratorV220(profile=ScenarioProfileV220.from_file(Path(args.config)), media_assets=assets, repo_root=Path("."))
    bench = ScenarioBenchmarkV220(
        generator=generator,
        config=ScenarioBenchmarkConfigV220(
            seeds=args.count,
            first_seed=args.seed,
            split=args.split,
            candidate_limit=args.candidate_limit,
            use_overlay=not args.no_overlay,
        ),
    )
    report = bench.run()
    out = Path(args.out)
    write_benchmark(out, report)
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
