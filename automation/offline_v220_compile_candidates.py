from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.engine.offline.media_bank_v219 import read_media_manifest
from src.engine.offline.scenario_candidate_compiler_v220 import ScenarioCandidateCompilerV220
from src.engine.offline.scenario_generator_v220 import OfflineScenarioGeneratorV220, ScenarioProfileV220


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile V2.20 generated worlds into V2.16-compatible candidate packs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--split", choices=["train", "validation", "holdout"], default="train")
    parser.add_argument("--manifest", default="content/ai/media_bank_v219/media_manifest.jsonl")
    parser.add_argument("--config", default="content/ai/offline_v220.json")
    parser.add_argument("--out", default="content/ai/candidate_synthetic_v220")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--candidate-limit", type=int, default=384)
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()

    assets = read_media_manifest(Path(args.manifest))
    generator = OfflineScenarioGeneratorV220(profile=ScenarioProfileV220.from_file(Path(args.config)), media_assets=assets, repo_root=Path("."))
    compiler = ScenarioCandidateCompilerV220(generator=generator, output_root=Path(args.out), candidate_limit=args.candidate_limit, include_overlay=not args.no_overlay)
    report = compiler.compile(first_seed=args.seed, count=args.count, split=args.split, session_id=args.session_id)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
