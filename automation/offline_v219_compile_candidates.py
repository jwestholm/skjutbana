from __future__ import annotations

import argparse
from pathlib import Path

from src.engine.offline.media_bank_v219 import read_media_manifest
from src.engine.offline.scenario_candidate_compiler_v219 import ScenarioCandidateCompilerV219
from src.engine.offline.scenario_generator_v219 import OfflineScenarioGeneratorV219, ScenarioProfileV219


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile seed-generated V2.19 worlds into V2.16-compatible candidate packs")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--split", choices=["train", "validation", "holdout"], default="train")
    parser.add_argument("--manifest", default="content/ai/media_bank_v219/media_manifest.jsonl")
    parser.add_argument("--config", default="content/ai/offline_v219.json")
    parser.add_argument("--out-root", default="content/ai/candidate_synthetic_v219")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--save-full-frames", action="store_true")
    args = parser.parse_args()

    assets = read_media_manifest(Path(args.manifest))
    generator = OfflineScenarioGeneratorV219(
        profile=ScenarioProfileV219.from_file(Path(args.config)),
        media_assets=assets,
        repo_root=Path("."),
    )
    compiler = ScenarioCandidateCompilerV219(
        generator=generator,
        output_root=Path(args.out_root),
        include_overlay=not args.no_overlay,
        save_full_frames=args.save_full_frames,
    )
    report = compiler.compile(first_seed=args.first_seed, count=args.count, split=args.split)
    print("\nV2.19 GENERATED CANDIDATE PACKS")
    print("================================")
    print(f"Session  : {report['session_id']}")
    print(f"Split    : {report['split']}")
    print(f"Worlds   : {report['count']}")
    print(f"Saved    : {report['saved']}")
    print(f"Root     : {report['capture'].get('root')}")
    print(f"Report   : {report['report_path']}")
    print("These packs can be passed to V2.17/V2.18 with --root, but are synthetic training data only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
