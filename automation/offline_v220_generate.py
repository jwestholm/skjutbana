from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from src.engine.offline.media_bank_v219 import read_media_manifest
from src.engine.offline.scenario_generator_v220 import OfflineScenarioGeneratorV220, ScenarioProfileV220, scenario_fingerprint


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic V2.20 before/after scenarios")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--split", choices=["train", "validation", "holdout"], default="train")
    parser.add_argument("--manifest", default="content/ai/media_bank_v219/media_manifest.jsonl")
    parser.add_argument("--config", default="content/ai/offline_v220.json")
    parser.add_argument("--out", default="content/ai/reports/v220/generated_examples")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--save-gray", action="store_true")
    args = parser.parse_args()

    assets = read_media_manifest(Path(args.manifest))
    generator = OfflineScenarioGeneratorV220(profile=ScenarioProfileV220.from_file(Path(args.config)), media_assets=assets, repo_root=Path("."))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset in range(args.count):
        scenario = generator.generate(args.seed + offset, split=args.split)
        row = scenario.spec.to_dict()
        row["fingerprint"] = scenario_fingerprint(scenario.spec)
        rows.append(row)
        if args.save_images:
            prefix = out / f"seed_{scenario.spec.seed:09d}"
            cv2.imwrite(str(prefix.with_name(prefix.name + "_pre.png")), scenario.pre_frames_rgb[-1])
            cv2.imwrite(str(prefix.with_name(prefix.name + "_post.png")), scenario.post_frames_rgb[-1])
            if args.save_gray:
                cv2.imwrite(str(prefix.with_name(prefix.name + "_pre_gray.png")), scenario.pre_frames[-1])
                cv2.imwrite(str(prefix.with_name(prefix.name + "_post_gray.png")), scenario.post_frames[-1])
    (out / "scenarios.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("V2.20 OFFLINE SCENARIO GENERATION")
    print("================================")
    print(f"Scenarios : {len(rows)}")
    print(f"Split     : {args.split}")
    print(f"Seeds     : {args.seed}..{args.seed + args.count - 1}")
    print(f"Media     : {len(assets)} indexed assets (procedural fallback is always available)")
    print(f"Output    : {out / 'scenarios.json'}")
    print("Saved images default to RGB observed frames; grayscale remains available for legacy inspection.")
    print("Synthetic scenarios are training/research data, not physical acceptance data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
