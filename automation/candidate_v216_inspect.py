from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.engine.offline.candidate_pack_v216 import CandidatePackV216, DEFAULT_DATA_ROOT, discover_candidate_packs


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect V2.16 candidate-shadow capture packs")
    parser.add_argument("--root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--max-shots", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    paths = discover_candidate_packs(root)
    if args.max_shots is not None:
        paths = paths[: max(0, args.max_shots)]
    sessions = Counter()
    backgrounds = Counter()
    candidates = []
    post_frames = []
    forced = 0
    labelled = 0
    for path in paths:
        pack = CandidatePackV216.load(path)
        sessions[str(pack.metadata.get("session_id", "unknown"))] += 1
        backgrounds[str(pack.metadata.get("background", "unknown"))] += 1
        candidates.append(len(pack.candidates))
        post_frames.append(pack.post_patches.shape[1] if pack.post_patches.ndim == 4 else 0)
        forced += sum(1 for row in pack.candidates if row.get("capture_forced_gt_nearest"))
        labelled += int(pack.gt_xy is not None)

    print("V2.16 CANDIDATE PACK INSPECTION")
    print("===============================")
    print(f"Root               : {root.resolve()}")
    print(f"Shot packs         : {len(paths)}")
    print(f"Labelled GT        : {labelled}")
    print(f"Sessions           : {len(sessions)}")
    print(f"Candidates total   : {sum(candidates)}")
    print(f"Candidates avg     : {(sum(candidates)/len(candidates)) if candidates else 0:.2f}")
    print(f"Candidates max     : {max(candidates) if candidates else 0}")
    print(f"Post frames avg    : {(sum(post_frames)/len(post_frames)) if post_frames else 0:.2f}")
    print(f"Forced GT-nearest  : {forced} (diagnostic-only rows)")
    print("Backgrounds:")
    for name, count in backgrounds.most_common():
        print(f"  {name:20} {count}")
    print("Sessions:")
    for name, count in sessions.most_common():
        print(f"  {name:36} {count}")
    if not paths:
        print("\nNo captures yet. Run one automation F2 session after installing V2.16.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
