from __future__ import annotations

import json
from pathlib import Path

from src.engine.ai.hole_patch_ensemble_v215 import HolePatchEnsembleV215, DEFAULT_CONFIG_PATH as V215_CONFIG
from src.engine.offline.candidate_pack_v216 import CandidateCaptureConfigV216, DEFAULT_CONFIG_PATH, DEFAULT_DATA_ROOT, discover_candidate_packs


def main() -> int:
    print("V2.16 CANDIDATE SHADOW VERIFY")
    print("============================")
    cfg = CandidateCaptureConfigV216.load(DEFAULT_CONFIG_PATH)
    print(f"Capture config     : {DEFAULT_CONFIG_PATH}")
    print(f"Capture enabled    : {cfg.enabled}")
    print(f"Patch size         : {cfg.patch_size}")
    print(f"Post frames        : {cfg.max_post_frames}")
    print(f"Candidate cap      : {cfg.max_candidates}")
    ensemble = HolePatchEnsembleV215.load(V215_CONFIG)
    print(f"V2.15 ensemble     : {V215_CONFIG}")
    print(f"Shadow only        : {ensemble.config.shadow_only}")
    packs = discover_candidate_packs(DEFAULT_DATA_ROOT)
    print(f"Candidate packs    : {len(packs)}")
    report = Path("content/ai/reports/v216/candidate_shadow_report.json")
    if report.exists():
        payload = json.loads(report.read_text(encoding="utf-8"))
        print(f"Benchmark report   : {report}")
        print(f"Provisional split  : {payload.get('split_is_provisional')}")
        print(f"Gate               : {payload.get('gate')}")
    else:
        print("Benchmark report   : not created yet")
    print("Authority           : SHADOW/OFFLINE ONLY")
    print("No V2.16 verifier path changes candidate order or live hit coordinates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
