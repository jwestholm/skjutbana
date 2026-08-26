from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.engine.ai.hole_patch_ensemble_v215 import HolePatchEnsembleV215


def main() -> int:
    p = argparse.ArgumentParser(description="Verify a generated V2.15 shadow ensemble config and both V2.14 models.")
    p.add_argument("--config", type=Path, default=Path("content/ai/reports/v215/hole_v215_ensemble.json"))
    args = p.parse_args()
    if not args.config.exists():
        print(f"ERROR: missing {args.config}")
        print("Run: python3 -m automation.hole_v215_ensemble")
        return 2
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    ensemble = HolePatchEnsembleV215.load(args.config)
    cfg = ensemble.config
    print("V2.15 SHADOW ENSEMBLE VERIFY")
    print("============================")
    print(f"Config            : {args.config}")
    print(f"Standard model    : {cfg.standard_model_path}")
    print(f"Mild model        : {cfg.mild_model_path}")
    print(f"Standard weight   : {cfg.standard_weight:.6f}")
    print(f"Mild weight       : {cfg.mild_weight:.6f}")
    print(f"Fused threshold   : {cfg.fused_threshold:.6f}")
    print(f"Shadow only       : {cfg.shadow_only}")
    print(f"Gate               : {payload.get('gate')}")
    print("Loaded both models successfully. No candidate order/live authority is changed by this verifier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
