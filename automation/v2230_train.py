from __future__ import annotations

import argparse
import json
from src.engine.ai.training_v223.trainer import train_once_v223


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one V2.23 champion/challenger portfolio offline")
    parser.add_argument("--quick", action="store_true", help="shorter engineering run")
    parser.add_argument("--no-legacy", action="store_true", help="use only native V2.23 captures")
    parser.add_argument("--seed", type=int, default=2230)
    args = parser.parse_args()
    report = train_once_v223(trigger="offline_cli", quick=args.quick, include_legacy=not args.no_legacy, seed_base=args.seed)
    print("V2.23.0 OFFLINE TRAIN")
    print("====================")
    print(f"Status: {report.get('status')}")
    print(f"Dataset: {report.get('dataset')}")
    print(f"Split: {report.get('split')}")
    print(f"Baseline validation: {report.get('baseline_validation')}")
    for trial in report.get("trials", []):
        print(f"  {trial.get('trial_id')} -> {trial.get('metrics')}")
    print(f"Best: {report.get('best_trial_id')}")
    print(f"Research champion promoted: {report.get('research_champion_promoted')} ({report.get('promotion_reason')})")
    print("Live authority: unchanged / NO")


if __name__ == "__main__":
    main()
