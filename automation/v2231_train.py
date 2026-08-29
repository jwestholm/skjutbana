from __future__ import annotations

import argparse
from src.engine.ai.training_v223.trainer import train_once_v223


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one V2.23.1 support-gated challenger portfolio offline")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-legacy", action="store_true")
    parser.add_argument("--seed", type=int, default=2231)
    args = parser.parse_args()
    report = train_once_v223(
        trigger="offline_cli_v2231", quick=args.quick,
        include_legacy=not args.no_legacy, seed_base=args.seed,
    )
    print("V2.23.1 OFFLINE TRAIN")
    print("=====================")
    print(f"Status: {report.get('status')}")
    print(f"Dataset: {report.get('dataset')}")
    print(f"Support: {report.get('support')}")
    print(f"Split: {report.get('split')}")
    print(f"Baseline validation: {report.get('baseline_validation')}")
    if report.get("reasons"):
        print(f"Gate reasons: {report.get('reasons')}")
    for trial in report.get("trials", []):
        print(f"  {trial.get('trial_id')} -> {trial.get('metrics')} gate={trial.get('research_promotion_gate',{}).get('passed')}")
    print(f"Best: {report.get('best_trial_id')}")
    print(f"Research champion promoted: {report.get('research_champion_promoted')} ({report.get('promotion_reason')})")
    print("Protected holdout evaluated for selection: NO")
    print("Live authority: unchanged / NO")


if __name__ == "__main__":
    main()
