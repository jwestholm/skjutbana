from __future__ import annotations

import argparse
from src.engine.ai.training_v223.trainer import train_once_v223


def main() -> None:
    p = argparse.ArgumentParser(description="V2.23.2 offline challenger training with fresh-F2 domain gate")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-legacy", action="store_true")
    p.add_argument("--seed", type=int, default=2232)
    args = p.parse_args()
    report = train_once_v223(trigger="offline_cli_v2232", quick=args.quick, include_legacy=not args.no_legacy, seed_base=args.seed)
    print("V2.23.2 OFFLINE TRAIN")
    print("=====================")
    print(f"Status: {report.get('status')}")
    print(f"Dataset: {report.get('dataset')}")
    print(f"Engineering support: {report.get('support')}")
    print(f"Fresh F2 domain: {report.get('domain')}")
    print(f"Split: {report.get('split')}")
    print(f"Reference baseline validation: {report.get('baseline_validation')}")
    for trial in report.get('trials', []):
        print(f"  {trial.get('trial_id')}")
        print(f"    validation={trial.get('metrics')}")
        print(f"    fresh_domain={trial.get('domain_metrics')}")
        print(f"    gate={trial.get('research_promotion_gate',{}).get('passed')} reasons={trial.get('research_promotion_gate',{}).get('reasons',[])}")
    print(f"Best: {report.get('best_trial_id')}")
    print(f"Research champion promoted: {report.get('research_champion_promoted')} ({report.get('promotion_reason')})")
    print("Protected holdout evaluated for selection: NO")
    print("Live authority: unchanged / NO")

if __name__ == "__main__":
    main()
