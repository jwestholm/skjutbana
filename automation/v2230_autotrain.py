from __future__ import annotations

import argparse
import time
from src.engine.ai.training_v223.trainer import train_once_v223


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-budgeted V2.23 autonomous challenger loop")
    parser.add_argument("--hours", type=float, default=1.0, help="wall-clock training budget")
    parser.add_argument("--quick", action="store_true", help="use short portfolios per trial")
    parser.add_argument("--no-legacy", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=0, help="0 = only time budget")
    parser.add_argument("--seed", type=int, default=22300)
    args = parser.parse_args()
    budget_s = max(0.01, args.hours) * 3600.0
    deadline = time.time() + budget_s
    round_no = 0
    print("V2.23.0 AUTONOMOUS CHALLENGER LOOP")
    print("=================================")
    print(f"Budget: {args.hours:.3f} h | protected holdout: NEVER used for selection")
    while time.time() < deadline and (args.max_rounds <= 0 or round_no < args.max_rounds):
        round_no += 1
        seed = args.seed + round_no * 10
        remaining = max(0.0, deadline - time.time())
        print(f"\n[AUTOTRAIN] round={round_no} seed={seed} remaining={remaining/60.0:.1f} min")
        report = train_once_v223(
            trigger=f"autotrain_round_{round_no}", quick=args.quick,
            include_legacy=not args.no_legacy, seed_base=seed,
        )
        print(f"[AUTOTRAIN] status={report.get('status')} best={report.get('best_trial_id')} promoted={report.get('research_champion_promoted')}")
        if report.get("status") == "insufficient_data":
            print("[AUTOTRAIN] stopping: insufficient usable data")
            break
    print(f"\n[AUTOTRAIN] finished rounds={round_no}")


if __name__ == "__main__":
    main()
