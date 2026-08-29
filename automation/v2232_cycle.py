from __future__ import annotations

import argparse
import json

from src.engine.ai.training_v223.proposal import expand_session
from src.engine.ai.training_v223.trainer import train_once_v223


def main() -> None:
    parser = argparse.ArgumentParser(description="V2.23.2 offline proposal + challenger cycle")
    parser.add_argument("--session", default="latest", help="Framepack session id or 'latest'")
    parser.add_argument("--limit", type=int, default=None, help="Limit proposal expansion for smoke testing")
    parser.add_argument("--force", action="store_true", help="Rebuild cached proposal sidecars")
    parser.add_argument("--quick", action="store_true", help="Use short challenger training")
    parser.add_argument("--no-legacy", action="store_true", help="Do not import V2.16/V2.20 legacy candidate packs")
    args = parser.parse_args()

    print("V2.23.2 OFFLINE CYCLE")
    print("=====================")
    proposal = expand_session(args.session, force=args.force, limit=args.limit)
    print("Proposal:", json.dumps(proposal, indent=2, sort_keys=True))
    if proposal.get("status") not in ("ok", "no_framepacks"):
        raise SystemExit(2)
    if args.limit is not None:
        print("\n[INFO] --limit was used; proposal smoke-test complete. Training is skipped so a partial session cannot masquerade as full proposal expansion.")
        return
    report = train_once_v223(
        trigger=f"offline_cycle:{proposal.get('session_id', args.session)}",
        quick=bool(args.quick),
        include_legacy=not args.no_legacy,
    )
    print("\nTraining status:", report.get("status"))
    print("Fresh domain:", json.dumps(report.get("domain", {}), indent=2, sort_keys=True))
    print("Best:", report.get("best_trial_id"))
    print("Promoted:", report.get("research_champion_promoted"), report.get("promotion_reason"))
    print("Live authority: unchanged / NO")


if __name__ == "__main__":
    main()
