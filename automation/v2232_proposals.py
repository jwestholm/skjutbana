from __future__ import annotations

import argparse
from src.engine.ai.training_v223.proposal import expand_session


def main() -> None:
    p = argparse.ArgumentParser(description="V2.23.2 offline dense proposal expansion for captured F2 framepacks")
    p.add_argument("--session", default="latest", help="session id or 'latest'")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    report = expand_session(args.session, force=args.force, limit=args.limit)
    print("\nV2.23.2 PROPOSAL SUMMARY")
    print("========================")
    print(f"Status: {report.get('status')}")
    print(f"Session: {report.get('session_id')}")
    print(f"Processed: {report.get('processed')}")
    print(f"Oracle20: {report.get('oracle20')}")
    print(f"Oracle42: {report.get('oracle42')}")
    print(f"Mean candidates: {report.get('mean_candidates')}")
    if report.get('errors'):
        print(f"Errors: {len(report.get('errors', []))}")

if __name__ == "__main__":
    main()
