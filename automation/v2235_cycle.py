from __future__ import annotations
import argparse
from src.engine.ai.training_v223.trainer_v2235 import cycle_v2235

def main() -> int:
    ap = argparse.ArgumentParser(description='Run complete V2.23.5 proposal/evidence/training cycle.')
    ap.add_argument('--session', default='latest')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    r = cycle_v2235(session=args.session, quick=args.quick)
    status = (r.get('train') or {}).get('status')
    print(f"\nV2.23.5 CYCLE COMPLETE: train_status={status}")
    return 0 if status == 'ok' else 2

if __name__ == '__main__':
    raise SystemExit(main())
