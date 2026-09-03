from __future__ import annotations
import argparse
from src.engine.ai.training_v223.trainer_v2235 import prepare_evidence_sessions

def main() -> int:
    ap = argparse.ArgumentParser(description='Prepare V2.23.5 registered-evidence patch banks.')
    ap.add_argument('--session', default='latest')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    r = prepare_evidence_sessions(session=args.session, force=args.force)
    print('\nV2.23.5 PREPARE SUMMARY\n=======================')
    print(f"Status: {r.get('status')}")
    for sid, row in (r.get('evidence') or {}).items():
        print(f"  {sid}: processed={row.get('processed',0)} cached={row.get('cached',0)} cache={row.get('cache_bytes',0)/1024/1024:.1f}MB errors={len(row.get('errors',[]))}")
    return 0 if r.get('status') == 'ok' else 2

if __name__ == '__main__':
    raise SystemExit(main())
