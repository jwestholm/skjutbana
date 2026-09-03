from __future__ import annotations
import argparse
from src.engine.ai.training_v223.trainer_v2234 import prepare_patch_sessions

def main() -> int:
    ap=argparse.ArgumentParser(description='Prepare V2.23.4 patch banks from existing proposal/framepack sessions.')
    ap.add_argument('--session',default='latest',help='session id, latest, or all')
    ap.add_argument('--force',action='store_true',help='rebuild V2.23.4 patch banks')
    args=ap.parse_args()
    session=None if args.session=='all' else args.session
    result=prepare_patch_sessions(session=session,force=args.force)
    print('\nV2.23.4 PREPARE SUMMARY\n=======================')
    print(f"Status: {result.get('status')}")
    for sid,row in result.get('patch',{}).items():
        print(f"  {sid}: processed={row.get('processed',0)} cached={row.get('cached',0)} cache={row.get('cache_bytes',0)/(1024**2):.1f}MB errors={len(row.get('errors',[]))}")
    return 0 if result.get('status')=='ok' else 2

if __name__=='__main__': raise SystemExit(main())
