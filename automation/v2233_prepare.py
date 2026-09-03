from __future__ import annotations
import argparse
from src.engine.ai.training_v223.trainer_v2233 import prepare_dense_sessions

def main() -> None:
    p=argparse.ArgumentParser(description='V2.23.3 rich PRE/POST evidence + dense cache preparation')
    p.add_argument('--session', default='all')
    p.add_argument('--force-rich', action='store_true')
    p.add_argument('--force-cache', action='store_true')
    args=p.parse_args()
    report=prepare_dense_sessions(session=args.session, force_rich=args.force_rich, force_cache=args.force_cache)
    print('\nV2.23.3 PREPARE SUMMARY')
    print('======================')
    print(f"Status: {report.get('status')}")
    for sid, item in report.get('sessions',{}).items():
        rich=item.get('rich',{}); cache=item.get('cache',{})
        print(f"{sid}: rich={rich.get('processed',0)} cached-rich={rich.get('cached',0)} dense-cache={cache.get('processed',0)} errors={len(rich.get('errors',[]))+len(cache.get('errors',[]))}")

if __name__=='__main__': main()
