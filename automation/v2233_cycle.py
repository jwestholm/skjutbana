from __future__ import annotations
import argparse
from src.engine.ai.training_v223.trainer_v2233 import cycle_v2233

def main() -> None:
    p=argparse.ArgumentParser(description='V2.23.3 proposal -> rich evidence -> reducer cache -> train')
    p.add_argument('--session', default='latest')
    p.add_argument('--quick', action='store_true')
    args=p.parse_args()
    result=cycle_v2233(session=args.session, quick=args.quick)
    train=result.get('train',{})
    print('\nV2.23.3 CYCLE SUMMARY')
    print('=====================')
    print(f"Proposal: {result.get('proposal',{}).get('status')} processed={result.get('proposal',{}).get('processed')}")
    print(f"Rich: {result.get('rich',{}).get('status')} processed={result.get('rich',{}).get('processed')} cached={result.get('rich',{}).get('cached')}")
    print(f"Cache: {result.get('cache',{}).get('status')} processed={result.get('cache',{}).get('processed')}")
    print(f"Train: {train.get('status')} mode={train.get('split',{}).get('mode')} gate={train.get('research_gate_passed')}")

if __name__=='__main__': main()
