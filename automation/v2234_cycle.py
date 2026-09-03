from __future__ import annotations
import argparse
from src.engine.ai.training_v223.trainer_v2234 import cycle_v2234

def main()->int:
    ap=argparse.ArgumentParser(description='Run V2.23.4 proposal->patch->training cycle.')
    ap.add_argument('--session',default='latest')
    ap.add_argument('--quick',action='store_true')
    args=ap.parse_args(); r=cycle_v2234(session=args.session,quick=args.quick)
    return 0 if r.get('train',{}).get('status')=='ok' else 2
if __name__=='__main__': raise SystemExit(main())
