from __future__ import annotations
import argparse
from src.engine.ai.training_v223.trainer_v2233 import train_cascade_v2233

def main() -> None:
    p=argparse.ArgumentParser(description='V2.23.3 learned dense reducer + final ranker')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--no-prepare', action='store_true')
    args=p.parse_args()
    report=train_cascade_v2233(quick=args.quick, prepare=not args.no_prepare)
    print('\nV2.23.3 TRAIN SUMMARY')
    print('=====================')
    print(f"Status: {report.get('status')}")
    if report.get('status')!='ok':
        print(f"Split mode: {report.get('split_mode')}")
        print(f"Notes: {report.get('notes')}")
        return
    split=report.get('split',{})
    print(f"Split: mode={split.get('mode')} train={split.get('train')} validation={split.get('validation')} fresh_domain={split.get('fresh_domain')}")
    br=report.get('best_reducer',{})
    print(f"Best reducer: kind={br.get('kind')} hidden={br.get('hidden')}")
    print(f"  validation={br.get('validation')}")
    if br.get('fresh_domain') is not None: print(f"  fresh_domain={br.get('fresh_domain')}")
    bf=report.get('best_final_ranker',{})
    print(f"Best final ranker: kind={bf.get('kind')} hidden={bf.get('hidden')}")
    print(f"  validation={bf.get('validation')}")
    if bf.get('fresh_domain') is not None: print(f"  fresh_domain={bf.get('fresh_domain')}")
    print(f"Domain validated: {report.get('domain_validated')}")
    print(f"Research cascade gate: {report.get('research_gate_passed')}")
    print(f"Elapsed: {report.get('elapsed_seconds',0):.1f}s")
    print('Live authority: unchanged / NO')

if __name__=='__main__': main()
