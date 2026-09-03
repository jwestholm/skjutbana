from __future__ import annotations
import argparse
from src.engine.ai.training_v223.trainer_v2234 import train_patch_cascade_v2234

def main() -> int:
    ap=argparse.ArgumentParser(description='Train V2.23.4 patch NewHole cascade.')
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--no-prepare',action='store_true')
    args=ap.parse_args()
    r=train_patch_cascade_v2234(quick=args.quick,prepare=not args.no_prepare)
    print('\nV2.23.4 TRAIN SUMMARY\n=====================')
    print(f"Status: {r.get('status')}")
    sp=r.get('split',{})
    print(f"Split: mode={sp.get('mode')} train={sp.get('train')} validation={sp.get('validation')} fresh_domain={sp.get('fresh_domain')}")
    bp=r.get('best_patch_model') or {}
    if bp:
        v=bp.get('validation',{})
        print(f"Best patch model: kind={bp.get('kind')} hidden={bp.get('hidden')} filters={bp.get('filters')}")
        print(f"  validation: oracle20={v.get('oracle20')}/{v.get('shots')} R128={v.get('retention20_at_k',{}).get('128')} R512={v.get('retention20_at_k',{}).get('512')} median_rank={v.get('median_positive_rank20')} top1={v.get('conditional_top1_20_rate')}")
    d=r.get('best_patch_fresh_domain')
    if d:
        print(f"  fresh-domain: oracle20={d.get('oracle20')}/{d.get('shots')} R128={d.get('retention20_at_k',{}).get('128')} R512={d.get('retention20_at_k',{}).get('512')} median_rank={d.get('median_positive_rank20')} top1={d.get('conditional_top1_20_rate')}")
    bf=r.get('best_final_ranker') or {}
    if bf:
        v=bf.get('validation',{})
        print(f"Best final ranker: kind={bf.get('kind')} conditionalTop1={v.get('conditional_top1_20_rate')} Top3={v.get('conditional_top3_20_rate')} median_rank={v.get('median_positive_rank')}")
    if r.get('best_final_fresh_domain'):
        d=r['best_final_fresh_domain']; print(f"  final fresh-domain: conditionalTop1={d.get('conditional_top1_20_rate')} Top3={d.get('conditional_top3_20_rate')} median_rank={d.get('median_positive_rank')}")
    print(f"Bootstrap learnability gate: {r.get('bootstrap_learnability_gate_passed',False)}")
    print(f"Domain validated: {r.get('domain_validated',False)}")
    print(f"Research cascade gate: {r.get('research_gate_passed',False)}")
    print(f"Elapsed: {r.get('elapsed_seconds',0):.1f}s")
    print('Live authority: unchanged / NO')
    return 0 if r.get('status')=='ok' else 2

if __name__=='__main__': raise SystemExit(main())
