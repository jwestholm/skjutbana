from __future__ import annotations
import argparse
from src.engine.ai.training_v223.trainer_v2235 import train_registered_evidence_cascade_v2235

def main() -> int:
    ap = argparse.ArgumentParser(description='Train V2.23.5 registered-evidence hard-mined cascade.')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--no-prepare', action='store_true')
    args = ap.parse_args()
    r = train_registered_evidence_cascade_v2235(quick=args.quick, prepare=not args.no_prepare)
    print('\nV2.23.5 TRAIN SUMMARY\n=====================')
    print(f"Status: {r.get('status')}")
    sp = r.get('split', {})
    print(f"Split: mode={sp.get('mode')} train={sp.get('train')} validation={sp.get('validation')} fresh_domain={sp.get('fresh_domain')}")
    be = r.get('best_evidence_model') or {}
    if be:
        v = be.get('validation', {})
        print(f"Best evidence model: kind={be.get('kind')} hidden={be.get('hidden')}")
        print(f"  validation: oracle20={v.get('oracle20')}/{v.get('shots')} R128={v.get('retention20_at_k',{}).get('128')} R512={v.get('retention20_at_k',{}).get('512')} median_rank={v.get('median_positive_rank20')} top1={v.get('conditional_top1_20_rate')}")
        tr = be.get('training', {})
        print(f"  hard-mine round1={tr.get('mine_round1')} round2={tr.get('mine_round2')}")
    d = r.get('best_evidence_fresh_domain')
    if d:
        print(f"  fresh-domain: oracle20={d.get('oracle20')}/{d.get('shots')} R128={d.get('retention20_at_k',{}).get('128')} R512={d.get('retention20_at_k',{}).get('512')} median_rank={d.get('median_positive_rank20')} top1={d.get('conditional_top1_20_rate')}")
    bf = r.get('best_final_ranker') or {}
    if bf:
        v = bf.get('validation', {})
        print(f"Best final ranker: kind={bf.get('kind')} conditionalTop1={v.get('conditional_top1_20_rate')} Top3={v.get('conditional_top3_20_rate')} median_rank={v.get('median_positive_rank')}")
    if r.get('best_final_fresh_domain'):
        d = r['best_final_fresh_domain']
        print(f"  final fresh-domain: conditionalTop1={d.get('conditional_top1_20_rate')} Top3={d.get('conditional_top3_20_rate')} median_rank={d.get('median_positive_rank')}")
    print(f"Bootstrap learnability gate: {r.get('bootstrap_learnability_gate_passed', False)}")
    print(f"Domain validated: {r.get('domain_validated', False)}")
    print(f"Research cascade gate: {r.get('research_gate_passed', False)}")
    print(f"Elapsed: {r.get('elapsed_seconds', 0):.1f}s")
    print('Live authority: unchanged / NO')
    return 0 if r.get('status') == 'ok' else 2

if __name__ == '__main__':
    raise SystemExit(main())
