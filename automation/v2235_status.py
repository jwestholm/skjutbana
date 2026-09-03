from __future__ import annotations
import json
from pathlib import Path
from src.engine.ai.training_v223.evidence_patch_v2235 import discover_evidence_sessions
from src.engine.ai.training_v223.trainer_v2235 import REGISTRY_PATH, REPORT_ROOT, select_evidence_split

def main() -> int:
    print('V2.23.5 STATUS\n==============')
    groups = discover_evidence_sessions(min_shots=1)
    print('Evidence sessions:')
    if not groups:
        print('  none')
    for sid, refs in sorted(groups.items()):
        total = sum(r.cache_path.stat().st_size for r in refs if r.cache_path.exists())
        print(f"  {sid}: evidence_banks={len(refs)} cache={total/1024/1024:.1f}MB")
    split = select_evidence_split()
    print(f"Split: mode={split.mode} train={len(split.train_refs)} validation={len(split.validation_refs)} fresh_domain={len(split.domain_refs)} domain_session={split.domain_session}")
    for note in split.notes:
        print(f"  note: {note}")
    latest = REPORT_ROOT / 'latest.json'
    if latest.exists():
        r = json.loads(latest.read_text(encoding='utf-8'))
        print(f"Latest run: {r.get('run_id')} status={r.get('status')} bootstrap={r.get('bootstrap_learnability_gate_passed')} domain_validated={r.get('domain_validated')} research_gate={r.get('research_gate_passed')}")
        e = (r.get('best_evidence_model') or {}).get('validation', {})
        if e:
            print(f"Evidence validation: R128={e.get('retention20_at_k',{}).get('128')} R512={e.get('retention20_at_k',{}).get('512')} median_rank={e.get('median_positive_rank20')} top1={e.get('conditional_top1_20_rate')}")
        f = (r.get('best_final_ranker') or {}).get('validation', {})
        if f:
            print(f"Final validation: Top1={f.get('conditional_top1_20_rate')} Top3={f.get('conditional_top3_20_rate')} median_rank={f.get('median_positive_rank')}")
    else:
        print('Latest run: none')
    try:
        reg = json.loads(REGISTRY_PATH.read_text(encoding='utf-8'))
    except Exception:
        reg = {}
    print(f"Bootstrap best: {reg.get('bootstrap_best')}")
    print(f"Research evidence champion: {reg.get('research_evidence_champion')}")
    print('Live authority: unchanged / NO')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
