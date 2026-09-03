from __future__ import annotations
import json
from pathlib import Path
from src.engine.ai.training_v223.patch_v2234 import discover_patch_sessions
from src.engine.ai.training_v223.trainer_v2234 import REGISTRY_PATH, REPORT_ROOT, select_patch_split

def _load(path):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:return None

def main()->int:
    groups=discover_patch_sessions(min_shots=1); split=select_patch_split(); latest=_load(REPORT_ROOT/'latest.json'); reg=_load(REGISTRY_PATH) or {}
    print('V2.23.4 STATUS\n==============')
    print('Patch sessions:')
    if not groups: print('  none')
    for sid,refs in sorted(groups.items()):
        total=sum(r.patch_path.stat().st_size for r in refs if r.patch_path.exists())
        print(f"  {sid}: patch_banks={len(refs)} cache={total/(1024**2):.1f}MB")
    print(f"Split: mode={split.mode} train={len(split.train_refs)} validation={len(split.validation_refs)} fresh_domain={len(split.domain_refs)} domain_session={split.domain_session}")
    for note in split.notes: print(f"  note: {note}")
    if latest:
        print(f"Latest run: {latest.get('run_id')} status={latest.get('status')} bootstrap={latest.get('bootstrap_learnability_gate_passed',False)} domain_validated={latest.get('domain_validated',False)} research_gate={latest.get('research_gate_passed',False)}")
        bp=latest.get('best_patch_model',{}).get('validation',{})
        if bp: print(f"Patch validation: R128={bp.get('retention20_at_k',{}).get('128')} R512={bp.get('retention20_at_k',{}).get('512')} median_rank={bp.get('median_positive_rank20')} top1={bp.get('conditional_top1_20_rate')}")
        dp=latest.get('best_patch_fresh_domain')
        if dp: print(f"Patch fresh-domain: R128={dp.get('retention20_at_k',{}).get('128')} R512={dp.get('retention20_at_k',{}).get('512')} median_rank={dp.get('median_positive_rank20')} top1={dp.get('conditional_top1_20_rate')}")
        fv=latest.get('best_final_ranker',{}).get('validation',{})
        if fv: print(f"Final validation: Top1={fv.get('conditional_top1_20_rate')} Top3={fv.get('conditional_top3_20_rate')} median_rank={fv.get('median_positive_rank')}")
    else: print('Latest run: none')
    print(f"Bootstrap best: {reg.get('bootstrap_best')}")
    print(f"Research patch champion: {reg.get('research_patch_champion')}")
    print('Live authority: unchanged / NO')
    return 0
if __name__=='__main__': raise SystemExit(main())
