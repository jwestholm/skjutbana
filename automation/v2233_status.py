from __future__ import annotations
import json
from pathlib import Path
from src.engine.ai.training_v223.rich_v2233 import discover_proposal_sessions
from src.engine.ai.training_v223.dense_v2233 import discover_cached_sessions
from src.engine.ai.training_v223.trainer_v2233 import ROOT, REGISTRY_PATH, REPORT_ROOT, select_dense_split

def main() -> None:
    proposals=discover_proposal_sessions(); cached=discover_cached_sessions(min_shots=1); split=select_dense_split()
    print('V2.23.3 STATUS')
    print('==============')
    print('Proposal sessions:')
    for sid, paths in sorted(proposals.items()):
        rich=sum(int(p.with_name(p.stem+'.rich_v2233.npz').exists()) for p in paths)
        cache=len(cached.get(sid,[]))
        print(f'  {sid}: proposals={len(paths)} rich={rich} numeric_cache={cache}')
    print(f"Split: mode={split.mode} train={len(split.train_refs)} validation={len(split.validation_refs)} fresh_domain={len(split.domain_refs)} domain_session={split.domain_session}")
    for note in split.notes: print(f'  note: {note}')
    latest=REPORT_ROOT/'latest.json'
    if latest.exists():
        try:
            r=json.loads(latest.read_text())
            print(f"Latest run: {r.get('run_id')} status={r.get('status')} domain_validated={r.get('domain_validated')} gate={r.get('research_gate_passed')}")
            br=r.get('best_reducer',{})
            if br:
                print(f"Reducer validation: {br.get('validation')}")
                if br.get('fresh_domain') is not None: print(f"Reducer fresh-domain: {br.get('fresh_domain')}")
            bf=r.get('best_final_ranker',{})
            if bf:
                print(f"Final validation: {bf.get('validation')}")
                if bf.get('fresh_domain') is not None: print(f"Final fresh-domain: {bf.get('fresh_domain')}")
        except Exception as exc: print(f'Latest report unreadable: {exc}')
    else: print('Latest V2.23.3 run: none')
    if REGISTRY_PATH.exists():
        try:
            reg=json.loads(REGISTRY_PATH.read_text()); print(f"Research cascade champion: {reg.get('research_cascade_champion')}")
        except Exception: pass
    print('Live authority: unchanged / NO')

if __name__=='__main__': main()
