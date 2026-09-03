from __future__ import annotations
from pathlib import Path

SECTION='''\n\n## V2.23.3 — Learned candidate reduction + rich NEW-hole ranking\n\nV2.23.2 established on a fresh 100-shot F2 projector/camera session that the heavy offline dense V2.21.5 pool reaches **95% oracle <=20px** and **96% <=42px**, while current/live was only 6% <=20px and local 15%. The physical signal therefore exists; ranking/reduction is now the dominant problem.\n\nV2.23.3 keeps V2.22 live authority frozen and adds GT-free full-frame PRE/POST evidence maps sampled at every dense candidate, compact numeric NPZ caches, a pairwise learned reducer, top-K retention metrics, and a reducer→final-ranker cascade. Candidates 20..42px from GT are neutral during reducer training rather than incorrect hard negatives.\n\nWith only one substantial dense-expanded F2 session, training is explicitly `single_session_bootstrap` (deterministic ~80/20 shot split) and may prove learnability only. No domain/generalisation claim is permitted. Once a second independent dense F2 session exists, the newest session is reserved untouched as fresh-domain validation. Research-shadow gating then requires high reducer retention plus non-trivial final Top1 performance. Live authority remains NO.\n'''

def main() -> None:
    for name in ('CURRENT_STATE.md','HIT_DETECTION_PLAN.md','AI_CONTEXT.md'):
        p=Path(name)
        if not p.exists():
            print(f'[SKIP] {name}: not present')
            continue
        text=p.read_text(encoding='utf-8')
        if '## V2.23.3 — Learned candidate reduction + rich NEW-hole ranking' in text:
            print(f'[PASS] {name}: section already present')
            continue
        p.write_text(text.rstrip()+SECTION+'\n',encoding='utf-8')
        print(f'[PASS] {name}: V2.23.3 section appended')

if __name__=='__main__': main()
