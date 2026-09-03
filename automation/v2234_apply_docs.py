from __future__ import annotations
from pathlib import Path

MARK='## V2.23.4 — Patch NewHole Candidate Model'
BLOCK='''\n\n## V2.23.4 — Patch NewHole Candidate Model\n\nV2.23.3 established a clear decision point: on one 75/25 bootstrap split the dense proposal pool still had 92% oracle@20, but the best tabular reducer retained only 34.8% of those positives at Top512 and had median positive rank 939. V2.23.4 therefore stops adding hand-tuned scalar features to the first-stage reducer.\n\nV2.23.4 compiles candidate-centred PRE/POST patch banks from the existing V2.23.2 framepacks and dense V2.21.5 proposal coordinates. Each patch contains PRE gray, mean POST gray, amplified absolute change, signed PRE->POST change, and temporal persistence. A dependency-free learned image model (patch MLP and tiny CNN trials) ranks the full dense pool. GT-centred/jittered patches are allowed only as training positives; they are never inserted into candidate pools or proposal metrics.\n\nThe first gate is learnability, not live authority: on the existing one-session bootstrap, target retention is >=80% @512, >=60% @128, and median positive rank <=100. If a second substantial F2 dense session exists, the newest session is held out as fresh-domain validation. Model/trial selection is engineering-validation only; the fresh-domain session is evaluated only after selection. V2.23.4 remains shadow-only and cannot grant live authority.\n'''

def main()->int:
    for name in ('CURRENT_STATE.md','HIT_DETECTION_PLAN.md','AI_CONTEXT.md'):
        p=Path(name)
        if not p.exists(): print(f'[SKIP] {name}: not found');continue
        s=p.read_text(encoding='utf-8')
        if MARK in s: print(f'[PASS] {name}: section already present');continue
        p.write_text(s.rstrip()+BLOCK+'\n',encoding='utf-8');print(f'[PASS] {name}: V2.23.4 section appended')
    return 0
if __name__=='__main__': raise SystemExit(main())
