from __future__ import annotations
from pathlib import Path

MARK = '## V2.23.5 — Registered Evidence Patch Ranker'
BLOCK = '''\n\n## V2.23.5 — Registered Evidence Patch Ranker\n\nV2.23.4 falsified the raw-PRE/POST patch approach on the one-session bootstrap: dense proposal oracle@20 remained 92%, but the best raw patch model retained only 30.4% @512, 13.0% @128 and had median positive rank 949. The tiny CNN trials were worse. V2.23.5 therefore does not add more raw-image capacity.\n\nV2.23.5 trains on candidate-centred patches from the same registered and photometrically compensated V2.21 physical evidence maps that feed the successful dense teacher: blackhat/tophat gain, persistent absolute/dark/bright change, gradient gain, fused evidence and compact change. Candidate labels are tightened for patch learning: <=6px is positive, 6..42px is neutral, and >42px is negative. GT-centred/jittered anchors remain training-only and never enter proposal/oracle metrics.\n\nThe learner uses iterative model-hard-negative mining: an initial physical/dense/random negative set is trained, the model scores every training candidate, its highest-scoring false candidates are mined, and two additional training stages focus on those mistakes. The bootstrap gate is retention@512 >=70%, retention@128 >=45% and median positive rank <=200. A newer substantial F2 session remains untouched fresh-domain validation. V2.23.5 is shadow-only and cannot grant live authority.\n'''

def main() -> int:
    for name in ('CURRENT_STATE.md', 'HIT_DETECTION_PLAN.md', 'AI_CONTEXT.md'):
        p = Path(name)
        if not p.exists():
            print(f'[SKIP] {name}: not found')
            continue
        s = p.read_text(encoding='utf-8')
        if MARK in s:
            print(f'[PASS] {name}: section already present')
            continue
        p.write_text(s.rstrip() + BLOCK + '\n', encoding='utf-8')
        print(f'[PASS] {name}: V2.23.5 section appended')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
