from __future__ import annotations
from pathlib import Path

BLOCK = '''\n\n## V2.23.6 — Registered Evidence Direct Heatmap Localizer\n\nV2.23.6 leaves the global 9k-candidate ranking path as a diagnostic/fallback and advances Source 4/direct proposal work. The same GT-free registered physical evidence maps that gave V2.23.2 dense proposal recall are downsampled into an 8-channel spatial field. A small fully-convolutional localizer learns a heatmap directly over the frame, with GT used only for training labels and metrics. Validation reports Top1/Top3 localisation at 5/10/20/42 px, median/P95 XY error, deterministic evidence-map baselines and an optional dense-snap diagnostic. Hard-negative mining is checkpointed and is rejected automatically if it hurts engineering validation. Fresh F2 remains untouched until model selection; live authority remains NO.\n'''


def patch(path: Path) -> bool:
    if not path.exists(): return False
    text = path.read_text(encoding='utf-8')
    marker = '## V2.23.6 — Registered Evidence Direct Heatmap Localizer'
    if marker in text: return True
    path.write_text(text.rstrip() + BLOCK + '\n', encoding='utf-8')
    return True


def main() -> int:
    for name in ('CURRENT_STATE.md','HIT_DETECTION_PLAN.md','AI_CONTEXT.md'):
        ok = patch(Path(name)); print(f"{'[PASS]' if ok else '[SKIP]'} {name}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
