from __future__ import annotations
import inspect
from pathlib import Path

def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f'[PASS] {msg}')

def main() -> int:
    print('V2.23.5 INSTALL VERIFICATION\n============================')
    required = [
        'src/engine/ai/training_v223/evidence_patch_v2235.py',
        'src/engine/ai/training_v223/evidence_model_v2235.py',
        'src/engine/ai/training_v223/trainer_v2235.py',
        'automation/v2235_prepare.py', 'automation/v2235_train.py', 'automation/v2235_status.py',
    ]
    check(all(Path(p).exists() for p in required), 'required V2.23.5 files exist')
    from src.engine.ai.training_v223.evidence_patch_v2235 import EVIDENCE_CHANNEL_NAMES, PATCH_POSITIVE_RADIUS_PX, PATCH_NEGATIVE_RADIUS_PX
    check(tuple(EVIDENCE_CHANNEL_NAMES) == ('blackhat_gain','tophat_gain','persistent_abs','gradient_gain','persistent_dark','persistent_bright','fused','compact_change'), 'V2.23.5 uses V2.21 registered physical evidence families')
    check(PATCH_POSITIVE_RADIUS_PX == 6.0 and PATCH_NEGATIVE_RADIUS_PX == 42.0, 'tight patch labels import')
    from src.engine.ai.training_v223.evidence_model_v2235 import EvidenceModelV2235, mine_hard_negatives
    check(hasattr(EvidenceModelV2235, 'score_patches') and callable(mine_hard_negatives), 'evidence learner + model-hard-negative miner import')
    from src.engine.ai.training_v223.trainer_v2235 import train_registered_evidence_cascade_v2235
    src = inspect.getsource(train_registered_evidence_cascade_v2235)
    check('best_idx = max' in src and 'domain_evidence = evaluate_evidence_model(best_model' in src, 'fresh-domain is evaluated only after engineering model selection')
    check('eligible_for_live_authority' in src and 'False' in src, 'V2.23.5 hard-codes no live authority')
    integ = Path('src/engine/ai/training_v223/integration.py').read_text(encoding='utf-8')
    check('schedule_cycle_v2235' in integ and '[V2.23.5]' in integ, 'F2 scheduler points to V2.23.5 shadow cycle')
    from src.engine.offline.direct_proposal_v221 import propose_direct_v221
    from src.engine.offline.physical_dense_v2215 import propose_dense_pool_v2215
    check(callable(propose_direct_v221) and callable(propose_dense_pool_v2215), 'registered-map producer and V2.21.5 dense teacher remain available')
    print('\nV2.23.5 installation verification passed.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
