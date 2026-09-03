from __future__ import annotations
import inspect
from pathlib import Path

def check(cond,msg):
    if not cond: raise AssertionError(msg)
    print(f"[PASS] {msg}")

def main()->int:
    print('V2.23.4 INSTALL VERIFICATION\n============================')
    required=[
        'src/engine/ai/training_v223/patch_v2234.py','src/engine/ai/training_v223/patch_model_v2234.py',
        'src/engine/ai/training_v223/trainer_v2234.py','automation/v2234_train.py','automation/v2234_prepare.py'
    ]
    check(all(Path(p).exists() for p in required),'required V2.23.4 files exist')
    from src.engine.ai.training_v223.patch_v2234 import PATCH_CHANNEL_NAMES,PATCH_SIZE
    check(len(PATCH_CHANNEL_NAMES)==5 and PATCH_SIZE==16,'patch image contract imports')
    from src.engine.ai.training_v223.patch_model_v2234 import PatchModelV2234
    check(hasattr(PatchModelV2234,'score_patches'),'learned patch model imports')
    from src.engine.ai.training_v223.trainer_v2234 import train_patch_cascade_v2234,select_patch_split
    src=inspect.getsource(train_patch_cascade_v2234)
    check('best_idx=max' in src and 'domain_patch = evaluate_patch_model(best_patch' in src,'fresh-domain is evaluated after validation-only patch-model selection')
    check('eligible_for_live_authority":False' in src or 'eligible_for_live_authority": False' in src,'V2.23.4 hard-codes no live authority')
    integ=Path('src/engine/ai/training_v223/integration.py').read_text(encoding='utf-8')
    check('schedule_cycle_v2234' in integ and '[V2.23.4]' in integ,'F2 scheduler points to V2.23.4 shadow cycle')
    from src.engine.offline.physical_dense_v2215 import propose_dense_pool_v2215
    check(callable(propose_dense_pool_v2215),'V2.21.5 dense teacher remains available')
    print('\nV2.23.4 installation verification passed.')
    return 0
if __name__=='__main__': raise SystemExit(main())
