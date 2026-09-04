from __future__ import annotations
import inspect
from pathlib import Path


def check(cond, msg):
    if not cond: raise AssertionError(msg)
    print(f'[PASS] {msg}')


def main() -> int:
    print('V2.23.6 INSTALL VERIFICATION\n============================')
    required = [
        'src/engine/ai/training_v223/heatmap_v2236.py',
        'src/engine/ai/training_v223/heatmap_model_v2236.py',
        'src/engine/ai/training_v223/trainer_v2236.py',
        'automation/v2236_prepare.py','automation/v2236_train.py','automation/v2236_status.py',
    ]
    check(all(Path(p).exists() for p in required), 'required V2.23.6 files exist')
    from src.engine.ai.training_v223.heatmap_v2236 import HEATMAP_STRIDE, HEATMAP_CHANNELS
    check(HEATMAP_STRIDE == 4 and HEATMAP_CHANNELS == 8, 'direct heatmap geometry imports')
    from src.engine.ai.training_v223.heatmap_model_v2236 import HeatmapModelV2236, peak_camera_xy
    check(hasattr(HeatmapModelV2236,'score_map') and callable(peak_camera_xy), 'full-frame localizer imports')
    from src.engine.ai.training_v223.trainer_v2236 import train_direct_heatmap_v2236
    src = inspect.getsource(train_direct_heatmap_v2236)
    check('objective(stage2_val) > objective(stage1_val)' in src, 'hard-negative checkpoint may be rejected when validation worsens')
    check('domain_metrics = evaluate_heatmap_model(best_model' in src, 'fresh domain is touched only after engineering model selection')
    check('eligible_for_live_authority' in src and 'False' in src, 'V2.23.6 hard-codes no live authority')
    integ = Path('src/engine/ai/training_v223/integration.py').read_text(encoding='utf-8')
    check('schedule_cycle_v2236' in integ and '[V2.23.6]' in integ, 'F2 scheduler points at direct-heatmap shadow cycle')
    from src.engine.offline.direct_proposal_v221 import propose_direct_v221
    check(callable(propose_direct_v221), 'registered evidence map producer is available')
    print('\nV2.23.6 installation verification passed.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
