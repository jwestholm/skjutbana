from __future__ import annotations
from pathlib import Path
import importlib

REQUIRED = [
    'src/engine/ai/training_v223/rich_v2233.py',
    'src/engine/ai/training_v223/dense_v2233.py',
    'src/engine/ai/training_v223/reducer_v2233.py',
    'src/engine/ai/training_v223/trainer_v2233.py',
    'automation/v2233_prepare.py',
    'automation/v2233_train.py',
    'automation/v2233_cycle.py',
    'automation/v2233_status.py',
    'automation/v2233_selftest.py',
]

def check(cond: bool, msg: str) -> None:
    if not cond: raise AssertionError(msg)
    print(f'[PASS] {msg}')

def main() -> None:
    print('V2.23.3 INSTALL VERIFICATION')
    print('============================')
    check(all(Path(p).exists() for p in REQUIRED),'required V2.23.3 files exist')
    schema=importlib.import_module('src.engine.ai.training_v223.schema')
    check(str(schema.SCHEMA_VERSION)=='2.23.3' and 'v2233_newhole_heuristic' in schema.FEATURE_NAMES,'schema exposes rich GT-free V2.23.3 evidence')
    dense=importlib.import_module('src.engine.ai.training_v223.dense_v2233')
    check(len(dense.REDUCER_FEATURE_NAMES)>=30,'compact reducer feature contract imports')
    reducer=importlib.import_module('src.engine.ai.training_v223.reducer_v2233')
    check(hasattr(reducer,'train_reducer') and hasattr(reducer,'evaluate_reducer'),'pairwise reducer imports')
    trainer=importlib.import_module('src.engine.ai.training_v223.trainer_v2233')
    check(hasattr(trainer,'cycle_v2233') and hasattr(trainer,'schedule_cycle_v2233'),'offline/F2 unified V2.23.3 cycle imports')
    physical=importlib.import_module('src.engine.offline.physical_dense_v2215')
    check(hasattr(physical,'propose_dense_pool_v2215') and hasattr(physical,'extract_candidate_features_v2215'),'V2.21.5 dense teacher API remains available')
    integration=Path('src/engine/ai/training_v223/integration.py').read_text(encoding='utf-8')
    check('schedule_cycle_v2233' in integration and 'V2.23.3' in integration,'F2 completion schedules V2.23.3 shadow cycle')
    main_text=Path('main.py').read_text(encoding='utf-8')
    check(main_text.index('install_v2226_runtime(App)') < main_text.index('install_v2230_training_pipeline()'),'frozen V2.22.6 runtime chain still precedes AI training integration')
    source='\n'.join(Path(p).read_text(encoding='utf-8') for p in REQUIRED if p.endswith('.py'))
    check('eligible_for_live_authority = True' not in source and 'live_authority": True' not in source,'V2.23.3 does not grant live authority')
    print('\nV2.23.3 installation verification passed.')

if __name__=='__main__': main()
