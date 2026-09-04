from __future__ import annotations
import json
from pathlib import Path
from src.engine.ai.training_v223.heatmap_v2236 import HEATMAP_CACHE_ROOT, discover_heatmap_sessions
from src.engine.ai.training_v223.trainer_v2236 import REGISTRY_PATH, REPORT_ROOT, select_heatmap_split


def _json(path: Path):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except Exception: return {}


def main() -> int:
    print('V2.23.6 STATUS\n==============')
    groups = discover_heatmap_sessions(min_shots=1)
    print('Heatmap sessions:')
    if not groups: print('  none')
    for sid, refs in groups.items():
        size = sum(r.cache_path.stat().st_size for r in refs if r.cache_path.exists()) / 1024 / 1024
        print(f'  {sid}: heatmaps={len(refs)} cache={size:.1f}MB')
    split = select_heatmap_split()
    print(f'Split: mode={split.mode} train={len(split.train_refs)} validation={len(split.validation_refs)} fresh_domain={len(split.domain_refs)} domain_session={split.domain_session}')
    for note in split.notes: print(f'  note: {note}')
    latest = _json(REPORT_ROOT / 'latest.json')
    if latest:
        print(f"Latest run: {latest.get('run_id')} status={latest.get('status')} signal={latest.get('bootstrap_signal')} direct_gate={latest.get('direct_path_gate')} learned_gate={latest.get('bootstrap_learnability_gate')} policy={latest.get('selected_policy')} domain_validated={latest.get('domain_validated')} research_gate={latest.get('research_gate_passed')}")
        bb = latest.get('best_validation_baseline', {}); bm = bb.get('metrics') or {}
        print(f"Baseline validation: {bb.get('name')} Top1@20={bm.get('top1_at20')} Top3@20={bm.get('top3_at20')} median={bm.get('median_error_px')}")
        best = latest.get('best_model', {}); vm = best.get('validation') or {}
        print(f"Heatmap validation: kind={best.get('kind')} stage={best.get('chosen_stage')} Top1@20={vm.get('top1_at20')} Top3@20={vm.get('top3_at20')} median={vm.get('median_error_px')} p95={vm.get('p95_error_px')}")
        pm = latest.get('selected_policy_validation') or {}
        print(f"Selected policy validation: Top1@20={pm.get('top1_at20')} Top3@20={pm.get('top3_at20')} median={pm.get('median_error_px')}")
        if latest.get('fresh_domain'):
            dm = latest['fresh_domain']
            print(f"Fresh domain: Top1@20={dm.get('top1_at20')} Top3@20={dm.get('top3_at20')} median={dm.get('median_error_px')}")
    reg = _json(REGISTRY_PATH)
    print(f"Bootstrap best: {reg.get('bootstrap_best')}")
    print(f"Research direct policy: {reg.get('research_direct_policy')}")
    print(f"Research heatmap champion: {reg.get('research_heatmap_champion')}")
    print('Live authority: unchanged / NO')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
