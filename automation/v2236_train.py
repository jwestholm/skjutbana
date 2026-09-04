from __future__ import annotations
import argparse
from src.engine.ai.training_v223.trainer_v2236 import train_direct_heatmap_v2236


def main() -> int:
    ap = argparse.ArgumentParser(description='Train V2.23.6 direct registered-evidence heatmap localizer')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--no-prepare', action='store_true')
    args = ap.parse_args()
    report = train_direct_heatmap_v2236(quick=args.quick, prepare=not args.no_prepare)
    print('\nV2.23.6 TRAIN SUMMARY\n=====================')
    print(f"Status: {report.get('status')}")
    split = report.get('split', {})
    print(f"Split: mode={split.get('mode')} train={split.get('train',0)} validation={split.get('validation',0)} fresh_domain={split.get('fresh_domain',0)}")
    bb = report.get('best_validation_baseline', {})
    bm = bb.get('metrics') or {}
    print(f"Best deterministic baseline: {bb.get('name')} Top1@20={bm.get('top1_at20')} Top3@20={bm.get('top3_at20')} median={bm.get('median_error_px')}")
    best = report.get('best_model', {})
    vm = best.get('validation') or {}
    print(f"Best heatmap model: kind={best.get('kind')} hidden={best.get('hidden')} stage={best.get('chosen_stage')}")
    print(f"  validation: Top1@5={vm.get('top1_at5')} Top1@10={vm.get('top1_at10')} Top1@20={vm.get('top1_at20')} Top1@42={vm.get('top1_at42')} Top3@20={vm.get('top3_at20')} median={vm.get('median_error_px')} p95={vm.get('p95_error_px')}")
    print(f"  dense-snap diagnostic: Top1@20={vm.get('snap_top1_at20')} Top3@20={vm.get('snap_top3_at20')} median={vm.get('snap_median_error_px')}")
    dm = report.get('fresh_domain') or {}
    if dm:
        print(f"Fresh domain: Top1@20={dm.get('top1_at20')} Top3@20={dm.get('top3_at20')} median={dm.get('median_error_px')}")
    pm = report.get('selected_policy_validation') or {}
    print(f"Selected direct policy: {report.get('selected_policy')} Top1@20={pm.get('top1_at20')} Top3@20={pm.get('top3_at20')} median={pm.get('median_error_px')}")
    print(f"Direct-path gate: {report.get('direct_path_gate')}")
    print(f"Bootstrap signal: {report.get('bootstrap_signal')}")
    print(f"Bootstrap learnability gate: {report.get('bootstrap_learnability_gate')}")
    print(f"Domain validated: {report.get('domain_validated')}")
    print(f"Research heatmap gate: {report.get('research_gate_passed')}")
    print(f"Elapsed: {float(report.get('elapsed_seconds',0)):.1f}s")
    print('Live authority: unchanged / NO')
    return 0 if report.get('status') == 'ok' else 1

if __name__ == '__main__':
    raise SystemExit(main())
