"""Package saved experiment results; never loads or scores holdout samples."""
import argparse
import shlex
import tarfile
from pathlib import Path

from automation.offline_10iter import ROOT, read, write
from src.engine.offline.evaluation import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    report=read(out/'experiment_report.json')
    best=report['iterations'][report['best_development_iteration']]
    base=report['iterations'][0]
    hold=report['protected_holdout']
    dev_delta=best['metrics']['tolerances']['20']['top1']-base['metrics']['tolerances']['20']['top1']
    hold_delta=hold['challenger']['tolerances']['20']['top1']-hold['baseline']['tolerances']['20']['top1']
    report['conclusion']={
        'development_primary_delta':best['metrics']['tolerances']['20']['top1']-base['metrics']['tolerances']['20']['top1'],
        'holdout_primary_delta':hold['challenger']['tolerances']['20']['top1']-hold['baseline']['tolerances']['20']['top1'],
        'physical_live_improvement_established':False,
        'interpretation':f"Development Top1@20 delta: {dev_delta:+d}; holdout delta: {hold_delta:+d}. "
            + ("No primary accuracy improvement was achieved. " if dev_delta <= 0 and hold_delta <= 0 else "")
            + f"Holdout MRR20: {hold['baseline']['tolerances']['20']['mrr_all']:.6f} -> {hold['challenger']['tolerances']['20']['mrr_all']:.6f}; "
            + f"Top10@20: {hold['baseline']['tolerances']['20']['top10']} -> {hold['challenger']['tolerances']['20']['top10']}; "
            + f"Top1@42: {hold['baseline']['tolerances']['42']['top1']} -> {hold['challenger']['tolerances']['42']['top1']}. "
            + "The selected iteration is NOT approved for live use. Projected F2 data does not establish physical/live performance.",
        'holdout_scored_once_per_configuration':True,
        'largest_failure':f"Missing saved candidates within 20px: development {best['metrics']['samples']-best['metrics']['tolerances']['20']['oracle']}/{best['metrics']['samples']}; "
            + f"holdout {hold['challenger']['samples']-hold['challenger']['tolerances']['20']['oracle']}/{hold['challenger']['samples']}.",
        'statistical_limit':'One small development session and one holdout session; ten model comparisons; insufficient evidence for a robust ranking gain.'}
    source=[ROOT/'automation'/f for f in ('offline_10iter.py','offline_10iter_prepare.py','offline_10iter_selftest.py','offline_10iter_report.py')]
    report['experiment_source_files']=[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in source]
    with tarfile.open(out/'experiment_source.tar.gz','x:gz') as archive:
        for p in source:archive.add(p,arcname=str(p.relative_to(ROOT)))
    write(out/'final_report.json',report)
    lines=['# Offline ten-iteration experiment','',
        f"Evaluated shots: development {base['metrics']['samples']}; holdout {hold['baseline']['samples']}. MRR includes zero for shots without a candidate at the tolerance. No live/physical inference.",'',
        '| Iteration | Hypothesis | Top1@20 | Top3@20 | Top10@20 | Oracle@20 | MRR20 | Delta Top1 | Decision |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in report['iterations']:
        m=r['metrics']['tolerances']['20']
        lines.append(f"| {r['iteration']} | {r['hypothesis']} | {m['top1']} | {m['top3']} | {m['top10']} | {m['oracle']} | {m['mrr_all']:.6f} | {m['top1']-base['metrics']['tolerances']['20']['top1']:+d} | {r['decision']} |")
    lines+=['','## Original development baseline, selected development challenger, and final holdout','',
            '| Split / policy | Tolerance px | Top1 | Top3 | Top10 | Oracle | MRR (all) | Mean positive rank |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,metrics in (('Development baseline',base['metrics']),(f"Development iteration {best['iteration']}",best['metrics']),
                          ('Holdout baseline',hold['baseline']),('Holdout challenger',hold['challenger'])):
        for t in ('5','10','20','42'):
            m=metrics['tolerances'][t]
            mean_rank=m['mean_rank_when_present']
            lines.append(f"| {label} | {t} | {m['top1']} | {m['top3']} | {m['top10']} | {m['oracle']} | {m['mrr_all']:.6f} | {mean_rank} |")
    lines+=['',report['conclusion']['interpretation'],'',report['conclusion']['largest_failure'],'',
            'Protected here means withheld from this optimization loop. Historical aggregate pool recall was already exposed in earlier work. All sessions share date/camera/background; absent imagery prevents visual duplicate certification.']
    (out/'report.md').open('x').write('\n'.join(lines)+'\n')
    commands=['#!/usr/bin/env bash','set -euo pipefail',
              'experiment_output="${1:?Supply a NEW output directory}"',
              'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1',
              'python3 -m automation.offline_10iter_prepare --output "$experiment_output"',
              'python3 -m automation.offline_10iter baseline --output "$experiment_output"']
    import json
    for r in report['iterations'][1:]:
        commands.append('python3 -m automation.offline_10iter trial --output "$experiment_output" '+
            f"--iteration {r['iteration']} --parent {r['parent']} --hypothesis {shlex.quote(r['hypothesis'])} --change {shlex.quote(json.dumps(r['change']))}")
    commands+=['python3 -m automation.offline_10iter finalize --output "$experiment_output"',
               'python3 -m automation.offline_10iter_report --output "$experiment_output"']
    (out/'reproduce.sh').open('x').write('\n'.join(commands)+'\n')


if __name__=='__main__':main()
