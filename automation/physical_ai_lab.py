"""Resumable offline physical research. Never activates or captures a model.

Extends V2.23's model/negative miner and cached shared physical witnesses.
Completed trial files are the journal authority; indexes are rebuildable.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import numpy as np

from automation import physical_ai_dataset as ds
from automation.ai_training_results import get_git_commit, utc_iso
from src.engine.ai.canonical_challenger import transform_matrix
from src.engine.ai.training_v223.model import RankModelV223, train_fusion_model

ROOT = Path('/data/skjutbana/physical_ai_lab')
CODE_FILES = ('automation/physical_ai_dataset.py', 'automation/physical_ai_lab.py',
              'src/engine/ai/training_v223/model.py',
              'src/engine/ai/training_v223/evidence_model_v2235.py',
              'src/engine/ai/canonical_challenger.py')


def code_identity():
    repo=Path(__file__).resolve().parents[1]
    return dict(revision=get_git_commit(), source_hashes={x:ds.digest(repo/x) for x in CODE_FILES})


def location(root, name):
    ds.guard(name)
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',name): raise ValueError('Invalid experiment name')
    return ds.output_root(root)/name


@contextmanager
def exclusive(root):
    with (root/'.lock').open('a') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Experiment already owned by another worker')
        yield


def config_for(seed, trial):
    rng=np.random.default_rng(np.random.SeedSequence([seed,trial]))
    return dict(seed=int(rng.integers(1,2**30)), hidden=int(rng.choice([12,24,48,64])),
                epochs=int(rng.choice([12,24,48,72])), learning_rate=float(10**rng.uniform(-3.5,-1.8)),
                l2=float(10**rng.uniform(-5,-2)), threshold=float(rng.choice([.35,.5,.7,.9])),
                hard_negative_count=int(rng.choice([0,4,8])), excluded_families=[])


def columns(config):
    excluded=set(config['excluded_families'])
    if not excluded <= set(ds.FAMILIES): raise ValueError('Unknown ablation family')
    names=[n for n in ds.RAW_NAMES if not any(n in ds.FAMILIES[f] for f in excluded)]
    names+= [n+'__missing' for n in names]
    return names,[ds.FEATURE_NAMES.index(n) for n in names]


def samples(events,x):
    out=[]
    for e in events:
        if e['truth']=='UNKNOWN': continue
        distances=[ds.distance(c['xy'],e['gt_xy']) for c in e['candidates']] if e['truth']=='HIT' else [float('inf')]*len(e['indices'])
        out.append((x[e['indices']],np.asarray(distances)))
    return out


def predict(model, events, x, threshold):
    predictions={}; detail=[]
    for e in events:
        scores=model.score_matrix(x[e['indices']])
        if not np.isfinite(scores).all(): raise ValueError('Nonfinite model scores')
        prob=1/(1+np.exp(-np.clip(scores,-40,40)))
        j=int(np.argmax(scores)) if len(scores) else None
        xy=e['candidates'][j]['xy'] if j is not None and prob[j]>=threshold else None
        predictions[e['event']]=xy
        detail.append(dict(event=e['event'], selected_xy=xy, scores=scores.tolist(),
                           uncalibrated_sigmoid=prob.tolist(),
                           candidate_ids=[c['id'] for c in e['candidates']]))
    return predictions,detail


def train_trial(directory, data, x, cfg, stop, *, training_filter=None):
    names,keep=columns(cfg); x=transform_matrix(x[:,keep],'signed_log')
    predictions={}; fold_reports=[]
    for i,fold in enumerate(data['folds']):
        out=directory/f'fold_{i:02d}'; out.mkdir(exist_ok=True)
        done=out/'complete.json'
        if done.exists():
            saved=ds.read(done)
            if saved['config']!=cfg or saved['dataset_hash']!=data['dataset_hash']:
                raise ValueError('Fold cache identity changed')
            for name,h in saved['model_hashes'].items():
                if ds.digest(out/name)!=h: raise ValueError('Fold model changed')
        else:
            if stop(): raise InterruptedError('budget/signal')
            train=[e for e in data['events'] if e['session'] in fold['train']]
            dev=[e for e in data['events'] if e['session'] in fold['dev']]
            held=[e for e in data['events'] if e['session'] in fold['held']]
            if training_filter is not None:
                train=training_filter(train,fold)
            kw={k:cfg[k] for k in ('hidden','epochs','learning_rate','l2','hard_negative_count')}
            model,history=train_fusion_model(samples(train,x), feature_names=names,seed=cfg['seed']+i,
                checkpoint=out/'training_checkpoint.npz',
                checkpoint_identity=dict(dataset=data['dataset_hash'],fold=fold,excluded=cfg['excluded_families']),
                stop=stop,**kw)
            model.metadata.update(dataset_hash=data['dataset_hash'],fold=fold,
                                  feature_schema=ds.SCHEMA,transform='signed_log',vision_features=False)
            model.save(out)
            pred,detail=predict(model,held,x,cfg['threshold'])
            dev_pred,_=predict(model,dev,x,cfg['threshold'])
            saved=dict(config=cfg,dataset_hash=data['dataset_hash'],fold=fold,predictions=pred,
                       detail=detail,training=history,dev_metrics=ds.evaluate(dev,dev_pred),
                       training_rows={e['event']:len(e['indices']) for e in train},
                       model_hashes={n:ds.digest(out/n) for n in ('model.npz','model.json')})
            ds.atomic_json(done,saved)
        if set(predictions)&set(saved['predictions']): raise ValueError('Duplicate outer prediction')
        predictions.update(saved['predictions']); fold_reports.append(saved)
    return dict(metrics=ds.evaluate(data['events'],predictions), predictions=predictions,folds=fold_reports)


def champion_compare(challenger, incumbent):
    """Research-only policy. Guardrails precede aggregate ranking."""
    if challenger['protected_failures']:
        return False,'protected CURRENT-correct @42 event lost'
    old={r['event']:r for r in incumbent['per_event']}
    new_cats=[r['event'] for r in challenger['per_event'] if r['error'] is not None and r['error']>100
              and (old[r['event']]['error'] is None or old[r['event']]['error']<=100)]
    if new_cats: return False,'new catastrophic >100px event: '+','.join(new_cats)
    if challenger['control_false_emissions']>incumbent['control_false_emissions']:
        return False,'control false emissions increased'
    def order(m):
        return (m['selected']['5'], -m['catastrophic_gt100'], m['worst_session_selected5'] or 0,
                m['selected']['10'], -m['control_false_emissions'], -m['unknown'],
                -(m['median_error'] if m['median_error'] is not None else float('inf')),
                -(m['p95_error'] if m['p95_error'] is not None else float('inf')))
    win=order(challenger)>order(incumbent)
    return win,'strict lexicographic improvement after guardrails' if win else 'no strict research-policy improvement'


def completed(root):
    # Compact result records omit per-fold model predictions/training histories.
    # Old/incomplete summaries are reconstructed from their authoritative result.
    rows=[]
    for p in sorted((root/'trials').glob('trial_*/result.json')):
        summary=p.with_name('summary.json')
        if summary.exists(): rows.append(ds.read(summary))
        else:
            r=ds.read(p); rows.append({k:r[k] for k in ('trial','status','metrics','finished_at','error') if k in r})
    return rows


def champion_entry(root, trial, metrics, data, reason):
    return dict(trial=trial,reason=reason,metrics=metrics,model_directory=str(root/'trials'/trial),
                dataset_hash=data['dataset_hash'],live_authority=False)


def publish(root, result, data, *, compare=champion_compare):
    """O(1) trial publication; full leaderboard is rebuilt at durable boundaries."""
    current=ds.read(root/'champion.json'); accepted=False
    why=result.get('error','trial failed')
    if result['status']=='COMPLETE':
        accepted,why=compare(result['metrics'],current['metrics'])
        if accepted:
            entry=champion_entry(root,result['trial'],result['metrics'],data,why)
            path=root/'champions'/(result['trial']+'.json')
            if path.exists() and ds.read(path)!=entry: raise ValueError('Champion overwrite refused')
            if not path.exists(): ds.atomic_json(path,entry)
            current.update(champion=result['trial'],metrics=result['metrics'],history=current['history']+[result['trial']])
            ds.atomic_json(root/'champion.json',current)
    compact=dict(trial=result['trial'],status=result['status'],accepted=accepted,reason=why,
                 selected=result.get('metrics',{}).get('selected'),finished_at=result['finished_at'])
    # This live progress stream is advisory. result.json is the journal authority;
    # an abrupt partial append never controls resume/champion decisions.
    with (root/'progress.jsonl').open('a') as f:
        f.write(json.dumps(compact,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
    return current


def refresh(root, data, *, compare=champion_compare):
    baseline=ds.evaluate(data['events'],{e['event']:e['baseline_xy'] for e in data['events']})
    incumbent=baseline; champion='frozen_exact_witness_baseline'; history=[]; leaderboard=[]
    records=completed(root)
    for r in records:
        if r['status']!='COMPLETE':
            leaderboard.append(dict(trial=r['trial'],status=r['status'],reason=r.get('error'))); continue
        win,why=compare(r['metrics'],incumbent)
        leaderboard.append(dict(trial=r['trial'],status=r['status'],metrics=r['metrics'],accepted=win,reason=why))
        if win:
            champion=r['trial']; incumbent=r['metrics']
            entry=champion_entry(root,champion,incumbent,data,why)
            dst=root/'champions'/(champion+'.json')
            if dst.exists():
                if ds.read(dst)!=entry: raise ValueError('Immutable champion record changed')
            else: ds.atomic_json(dst,entry)
            history.append(champion)
    summary=dict(champion=champion,metrics=incumbent,history=history,live_authority=False,
                 policy='Selected@5 primary; protected/catastrophe/control guardrails; then worst session/@10/coverage/tails',
                 repeated_cv_is_development=True)
    ds.atomic_json(root/'champion.json',summary)
    ds.atomic_json(root/'leaderboard.json',leaderboard)
    successes=[r for r in records if r['status']=='COMPLETE']
    ranked=sorted(successes,key=lambda r:(r['metrics']['selected']['5'],r['metrics']['worst_session_selected5'] or 0),reverse=True)[:5]
    failure_sets=[{q['event'] for q in r['metrics']['per_event'] if q['truth']=='HIT' and (q['error'] is None or q['error']>5)} for r in ranked]
    common=sorted(set.intersection(*failure_sets)) if failure_sets else []
    milestones={str(n):max(r['metrics']['selected']['5'] for r in successes[:n]) for n in (100,1000,len(successes)) if n and len(successes)>=n}
    last=next((r['finished_at'] for r in reversed(successes) if r['trial']==champion),None)
    ds.atomic_json(root/'failure_wall.json',dict(completed_trials=len(successes),best_after=milestones,
                   top_model_common_failures=common,top_model_count=len(ranked),
                   seconds_since_champion_improvement=time.time()-last if last else None,
                   metadata={e['event']:dict(session=e['session'],truth=e['truth'],status=e['status'],caveats=e.get('caveats',[])) for e in data['events'] if e['event'] in common},
                   interpretation='Repeated small development set can overfit; plateau is not evidence of physical sufficiency'))
    # Rebuildable journal derived from atomic result files; no partial append recovery.
    ds.atomic_json(root/'journal.json',[{k:r[k] for k in ('trial','status','finished_at','error') if k in r} for r in records])
    return summary


def run(root,dataset,*,hours,seed=1729,max_trials=None,resume=False,adapter=None):
    # Same lifecycle for evidence and visual proposal trials; role-specific
    # data/model/scoreboards are supplied by the existing Vision module.
    identity_fn=adapter.code_identity if adapter else code_identity
    load_fn=adapter.load if adapter else ds.load
    refresh_fn=adapter.refresh if adapter else refresh
    publish_fn=adapter.publish if adapter else publish
    config_fn=adapter.config_for if adapter else config_for
    train_fn=adapter.train_trial if adapter else train_trial
    ds.guard(root)
    if dataset is not None: ds.guard(dataset)
    if hours is not None and (hours<=0 or not np.isfinite(hours)): raise ValueError('hours must be positive')
    if max_trials is not None and max_trials<1: raise ValueError('max_trials must be positive')
    identity=identity_fn()
    if not resume:
        root.mkdir(parents=True,exist_ok=False)
    elif not root.is_dir(): raise ValueError('Unknown experiment')
    with exclusive(root):
        if resume:
            manifest=ds.read(root/'manifest.json'); dataset=Path(manifest['dataset'])
            if manifest['code']!=identity: raise ValueError('Code changed: use a new experiment, not mixed-code resume')
            if manifest.get('active_started_at') is not None:
                # Abrupt kill cannot flush finally. Conservatively charge wall
                # time since the durable heartbeat, including downtime.
                manifest['spent_seconds']+=max(0,time.time()-manifest['active_started_at'])
            if hours is not None: manifest['budget_hours']+=hours
            if max_trials is not None: manifest['max_trials']=max_trials
        else:
            manifest=dict(schema='physical-ai-lab-1',dataset=str(Path(dataset).resolve()),code=identity,
                          seed=seed,budget_hours=hours,max_trials=max_trials,spent_seconds=0,
                          created=utc_iso(),live_authority=False,vision_features=False,
                          runtime=dict(python=platform.python_version(),numpy=np.__version__,machine=platform.machine(),
                                       cpu_count=os.cpu_count(),device='CPU NumPy',threads=os.environ.get('OPENBLAS_NUM_THREADS','environment default')))
        data,x=load_fn(dataset)
        if resume and manifest['dataset_hash']!=data['dataset_hash']: raise ValueError('Dataset changed on resume')
        manifest['dataset_hash']=data['dataset_hash']; manifest['folds']=data['folds']
        manifest['active_started_at']=time.time()
        ds.atomic_json(root/'manifest.json',manifest)
        (root/'trials').mkdir(exist_ok=True)
        started=time.monotonic(); spent_before=manifest['spent_seconds']
        remaining=manifest['budget_hours']*3600-spent_before
        interrupted=[False]
        old_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
        for sig in old_handlers: signal.signal(sig,lambda *_:interrupted.__setitem__(0,True))
        stop=lambda:interrupted[0] or time.monotonic()-started>=remaining
        try:
            refresh_fn(root,data)
            trial=len(completed(root)); failures=0
            while not stop() and (manifest['max_trials'] is None or trial<manifest['max_trials']):
                name=f'trial_{trial:06d}'; directory=root/'trials'/name; directory.mkdir(exist_ok=True)
                cfg=config_fn(manifest['seed'],trial)
                if (directory/'config.json').exists() and ds.read(directory/'config.json')!=cfg: raise ValueError('Trial configuration changed')
                ds.atomic_json(directory/'config.json',cfg)
                t=time.monotonic()
                try:
                    result=train_fn(directory,data,x,cfg,stop)
                    result.update(status='COMPLETE',trial=name,config=cfg,dataset_hash=data['dataset_hash'],
                                  code=identity,elapsed_seconds=time.monotonic()-t,finished_at=time.time())
                    failures=0
                except InterruptedError:
                    break
                except Exception as exc:
                    result=dict(status='FAILED',trial=name,config=cfg,error=f'{type(exc).__name__}: {exc}',finished_at=time.time())
                    failures+=1
                target=directory/'result.json'
                if target.exists(): raise ValueError('Refusing completed trial overwrite')
                ds.atomic_json(target,result)
                ds.atomic_json(directory/'summary.json',{k:result[k] for k in ('trial','status','metrics','finished_at','error') if k in result})
                summary=publish_fn(root,result,data)
                # Full history scans are not on every trial's critical path.
                if (trial+1)%100==0: refresh_fn(root,data)
                manifest['spent_seconds']=spent_before+time.monotonic()-started
                manifest['active_started_at']=time.time()
                ds.atomic_json(root/'manifest.json',manifest)
                print(name,result['status'],'champion',summary['champion'],flush=True)
                trial+=1
                if failures>=3: raise RuntimeError('Three consecutive failed trials; stopping unattended loop')
        finally:
            for sig,handler in old_handlers.items(): signal.signal(sig,handler)
            manifest['spent_seconds']=spent_before+time.monotonic()-started
            manifest['active_started_at']=None
            manifest['last_stopped']=utc_iso(); ds.atomic_json(root/'manifest.json',manifest)
            refresh_fn(root,data)
    return ds.read(root/'champion.json')


def queue_ablations(root):
    champion=ds.read(root/'champion.json')['champion']
    if champion=='frozen_exact_witness_baseline':
        raise ValueError('No learned research champion yet; no fake ablation of the deterministic baseline')
    cfg=ds.read(root/'trials'/champion/'config.json')
    jobs=[]
    for family in ds.FAMILIES:
        job=dict(parent=champion,family=family,config=dict(cfg,excluded_families=[family]),
                 action='full retraining on identical folds/seeds; never zero-at-inference importance')
        path=root/'ablations'/champion/(family+'.job.json')
        if not path.exists(): ds.atomic_json(path,job)
        jobs.append(str(path))
    return jobs


def run_ablations(root,hours):
    manifest=ds.read(root/'manifest.json')
    if manifest['code']!=code_identity(): raise ValueError('Ablation code differs')
    data,x=ds.load(manifest['dataset']); start=time.monotonic()
    with exclusive(root):
        for path in sorted((root/'ablations').glob('*/*.job.json')):
            job=ds.read(path); out=path.parent/job['family']; out.mkdir(exist_ok=True)
            if (out/'result.json').exists(): continue
            try: result=train_trial(out,data,x,job['config'],lambda:time.monotonic()-start>=hours*3600)
            except InterruptedError: break
            parent=ds.read(root/'trials'/job['parent']/'result.json')
            def good(r): return {e['event'] for e in r['metrics']['per_event'] if e['error'] is not None and e['error']<=5}
            a,b=good(parent),good(result)
            all_hits={e['event'] for e in data['events'] if e['truth']=='HIT'}
            result.update(unique_saves=sorted(b-a),unique_regressions=sorted(a-b),failure_overlap=sorted(all_hits-(a|b)),job=job)
            ds.atomic_json(out/'result.json',result)


def main():
    ds.install_guard()
    import sys
    if len(sys.argv)>1 and sys.argv[1].startswith('fusion-'):
        from automation.physical_ai_fusion import main as fusion_main
        return fusion_main(sys.argv[1:])
    if len(sys.argv)>1 and sys.argv[1].startswith('vision-repr-'):
        from automation.physical_ai_vision_repr import main as representation_main
        return representation_main(sys.argv[1:])
    if len(sys.argv)>1 and sys.argv[1].startswith('peak-'):
        from automation.physical_ai_vision_peaks import main as peak_main
        return peak_main(sys.argv[1:])
    if len(sys.argv)>1 and sys.argv[1].startswith('vision-'):
        from automation.physical_ai_vision import lab_main
        return lab_main(sys.argv[1:])
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build'); b.add_argument('--output',type=Path,required=True)
    for command in ('run','resume','status','queue-ablations','run-ablations'):
        q=sub.add_parser(command); q.add_argument('--experiment',required=True); q.add_argument('--root',type=Path,default=ROOT)
        if command in ('run','resume','run-ablations'): q.add_argument('--hours',type=float,default=None)
        if command in ('run','resume'): q.add_argument('--max-trials',type=int,default=None)
        if command=='run':
            q.add_argument('--dataset',type=Path,required=True); q.add_argument('--seed',type=int,default=1729)
    args=p.parse_args()
    if args.command=='build':
        m=ds.build(args.output); print(m['dataset_hash'],len(m['events'])); return
    root=location(args.root,args.experiment)
    if args.command=='run':
        if args.hours is None: p.error('run requires --hours')
        run(root,args.dataset,hours=args.hours,seed=args.seed,max_trials=args.max_trials)
    elif args.command=='resume': run(root,None,hours=args.hours,max_trials=args.max_trials,resume=True)
    elif args.command=='status':
        for name in ('champion.json','failure_wall.json'):
            value=ds.read(root/name)
            if name=='champion.json':
                value=dict(champion=value['champion'],history=value['history'],live_authority=False,
                           metrics={k:value['metrics'][k] for k in ('impacts','selected','oracle','unknown',
                                      'control_false_emissions','catastrophic_gt100','worst_session_selected5')})
            print(name,json.dumps(value,indent=2))
        print('Full leaderboard:',root/'leaderboard.json')
    elif args.command=='queue-ablations': print('\n'.join(queue_ablations(root)))
    else: run_ablations(root,args.hours or .1)


if __name__=='__main__': main()
