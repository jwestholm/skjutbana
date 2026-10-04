"""Representation/full-map mining adapter for the existing bounded Vision lab.

No runtime authority. Same V2236 models, nested peak adapter, runner and metrics.
"""
import argparse
import json
import sys
import time
import resource
from pathlib import Path
import numpy as np
from automation import physical_ai_dataset as ds, physical_ai_lab as lab
from automation import physical_ai_vision as v, physical_ai_vision_peaks as p
from src.engine.ai.training_v223.heatmap_model_v2236 import HeatmapModelV2236

HARD=('POST_FIX:10','S01:3','S01:5','S02:4','S02:6')
PROTECTED=(*p.PROTECTED,'POST_FIX:3')
SCHEMA='vision-representation-1'


def code_identity():
    value=p.code_identity();value['role']='VISION_REPRESENTATION'
    value['source_hashes']['automation/physical_ai_vision_repr.py']=ds.digest(Path(__file__))
    return value


def prepare(dataset,baseline_cache,baseline_result,output):
    data,cache=v.load(dataset);base,shards=p.load(baseline_cache);r=ds.read(baseline_result)
    if base['peak_manifest']['intent']['vision_cache_hash']!=data['dataset_hash'] or r['dataset_hash']!=base['dataset_hash']:
        raise ValueError('Frozen reference dataset mismatch')
    if r['status']!='COMPLETE' or v.evaluate_union(data['events'],r['outputs'])!=r['metrics']:
        raise ValueError('Frozen baseline metrics do not reproduce')
    m=dict(schema=SCHEMA,vision_cache=str(cache.resolve()),vision_hash=data['dataset_hash'],
           baseline_cache=str(Path(baseline_cache).resolve()),baseline_cache_hash=base['dataset_hash'],
           baseline_result=str(Path(baseline_result).resolve()),baseline_sha256=ds.digest(baseline_result),
           baseline_metrics=r['metrics'],baseline_pool=pool_metrics(data,shards),
           base_config=base['peak_manifest']['intent']['config'],reranker_config=r['config'],
           pool=32,primary_budget=3,budgets=[1,3,8],live_authority=False)
    m['dataset_hash']=ds.canonical_hash(m)
    out=ds.output_root(output);out.mkdir(parents=True,exist_ok=False);ds.atomic_json(out/'representation.json',m)
    return m


def load(root):
    m=ds.read(Path(root)/'representation.json');expected=m.pop('dataset_hash')
    if ds.canonical_hash(m)!=expected or m['schema']!=SCHEMA or m['pool']!=32 or m['primary_budget']!=3:
        raise ValueError('Representation manifest/budget changed')
    if ds.digest(m['baseline_result'])!=m['baseline_sha256']:raise ValueError('Frozen reference changed')
    data,cache=v.load(m['vision_cache'])
    if data['dataset_hash']!=m['vision_hash']:raise ValueError('Visual cache changed')
    return dict(data,dataset_hash=expected,repr_manifest=m),cache


def config_for(seed,trial):
    # Four paired smoke configurations: equal total epochs and seed. This is a
    # predeclared mining ablation + two small representation families, not tuning
    # individual failure events. Later trials search only this bounded space.
    fixed=[('native','persistent',0),('native','persistent',64),
           ('native','all',64),('context9','all',64)]
    cfg=dict(seed=1018197377,kind='spatial_conv',hidden=4,epochs=3,refine_epochs=2,
             learning_rate=.0009756585653263493,l2=.0018737355850663082,stride=1,
             channels='persistent',positive_radius=6.5,loss_kind='pair',hard_count=64,
             nms_radius_px=6.,budgets=[1,3,8],temporal='first3_last3',normalization='train_channel_standardization')
    if trial<4:
        cfg['representation'],cfg['channels'],cfg['full_map_negatives']=fixed[trial]
    else:
        rng=np.random.default_rng(np.random.SeedSequence([seed,trial]))
        rep,ch,_=fixed[int(rng.integers(1,4))]
        cfg.update(representation=rep,channels=ch,full_map_negatives=int(rng.choice([32,64])),
                   seed=int(rng.integers(1,2**30)),hidden=int(rng.choice([4,8])),stride=int(rng.choice([1,2])),
                   positive_radius=float(rng.choice([3.5,4.5,6.5])),loss_kind=str(rng.choice(['pair','bce'])),
                   learning_rate=float(10**rng.uniform(-3.4,-2.7)),l2=float(10**rng.uniform(-3.5,-2.5)))
    return cfg


def pool_metrics(data,shards):
    rows=[]
    for shard in shards:
        for e in data['events']:
            if e['event'] not in shard['held']:continue
            pp=shard['held'][e['event']][p.ORDERS[1]]
            if len(pp)>32:raise ValueError('Internal pool overflow')
            ve=min((ds.distance(q['camera_xy'],e['gt_xy']) for q in pp),default=None) if e['truth']=='HIT' else None
            ce=min((ds.distance(q['xy'],e['gt_xy']) for q in e['candidates']),default=None) if e['truth']=='HIT' else None
            ue=min([x for x in (ve,ce) if x is not None],default=None)
            rows.append(dict(event=e['event'],session=e['session'],truth=e['truth'],vision_error=ve,combined_error=ue,count=len(pp)))
    def oracle(rows,field):return {str(t):sum(r[field] is not None and r[field]<=t for r in rows if r['truth']=='HIT') for t in (5,10,20,42)}
    per={s:dict(vision=oracle([r for r in rows if r['session']==s],'vision_error'),
                combined=oracle([r for r in rows if r['session']==s],'combined_error')) for s in data['sessions']}
    return dict(vision=oracle(rows,'vision_error'),combined=oracle(rows,'combined_error'),per_session=per,per_event=rows,max_count=max(r['count'] for r in rows))


def audit_hard_events(data,cache,base_dir,cfg):
    # Diagnostic only, AFTER all model training and final proposal decisions.
    out={}
    for i,fold in enumerate(data['folds']):
        events=[e for e in data['events'] if e['event'] in HARD and e['session'] in fold['held']]
        if not events:continue
        model=HeatmapModelV2236.load(base_dir/f'fold_{i:02d}')
        for e in events:
            row=data['vision_rows'][e['event']];maps=v.event_maps(cache,row,cfg['stride'],cfg['representation'])
            score=model.score_map(v.normalize(maps,model.metadata['normalization']))
            out[e['event']]=p.audit_peak(score,stride=cfg['stride'],origin=row['origin'],nms_radius_px=cfg['nms_radius_px'],gt_xy=e['gt_xy'])
    return out


def train_trial(directory,data,cache,cfg,stop):
    if cfg['budgets']!=[1,3,8] or cfg['nms_radius_px']!=6. or cfg['stride'] not in (1,2):raise ValueError('Frozen proposal geometry/budget')
    start=time.perf_counter();timing={}
    # Reuse the original adapter's actual five outer fits, then its 15 nested
    # fits. No global OOF row is passed as a TRAIN feature to the next stage.
    visual_data=dict(data,dataset_hash=data['repr_manifest']['vision_hash'])
    base=directory/'base';source=base/'trials/trial_000000';source.mkdir(parents=True,exist_ok=True)
    ds.atomic_json(base/'manifest.json',dict(dataset=str(cache)))
    result_path=source/'result.json'
    if not result_path.exists():
        raw=v.train_trial(source,visual_data,cache,cfg,stop)
        raw.update(config=cfg,dataset_hash=visual_data['dataset_hash']);ds.atomic_json(result_path,raw)
    timing['outer_vision_seconds']=time.perf_counter()-start;t=time.perf_counter()
    nested=directory/'nested'
    if not (nested/'peaks.json').exists():p.prepare(base,'trial_000000',nested,resume=(nested/'intent.json').exists(),stop=stop)
    pdata,shards=p.load(nested)
    timing['nested_vision_extraction_seconds']=time.perf_counter()-t;t=time.perf_counter()
    dest=directory/'reranker';dest.mkdir(exist_ok=True)
    rcfg=dict(data['repr_manifest']['reranker_config']);rcfg.update(order=p.ORDERS[1],pool=32)
    result=p.train_trial(dest,pdata,shards,rcfg,stop)
    timing['reranker_seconds']=time.perf_counter()-t;t=time.perf_counter()
    result['metrics']['pool']=pool_metrics(data,shards)
    result['hard_events']=audit_hard_events(data,cache,source,cfg)
    timing['diagnostic_audit_seconds']=time.perf_counter()-t
    def good(m):return {e['event'] for e in m['per_event'] if e['combined_error'] is not None and e['combined_error']<=5}
    a=good(data['repr_manifest']['baseline_metrics']['3']);b=good(result['metrics']['3'])
    ap={r['event'] for r in data['repr_manifest']['baseline_pool']['per_event'] if r['combined_error'] is not None and r['combined_error']<=5}
    bp={r['event'] for r in result['metrics']['pool']['per_event'] if r['combined_error'] is not None and r['combined_error']<=5}
    result['baseline_delta']=dict(saves=sorted(b-a),regressions=sorted(a-b),protected_lost=sorted(set(PROTECTED)-b),pool_saves=sorted(bp-ap),pool_regressions=sorted(ap-bp))
    result['performance']=dict(**timing,total_seconds=time.perf_counter()-start,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result['metrics']['runtime_seconds']=result['performance']['total_seconds']
    result['lineage']=dict(representation=cfg,cache=pdata['dataset_hash'],nested_directory=str(nested),reranker_config=rcfg,code=code_identity())
    return result


def policy(metrics):
    m=metrics['3'];base=v.policy(m)
    return (base[0],metrics['pool']['combined']['5'],base[2],metrics['1']['combined_oracle']['5'],*base[3:],-metrics.get('runtime_seconds',1e30))


def comparison(metrics,old,data):
    lost=set(data['repr_manifest']['baseline_metrics']['3']['unique_saves'])-set(metrics['3']['unique_saves'])
    # Conservative research rule: trade at most one protected save only when
    # primary coverage is strictly greater than the incumbent. Always report it.
    if lost and (len(lost)>1 or metrics['3']['combined_oracle']['5']<=old['3']['combined_oracle']['5']):
        return dict(accepted=False,reason='protected saves lost',lost=sorted(lost))
    win=policy(metrics)>policy(old)
    return dict(accepted=win,reason='fixed top3/pool/session/burden/runtime lexicographic '+('gain' if win else 'no gain'),lost=sorted(lost))


def initial(data):
    m=dict(data['repr_manifest']['baseline_metrics'],pool=data['repr_manifest']['baseline_pool'])
    return dict(champion='frozen_32',metrics=m,history=[],live_authority=False)


def accept(root,state,r,data):
    if r['status']!='COMPLETE':return dict(accepted=False,reason=r.get('error','failed'))
    decision=comparison(r['metrics'],state['metrics'],data)
    if decision['accepted']:
        entry=dict(trial=r['trial'],metrics=r['metrics'],dataset_hash=data['dataset_hash'],reason=decision,
                   directory=str(root/'trials'/r['trial']),live_authority=False)
        target=root/'champions'/(r['trial']+'.json')
        if target.exists() and ds.read(target)!=entry:raise ValueError('Champion overwrite')
        if not target.exists():ds.atomic_json(target,entry)
        state.update(champion=r['trial'],metrics=r['metrics']);state['history'].append(r['trial'])
    return decision


def publish(root,result,data):
    state=ds.read(root/'champion.json');decision=accept(root,state,result,data)
    ds.atomic_json(root/'champion.json',state)
    with (root/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(trial=result['trial'],decision=decision))+'\n')
    return state


def refresh(root,data):
    state=initial(data);records=lab.completed(root);board=[]
    for r in records:board.append(dict(trial=r['trial'],status=r['status'],decision=accept(root,state,r,data),metrics=r.get('metrics')))
    ds.atomic_json(root/'champion.json',state);ds.atomic_json(root/'leaderboard.json',board)
    top=sorted((r for r in records if r['status']=='COMPLETE'),key=lambda r:policy(r['metrics']),reverse=True)[:5]
    wall=[]
    for r in top:
        full=ds.read(root/'trials'/r['trial']/'result.json')
        wall.append(dict(trial=r['trial'],hard_events=full['hard_events'],delta=full['baseline_delta'],
                         unresolved=r['metrics']['3']['still_unavailable5'],diagnostics={e:full['outputs'][e]['diagnostic'] for e in r['metrics']['3']['still_unavailable5']}))
    ds.atomic_json(root/'failure_wall.json',dict(top_models=wall,completed=len(top) if len(records)<5 else sum(r['status']=='COMPLETE' for r in records),live_authority=False))
    ds.atomic_json(root/'journal.json',[{k:r[k] for k in ('trial','status','finished_at','error') if k in r} for r in records]);return state


def main(argv=None):
    ds.install_guard();parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('vision-repr-prepare')
    for name in ('dataset','baseline-cache','baseline-result','output'):q.add_argument('--'+name,type=Path,required=True)
    for name in ('vision-repr-run','vision-repr-resume','vision-repr-status'):
        q=sub.add_parser(name);q.add_argument('--experiment',required=True);q.add_argument('--root',type=Path,default=lab.ROOT)
        if name!='vision-repr-status':q.add_argument('--hours',type=float);q.add_argument('--max-trials',type=int)
        if name=='vision-repr-run':q.add_argument('--dataset',type=Path,required=True);q.add_argument('--seed',type=int,default=2381)
    a=parser.parse_args(argv)
    if a.cmd=='vision-repr-prepare':print(prepare(a.dataset,a.baseline_cache,a.baseline_result,a.output)['dataset_hash']);return
    root=lab.location(a.root,a.experiment)
    if a.cmd=='vision-repr-status':
        c=ds.read(root/'champion.json');print(json.dumps(dict(champion=c['champion'],history=c['history'],combined=c['metrics']['3']['combined_oracle'],pool=c['metrics']['pool'],failure_wall=str(root/'failure_wall.json')),indent=2));return
    if a.cmd=='vision-repr-run' and a.hours is None:parser.error('--hours required')
    lab.run(root,getattr(a,'dataset',None),hours=a.hours,seed=getattr(a,'seed',2381),max_trials=a.max_trials,resume=a.cmd=='vision-repr-resume',adapter=sys.modules[__name__])


if __name__=='__main__':main()
