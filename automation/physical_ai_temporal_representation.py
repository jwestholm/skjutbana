"""One fixed temporal-observation sidecar + unchanged linear selector seeds.

Reuses cached candidate-local signals and nested TRAIN lineage. No source image
search, proposal generation, support change, live hook or hyperparameter search.
"""
import argparse
from pathlib import Path
import signal
import time
import numpy as np
from automation import physical_ai_dataset as ds,physical_ai_fusion as fusion
from automation import physical_ai_fusion_crossfit as crossfit
from automation import physical_ai_selector_gap as gap,physical_ai_event_selector as selector
from src.engine.ai.canonical_challenger import transform_matrix
from src.engine.offline.common_verifier import query_temporal_observation

BASELINE=Path('/data/skjutbana/evaluation_runs/selector_gap_20261007_01')
EXTRA=('observed_window_signed_change','observed_window_coherent_strength')
PHYSICAL=gap.PHYSICAL+EXTRA
FEATURE_NAMES=PHYSICAL+tuple(n+'__missing' for n in PHYSICAL)+tuple(n+'__event_midrank' for n in PHYSICAL)


def source_code():
    files=[Path(__file__),Path(selector.__file__),Path(gap.__file__),Path('src/engine/offline/common_verifier.py'),Path('src/engine/offline/clean_physical_change.py')]
    return {str(p):ds.digest(p) for p in files}


def observation_vector(witness):
    result=query_temporal_observation(witness).metadata['observed_window']
    if not result['available']:return [0.,0.,1.,1.],result
    return [result['signed_change'],result['coherent_strength'],0.,0.],result


def build(output):
    output=ds.output_root(output);data,x=fusion.load(crossfit.DEFAULT);dst=output/'observation_cache.json'
    identity=dict(dataset=data['dataset_hash'],code=source_code(),contract='observed-first3-last3-v1',features=list(EXTRA))
    if dst.exists():
        saved=ds.read(dst)
        content={k:v for k,v in saved.items() if k!='cache_hash'}
        if ds.canonical_hash(content)!=saved['cache_hash']:raise ValueError('Observation manifest changed')
        if saved['identity']!=identity or ds.digest(output/'observations.npy')!=saved['feature_sha256']:raise ValueError('Observation cache identity changed')
        for name,h in saved['source_hashes'].items():
            if ds.digest(name)!=h:raise ValueError('Witness source changed')
        return data,x,np.load(output/'observations.npy',allow_pickle=False),saved
    output.mkdir(parents=True,exist_ok=True);hashes={};classic=fusion.CLASSIC/'provenance.json';vision=fusion.DEFAULT/'provenance.json'
    def read(path):hashes[str(path)]=ds.digest(path);return ds.read(path)
    proof={(r['event'],str(r['candidate_id'])):r['witness'] for r in read(classic)+read(vision)}
    witnesses={};owners={}
    for e in data['events']:
        for i,c in zip(e['indices'],e['candidates']):witnesses[i]=proof[(e['event'],str(c['id']))];owners[i]=(e,c)
    for record in data['nested_training']:
        for e in record['events']:
            n=e['classic_count'];directory=Path(e['vision_top3'][0]['reranker_model_id']);ev=read(directory/(e['event'].replace(':','_')+'_evidence.json'))
            if not np.array_equal(np.asarray(ev['vectors'],np.float32).reshape(-1,x.shape[1]),x[e['indices'][n:]]):raise ValueError('Nested evidence association changed')
            if len(ev['witnesses'])!=len(e['indices'][n:]):raise ValueError('Missing nested witnesses')
            for i,c,w in zip(e['indices'][n:],e['candidates'][n:],ev['witnesses']):witnesses[i]=w;owners[i]=(e,c)
    if sorted(witnesses)!=list(range(len(x))):raise ValueError('Incomplete evidence sidecar')
    vectors=[];rows=[]
    for i in range(len(x)):
        w=witnesses[i];e,c=owners[i]
        if (w['event_id'],w['candidate_id'],w['camera_xy'],w['frames'])!=(e['event'],str(c['id']),c['xy'],e['frames']):raise ValueError('Candidate identity/frame mismatch')
        old=np.asarray(ds.vector(dict(physical_witness=w)),np.float32)
        # Validate original seven temporal scalars and their explicit missingness.
        columns=list(range(7))+list(range(len(ds.RAW_NAMES),len(ds.RAW_NAMES)+7))
        if not np.array_equal(old[columns],x[i,columns]):raise ValueError('Temporal evidence differs from frozen feature row')
        v,measurements=observation_vector(w);vectors.append(v)
        rows.append(dict(index=i,event=e['event'],candidate_id=c['id'],camera_xy=c['xy'],witness_hash=ds.canonical_hash(w),measurements=measurements))
    values=np.asarray(vectors,np.float32);np.save(output/'observations.npy',values,allow_pickle=False)
    saved=dict(identity=identity,feature_sha256=ds.digest(output/'observations.npy'),source_hashes=hashes,rows=rows)
    saved['cache_hash']=ds.canonical_hash(saved);ds.atomic_json(dst,saved)
    return data,x,values,saved


def matrix(raw,observations):
    """Feature projection has no event/coordinate/label/source arguments."""
    if observations.shape!=(len(raw),4) or not np.isfinite(observations).all():raise ValueError('Invalid observation sidecar')
    if not np.isin(observations[:,2:],[0,1]).all():raise ValueError('Explicit missing flags required')
    n=len(gap.PHYSICAL);base=gap.physical_matrix(raw)
    values=transform_matrix(observations[:,:2],'signed_log');missing=transform_matrix(observations[:,2:],'signed_log')
    return np.column_stack((base[:,:n],values,base[:,n:],missing)).astype(np.float32)


def run(output,stop=lambda:False):
    output=ds.output_root(output);data,raw,extra,cache=build(output);x=matrix(raw,extra)
    baseline=ds.read(BASELINE/'linear/result.json');reference=ds.read(gap.REFERENCE);config=baseline['config']
    assert config['kind']=='linear'
    identity=dict(cache_hash=cache['cache_hash'],dataset=data['dataset_hash'],code=source_code(),config=config,seeds=[1729,1730,1731],names=list(FEATURE_NAMES))
    intent=output/'training_intent.json'
    if intent.exists() and ds.read(intent)!=identity:raise ValueError('Trial intent changed')
    if not intent.exists():ds.atomic_json(intent,identity)
    results=[]
    for seed in identity['seeds']:
        root=output/f'seed_{seed}';root.mkdir(exist_ok=True);final=root/'result.json'
        if final.exists():
            old=ds.read(final)
            if old['identity']!=identity:raise ValueError('Completed trial identity changed')
            results.append(old);continue
        started=time.monotonic();predictions={};reports=[]
        for i,fold in enumerate(data['folds']):
            path=root/f'fold_{i:02d}';path.mkdir(exist_ok=True);done=path/'complete.json'
            if done.exists():
                saved=ds.read(done)
                if saved['identity']!=identity or saved['fold']!=fold:raise ValueError('Fold identity differs')
                for name,h in saved['hashes'].items():
                    if ds.digest(path/name)!=h:raise ValueError('Fold model drift')
            else:
                if stop():raise InterruptedError('Paused before fold')
                train=gap.train_events(data,fold);cfg=dict(config,seed=seed+i)
                samples=[(selector.relative_features(x[e['indices']],PHYSICAL),gap.positive_mask(e)) for e in train]
                model,history=selector.fit(samples,cfg,path,dict(run=identity,fold=fold,training_events=[e['event'] for e in train]),stop,feature_names=FEATURE_NAMES)
                model.save(path);pp={};detail=[]
                for e in data['events']:
                    if e['session'] not in fold['held']:continue
                    j,scores=selector.predict(model,x[e['indices']],PHYSICAL);xy=None if j is None else e['candidates'][j]['xy'];pp[e['event']]=xy
                    detail.append(dict(event=e['event'],candidate_ids=[c['id'] for c in e['candidates']],scores=scores.tolist(),selected_xy=xy))
                saved=dict(identity=identity,fold=fold,training_events=[e['event'] for e in train],predictions=pp,detail=detail,history=history,hashes={name:ds.digest(path/name) for name in ('model.json','model.npz')})
                ds.atomic_json(done,saved)
            predictions.update(saved['predictions']);reports.append(saved)
        metrics=selector.extra_metrics(ds.evaluate(data['events'],predictions));old={r['event']:r for r in baseline['metrics']['per_event']}
        yes=lambda r:r['error'] is not None and r['error']<=5
        result=dict(identity=identity,seed=seed,config=config,metrics=metrics,predictions=predictions,folds=reports,elapsed_seconds=time.monotonic()-started,
            gains=[r['event'] for r in metrics['per_event'] if yes(r) and not yes(old[r['event']])],losses=[r['event'] for r in metrics['per_event'] if not yes(r) and yes(old[r['event']])],
            comparison=fusion.compare(metrics,baseline['metrics']))
        ds.atomic_json(final,result);results.append(result)
        print(seed,metrics['selected'],'controls',metrics['control_false_emissions'],'cats',metrics['catastrophic_gt100'],'UNKNOWN',metrics['unknown'],flush=True)
    ds.atomic_json(output/'comparison.json',dict(linear_baseline=baseline['metrics'],champion80k=reference['metrics'],results=results))
    return results


def main():
    ds.install_guard();p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['build','run']);p.add_argument('--output',type=Path,required=True);a=p.parse_args();flag=[False]
    old={s:signal.signal(s,lambda *_:flag.__setitem__(0,True)) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        if a.command=='build':data,x,v,c=build(a.output);print('rows',len(v),'hash',c['cache_hash'])
        else:run(a.output,lambda:flag[0])
    except InterruptedError:print('Paused; repeat command to resume.')
    finally:
        for s,h in old.items():signal.signal(s,h)

if __name__=='__main__':main()
