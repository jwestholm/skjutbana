"""Two fixed listwise loss probes using existing RankModel/evidence gradients.

No lab search adapter or live authority. Resume hash-bound folds/epoch checkpoints.
UNKNOWN is a fixed zero-logit alternative learned against every candidate.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
from automation import physical_ai_dataset as ds, physical_ai_fusion as fusion
from automation import physical_ai_selector_gap as gap
from src.engine.ai.training_v223.model import RankModelV223,_softmax
from src.engine.ai.training_v223.evidence_model_v2235 import EvidenceModelV2235,_grads

NAMES=gap.NAMES+tuple(n+'__event_midrank' for n in gap.PHYSICAL)


def relative_features(physical):
    """Runtime candidate matrix only: no labels/XY/IDs/source arguments."""
    if physical.ndim!=2 or physical.shape[1]!=len(gap.NAMES):raise ValueError('Physical schema required')
    n=len(gap.PHYSICAL);r=np.zeros((len(physical),n),np.float32)
    for j in range(n):
        valid=physical[:,j+n]==0;v=physical[valid,j]
        if len(v):r[valid,j]=[(np.sum(v<t)+.5*np.sum(v==t))/len(v) for t in v]
    return np.column_stack((physical,r)).astype(np.float32)


def target(positive):
    p=np.asarray(positive,bool);q=np.zeros(len(p)+1,np.float32)
    if p.any():q[:-1]=p/p.sum()
    else:q[-1]=1
    return q


def loss_gradient(scores,positive):
    q=target(positive);p=_softmax(np.append(scores,0))
    return float(-np.sum(q*np.log(np.maximum(p,1e-12)))),(p-q)[:-1]


def fit(samples,cfg,path,identity,stop=lambda:False):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    joined=np.concatenate([x for x,_ in samples]);mean=joined.mean(0);scale=joined.std(0);scale[scale<1e-6]=1
    rng=np.random.default_rng(cfg['seed']);d=len(NAMES);h=cfg['hidden']
    if joined.shape[1]!=d or not np.isfinite(joined).all():raise ValueError('Invalid features')
    if cfg['kind']=='linear':params=dict(w=np.zeros(d,np.float32),b=np.zeros(1,np.float32))
    elif cfg['kind']=='mlp':params=dict(w1=rng.normal(0,np.sqrt(2/d)*.45,(d,h)).astype(np.float32),b1=np.zeros(h,np.float32),w2=rng.normal(0,.1,(h,1)).astype(np.float32),b2=np.zeros(1,np.float32))
    else:raise ValueError('Only predeclared linear/MLP')
    m={k:np.zeros_like(v) for k,v in params.items()};v={k:np.zeros_like(a) for k,a in params.items()};start=step=0;history=[]
    checkpoint=path/'checkpoint.npz';binding=dict(config=cfg,identity=identity)
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as old:
            state=json.loads(str(old['metadata']))
            if state['binding']!=binding or not np.array_equal(old['mean'],mean) or not np.array_equal(old['scale'],scale):raise ValueError('Checkpoint identity/scaler differs')
            for k in params:params[k]=old['p_'+k].copy();m[k]=old['m_'+k].copy();v[k]=old['v_'+k].copy()
            start=state['epoch'];step=state['step'];history=state['history'];rng.bit_generator.state=state['rng']
    model=EvidenceModelV2235(cfg['kind'],params)
    for epoch in range(start,cfg['epochs']):
        if stop():raise InterruptedError('Epoch checkpoint preserved')
        losses=[]
        for index in rng.permutation(len(samples)):
            raw,positive=samples[index];x=(raw-mean)/scale;scores=model._forward_features(x)
            loss,grad=loss_gradient(scores,positive);losses.append(loss);grads=_grads(model,x,grad,l2=cfg['l2']);step+=1
            for k,g in grads.items():
                g=np.clip(g,-5,5).astype(np.float32);m[k]=.9*m[k]+.1*g;v[k]=.999*v[k]+.001*g*g
                params[k]-=cfg['learning_rate']*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-7)
        history.append(float(np.mean(losses)))
        if not np.isfinite(history[-1]):raise ValueError('Nonfinite training')
        state=dict(binding=binding,epoch=epoch+1,step=step,history=history,rng=rng.bit_generator.state)
        tmp=checkpoint.with_suffix('.tmp')
        with tmp.open('wb') as f:
            np.savez(f,metadata=json.dumps(state),mean=mean,scale=scale,**{'p_'+k:a for k,a in params.items()},**{'m_'+k:a for k,a in m.items()},**{'v_'+k:a for k,a in v.items()});f.flush();os.fsync(f.fileno())
        os.replace(tmp,checkpoint)
    return RankModelV223(cfg['kind'],NAMES,mean,scale,params,dict(binding,role='OFFLINE_EVENT_LISTWISE',unknown_logit=0,live_authority=False)),history


def predict(model,matrix):
    scores=model.score_matrix(relative_features(matrix))
    # Deterministic tie with UNKNOWN is abstention, never forced emission.
    best=int(np.argmax(scores)) if len(scores) else None
    return (best if best is not None and scores[best]>0 else None),scores


def extra_metrics(metrics):
    errors=[r['error'] for r in metrics['per_event'] if r['error'] is not None]
    return dict(metrics,errors_gt42=sum(e>42 for e in errors),max_error=max(errors) if errors else None,
                wrong_emissions_at5=sum(e>5 for e in errors))


def run(output,stop=lambda:False):
    output=ds.output_root(output);data,x=fusion.load(gap.crossfit.DEFAULT);physical=gap.physical_matrix(x)
    declared=ds.read(output/'decision_gate.json')
    if not declared['phase2']:raise ValueError('Audit gate did not authorize model probe')
    configs=declared['models'];reference=ds.read(gap.REFERENCE)['metrics']
    identity=dict(dataset=data['dataset_hash'],reference=ds.digest(gap.REFERENCE),configs=configs,
                  code={str(p):ds.digest(p) for p in (Path(__file__),Path(gap.__file__))},features=list(NAMES),decision='argmax candidates vs UNKNOWN logit0; no fitted threshold')
    manifest=output/'model_intent.json'
    if manifest.exists() and ds.read(manifest)!=identity:raise ValueError('Run identity changed')
    if not manifest.exists():ds.atomic_json(manifest,identity)
    results=[]
    for config in configs:
        root=output/config['kind'];root.mkdir(exist_ok=True);predictions={};reports=[];started=time.monotonic()
        for i,fold in enumerate(data['folds']):
            path=root/f'fold_{i:02d}';path.mkdir(exist_ok=True);done=path/'complete.json'
            if done.exists():
                saved=ds.read(done)
                if saved['identity']!=identity or saved['fold']!=fold:raise ValueError('Fold identity changed')
                for name,h in saved['hashes'].items():
                    if ds.digest(path/name)!=h:raise ValueError('Fold model changed')
            else:
                if stop():raise InterruptedError('Fold boundary')
                train=gap.train_events(data,fold);cfg=dict(config,seed=config['seed']+i)
                samples=[(relative_features(physical[e['indices']]),gap.positive_mask(e)) for e in train]
                model,history=fit(samples,cfg,path,dict(run=identity,fold=fold,training_events=[e['event'] for e in train]),stop)
                model.save(path);detail=[];pp={}
                for e in data['events']:
                    if e['session'] not in fold['held']:continue
                    j,scores=predict(model,physical[e['indices']]);xy=e['candidates'][j]['xy'] if j is not None else None;pp[e['event']]=xy
                    detail.append(dict(event=e['event'],selected_xy=xy,scores=scores.tolist(),candidate_ids=[c['id'] for c in e['candidates']]))
                saved=dict(identity=identity,fold=fold,training_events=[e['event'] for e in train],history=history,predictions=pp,detail=detail,
                           training_target_counts=dict(positive_events=sum(bool(p.any()) for _,p in samples),unknown_events=sum(not p.any() for _,p in samples)),
                           hashes={n:ds.digest(path/n) for n in ('model.json','model.npz')})
                ds.atomic_json(done,saved)
            predictions.update(saved['predictions']);reports.append(saved)
        metrics=extra_metrics(ds.evaluate(data['events'],predictions));old={r['event']:r for r in reference['per_event']}
        gain=[];loss=[]
        for r in metrics['per_event']:
            success=lambda a:a['error'] is not None and a['error']<=5
            if success(r) and not success(old[r['event']]):gain.append(r['event'])
            if success(old[r['event']]) and not success(r):loss.append(r['event'])
        result=dict(config=config,metrics=metrics,predictions=predictions,folds=reports,comparison=fusion.compare(metrics,reference),unique_saves=gain,regressions=loss,elapsed_seconds=time.monotonic()-started)
        dest=root/'result.json'
        if dest.exists():result=ds.read(dest)
        else:ds.atomic_json(dest,result)
        results.append(result);print(config['kind'],metrics['selected'],'false',metrics['control_false_emissions'],'cat',metrics['catastrophic_gt100'],'unknown',metrics['unknown'],flush=True)
    ds.atomic_json(output/'model_comparison.json',dict(reference=extra_metrics(reference),results=results))
    return results


def main():
    ds.install_guard();p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();flag=[False]
    old={s:signal.signal(s,lambda *_:flag.__setitem__(0,True)) for s in (signal.SIGINT,signal.SIGTERM)}
    try:run(args.output,lambda:flag[0])
    except InterruptedError:print('Paused; repeat the same command to resume.')
    finally:
        for s,h in old.items():signal.signal(s,h)

if __name__=='__main__':main()
