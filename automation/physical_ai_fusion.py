"""Frozen union adapter to the existing Evidence lab; no proposal/model promotion.

Standalone OOF is not nested OOF. Unsafe upstream training rows are excluded,
never relabelled; held scoring always uses the complete frozen top3 union.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from automation import physical_ai_dataset as ds, physical_ai_lab as lab

CLASSIC=lab.ROOT/'datasets/witness_v1_20261003'
VISION=lab.ROOT/'vision_peak_smoke_final_20261003/trials/trial_000001/result.json'
PEAKS=lab.ROOT/'datasets/vision_peaks32_nested_20261003'
DEFAULT=lab.ROOT/'datasets/fusion_frozen_top3_v1'
REVISION='frozen-union-evidence-1'


def code_identity():
    identity=lab.code_identity();repo=Path(__file__).resolve().parents[1]
    for name in ('automation/physical_ai_fusion.py','src/engine/offline/common_verifier.py','src/engine/offline/clean_physical_change.py'):
        identity['source_hashes'][name]=ds.digest(repo/name)
    return dict(identity,role='FROZEN_UNION_EVIDENCE_SELECTOR')


def validate_proposal(p,event):
    for key in ('training_sessions','vision_training_sessions','reranker_training_sessions'):
        if not p.get(key):raise ValueError('Missing upstream lineage')
        for session in p[key]:ds.guard(session)
        if event['session'] in p[key]:raise ValueError('Upstream used its own held session')
    ds.guard(p['event']);ds.guard(p['held_session'])
    if p['event']!=event['event'] or p['held_session']!=event['session'] or p['source']!='VISION':raise ValueError('Foreign proposal')
    if len(p['camera_xy'])!=2 or not np.isfinite(p['camera_xy']).all():raise ValueError('Invalid XY')
    if not p.get('model_sha256') or not p.get('reranker_model_sha256'):raise ValueError('Missing model identity')


def safe_training_proposal(p,event,fold):
    validate_proposal(p,event)
    allowed=set(fold['train'])-{event['session']}
    return all(set(p[key])<=allowed for key in ('training_sessions','vision_training_sessions','reranker_training_sessions'))


def union_event(event,proposals):
    if len(proposals)!=3:raise ValueError('Frozen Vision final budget must be three')
    result=copy.deepcopy(event);result['classic_count']=len(event['candidates'])
    result['vision_top3']=copy.deepcopy(proposals)
    for i,p in enumerate(proposals):
        validate_proposal(p,event)
        if any(c['xy']==p['camera_xy'] for c in result['candidates']):continue
        result['candidates'].append(dict(id=f'VISION:{i+1}',xy=list(p['camera_xy']),vision=copy.deepcopy(p)))
    return result


def training_filter(events,fold):
    rows=[]
    for e in events:
        if e['session'] not in fold['train']:raise ValueError('Held/development event in training')
        keep=[i for i,c in enumerate(e['candidates']) if i<e['classic_count'] or safe_training_proposal(c['vision'],e,fold)]
        rows.append(dict(e,candidates=[e['candidates'][i] for i in keep],indices=[e['indices'][i] for i in keep]))
    return rows


def existing_evidence(event,added,hashes):
    """Query the unchanged batched witness primitive at exact frozen Vision XY."""
    from src.engine.offline.common_verifier import camera_local_contrasts,query_physical_witnesses
    frames=event['frames'];signals=[]
    if not 1<=len(frames)<=50 or frames[-1]['ordinal']>event['cutoff_ordinal']:raise ValueError('Invalid saved cutoff')
    if event['cutoff_receipt'] is not None and frames[-1]['receipt_timestamp']>event['cutoff_receipt']:raise ValueError('Future frame')
    for f in frames:
        path=f['frame_id']
        if ds.digest(path)!=hashes[path]:raise ValueError('Source frame changed')
        frame=np.load(path,mmap_mode='r',allow_pickle=False)
        signals.append(camera_local_contrasts(frame,[c['xy'] for c in added]))
    signals=np.asarray(signals);previous=[None]*len(added)
    for n in range(1,len(frames)+1):
        identities=[dict(candidate_id=c['id'],coordinate_version=c['id'],camera_xy=c['xy'],previous=previous[j],
            source_provenance=dict(producer='FROZEN_OOF_VISION',offline_query=True)) for j,c in enumerate(added)]
        bundles=query_physical_witnesses(signals[:n],frames[:n],event_id=event['event'],candidates=identities,
            cutoff_frame_id=frames[n-1]['frame_id'],cutoff_ts=frames[n-1]['receipt_timestamp'])
        previous=[b.witnesses[0] for b in bundles]
    return [ds.vector(dict(physical_witness=w)) for w in previous],previous


def build(output):
    from automation import physical_ai_vision_peaks as peaks
    output=ds.output_root(output)
    if output.exists():raise FileExistsError('Existing dataset is immutable')
    classic,x=ds.load(CLASSIC);upstream,shards=peaks.load(PEAKS);result=ds.read(VISION)
    if result['dataset_hash']!=upstream['dataset_hash']:raise ValueError('Wrong frozen peak cache')
    if set(result['outputs'])!={e['event'] for e in classic['events']}:raise ValueError('Incomplete frozen OOF events')
    # Validate actual serialized upstream models, not just proposal assertions.
    source_hashes=dict(classic['source_hashes']);source_hashes[str(VISION)]=ds.digest(VISION)
    for i,shard in enumerate(shards):
        saved=result['folds'][i];fold=classic['folds'][i]
        if saved['identity']['fold']!=fold:raise ValueError('Fold mismatch')
        path=VISION.parent/f'fold_{i:02d}'
        for name,h in saved['model_hashes'].items():
            if ds.digest(path/name)!=h:raise ValueError('Reranker model changed')
            source_hashes[str(path/name)]=h
        meta=ds.read(path/'model.json')['metadata']
        if meta['outer_fold']!=fold or set(meta['training_events'])!={e['event'] for e in classic['events'] if e['session'] in fold['train']}:raise ValueError('Reranker training lineage changed')
        for e in classic['events']:
            if e['session'] not in fold['held']:continue
            for p in result['outputs'][e['event']]['proposals'][:3]:
                if p['reranker_model_sha256']!=saved['model_hashes']['model.npz'] or p['reranker_training_sessions']!=fold['train']:raise ValueError('Reranker lineage mismatch')
                peaks.validate_upstream(p,e,fold,training=False)
                actual=shard['held'][e['event']][result['config']['order']]
                if not any(q['camera_xy']==p['camera_xy'] and q['model_sha256']==p['model_sha256'] for q in actual):raise ValueError('Proposal not from frozen upstream pool')
    output.mkdir(parents=True);vectors=[];events=[];proof=[]
    for e in classic['events']:
        item=union_event(e,result['outputs'][e['event']]['proposals'][:3]);added=item['candidates'][item['classic_count']:]
        extra,witnesses=existing_evidence(e,added,source_hashes) if added else ([],[])
        item['indices']=list(range(len(vectors),len(vectors)+len(item['candidates'])))
        vectors.extend(x[e['indices']]);vectors.extend(extra);events.append(item)
        proof.extend(dict(event=e['event'],candidate_id=c['id'],witness=w) for c,w in zip(added,witnesses))
    np.save(output/'features.npy',np.asarray(vectors,np.float32),allow_pickle=False);ds.atomic_json(output/'provenance.json',proof)
    data=dict(classic,events=events,source_hashes=source_hashes,feature_sha256=ds.digest(output/'features.npy'),provenance_sha256=ds.digest(output/'provenance.json'))
    data.pop('dataset_hash');data['fusion']=dict(revision=REVISION,classic_dataset_hash=classic['dataset_hash'],vision_result_hash=ds.digest(VISION),
        top_k=3,code=code_identity(),training_policy='exclude upstream-contaminated Vision TRAIN rows; full frozen union at HELD',
        missing='Vision producer morphology unavailable; explicit existing missing indicators; no learned Vision scores as features',
        no_impact='No validated rejection head; abstention remains UNKNOWN')
    data['dataset_hash']=ds.canonical_hash(data);ds.atomic_json(output/'dataset.json',data)
    return data


def load(path):
    data,x=ds.load(path)
    if data.get('fusion',{}).get('revision')!=REVISION:raise ValueError('Not a frozen union cache')
    for e in data['events']:
        if not 0<=e['classic_count']<=len(e['candidates'])<=e['classic_count']+3:raise ValueError('Candidate cap changed')
        for p in e['vision_top3']:validate_proposal(p,e)
        allowed={tuple(p['camera_xy']) for p in e['vision_top3']}
        for c in e['candidates'][e['classic_count']:]:
            if tuple(c['xy']) not in allowed or c['xy']!=c['vision']['camera_xy']:raise ValueError('Unfrozen coordinate')
    return data,x


def config_for(seed,trial):
    return dict(lab.config_for(seed,trial),threshold=.5)  # Fixed, never fitted to held truth.


def train_trial(directory,data,x,cfg,stop):
    if cfg['threshold']!=.5:raise ValueError('Frozen decision threshold')
    result=lab.train_trial(directory,data,x,cfg,stop,training_filter=training_filter)
    result['scope']='FULL_FROZEN_UNION_HELD; UPSTREAM_SAFE_TRAIN_ROWS_ONLY'
    return result


def compare(a,b):
    if a['control_false_emissions']>b['control_false_emissions']:return False,'new control false emissions'
    if a['protected_failures']:return False,'protected physical regression'
    old={r['event']:r for r in b['per_event']}
    if any(r['error'] is not None and r['error']>100 and (old[r['event']]['error'] is None or old[r['event']]['error']<=100) for r in a['per_event']):return False,'new catastrophic selection'
    def key(m):
        return (m['selected']['5'],-m['catastrophic_gt100'],m['worst_session_selected5'] or 0,m['selected']['10'],
                -(m['median_error'] if m['median_error'] is not None else float('inf')),
                -(m['p95_error'] if m['p95_error'] is not None else float('inf')),-m['unknown'])
    return key(a)>key(b),'selected@5, safety, worst session, @10, tails, UNKNOWN'


def publish(root,result,data):return lab.publish(root,result,data,compare=compare)


def refresh(root,data):
    state=lab.refresh(root,data,compare=compare)
    state['policy']='Selected@5; no new control/catastrophe or protected loss; worst session/@10/tails/UNKNOWN'
    state['fusion']=data['fusion'];ds.atomic_json(root/'champion.json',state)
    return state


def status(root):
    state=ds.read(root/'champion.json');records=lab.completed(root)
    return dict(trials=len(records),champion=state['champion'],history=state['history'],metrics=state['metrics'],
                scope=state['fusion'],live_authority=False,failure_wall=str(root/'failure_wall.json'))


def main(argv):
    import sys
    ds.install_guard();p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for command in ('fusion-run','fusion-resume','fusion-status'):
        q=sub.add_parser(command);q.add_argument('--experiment',required=True)
        if command!='fusion-status':q.add_argument('--hours',type=float);q.add_argument('--max-trials',type=int)
        if command=='fusion-run':q.add_argument('--dataset',type=Path,default=DEFAULT);q.add_argument('--seed',type=int,default=1729)
    args=p.parse_args(argv);root=lab.location(lab.ROOT,args.experiment)
    if args.command=='fusion-status':print(json.dumps(status(root),indent=2));return
    resume=args.command=='fusion-resume'
    if not resume:
        if args.hours is None:p.error('--hours is required')
        ds.output_root(args.dataset)
        if not args.dataset.exists():build(args.dataset)
    lab.run(root,None if resume else args.dataset,hours=args.hours,max_trials=args.max_trials,resume=resume,
            seed=1729 if resume else args.seed,adapter=sys.modules[__name__])
