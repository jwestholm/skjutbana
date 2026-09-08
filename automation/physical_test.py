"""Physical-test lifecycle: start, check, label, classify, evaluate and restore settings."""
from __future__ import annotations
import argparse,json,os,socket,subprocess,sys,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from automation.physical_session_audit import audit,distance
from automation.physical_trace_export import export
from src.engine.ai.canonical_challenger import CanonicalChallenger

ROOT=Path(__file__).resolve().parents[1]
ACTIVE=ROOT/'evaluation_runs/physical_test_active.json'
SETTINGS=ROOT/'content/ai/settings.json'
DEFAULT_CHALLENGER=ROOT/'evaluation_runs/overnight_20260907/ranking/challenger.json'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def start(args):
    try:
        with socket.create_connection(('127.0.0.1',8765),timeout=.3):
            raise ValueError('Close the running application before start: settings are loaded at application startup.')
    except OSError: pass
    if ACTIVE.exists() and not json.loads(ACTIVE.read_text()).get('restored'):
        raise ValueError('A session is already active; use stop after closing the application, or pass --root to check an older session.')
    settings=json.loads(SETTINGS.read_text())
    if settings.get('mode') not in ('off','train_only','advisory'):
        raise ValueError('Physical test requires off, train_only or advisory mode; authority modes are not allowed.')
    challenger=args.challenger.resolve()
    CanonicalChallenger(challenger) # Validate pinned artifacts BEFORE settings changes.
    name=datetime.now(timezone.utc).strftime('session_%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
    root=ROOT/'content/ai/physical_traces'/name;root.mkdir(parents=True,exist_ok=False)
    changes={'physical_trace_capture_enabled':True,'physical_trace_root':str(root),
             'physical_trace_session_id':name,'canonical_challenger_manifest':str(challenger)}
    state={'root':str(root),'before':{k:{'present':k in settings,'value':settings.get(k)} for k in changes},'applied':changes,'restored':False}
    write(root/'test_setup.json',state);write(ACTIVE,state);settings.update(changes);write(SETTINGS,settings)
    print('Session:',root)
    if not args.prepare_only:
        with (root/'runtime.log').open('xb') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'main.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['pid']=proc.pid;write(ACTIVE,state);print('Application PID:',proc.pid,'— runtime log:',root/'runtime.log')
    else:print('Prepared. Start python3 main.py before shooting; capture is not yet verified.')


def health(root):
    results=[]
    for directory in sorted((root/'shots').glob('shot_*')):
        problems=[];p=directory/'trace.json'
        if not p.exists():results.append({'shot':directory.name,'problems':['trace.json missing; capture may still be running']});continue
        try:
            t=json.loads(p.read_text())
            if not t.get('completeness',{}).get('trace_complete'):problems.append('producer reports incomplete trace')
            frames=t.get('frames',[])
            for kind in ('pre','post'):
                if not any(f.get('kind','').startswith(kind) for f in frames):problems.append(kind+' frames missing')
            for f in frames:
                artifact=(directory/f['path']).resolve()
                if not artifact.is_relative_to(directory.resolve()):raise ValueError('artifact escapes shot directory')
                a=np.load(artifact,mmap_mode='r',allow_pickle=False)
                if list(a.shape)!=f['shape'] or str(a.dtype)!=f['dtype']:problems.append('frame metadata mismatch '+f['path'])
            for stage in t.get('stages',[]):
                for f in stage.get('evidence_maps',{}).values():
                    if not isinstance(f,dict) or 'path' not in f:problems.append('unpersisted evidence map');continue
                    artifact=(directory/f['path']).resolve()
                    if not artifact.is_relative_to(directory.resolve()):raise ValueError('map escapes shot directory')
                    a=np.load(artifact,mmap_mode='r',allow_pickle=False)
                    if list(a.shape)!=f['shape']:problems.append('map shape mismatch')
            results.append({'shot':directory.name,'peak_ts':t['peak_ts'],'labeled':(directory/'ground_truth.json').exists(),
                'mode':t.get('provenance',{}).get('effective_detector_settings',{}).get('mode'),
                'audio_diagnostics':bool(t.get('audio_trigger')),'challenger':t.get('canonical_challenger',{}).get('status'),'problems':problems})
        except (ValueError,OSError,KeyError,TypeError) as exc:results.append({'shot':directory.name,'problems':[str(exc)]})
    return {'session':str(root),'events':len(results),'healthy':bool(results) and all(not r['problems'] for r in results),'shots':results}


def evaluate_session(root,output,manifest,mapping=None):
    checked=health(root)
    if not checked['healthy']:raise ValueError('Trace health failed; run check and finish/repair capture first.')
    result=audit(root,mapping);challenger=CanonicalChallenger(manifest) if manifest else None
    output.mkdir(parents=True,exist_ok=False);write(output/'health.json',checked)
    export(root,output/'evaluation_trace.json');payload=json.loads((output/'evaluation_trace.json').read_text());false=[];real=[]
    for row,shot in zip(result['shots'],payload['shots']):
        if row['physical_state']=='NO_PHYSICAL_SHOT':false.append(row);continue
        gt = row['ground_truth']
        quality = row['physical_state'] if row['physical_state'] in ('precise','approximate') else 'unknown'
        if quality == 'approximate' and gt is not None and 'uncertainty_radius_px' not in gt:
            quality = 'unknown'  # Legacy clicks have no defensible uncertainty radius.
        shot['ground_truth'] = {**gt, 'quality': quality} if gt else None
        real.append(shot)
        trace=json.loads((root/'shots'/f"shot_{int(row['shot_id']):08d}"/'trace.json').read_text())
        pool=trace.get('decision_input',{}).get('retained_candidates')
        if pool is None:pool=next((s['candidates'] for s in reversed(trace['stages']) if s.get('candidates')),[])
        if challenger:
            ranked=challenger.rank(pool);row['challenger']=ranked
            chosen=pool[ranked['order'][0]] if ranked['order'] else None;gt=row['ground_truth']
            row['challenger_distance_px']=distance(chosen,gt) if chosen and gt else None
            row['challenger_positive_ranks']={str(r):next((j+1 for j,i in enumerate(ranked['order']) if gt and distance(pool[i],gt)<=r),None) for r in (5,10,20,42)}
            row['shadow_semantics']='post-decision retained-pool ranking; ignores confirmation eligibility; not live-path replay'
    payload['shots']=real;write(output/'evaluation_trace.json',payload);write(output/'false_events.json',false);write(output/'physical_comparison.json',result)
    from src.engine.offline.evaluation import evaluate
    write(output/'metrics.json',evaluate(real,mode='live_path_replay'))
    print('Evaluation:',output,'Real/unresolved events:',len(real),'Nonphysical events:',len(false))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['start','check','label','classify','evaluate','stop'])
    p.add_argument('--root',type=Path);p.add_argument('--output',type=Path);p.add_argument('--mapping',type=Path)
    p.add_argument('--challenger',type=Path,default=DEFAULT_CHALLENGER);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--shot-id',type=int);p.add_argument('--label-shot-id',type=int);p.add_argument('--no-physical-shot',action='store_true');p.add_argument('--reason')
    a=p.parse_args()
    try:
        if a.command=='start':start(a);return
        state=json.loads(ACTIVE.read_text()) if ACTIVE.exists() else {}
        root=a.root or (Path(state['root']) if state else None)
        if root is None:raise ValueError('No active session; specify --root.')
        if a.command=='check':
            h=health(root);print(json.dumps(h,indent=2));sys.exit(0 if h['healthy'] else 1)
        elif a.command=='label':subprocess.run([sys.executable,'-m','automation.physical_trace_label','--root',str(root)],cwd=ROOT,check=True)
        elif a.command=='classify':
            if not a.shot_id or not a.reason or (a.no_physical_shot==(a.label_shot_id is not None)):raise ValueError('Provide --shot-id, --reason and exactly one of --no-physical-shot or --label-shot-id.')
            if not (root/'shots'/f'shot_{a.shot_id:08d}'/'trace.json').exists():raise ValueError('Unknown event')
            if a.label_shot_id is not None and not (root/'shots'/f'shot_{a.label_shot_id:08d}'/'ground_truth.json').exists():raise ValueError('Label file does not exist')
            path=root/'physical_assignments.json';m=json.loads(path.read_text()) if path.exists() else {}
            m[str(a.shot_id)]={'label_shot_id':a.label_shot_id,'reason':a.reason}
            if a.no_physical_shot:m[str(a.shot_id)]['state']='NO_PHYSICAL_SHOT'
            write(path,m);print(path)
        elif a.command=='evaluate':
            path=a.mapping or root/'physical_assignments.json';m=json.loads(path.read_text()) if path.exists() else None
            output=a.output or ROOT/'evaluation_runs'/('physical_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
            evaluate_session(root,output,a.challenger,m)
        elif a.command=='stop':
            if not state or root!=Path(state['root']):raise ValueError('stop only restores the active helper session')
            if state.get('pid'):
                try:os.kill(state['pid'],0)
                except ProcessLookupError:pass
                else:raise ValueError('Close the helper-started application first so its trace is flushed.')
            current=json.loads(SETTINGS.read_text());conflicts=[k for k,v in state['applied'].items() if current.get(k)!=v]
            if conflicts:raise ValueError('Trace settings changed independently; refusing to overwrite: '+', '.join(conflicts))
            for k,before in state['before'].items():
                if before['present']:current[k]=before['value']
                else:current.pop(k,None)
            write(SETTINGS,current);state['restored']=True;write(ACTIVE,state);print('Previous trace settings restored; unrelated settings preserved.')
    except (ValueError,OSError) as exc:p.exit(1,str(exc)+'\n')

if __name__=='__main__':main()
