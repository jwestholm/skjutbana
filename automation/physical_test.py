"""Physical-test lifecycle: start, check, label, classify, evaluate and restore settings."""
from __future__ import annotations
import argparse,json,os,socket,subprocess,sys,uuid,hashlib
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from automation.physical_session_audit import audit,distance
from automation.physical_trace_export import export
from src.engine.ai.canonical_challenger import CanonicalChallenger
from src.engine.ai.confirmation_selection_shadow import CONFIG_HASH
from src.engine.physical_trace import selector_snapshot

ROOT=Path(__file__).resolve().parents[1]
ACTIVE=ROOT/'evaluation_runs/physical_test_active.json'
SETTINGS=ROOT/'content/ai/settings.json'
DEFAULT_CHALLENGER=ROOT/'evaluation_runs/overnight_20260907/ranking/challenger.json'
FROZEN_SHADOW_HASH='123a2e510f545895adbee1def7c1a29e17e860af8bad050cfad28cfee94c1d9f'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def _digest(path):
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def _git_commit():
    try: return subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    except Exception: return None


def _xy(value):
    return value if isinstance(value,dict) and 'camera_x' in value and 'camera_y' in value else None


def _selector_metrics(rows):
    metrics={}
    for name in ('CURRENT_DETERMINISTIC','CONFIRMATION_SELECTION_SHADOW','CANONICAL_AI_SHADOW'):
        errors=[]; oracle={str(r):0 for r in (5,10,20,42)}; hits={str(r):0 for r in (5,10,20,42)}
        conditional_hits={str(r):0 for r in (5,10,20,42)}; conditional_den={str(r):0 for r in (5,10,20,42)}
        misses=0
        for row in rows:
            gt=row.get('ground_truth'); selected=(row.get('selectors',{}).get(name) or {}).get('selected')
            if not gt: continue
            err=distance(selected,gt) if _xy(selected) else None
            if err is None: misses+=1; continue
            errors.append(float(err))
            for radius in (5,10,20,42):
                key=str(radius); available=row.get('oracle_candidate_available',{}).get(key)
                if available: oracle[key]+=1; conditional_den[key]+=1
                if err<=radius:
                    hits[key]+=1
                    if available: conditional_hits[key]+=1
        errors_sorted=sorted(errors)
        p95=errors_sorted[max(0,int(np.ceil(.95*len(errors_sorted)))-1)] if errors_sorted else None
        metrics[name]={'shots_with_ground_truth':sum(bool(r.get('ground_truth')) for r in rows),
            'selected_evaluated':len(errors),'misses_or_timeouts':misses,'oracle_candidate_available':oracle,
            'accuracy':{str(r):hits[str(r)]/len(errors) if errors else None for r in (5,10,20,42)},
            'conditional_accuracy':{str(r):conditional_hits[str(r)]/conditional_den[str(r)] if conditional_den[str(r)] else None for r in (5,10,20,42)},
            'conditional_denominator':conditional_den,'errors_gt_100_px':sum(e>100 for e in errors),
            'mean_error_px':float(np.mean(errors)) if errors else None,'median_error_px':float(np.median(errors)) if errors else None,'p95_error_px':p95}
    return metrics


def _validation_status(root, rows, real, manifest):
    setup_path=root/'test_setup.json'; setup=json.loads(setup_path.read_text()) if setup_path.exists() else {}
    if not setup or not setup.get('session_git_commit') or not setup.get('frozen_shadow_config_hash') or not setup.get('canonical_manifest_sha256'):
        return {'status':'DEVELOPMENT','reason':'session setup metadata lacks frozen provenance; validation cannot be established','event_count':len(rows)}
    expected=setup.get('frozen_shadow_config_hash',FROZEN_SHADOW_HASH)
    hashes=[]; source_commits=[]; labels_before_selection=[]
    for row,shot in zip(rows,real):
        shadow=(shot.get('selectors',{}).get('CONFIRMATION_SELECTION_SHADOW') or {})
        if shadow.get('config_hash') is not None: hashes.append(shadow.get('config_hash'))
        trace_path=root/'shots'/f"shot_{int(row['shot_id']):08d}"/'trace.json'
        trace=json.loads(trace_path.read_text())
        source_commits.append(trace.get('provenance',{}).get('git_commit'))
        gt=row.get('ground_truth') or {}; attached=gt.get('attached_at')
        if attached is not None and float(attached) < float(trace.get('created_at',0.0)): labels_before_selection.append(row['shot_id'])
    hash_ok=bool(hashes) and all(value==expected for value in hashes)
    commit_ok=bool(set(source_commits)) and len(set(source_commits))==1 and source_commits[0]==setup.get('session_git_commit')
    labels_ok=not labels_before_selection
    manifest_ok=bool(manifest and setup.get('canonical_manifest_sha256') in (None,_digest(Path(manifest))) )
    status='INDEPENDENT_PHYSICAL_VALIDATION' if hash_ok and commit_ok and labels_ok and manifest_ok else ('INVALIDATED' if not hash_ok or not commit_ok or not labels_ok else 'DEVELOPMENT')
    return {'status':status,'selector_hash_expected':expected,'selector_hashes_observed':sorted(set(hashes)),
            'selector_hash_match':hash_ok,'session_git_commit':setup.get('session_git_commit'),
            'trace_git_commits':sorted(set(source_commits)),'source_commit_match':commit_ok,
            'labels_before_selection':labels_before_selection,'labels_frozen_after_selection':labels_ok,
            'canonical_manifest_hash_match':manifest_ok,'event_count':len(rows)}


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
    state={'root':str(root),'before':{k:{'present':k in settings,'value':settings.get(k)} for k in changes},'applied':changes,'restored':False,
           'session_git_commit':_git_commit(),'frozen_shadow_config_hash':FROZEN_SHADOW_HASH,
           'canonical_manifest_sha256':_digest(challenger),'settings_snapshot':settings.copy(),
           'validation_label':'DEVELOPMENT'}
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
        selectors=selector_snapshot(trace, str(manifest) if manifest else None)
        canonical=selectors.get('CANONICAL_AI_SHADOW',{})
        order=canonical.get('order',[]); candidates=canonical.get('candidates',[])
        if order and candidates:
            chosen_index=order[0]
            chosen=next((c for c in candidates if c.get('input_index')==chosen_index),None)
            if chosen:
                canonical['selected']={'camera_x':chosen.get('camera_x'),'camera_y':chosen.get('camera_y'),**chosen}
        row['selectors']=selectors
        retained_pool=trace.get('decision_input',{}).get('retained_candidates') or []
        row['oracle_candidate_available']={str(r):any(gt and distance(c,gt)<=r for c in retained_pool) for r in (5,10,20,42)}
        for name,selector in selectors.items():
            chosen=_xy(selector.get('selected')) if isinstance(selector,dict) else None
            row.setdefault('selector_distances_px',{})[name]=distance(chosen,gt) if chosen and gt else None
        shot['selectors']=selectors
        shot['oracle_candidate_available']=row['oracle_candidate_available']
        shot['selector_distances_px']=row['selector_distances_px']
    payload['shots']=real;write(output/'evaluation_trace.json',payload);write(output/'false_events.json',false)
    result['selector_metrics']=_selector_metrics(real)
    result['confirmation_selection_shadow_config_hash']=CONFIG_HASH
    result['validation']=_validation_status(root,real,real,manifest)
    write(output/'physical_comparison.json',result)
    from src.engine.offline.evaluation import evaluate
    write(output/'metrics.json',evaluate(real,mode='live_path_replay'))
    print('Evaluation:',output,'Real/unresolved events:',len(real),'Nonphysical events:',len(false))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['start','check','label','classify','evaluate','stop'])
    location=p.add_mutually_exclusive_group()
    location.add_argument('--root',type=Path);location.add_argument('--binding',type=Path,help='Use this collection binding for check/label/classify/evaluate')
    p.add_argument('--output',type=Path);p.add_argument('--mapping',type=Path)
    p.add_argument('--challenger',type=Path,default=DEFAULT_CHALLENGER);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--shot-id',type=int);p.add_argument('--label-shot-id',type=int);p.add_argument('--no-physical-shot',action='store_true');p.add_argument('--reason')
    a=p.parse_args()
    try:
        if a.command=='start':
            if a.binding:raise ValueError('A collection binding is already prepared; launch main.py instead of physical_test start.')
            start(a);return
        state=json.loads(ACTIVE.read_text()) if not a.binding and ACTIVE.exists() else {}
        if a.binding:
            from automation.physical_label_reset import resolve_session
            root,_=resolve_session(binding=a.binding)
        else:root=a.root or (Path(state['root']) if state else None)
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
