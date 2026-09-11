"""Reproduce the label-blind split audit and original baseline before fitting."""
import json, hashlib, math
from pathlib import Path
from src.engine.offline.evaluation import evaluate, encoded, digest, provenance
import argparse
parser=argparse.ArgumentParser(description='Freeze chronological native splits and pre-experiment reference, without model training.')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]; out=args.output
out.mkdir(parents=True,exist_ok=False)
dirs=sorted((root/'content/ai/training_v223/sessions').iterdir())
sessions=[d for d in dirs if len(list(d.glob('shot_*.json')))>=50]
assert len(sessions)==3
splits={}; prior=[]; excluded=[]
# Later sessions take precedence in label-blind duplicate purging.
for label,d in reversed(list(zip(('TRAIN','DEVELOPMENT','PROTECTED_FINAL_HOLDOUT'),sessions))):
 entries=[]
 for p in sorted(d.glob('shot_*.json')):
  r=json.loads(p.read_text()); cs=r['candidates']
  signature=frozenset((round(c['camera_x']/8),round(c['camera_y']/8)) for c in cs)
  exact=hashlib.sha256(encoded(cs).encode()).hexdigest()
  duplicate=next((old for old,sig,h in prior if exact==h or (signature and sig and len(signature&sig)/len(signature|sig)>=.9)),None)
  if duplicate:
   excluded.append({'path':str(p.relative_to(root)),'reason':'cross_split_candidate_snapshot_duplicate','matches':duplicate});continue
  entries.append({'path':str(p.relative_to(root)), 'sha256':digest(p),'session_id':r['session_id'],'shot_id':r['shot_id'], 'candidate_fingerprint':exact})
 # Do not dedupe within session: it stays together and represents the captured sequence.
 for e in entries:
  r=json.loads((root/e['path']).read_text())
  prior.append((e['path'],frozenset((round(c['camera_x']/8),round(c['camera_y']/8)) for c in r['candidates']),e['candidate_fingerprint']))
 splits[label]=entries
manifest={'schema_version':'offline-10iter-1','base_commit':'ad03f4c','splits':splits,'excluded':excluded,
 'excluded_sessions':[d.name for d in dirs if d not in sessions],
 'split_policy':'chronological whole sessions; purge earlier snapshots if exact candidate hash or 8px quantized coordinate-set Jaccard >= 0.9 matches later split; label-blind',
 'limitations':['Same camera/background/date; unavailable PRE/POST imagery in oldest sessions prevents visual near-duplicate certification.', 'Newest session historical aggregate pool recall exposed before this experiment; no holdout ranking/model results used in tuning.']}
(out/'splits.json').write_text(encoded(manifest))
config={'primary_metric':'DEVELOPMENT Top-1 <=20 camera px / all included shots','top_k':10,'tolerances':[5,10,20,42],
 'guardrails':'identical full candidate IDs/XY/order membership; Top-10 recall at each tolerance >= original baseline; select strictly higher Top1@20, ties use all-shot MRR@20 then Top3@20; exact ties keep incumbent',
 'baseline':'complete descending baseline_score (stable ties); offline reference, not live final rank',
 'training':'TRAIN only, existing train_rank_model; seed=2230; no pretrained weights; no model registry calls',
 'holdout':'open for scoring only after iteration 10 and freeze; baseline and selected challenger exactly once each'}
(out/'protocol.json').write_text(encoded(config))
shots=[]
for e in splits['DEVELOPMENT']:
 r=json.loads((root/e['path']).read_text());cs=r['candidates']
 assert all(c['baseline_score'] is not None for c in cs)
 order=sorted(cs,key=lambda c:-c['baseline_score'])
 shots.append({'session_id':r['session_id'],'shot_id':r['shot_id'],'coordinate_space':'camera','source_kind':r['source_kind'], 'ground_truth':{'camera_x':r['gt_camera_x'],'camera_y':r['gt_camera_y']},'saved_pool':cs,'ranked':order})
r=evaluate(shots,mode='offline_candidate',top_k=10)
r['provenance']=provenance(root,[root/e['path'] for e in splits['DEVELOPMENT']],'offline-10iter-development',{'effective_detector_settings':None,'models':None,'calibration':None,'evaluation_reference':config['baseline']})
(out/'baseline_before_changes.json').write_text(encoded(r))
print('Split counts:',{k:len(v) for k,v in splits.items()},'purged',len(excluded))
for tol,v in r['tolerances_camera_px'].items():
 print(tol,{k:v['stages'][k]['correct'] for k in ('top_1','top_3','top_k','saved_pool')})
