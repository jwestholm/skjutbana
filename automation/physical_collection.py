"""Safe research collection orchestration and validation helpers."""
from __future__ import annotations
import argparse,json,uuid,subprocess,os,hashlib,shutil
import sys,signal,re
from datetime import datetime,timezone
from pathlib import Path

def load_plan(path):
 d=json.loads(Path(path).read_text());rows=d.get('rows',[])
 if not rows:raise ValueError('empty capture plan')
 sessions=sorted(set(r['session'] for r in rows));
 if len(sessions)!=d.get('sessions'):raise ValueError('session count mismatch')
 return d
def bind(plan,session,trace_root,source_commit='unknown'):
 rows=[r for r in plan['rows'] if r['session']==session]
 if not rows:raise ValueError('unknown session')
 existing=Path(trace_root)/f'planned_{session}.json'
 if existing.exists():raise FileExistsError(f'{session} already has a binding at {existing}; choose a new plan/session')
 if session=='S03' and any(r.get('session_class')!='VALIDATION_UNTOUCHED' for r in rows):raise ValueError('validation binding mismatch')
 meta=dict(collection_plan_id=plan.get('collection_plan_id',plan.get('plan_id','local-plan')),planned_session_id=session,session_class=rows[0]['session_class'],created_at=datetime.now(timezone.utc).isoformat(),source_commit=source_commit,trace_root=str(trace_root),rows=rows)
 return meta
def guard_training(session_class,operation):
 if session_class=='VALIDATION_UNTOUCHED' and operation in ('train','tune','fit','select_config','model_select','threshold_select','research_calibrate'):
  raise PermissionError('validation data is evaluation-only; training/tuning refused')
 return True
def _settings(path):
 p=Path(path); return json.loads(p.read_text())

def _development_path(path):
 path=Path(path).absolute()
 def forbidden(candidate):
  return any(re.search(r'(^|_)S03(_|\.|$)',part,re.I) for part in candidate.parts)
 if forbidden(path):raise ValueError('Validation paths are forbidden for development capture')
 if forbidden(path.resolve()):raise ValueError('Validation paths are forbidden for development capture')
 return path

def verify_capture_launch(binding_path, *, claim=False, expected_session=None):
 """Fail closed before importing App. Process-scoped capture; no settings write."""
 binding_path=_development_path(binding_path)
 b=json.loads(binding_path.read_text())
 session=b.get('planned_session_id')
 if session not in ('CAN_OBJECT','D02','KOMPETENSPROV_AK') or b.get('capture_configuration')!='PROCESS_ONLY' or (expected_session and session!=expected_session):
  raise ValueError('launch requires an explicit supported process-only development binding')
 if b.get('session_class')!='DEVELOPMENT':raise ValueError('development capture required')
 root=_development_path(b['trace_root'])
 if not Path(b['trace_root']).is_absolute() or not root.name.startswith('session_') or f'_{session}_' not in root.name:
  raise ValueError('not a fresh explicit development root')
 if root.is_symlink() or root.resolve()!=root or not root.is_dir():
  raise ValueError('capture root missing or indirect')
 if json.loads((root/'collection_binding.json').read_text())!=b:
  raise ValueError('root binding differs from launch binding')
 plan_path=_development_path(b['plan_path']);plan=load_plan(plan_path)
 if plan.get('collection_plan_id')!=b['collection_plan_id'] or plan['rows']!=b['rows']:
  raise ValueError('capture plan differs from binding')
 if b.get('plan_sha256') and hashlib.sha256(plan_path.read_bytes()).hexdigest()!=b['plan_sha256']:
  raise ValueError('capture plan hash differs from binding')
 if session in ('D02','KOMPETENSPROV_AK'):
  from src.engine.physical_trace import validate_capture_history
  validate_capture_history(b.get('capture_history'))
  if not b.get('capture_history'):raise ValueError('Development capture history configuration missing')
  settings_path=Path(__file__).resolve().parents[1]/'content/ai/settings.json'
  if hashlib.sha256(settings_path.read_bytes()).hexdigest()!=b.get('saved_settings_sha256'):
   raise ValueError('Saved settings changed since preparation; create a fresh launch')
 if (root/'shots').exists() and any((root/'shots').iterdir()):
  raise ValueError('capture root already contains shots; create a new launch')
 if (root/'runtime_claim.json').exists():
  raise ValueError('this binding was already launched; create a new launch')
 if not os.access(root,os.W_OK):raise ValueError('capture root not writable')
 if claim:
  with (root/'runtime_claim.json').open('x') as f:
   json.dump(dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat()),f)
 return b

def verify_can_launch(binding_path, *, claim=False):
 return verify_capture_launch(binding_path,claim=claim,expected_session='CAN_OBJECT')

def _write_process_binding(output,trace_root,plan_path,session,**extra):
 root=_development_path(trace_root).resolve()/f'session_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{session}_{uuid.uuid4().hex[:8]}'
 root.mkdir(parents=True,exist_ok=False)
 head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 b=bind(load_plan(plan_path),session,root,head)
 b.update(capture_configuration='PROCESS_ONLY',plan_path=str(plan_path),status='PREPARED_NOT_CAPTURED',**extra)
 for p in (root/'collection_binding.json',output/'binding.json'):
  with p.open('x') as f:json.dump(b,f,indent=2)
 return output/'binding.json'

def prepare_can_launch(output, trace_root):
 """Prepare the operator's fresh three-shot retry; never invoked on import."""
 output=Path(output).absolute();output.mkdir(parents=True,exist_ok=False)
 plan=load_plan(Path(__file__).resolve().parents[1]/'research/physical_capture_plans/CAN_OBJECT.json')
 plan['collection_plan_id']='CAN_OBJECT_retry_'+uuid.uuid4().hex
 plan['rows']=plan['rows'][:3]
 plan['protocol']=[
  'SFX OFF, 20 m. Wait for LIVE and settled projection before each shot.',
  '1: intentional can hit. 2: intentional duck hit. 3: physical impact just outside an object.',
  'Independently note actual damage, event IDs and ambiguity; intended targets are not XY truth.',
  'No capture/no audio means NOT_CAPTURED. Extra audio events need separate notes.',
  'OBJECT reacts immediately; later CURRENT may see reaction pixels and is comparison only.']
 plan_path=output/'plan.json';plan_path.write_text(json.dumps(plan,indent=2)+'\n')
 binding=_write_process_binding(output,trace_root,plan_path,'CAN_OBJECT')
 head=json.loads(binding.read_text())['source_commit']
 (output/'event_notes.json').write_text(json.dumps(dict(truth_source='independent human observation',
  events=[dict(planned_physical_shot=i,actual_event_ids=[],human_observation=None,status='NOT_OBSERVED') for i in (1,2,3)]),indent=2))
 (output/'geometry_notes.json').write_text(json.dumps(dict(observer=None,board_revision=None,
  board_bounds_camera_px=None,seam_or_unavailable_reason=None,calibration_identity=None,
  camera_projection_setup_notes=None,status='HUMAN_OBSERVATION_PENDING'),indent=2))
 source_files=['main.py','automation/physical_collection.py','content/games/can_shootout_game.py',
  'src/engine/camera/object_physical_query.py','src/engine/input/object_hit_v2223.py',
  'src/engine/game_objects/geometry.py','src/engine/game_objects/manager.py','src/engine/physical_trace.py']
 (output/'source_provenance.json').write_text(json.dumps(dict(head=head,
  source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_files}),indent=2))
 verify_can_launch(output/'binding.json')
 return output/'binding.json'

def _prepare_history_launch(output,trace_root,plan_path,session,*,capture_history=None):
 """Copy the accepted plan into a fresh run; never edit saved settings."""
 plan_path=_development_path(plan_path);output=_development_path(output)
 _development_path(trace_root)
 plan=load_plan(plan_path)
 if session not in ('D02','KOMPETENSPROV_AK') or plan.get('sessions')!=1 or any(r.get('session')!=session or r.get('session_class')!='DEVELOPMENT' for r in plan['rows']):
  raise ValueError('An explicit matching DEVELOPMENT plan is required')
 from automation.physical_capture_provenance import saved_configuration
 from src.engine.physical_trace import validate_capture_history
 history=validate_capture_history(capture_history if capture_history is not None else
  dict(pre_seconds=4.,post_seconds=5.,pre_max_frames=240,post_max_frames=300,max_events=16))
 settings_path=Path(__file__).resolve().parents[1]/'content/ai/settings.json'
 settings_bytes=settings_path.read_bytes()
 if json.loads(settings_bytes).get('mode','off') not in ('off','train_only','advisory'):
  raise ValueError('Development capture requires existing CURRENT authority; saved AI mode is not compatible')
 output.mkdir(parents=True,exist_ok=False)
 shutil.copyfile(plan_path,output/'plan.json')
 for name in ('geometry_notes.json','event_notes.json'):
  source=_development_path(plan_path.parent/name)
  if source.exists():shutil.copyfile(source,output/name)
 binding=_write_process_binding(output,trace_root,output/'plan.json',session,capture_history=history,
  plan_sha256=hashlib.sha256((output/'plan.json').read_bytes()).hexdigest(),
  saved_settings_sha256=hashlib.sha256(settings_bytes).hexdigest())
 (output/'saved_configuration.json').write_text(json.dumps(saved_configuration(),indent=2)+'\n')
 verify_capture_launch(binding)
 return binding

def prepare_d02_launch(output,trace_root,plan_path,*,capture_history=None):
 return _prepare_history_launch(output,trace_root,plan_path,'D02',capture_history=capture_history)

def prepare_kompetensprov_launch(output,trace_root):
 repo=Path(__file__).resolve().parents[1]
 binding=_prepare_history_launch(output,trace_root,repo/'research/physical_capture_plans/KOMPETENSPROV_AK.json','KOMPETENSPROV_AK')
 (binding.parent/'event_notes.json').write_text(json.dumps(dict(truth_source='independent human observation; intent/game output is not truth',
  events=[dict(planned_physical_shot=i,actual_event_ids=[],human_observation=None,actual_camera_xy=None,status='NOT_OBSERVED') for i in range(1,7)],
  unexpected_audio_events=[],geometry_notes=None),indent=2)+'\n')
 files=['content/games/kompetensprov_ak/game.py','content/games/kompetensprov_ak/rules.py',
  'assets/games/kompetensprov_ak/hf2020.png','assets/games/kompetensprov_ak/hf2020_zone_mask.png',
  'automation/physical_collection.py','main.py']
 (binding.parent/'source_provenance.json').write_text(json.dumps({f:hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
 return binding

def launch_kompetensprov(output,trace_root):
 binding=prepare_kompetensprov_launch(output,trace_root)
 print(f'KOMPETENSPROV DEVELOPMENT: {binding}\nDIAGNOSTIC MODE: silent/static, no spoken commands or normal competition sequence.\nSelect Kompetensprov AK. ENTER begins test; wait 5 seconds before/after shots. ENTER ends after final POST.\nNormal game mode: start python3 main.py without capture environment overrides.',flush=True)
 return _run_capture(binding,dict(os.environ,SKJUTBANA_PHYSICAL_CAPTURE_BINDING=str(binding),SKJUTBANA_KOMPETENSPROV_TEST='1'))

def require_can_foundations(report_path):
 if report_path is None:
  raise ValueError('OBJECT foundation preflight required before shooting: run no-shoot geometry/sync checks; live PRE is not validated')
 report=json.loads(Path(report_path).read_text())
 if report.get('status')!='READY' or report.get('ready_to_shoot') is not True:
  raise ValueError('OBJECT foundation preflight BLOCKED: '+ '; '.join(report.get('reasons', ['geometry/timing/PRE not validated'])))
 return report

def launch_can(output,trace_root,foundation_report=None):
 # Check before creating a binding or starting any hardware. Current foundation
 # diagnostics intentionally return BLOCKED until a live PRE policy is validated.
 require_can_foundations(foundation_report)
 binding=prepare_can_launch(output,trace_root)
 print(f'CAPTURE PREFLIGHT PASSED: {binding}\nSelect Skjut burkar, 20 m; SFX OFF. No CURRENT visual hold.',flush=True)
 env=dict(os.environ,SKJUTBANA_PHYSICAL_CAPTURE_BINDING=str(binding),SKJUTBANA_CAN_SFX='0',SKJUTBANA_OBJECT_COMPARE='0')
 return _run_capture(binding,env)

def prepare_can_play(output):
 """Compact gameplay acceptance, not a full research-capture binding."""
 output=_development_path(output)
 if not output.resolve().is_relative_to(Path('/data/skjutbana')):
  raise ValueError('Gameplay diagnostics must be under /data/skjutbana')
 output.mkdir(parents=True,exist_ok=False)
 targets=['mini_can','soda_can','gold_can','beans_xxl','sardines']
 notes=[dict(shot=i+1,player=1+i//5,intended_object=targets[i%5],
   observed_physical_object=None,shot_id=None,result=None,notes='') for i in range(10)]
 notes.extend(dict(shot=i+11,player=1,intended_object=None,observed_physical_object=None,
   shot_id=None,result=None,notes='deliberate outside/between active objects') for i in range(5))
 manifest=dict(mode='OBJECT_HIT_GAMEPLAY_ACCEPTANCE',game='shoot_stuff_range',
   menu_title='Skjutprylar',full_frame_capture=False,live_accuracy_validated=False,
   head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
   settings_sha256=hashlib.sha256(Path('content/ai/settings.json').read_bytes()).hexdigest(),
   source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
     'src/engine/camera/object_physical_query.py','src/engine/input/hit_regions.py',
     'src/engine/input/object_hit_v2223.py','src/engine/game_objects/manager.py',
     'src/engine/physical_trace.py','content/games/can_shootout_game.py',__file__)})
 path=output/'play.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
 (output/'event_notes.json').write_text(json.dumps(dict(
   truth='Independent observed impact/object, not intended aim or game reaction',events=notes,
   target_results=['correct_object','wrong_object','no_object_detected'],
   control_results=['correctly_no_object','false_object_hit'],
   unexpected_audio_events=[]),indent=2)+'\n')
 env=dict(os.environ)
 for key in ('SKJUTBANA_PHYSICAL_CAPTURE_BINDING','SKJUTBANA_PHYSICAL_TRACE_ROOT',
             'SKJUTBANA_PHYSICAL_CAPTURE_HISTORY','SKJUTBANA_KOMPETENSPROV_TEST'):
  env.pop(key,None)
 env.update(SKJUTBANA_PHYSICAL_TRACE_DISABLE='1',SKJUTBANA_CAN_SFX='0',SKJUTBANA_OBJECT_COMPARE='0')
 return path,env

def play_can(output):
 manifest,env=prepare_can_play(output)
 print('OBJECT_HIT gameplay test: select Spel > Skjutprylar, 20 m. Normal game, SFX OFF.\n'
       'Shoot mini/soda/gold/beans/sardines once in player 1, then once each in player 2.\n'
       'Press R after the match: five deliberate outside shots. Wait for LIVE/settled image each time.\n'
       'Record observed physical object, shot_id and result in '+str(manifest.parent/'event_notes.json')+'\n'
       'Compact logs only; full trace disabled for this process. Physical reliability is unverified.',flush=True)
 return _run_capture(manifest,env,compact=True)

def launch_d02(output,trace_root,plan_path,*,capture_history=None):
 binding=prepare_d02_launch(output,trace_root,plan_path,capture_history=capture_history)
 print(f'D02 CAPTURE BINDING: {binding}\nCURRENT unchanged. Use a static scene; complete geometry/event notes before shooting. Warm up at least 5 seconds.',flush=True)
 return _run_capture(binding,dict(os.environ,SKJUTBANA_PHYSICAL_CAPTURE_BINDING=str(binding)))

def _run_capture(binding,env,*,compact=False):
 with (binding.parent/'runtime.log').open('x') as log:
  process=subprocess.Popen([sys.executable,'-u','main.py'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  try:
   for line in process.stdout:print(line,end='');log.write(line);log.flush()
  except KeyboardInterrupt:
   if process.poll() is None:
    process.send_signal(signal.SIGINT)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:process.terminate()
  finally:
   if process.poll() is None:process.terminate()
   process.wait()
   exit_record=dict(returncode=process.returncode,finished_at=datetime.now(timezone.utc).isoformat(),
    metadata_finalization=('Compact gameplay log only; no trace requested' if compact else
      'Verify trace.json / trace_error.json; process exit alone is not trace completeness'))
   (binding.parent/'runtime_exit.json').write_text(json.dumps(exit_record,indent=2)+'\n')
   print(f'[CAPTURE] process exited returncode={process.returncode}',flush=True)
 return process.returncode
def _set_trace(settings_path,root,backup):
 p=Path(settings_path); original=p.read_bytes(); Path(backup).write_bytes(original)
 data=json.loads(original); data['physical_trace_capture_enabled']=True; data['physical_trace_root']=str(root)
 p.write_text(json.dumps(data,indent=2)+'\n'); return hashlib.sha256(original).hexdigest()
def preflight(plan_path,session,trace_root,binding=None,settings_path='content/ai/settings.json'):
 reasons=[]
 try: plan=load_plan(plan_path);rows=[r for r in plan['rows'] if r['session']==session]
 except Exception as e:return False,[f'plan unreadable: {e}']
 if not rows:reasons.append(f'plan has no session {session}')
 if (Path(trace_root)/f'planned_{session}.json').exists():reasons.append('session binding already exists; refusing overwrite')
 if not os.access(Path(trace_root),os.W_OK):reasons.append(f'trace root is not writable: {trace_root}')
 if binding:
  try:
   b=json.loads(Path(binding).read_text()); configured=_settings(settings_path)
   expected=str(Path(b['trace_root']))
   if b.get('capture_configuration')=='PROCESS_ONLY':
    verify_capture_launch(binding)
   else:
    if not configured.get('physical_trace_capture_enabled'): reasons.append('physical trace capture is disabled; run start again')
    if str(configured.get('physical_trace_root','')) != expected: reasons.append(f'physical trace root is {configured.get("physical_trace_root")!r}, expected {expected!r}; run start or restore binding')
   if Path(expected).name.startswith('session_20260907_'): reasons.append('refusing historical trace root; use the unique root in the binding')
  except Exception as e: reasons.append(f'binding/settings unreadable: {e}')
 try: subprocess.run(['git','rev-parse','--abbrev-ref','HEAD'],check=True,capture_output=True,text=True)
 except Exception:reasons.append('git branch cannot be verified')
 return not reasons,reasons
def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
 play=sub.add_parser('play-can',help='Existing Skjutprylar gameplay; compact 15-shot notes, no full traces/settings writes')
 play.add_argument('--output-dir',type=Path,required=True)
 c=sub.add_parser('launch-can',help='Fresh captured three-shot Can retry; never edits settings.json')
 c.add_argument('--foundation-report',type=Path,help='Required READY no-shoot geometry/timing/PRE preflight; current policy remains blocked')
 c.add_argument('--output-dir',type=Path,required=True);c.add_argument('--trace-root',type=Path,default=Path('/data/skjutbana/physical_traces'))
 d=sub.add_parser('launch-d02',help='Fresh process-only D02 capture; never edits settings.json')
 d.add_argument('--plan',type=Path,required=True);d.add_argument('--output-dir',type=Path,required=True)
 d.add_argument('--trace-root',type=Path,default=Path('/data/skjutbana/physical_traces'))
 k=sub.add_parser('launch-kompetensprov',help='Fresh six-shot physical diagnostic; process-only, CURRENT unchanged')
 k.add_argument('--output-dir',type=Path,required=True)
 k.add_argument('--trace-root',type=Path,default=Path('/data/skjutbana/physical_traces'))
 s=sub.add_parser('start');s.add_argument('--plan',type=Path,required=True);s.add_argument('--session',required=True);s.add_argument('--trace-root',type=Path,default=Path('content/ai/physical_traces'));s.add_argument('--source-commit',default='unknown');s.add_argument('--settings',type=Path,default=Path('content/ai/settings.json'));s.add_argument('--output',type=Path,required=True)
 f=sub.add_parser('preflight');f.add_argument('--plan',type=Path,required=True);f.add_argument('--session',required=True);f.add_argument('--trace-root',type=Path,default=Path('content/ai/physical_traces'));f.add_argument('--binding',type=Path);f.add_argument('--settings',type=Path,default=Path('content/ai/settings.json'))
 g=sub.add_parser('guard');g.add_argument('--class',dest='cls',required=True);g.add_argument('--operation',required=True)
 from automation.physical_finalize_manifest import add_arguments,finalize
 add_arguments(sub.add_parser('finalize',help='Finalize saved labels using an exact binding; --preview shows the mapping'))
 r=sub.add_parser('restore');r.add_argument('--binding',type=Path,required=True)
 from automation.physical_label_reset import add_reset_arguments,reset_labels
 add_reset_arguments(sub.add_parser('reset-labels',help='Preview or archive/reset labels; preserve captured evidence'))
 a=p.parse_args()
 if a.cmd=='play-can':
  try:raise SystemExit(play_can(a.output_dir))
  except (ValueError,OSError,KeyError) as exc:p.exit(2,f'CAN GAMEPLAY LAUNCH REFUSED: {exc}\n')
 elif a.cmd=='launch-kompetensprov':
  try:raise SystemExit(launch_kompetensprov(a.output_dir,a.trace_root))
  except (ValueError,OSError,KeyError) as exc:p.exit(2,f'KOMPETENSPROV LAUNCH REFUSED: {exc}\n')
 elif a.cmd=='launch-d02':
  try:raise SystemExit(launch_d02(a.output_dir,a.trace_root,a.plan))
  except (ValueError,OSError,KeyError) as exc:p.exit(2,f'D02 LAUNCH REFUSED: {exc}\n')
 elif a.cmd=='launch-can':
  try:raise SystemExit(launch_can(a.output_dir,a.trace_root,a.foundation_report))
  except (ValueError,OSError,KeyError) as exc:p.exit(2,f'CAN LAUNCH REFUSED: {exc}\n')
 elif a.cmd=='reset-labels':
  try: reset_labels(binding=a.binding,root=a.root,session=a.session,apply=a.apply,backup_root=a.backup_root)
  except (ValueError,OSError,KeyError) as exc:p.exit(1,f'RESET REFUSED: {exc}\n')
 elif a.cmd=='preflight':
  ok,reasons=preflight(a.plan,a.session,a.trace_root,a.binding,a.settings);print(('READY TO SHOOT' if ok else 'NOT READY')+f' SESSION {a.session}');[print('- '+r) for r in reasons];raise SystemExit(0 if ok else 2)
 elif a.cmd=='start':
  backup=a.output.with_suffix('.settings.backup.json')
  if any(p.exists() or p.is_symlink() for p in (a.output,backup)):
   p.exit(2,'START REFUSED: binding or settings backup exists; choose a new output path\n')
  a.output.parent.mkdir(parents=True,exist_ok=True)
  plan=load_plan(a.plan); root=a.trace_root/f'session_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{a.session}_{uuid.uuid4().hex[:8]}'; root.mkdir(parents=True)
  backup=a.output.with_suffix('.settings.backup.json'); meta=bind(plan,a.session,root,a.source_commit); meta['settings_path']=str(a.settings); meta['settings_backup']=str(backup); meta['settings_sha256']=_set_trace(a.settings,root,backup); meta['status']='READY'; meta['trace_root']=str(root)
  (root/'collection_binding.json').write_text(json.dumps(meta,indent=2)+'\n'); a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(meta,indent=2)+'\n');print(f'STATUS: READY\nSESSION: {a.session}\nTRACE_ROOT: {root}\nPHYSICAL_TRACE_CAPTURE: ENABLED')
 elif a.cmd=='restore':
  b=json.loads(a.binding.read_text()); backup=Path(b['settings_backup']); target=Path(b['settings_path'])
  if not backup.exists(): raise FileNotFoundError(f'settings backup missing: {backup}; do not continue, recover it before restoring')
  shutil.copyfile(backup,target); print(f'RESTORED SETTINGS: {target}')
 elif a.cmd=='guard': guard_training(a.cls,a.operation);print('ALLOWED')
 else:
  try:finalize(a)
  except (ValueError,OSError,KeyError,TypeError) as exc:p.exit(1,f'FINALIZE REFUSED: {exc}\n')
if __name__=='__main__':main()
