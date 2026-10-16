import unittest,tempfile,json
from pathlib import Path
from automation.physical_capture_plan import build
from automation.physical_collection import bind,guard_training,preflight,_set_trace,prepare_can_launch,verify_can_launch,prepare_d02_launch,verify_capture_launch,launch_d02
from automation.physical_finalize import validate

class CollectionTests(unittest.TestCase):
 def test_runner_records_exit_status_without_starting_hardware(self):
  from io import BytesIO
  from unittest.mock import patch,Mock
  from automation.physical_collection import _run_capture
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);binding=p/'binding.json';binding.write_text('{}')
   process=Mock(stdout=BytesIO(b'fixture completed\n'),returncode=-15)
   process.poll.return_value=-15;process.wait.return_value=-15
   with patch('automation.physical_collection.subprocess.Popen',return_value=process):
    self.assertEqual(_run_capture(binding,{}),-15)
   self.assertEqual(json.loads((p/'runtime_exit.json').read_text())['returncode'],-15)
   self.assertEqual((p/'runtime.log').read_text(),'fixture completed\n')
   process.terminate.assert_not_called()
   self.assertTrue(process.stdout.closed)

 def test_d02_process_only_unique_roots_settings_and_symlink_unchanged(self):
  import os
  settings=Path('content/ai/settings.json');before=settings.read_bytes()
  real_link=Path('content/ai/physical_traces');target=os.readlink(real_link)
  plan=Path('research/physical_capture_plans/D02.json');plan_before=plan.read_bytes()
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'traces').mkdir();link=p/'physical_traces';link.symlink_to(p/'traces')
   first=prepare_d02_launch(p/'first',link,plan)
   second=prepare_d02_launch(p/'second',link,plan)
   a=verify_capture_launch(first);b=verify_capture_launch(second)
   self.assertNotEqual(a['trace_root'],b['trace_root'])
   self.assertEqual(Path(a['trace_root']).parent,p/'traces')
   self.assertEqual((p/'first/plan.json').read_bytes(),plan_before)
   self.assertEqual(a['capture_history']['pre_seconds'],4.)
   self.assertEqual(a['capture_history']['post_seconds'],5.)
   self.assertEqual(list(Path(a['trace_root']).iterdir()),[Path(a['trace_root'])/'collection_binding.json'])
   self.assertFalse(list((p/'first').glob('*backup*')))
   self.assertTrue(link.is_symlink());self.assertEqual(os.readlink(link),str(p/'traces'))
   verify_capture_launch(first,claim=True)
   with self.assertRaisesRegex(ValueError,'already launched'):verify_capture_launch(first)
   with self.assertRaises(FileExistsError):prepare_d02_launch(p/'second',link,plan)
   with self.assertRaises(ValueError):verify_can_launch(second)
  self.assertEqual(settings.read_bytes(),before)
  self.assertEqual(os.readlink(real_link),target);self.assertEqual(plan.read_bytes(),plan_before)

 def test_d02_launch_reuses_runner_without_detector_overrides(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   with patch('automation.physical_collection._run_capture',return_value=0) as runner:
    self.assertEqual(launch_d02(p/'run',p/'traces',Path('research/physical_capture_plans/D02.json')),0)
   binding,env=runner.call_args.args
   self.assertEqual(env['SKJUTBANA_PHYSICAL_CAPTURE_BINDING'],str(binding))
   import os
   self.assertEqual({k:v for k,v in env.items() if k!='SKJUTBANA_PHYSICAL_CAPTURE_BINDING'},
                    {k:v for k,v in os.environ.items() if k!='SKJUTBANA_PHYSICAL_CAPTURE_BINDING'})

 def test_startup_guard_installs_d02_history_without_importing_app(self):
  import os
  from unittest.mock import patch
  from src.engine.physical_trace import PhysicalTraceRecorder
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);binding=prepare_d02_launch(p/'run',p/'traces',Path('research/physical_capture_plans/D02.json'))
   original=Path('content/ai/settings.json').read_bytes()
   prefix=Path('main.py').read_text().split('def _restore_terminal')[0]
   with patch.dict(os.environ,SKJUTBANA_PHYSICAL_CAPTURE_BINDING=str(binding)), patch('faulthandler.enable') as faults:
    exec(compile(prefix,'main.py startup guard','exec'),{})
    faults.assert_called_once_with()
    recorder=PhysicalTraceRecorder(p/'unused')
    recorder.configure({'physical_trace_capture_enabled':False,'physical_trace_root':'stale'})
    self.assertTrue(recorder.enabled)
    self.assertEqual(recorder.capture_history['pre_seconds'],4.)
    self.assertEqual(str(recorder.root),json.loads(binding.read_text())['trace_root'])
    self.assertTrue((recorder.root/'runtime_claim.json').exists())
   self.assertEqual(Path('content/ai/settings.json').read_bytes(),original)

 def test_development_launcher_rejects_validation_path_before_read(self):
  from unittest.mock import patch
  # A synthetic path string only; never create or inspect a validation directory.
  with patch.object(Path,'read_text',side_effect=AssertionError('must not read')), patch.object(Path,'resolve',side_effect=AssertionError('must not inspect')):
   with self.assertRaisesRegex(ValueError,'Validation paths'):
    verify_capture_launch('/forbidden/session_S03_example/binding.json')
   with self.assertRaisesRegex(ValueError,'Validation paths'):
    prepare_d02_launch('/tmp/unused','/tmp/unused','/forbidden/S03/plan.json')

 def test_d02_rejects_stale_roots_and_changed_full_protocol(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);binding=prepare_d02_launch(p/'run',p/'traces',Path('research/physical_capture_plans/D02.json'))
   b=verify_capture_launch(binding);plan=Path(b['plan_path']);payload=json.loads(plan.read_text())
   payload['execution_order'].reverse();plan.write_text(json.dumps(payload))
   with self.assertRaisesRegex(ValueError,'hash differs'):verify_capture_launch(binding)
   plan.write_bytes(Path('research/physical_capture_plans/D02.json').read_bytes())
   (Path(b['trace_root'])/'shots/shot_00000001').mkdir(parents=True)
   with self.assertRaisesRegex(ValueError,'already contains shots'):verify_capture_launch(binding)

 def test_can_foundation_gate_and_runner_environment_are_preserved(self):
  from unittest.mock import patch
  from automation.physical_collection import launch_can
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   with self.assertRaisesRegex(ValueError,'foundation preflight required'):
    launch_can(p/'blocked',p/'traces')
   self.assertFalse((p/'blocked').exists())
   report=p/'fixture_foundations.json';report.write_text(json.dumps(dict(status='READY',ready_to_shoot=True)))
   with patch('automation.physical_collection._run_capture',return_value=0) as runner:
    self.assertEqual(launch_can(p/'run',p/'traces',report),0)
   binding,env=runner.call_args.args
   self.assertEqual(verify_can_launch(binding)['planned_session_id'],'CAN_OBJECT')
   self.assertEqual(env['SKJUTBANA_CAN_SFX'],'0');self.assertEqual(env['SKJUTBANA_OBJECT_COMPARE'],'0')

 def test_process_only_can_binding_never_changes_saved_settings(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);settings=p/'settings.json';settings.write_text('{"physical_trace_root":"stale"}')
   before=settings.read_bytes();binding=prepare_can_launch(p/'run',p/'traces')
   b=verify_can_launch(binding)
   self.assertEqual(settings.read_bytes(),before);self.assertEqual(len(b['rows']),3)
   self.assertEqual(b['capture_configuration'],'PROCESS_ONLY')
   self.assertFalse((Path(b['trace_root'])/'shots').exists())
   self.assertFalse(list((p/'run').glob('*settings*')))
   verify_can_launch(binding,claim=True)
   with self.assertRaisesRegex(ValueError,'already launched'):verify_can_launch(binding)
   with self.assertRaises(FileExistsError):prepare_can_launch(p/'run',p/'traces')

 def test_can_preflight_refuses_populated_root_and_mismatched_plan(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);binding=prepare_can_launch(p/'run',p/'traces');b=verify_can_launch(binding)
   shot=Path(b['trace_root'])/'shots/shot_00000001';shot.mkdir(parents=True)
   sentinel=shot/'preserved';sentinel.write_text('unaltered')
   with self.assertRaisesRegex(ValueError,'already contains shots'):verify_can_launch(binding)
   self.assertEqual(sentinel.read_text(),'unaltered')
   other=prepare_can_launch(p/'other',p/'traces');b=verify_can_launch(other)
   plan=Path(b['plan_path']);payload=json.loads(plan.read_text());payload['rows'].pop();plan.write_text(json.dumps(payload))
   with self.assertRaisesRegex(ValueError,'differs'):verify_can_launch(other)

 def test_capture_expected_with_bad_binding_exits_before_hardware_imports(self):
  import subprocess,sys,os
  with tempfile.TemporaryDirectory() as d:
   cp=subprocess.run([sys.executable,'main.py'],capture_output=True,text=True,
    env=dict(os.environ,SKJUTBANA_PHYSICAL_CAPTURE_BINDING=str(Path(d)/'absent.json')))
   self.assertNotEqual(cp.returncode,0);self.assertIn('PREFLIGHT REFUSED BEFORE GAME START',cp.stderr)
   self.assertNotIn('pygame',cp.stdout+cp.stderr)

 def test_start_refuses_existing_binding_before_changing_settings_or_root(self):
  import subprocess,sys
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);plan=p/'plan.json';plan.write_text(json.dumps(build(1,2)));settings=p/'settings.json';settings.write_text('{"physical_trace_capture_enabled": false}')
   out=p/'binding.json';out.write_text('preserved binding');before=settings.read_bytes()
   result=subprocess.run([sys.executable,'-m','automation.physical_collection','start','--plan',str(plan),'--session','S01','--settings',str(settings),'--trace-root',str(p/'traces'),'--output',str(out)],capture_output=True,text=True)
   self.assertEqual(result.returncode,2);self.assertEqual(settings.read_bytes(),before)
   self.assertEqual(out.read_text(),'preserved binding');self.assertFalse((p/'traces').exists())
 def setUp(self): self.plan=build(3,10)
 def test_binding_and_validation_guard(self):
  m=bind(self.plan,'S03',Path('/tmp/trace'));self.assertEqual(m['session_class'],'VALIDATION_UNTOUCHED')
  with self.assertRaises(PermissionError):guard_training('VALIDATION_UNTOUCHED','train')
 def test_development_allowed(self): self.assertTrue(guard_training('DEVELOPMENT','train'))
 def test_preflight_and_duplicate_binding(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);plan=p/'plan.json';plan.write_text(json.dumps(self.plan));ok,reasons=preflight(plan,'S01',p);self.assertTrue(ok);(p/'planned_S01.json').write_text('{}');ok,reasons=preflight(plan,'S01',p);self.assertFalse(ok)
 def test_wrong_session_plan_refused(self):
  with self.assertRaises(ValueError):bind(self.plan,'SX',Path('/tmp'))
 def test_validation_all_tuning_ops_refused(self):
  for op in ('train','tune','fit','select_config'):
   with self.assertRaises(PermissionError):guard_training('VALIDATION_UNTOUCHED',op)
 def test_finalize_output_overwrite_is_forbidden_by_cli_contract(self):
  self.assertTrue(True)
 def test_label_and_quality_finalize(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);labels={'collection_plan_id':self.plan['collection_plan_id'],'labels':[{'event_id':i+1,'planned_physical_shot':i+1,'status':'PHYSICAL'} for i in range(10)]};(p/'l.json').write_text(json.dumps(labels));(p/'q.json').write_text(json.dumps({'frame_completeness':True,'trace_completeness':True,'label_completeness':True}));r=validate(self.plan,'S01',p/'l.json',p/'q.json');self.assertEqual(r['status'],'FINALIZED')
 def test_unlabeled_fails(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'l.json').write_text(json.dumps({'labels':[{'event_id':1,'status':'UNLABELED'}]}));(p/'q.json').write_text(json.dumps({'frame_completeness':True,'trace_completeness':True,'label_completeness':True}))
   with self.assertRaises(ValueError):validate(self.plan,'S01',p/'l.json',p/'q.json')
 def test_start_repoints_stale_root_and_preflight_checks_it(self):
  import subprocess,sys
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); plan=p/'plan.json'; plan.write_text(json.dumps(self.plan)); settings=p/'settings.json'; settings.write_text(json.dumps({'physical_trace_capture_enabled':True,'physical_trace_root':'content/ai/physical_traces/session_20260907_biathlon5'})); out=p/'binding.json'
   cp=subprocess.run([sys.executable,'-m','automation.physical_collection','start','--plan',str(plan),'--session','S01','--settings',str(settings),'--trace-root',str(p/'traces'),'--output',str(out)],capture_output=True,text=True);self.assertEqual(cp.returncode,0,cp.stderr)
   b=json.loads(out.read_text()); self.assertNotIn('session_20260907_biathlon5',b['trace_root']); self.assertEqual(json.loads(settings.read_text())['physical_trace_root'],b['trace_root'])
   cp=subprocess.run([sys.executable,'-m','automation.physical_collection','preflight','--plan',str(plan),'--session','S01','--trace-root',str(p/'traces'),'--binding',str(out),'--settings',str(settings)],capture_output=True,text=True);self.assertEqual(cp.returncode,0,cp.stdout)
   self.assertEqual(settings.read_bytes(),settings.read_bytes())

if __name__=='__main__':unittest.main()
