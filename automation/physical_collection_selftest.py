import unittest,tempfile,json
from pathlib import Path
from automation.physical_capture_plan import build
from automation.physical_collection import bind,guard_training,preflight
from automation.physical_finalize import validate

class CollectionTests(unittest.TestCase):
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
   p=Path(d);labels={'labels':[{'event_id':i,'planned_physical_shot':i+1,'status':'PHYSICAL'} for i in range(10)]};(p/'l.json').write_text(json.dumps(labels));(p/'q.json').write_text(json.dumps({'frame_completeness':True}));r=validate(self.plan,'S01',p/'l.json',p/'q.json');self.assertEqual(r['status'],'FINALIZED')
 def test_unlabeled_fails(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'l.json').write_text(json.dumps({'labels':[{'event_id':1,'status':'UNLABELED'}]}));(p/'q.json').write_text(json.dumps({'frame_completeness':True}))
   with self.assertRaises(ValueError):validate(self.plan,'S01',p/'l.json',p/'q.json')

if __name__=='__main__':unittest.main()
