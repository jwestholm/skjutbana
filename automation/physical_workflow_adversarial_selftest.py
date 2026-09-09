"""Real JSON/CLI boundary checks for destructive collection workflow cases."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from automation.physical_capture_plan import build
from automation.physical_finalize import validate
from automation.physical_collection import bind,guard_training

class AdversarialWorkflowTests(unittest.TestCase):
 def setUp(self): self.plan=build(3,10)
 def labels(self,n=10,status='PHYSICAL'):
  return {'collection_plan_id':self.plan['collection_plan_id'],'labels':[{'planned_physical_shot':i+1,'event_id':i+1,'status:':status,'status':status} for i in range(n)]}
 def test_duplicate_mapping_refused(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);l=self.labels();l['labels'][1]['event_id']=1;(p/'l').write_text(json.dumps(l));(p/'q').write_text(json.dumps({'frame_completeness':True,'trace_completeness':True,'label_completeness':True}))
   with self.assertRaises(ValueError):validate(self.plan,'S01',p/'l',p/'q')
 def test_ambiguous_refused(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);l=self.labels();l['labels'][2]['status']='AMBIGUOUS';(p/'l').write_text(json.dumps(l));(p/'q').write_text(json.dumps({'frame_completeness':True,'trace_completeness':True,'label_completeness':True}))
   with self.assertRaises(ValueError):validate(self.plan,'S01',p/'l',p/'q')
 def test_wrong_plan_refused(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);l=self.labels();l['collection_plan_id']='wrong';(p/'l').write_text(json.dumps(l));(p/'q').write_text(json.dumps({'frame_completeness':True,'trace_completeness':True,'label_completeness':True}))
   with self.assertRaises(ValueError):validate(self.plan,'S01',p/'l',p/'q')
 def test_validation_matrix(self):
  for op in ('train','tune','fit','select_config','model_select'):
   with self.assertRaises(PermissionError):guard_training('VALIDATION_UNTOUCHED',op)
 def test_real_preflight_cli(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'plan.json';p.write_text(json.dumps(self.plan));r=subprocess.run([sys.executable,'-m','automation.physical_collection','preflight','--plan',str(p),'--session','S01','--trace-root',d],capture_output=True,text=True);self.assertEqual(r.returncode,0);self.assertIn('READY TO SHOOT',r.stdout)

if __name__=='__main__':unittest.main()
