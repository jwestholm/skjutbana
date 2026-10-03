"""Capture preparation regressions; no physical session is opened."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from automation.physical_capture_provenance import (template, validate_notes, geometry_settings,
    PLAN, snapshot, check, plan_matches_binding, geometry_postflight, saved_configuration)
from automation.physical_collection import bind, load_plan
from automation.physical_finalize_manifest import identity_rows
from automation.physical_label_reset import resolve_session


class Tests(unittest.TestCase):
    def notes(self):
        return dict(session='D02',observer='fixture',observed_at='fixture-time',board_revision='fixture-revision',
            board_bounds_basis='software fixture only',camera_frame_size_px=[100,100],
            board_bounds_camera_px=[[10,10],[90,10],[90,90],[10,90]],
            seam=dict(observable=False,reason='not visible in fixture'),
            display=dict(framebuffer_size_px=[800,600],viewport_xywh=[0,0,800,600],
                content_rect_viewport_xywh=[0,0,800,600],monitor_description='fixture',projection_description='fixture'),
            calibration_verified_unchanged_since_observation=True)

    def test_plan_has_two_physical_rows_and_six_protocol_events(self):
        plan=load_plan(PLAN)
        self.assertEqual(set(identity_rows(plan['rows'])),{1,2})
        self.assertEqual(len(plan['protocol_events']),6)
        self.assertEqual(plan['execution_order'],[3,1,2,4,5,6])
        self.assertTrue(all(row['session']=='D02' and row['actual_event_id'] is None for row in plan['rows']))

    def test_template_never_claims_capture_or_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'package';template(root)
            with self.assertRaises(ValueError):validate_notes(json.loads((root/'geometry_notes.json').read_text()))
            self.assertTrue(all(e['status']=='NOT_OBSERVED' and not e['actual_event_ids']
                for e in json.loads((root/'event_notes.json').read_text())['events']))
            with self.assertRaises(FileExistsError):template(root)

    def test_notes_require_finite_camera_bounds_and_explicit_unobserved_seam(self):
        good=self.notes();validate_notes(good)
        for key,value in [('camera_frame_size_px',[float('nan'),100]),('board_bounds_camera_px',[[0,0]]*4),
                          ('calibration_verified_unchanged_since_observation',False)]:
            bad=copy.deepcopy(good);bad[key]=value
            with self.assertRaises(ValueError):validate_notes(bad)
        bad=copy.deepcopy(good);bad['seam']={'observable':None}
        with self.assertRaises(ValueError):validate_notes(bad)

    def test_settings_export_excludes_unrelated_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'settings.json';path.write_text(json.dumps({'viewport':[0,0,800,600],'led':{'local_key':'secret'}}))
            exported=geometry_settings(path)
            self.assertNotIn('led',exported)
            self.assertNotIn('secret',json.dumps(exported))

    def test_bound_snapshot_check_and_refuse_reused_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);session=root/'session_D02_fixture';session.mkdir()
            plan=load_plan(PLAN);meta=bind(plan,'D02',session)
            settings=root/'settings.json';settings.write_text(json.dumps({'scanport':[0,0,100,100]}))
            ai=root/'ai.json';ai.write_text(json.dumps({'physical_trace_capture_enabled':True,'physical_trace_root':str(session)}))
            (session/'collection_binding.json').write_text(json.dumps(meta))
            binding=root/'binding.json';binding.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):resolve_session(binding=binding,session='D02')
            notes=root/'notes.json';notes.write_text(json.dumps(self.notes()))
            provenance=root/'provenance.json'
            snapshot(binding,notes,provenance,settings,ai)
            self.assertTrue(check(PLAN,binding,provenance,settings,ai)[0])
            altered=root/'altered_plan.json';changed=copy.deepcopy(plan);changed['execution_order'].reverse()
            altered.write_text(json.dumps(changed))
            ok,reasons=check(altered,binding,provenance,settings,ai)
            self.assertFalse(ok);self.assertIn('Capture plan changed since provenance snapshot',reasons)
            (session/'shots/shot_00000001').mkdir(parents=True)
            self.assertFalse(check(PLAN,binding,provenance,settings,ai)[0])
            with self.assertRaises(ValueError):snapshot(binding,notes,root/'again.json',settings,ai)

    def test_plan_rows_must_match_the_bound_plan(self):
        plan=load_plan(PLAN);meta=bind(plan,'D02',Path('/tmp/not_a_capture'))
        meta['rows']=copy.deepcopy(meta['rows']);meta['rows'][0]['category']='wrong protocol'
        with self.assertRaises(ValueError):plan_matches_binding(PLAN,meta)

    def test_saved_setup_does_not_copy_detector_secrets_or_claim_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);settings=root/'s.json';ai=root/'a.json'
            settings.write_text(json.dumps({'scanport':[0,0,100,100],'led':{'secret':'secret-value'}}))
            ai.write_text(json.dumps({'analysis_roi_crop_v2221_enabled':True,'secret':'secret-value'}))
            value=saved_configuration(settings,ai)
            self.assertNotIn('secret-value',json.dumps(value))
            self.assertEqual(value['status'],'SAVED_CONFIGURATION_ONLY_NOT_RUNTIME_VERIFICATION')

    def test_geometry_postflight_exports_metadata_without_mutating_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);session=root/'session_D02_fixture';session.mkdir()
            binding=root/'binding.json';binding.write_text(json.dumps(bind(load_plan(PLAN),'D02',session)))
            with self.assertRaises(ValueError):geometry_postflight(binding,root/'absent.json')
            shot=session/'shots/shot_00000007';shot.mkdir(parents=True)
            trace=shot/'trace.json';trace.write_text(json.dumps(dict(shot_id=7,frames=[{'shape':[100,200]}],
                stages=[{'window_debug':{'frame_ts':1.,'v2221_crop_x0':10,'v2221_crop_w':100}}])))
            original=trace.read_bytes();output=root/'geometry.json';geometry_postflight(binding,output)
            self.assertEqual(trace.read_bytes(),original)
            self.assertEqual(json.loads(output.read_text())['events'][0]['geometry_status'],'RECORDED')
            with self.assertRaises(FileExistsError):geometry_postflight(binding,output)

    def test_geometry_postflight_refuses_cross_session_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);session=root/'session_D02_fixture';session.mkdir()
            binding=root/'binding.json';binding.write_text(json.dumps(bind(load_plan(PLAN),'D02',session)))
            (session/'shots').symlink_to(root/'unavailable_other_session',target_is_directory=True)
            with self.assertRaises(ValueError):geometry_postflight(binding,root/'geometry.json')


if __name__=='__main__':unittest.main()
