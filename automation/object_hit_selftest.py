"""Opt-in OBJECT_HIT software checks; generated pixels are not physical validation."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from automation import object_physical_query_selftest as fx
from automation.physical_collection import prepare_can_play
from src.engine.camera.object_physical_query import ObjectPhysicalQuery
from src.engine.game_objects import ObjectManager, ObjectGeometry, make_breakable_object
from src.engine.input.hit_regions import HitRegion, object_hit_regions, object_query_regions
from src.engine.input.object_hit_v2223 import CameraHitRegionV240
from src.engine.physical_trace import PhysicalTraceRecorder


class ObjectHitTests(unittest.TestCase):
    def setup_query(self, boxes=((100,60,60,80),), *, strict=True, top_k=3):
        manager=ObjectManager()
        for i,box in enumerate(boxes):manager.add(make_breakable_object(str(i),ObjectGeometry(*box)))
        regions=manager.get_hit_regions()
        regions=(object_hit_regions if strict else object_query_regions)(regions,top_k=top_k)
        camera=tuple(CameraHitRegionV240(r.object_id,r.x,r.y,r.width,r.height) for r in regions)
        snap=fx.snapshot(camera,regions);manager._snapshot_provider=lambda sid:snap
        pre=np.full((240,480),140,np.uint8)
        q=ObjectPhysicalQuery(snap,[fx.frame(pre,1.94)],camera_to_game=lambda x,y:(x,y),top_k=top_k)
        return q,pre,manager

    def damage(self,q,pre,points):
        post=pre.copy()
        for x,y in points:post[y-2:y+3,x-2:x+3]=30
        return fx.hit_result(q,post)

    def test_one_object_and_actual_observed_xy(self):
        q,pre,m=self.setup_query();r=self.damage(q,pre,[(123,93)])
        d=r['object_decision'];self.assertEqual((d['state'],d['object_id']),('OBJECT_HIT','0'))
        self.assertEqual(d['hypothesis']['camera_xy'],[122.,92.])
        self.assertEqual(d['hypothesis']['rank'],1)
        resolved,h=m.resolve_object_query(r)
        self.assertEqual(resolved.interactions[0].object_id,'0')
        self.assertEqual([resolved.game_x,resolved.game_y],h['game_xy'])
        self.assertFalse(m.get('0').hittable)
        self.assertEqual(m.get_hit_regions(),())
        self.assertEqual(m.resolve_object_query(r),(None,None))

    def test_context_margin_never_expands_object(self):
        q,pre,m=self.setup_query();r=self.damage(q,pre,[(177,100)])
        self.assertEqual(r['status'],'PHYSICAL_HIT')
        self.assertEqual(r['object_decision']['state'],'UNKNOWN')
        self.assertIsNone(r['object_decision']['object_id'])
        self.assertEqual(m.resolve_object_query(r),(None,None))

    def test_far_outside_never_enters_object_query(self):
        q,pre,m=self.setup_query();post=pre.copy();post[190:195,400:405]=0
        q.observe(fx.frame(post,2.03),delivered_ts=2.032)
        r=q.observe(fx.frame(post,2.21),delivered_ts=2.212)
        self.assertEqual(r['object_decision']['state'],'NO_OBJECT_HIT')
        self.assertFalse(r['object_decision']['global_rescue_used'])
        self.assertFalse(r['object_decision']['no_physical_impact_established'])

    def test_two_supported_objects_abstain_before_top1(self):
        q,pre,m=self.setup_query(((80,60,40,80),(160,60,40,80)),top_k=1)
        r=self.damage(q,pre,[(100,93),(180,93)])
        self.assertEqual(len(r['hypotheses']),1)
        self.assertEqual(r['object_decision']['state'],'UNKNOWN')
        self.assertEqual(r['object_decision']['supported_object_ids'],['0','1'])
        self.assertEqual(m.resolve_object_query(r),(None,None))
        self.assertTrue(all(o.hittable for o in m.objects))

    def test_overlapping_exact_shapes_abstain(self):
        q,pre,m=self.setup_query(((100,60,60,80),(110,65,60,80)))
        r=self.damage(q,pre,[(130,93)])
        self.assertEqual(r['object_decision']['reason'],'multiple_supported_objects')

    def test_motion_after_freeze_cannot_rebind_object(self):
        q,pre,m=self.setup_query();m.get('0').geometry.x=300
        r=self.damage(q,pre,[(123,93)])
        self.assertEqual(r['object_decision']['object_id'],'0')
        self.assertIsNotNone(m.resolve_object_query(r)[0])

    def test_cancel_or_deadline_cannot_reuse_supported_rows(self):
        q,pre,m=self.setup_query();self.damage(q,pre,[(123,93)])
        r=q.finish('UNCERTAIN','cancelled')
        self.assertEqual(r['object_decision']['supported_object_ids'],[])
        self.assertEqual(m.resolve_object_query(r),(None,None))

    def test_plain_hitregion_uses_exact_game_rect(self):
        regions=object_hit_regions((HitRegion('plain',100,60,60,80),))
        pre=np.full((240,480),140,np.uint8)
        q=ObjectPhysicalQuery(fx.snapshot(game=regions,regions=(CameraHitRegionV240('plain',100,60,60,80),)),
            [fx.frame(pre,1.94)],camera_to_game=lambda x,y:(x,y))
        r=self.damage(q,pre,[(123,93)])
        self.assertEqual(r['object_decision']['object_id'],'plain')

    def test_legacy_opt_out_preserves_physical_candidates(self):
        q,p,_=self.setup_query(strict=False);a=self.damage(q,p,[(123,93)])
        q,p,_=self.setup_query();b=self.damage(q,p,[(123,93)])
        self.assertNotIn('object_decision',a)
        self.assertEqual(a['status'],b['status'])
        for x,y in zip(a['hypotheses'],b['hypotheses']):
            for k in ('camera_xy','game_xy','confirmed','evidence','previous_same_xy_evidence','evidence_score'):
                self.assertEqual(x[k],y[k])

    def test_compact_log_excludes_ledger_and_global_xy_cannot_change_object(self):
        q,p,m=self.setup_query();self.damage(q,p,[(123,93)])
        current=dict(camera_xy=[9999,9999],state='confirmed',emitted=True)
        record=q.compact_record(current=current)
        self.assertEqual(record['object_id'],'0');self.assertEqual(record['exact_coordinate_result'],current)
        self.assertNotIn('candidate_ledger',record)
        self.assertNotIn('hypotheses',record)
        self.assertLess(len(json.dumps(record)),6000)

    def test_invalid_semantic_rejected(self):
        with self.assertRaises(ValueError):object_query_regions([],result_semantics='snap')


class GameplayLaunchTests(unittest.TestCase):
    def test_compact_launch_prepares_15_notes_without_settings_write(self):
        original=Path('content/ai/settings.json').read_bytes()
        with tempfile.TemporaryDirectory(dir='/data/skjutbana') as root:
            output=Path(root)/'play'
            with patch.dict(os.environ,SKJUTBANA_PHYSICAL_CAPTURE_BINDING='old-binding'):
                manifest,env=prepare_can_play(output)
            self.assertNotIn('SKJUTBANA_PHYSICAL_CAPTURE_BINDING',env)
            self.assertEqual(env['SKJUTBANA_PHYSICAL_TRACE_DISABLE'],'1')
            self.assertEqual(env['SKJUTBANA_OBJECT_COMPARE'],'0')
            self.assertEqual(len(json.loads((output/'event_notes.json').read_text())['events']),15)
            self.assertFalse(json.loads(manifest.read_text())['full_frame_capture'])
            with self.assertRaises(FileExistsError):prepare_can_play(output)
        self.assertEqual(original,Path('content/ai/settings.json').read_bytes())

    def test_forbidden_and_non_data_paths_rejected(self):
        for path in ('/data/skjutbana/S03/no_access','/tmp/not_gameplay'):
            with self.assertRaises(ValueError):prepare_can_play(path)

    def test_compact_disables_saved_capture_only_for_process(self):
        with tempfile.TemporaryDirectory() as root:
            recorder=PhysicalTraceRecorder(root)
            with patch.dict(os.environ,{'SKJUTBANA_PHYSICAL_TRACE_DISABLE':'1'},clear=True):
                recorder.configure({'physical_trace_capture_enabled':True})
                self.assertFalse(recorder.enabled)
            recorder.configure({'physical_trace_capture_enabled':True})
            self.assertTrue(recorder.enabled)
            with patch.dict(os.environ,{'SKJUTBANA_PHYSICAL_TRACE_DISABLE':'1','SKJUTBANA_PHYSICAL_CAPTURE_BINDING':'explicit'},clear=True):
                with self.assertRaises(ValueError):recorder.configure({})


if __name__=='__main__':unittest.main()
