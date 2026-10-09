"""OBJECT_HIT observed-newness contract; synthetic tests, not physical accuracy."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch

import numpy as np

from automation import object_physical_query_selftest as fx
from automation.physical_collection import prepare_can_play
from src.engine.camera.object_physical_query import ObjectPhysicalQuery
from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2
from src.engine.input.hit_regions import HitRegion, object_hit_regions
from src.engine.physical_trace import PhysicalTraceRecorder, get_object_crop_recorder


class FreshnessTests(unittest.TestCase):
    @staticmethod
    def pixels(value=140):
        a=np.full((240,360),140,np.uint8);a[89:94,122:127]=value
        return a

    def query(self, values, **kw):
        regions=object_hit_regions([HitRegion('can',100,60,60,80)])
        pre=[fx.frame(self.pixels(v),1.84+.05*i) for i,v in enumerate(values)]
        return ObjectPhysicalQuery(fx.snapshot(game=regions),pre,camera_to_game=lambda x,y:(x,y),**kw)

    def finish(self,q,a,b=None):
        q.observe(fx.frame(self.pixels(a),2.03),delivered_ts=2.035)
        q.observe(fx.frame(self.pixels(a if b is None else b),2.065),delivered_ts=2.07)
        return q.result or q.finish_deadline(2.36)

    def identity_registration(self):
        # Isolate temporal semantics from the known uniform-fixture registration
        # artifact. Other tests and the benchmark exercise actual registration.
        return patch.object(CandidateGeneratorV2,'_register_current',
            lambda self,reference,current,**kw:(current,dict(applied=False,dx=0.,dy=0.)))

    def test_fresh_both_polarities_and_exact_xy(self):
        for level in (30,230):
            q=self.query([140,140,140]);r=self.finish(q,level)
            self.assertEqual(r['object_decision']['state'],'OBJECT_HIT')
            c=r['object_decision']['hypothesis']
            self.assertEqual(c['event_freshness']['state'],'SUPPORTED')
            self.assertTrue(c['event_freshness']['same_polarity'])
            self.assertFalse(c['event_freshness']['clean_pre_certified'])
            self.assertFalse(c['event_freshness']['physical_cause_certified'])
            x,y=map(int,c['camera_xy']);self.assertEqual(self.pixels(level)[y,x],level)

    def test_recurrent_pre_response_not_new_despite_post_persistence(self):
        with self.identity_registration():
            q=self.query([90,140,140]);r=self.finish(q,90)
        self.assertEqual(r['object_decision']['state'],'UNKNOWN')
        points=[c for row in r['per_object'] for c in row['candidates']]
        self.assertTrue(any(c['persistent_local_change'] for c in points))
        self.assertFalse(any(c['confirmed'] for c in points))
        self.assertTrue(any(c['event_freshness']['reason']=='post_overlaps_pre_variation' for c in points))

    def test_new_increment_at_recurrent_location_is_legal(self):
        with self.identity_registration():
            r=self.finish(self.query([110,140,140]),30)
        self.assertEqual(r['object_decision']['state'],'OBJECT_HIT')

    def test_opposite_post_polarities_are_not_persistence(self):
        with self.identity_registration():
            r=self.finish(self.query([140,140,140]),90,190)
        self.assertEqual(r['object_decision']['state'],'UNKNOWN')
        points=[c for row in r['per_object'] for c in row['candidates']]
        self.assertTrue(any(c['persistent_local_change'] for c in points))
        self.assertTrue(any(c['event_freshness']['reason']=='post_polarity_reversal' for c in points))

    def test_missing_pre_history_is_unknown_not_clean_baseline(self):
        r=self.finish(self.query([140]),30)
        self.assertEqual(r['object_decision']['state'],'UNKNOWN')
        c=r['per_object'][0]['candidates'][0]
        self.assertIsNone(c['event_freshness']['current'])
        self.assertEqual(c['event_freshness']['reason'],'pre_history_unavailable')

    def test_pre_existing_static_hole_does_not_create_hit(self):
        r=self.finish(self.query([30,30,30]),30)
        self.assertEqual(r['object_decision']['supported_object_ids'],[])

    def test_envelope_is_pixel_local_and_source_independent(self):
        latest=np.array([[5.,-5.,5.,-5.]],np.float32)
        earlier=np.array([[-1.,1.,3.,-3.]],np.float32)
        np.testing.assert_array_equal(ObjectPhysicalQuery._new_change(latest,[earlier]),[[0,0,3,-3]])
        np.testing.assert_array_equal(latest,[[5,-5,5,-5]])


class CropDiagnosticTests(unittest.TestCase):
    def test_bounded_writer_saves_only_actual_query_crops_and_result(self):
        with tempfile.TemporaryDirectory() as root:
            rec=PhysicalTraceRecorder(root,enabled=True,object_only=True,writer_workers=1,
                max_queued_bytes=16*1024*1024)
            self.addCleanup(rec.shutdown)
            # Global full-frame state deliberately exists: object-only must skip it.
            scanner=S(pre_shot_snapshot=np.zeros((100,100),np.uint8),pre_shot_snapshot_ts=1.9,
                frame_history=[S(timestamp=1.8,gray=np.zeros((100,100),np.uint8))])
            rec.start_from_scanner(scanner,S(shot_id=1,peak_ts=2.))
            fxcase=FreshnessTests()
            q=fxcase.query([140,140,140],capture_crop=rec.capture_object_crop,capture_result=rec.finish_object_query)
            result=fxcase.finish(q,30)
            rec._queue.join()
            trace=json.loads((Path(root)/'shots/shot_00000001/trace.json').read_text())
            self.assertTrue(trace['completeness']['trace_complete'])
            self.assertEqual(trace['frames'],[]);self.assertEqual(trace['stages'],[])
            self.assertEqual(len(trace['object_evidence_crops']),5)
            self.assertEqual(trace['outcome']['object_decision'],result['object_decision'])
            self.assertIsNone(trace['outcome']['ground_truth'])
            for crop in trace['object_evidence_crops']:
                pixels=np.load(Path(root)/'shots/shot_00000001'/crop['path'])
                self.assertEqual(list(pixels.shape),crop['shape'])
                self.assertLess(pixels.size,100000)

    def test_crop_queue_loss_is_explicit_not_complete(self):
        with tempfile.TemporaryDirectory() as root:
            rec=PhysicalTraceRecorder(root,enabled=True,object_only=True,max_queued_bytes=1,writer_workers=1)
            self.addCleanup(rec.shutdown)
            rec.start_from_scanner(S(),S(shot_id=1,peak_ts=2.))
            rec.capture_object_crop(1,'pre',1.9,[0,0,8,8],np.zeros((8,8),np.uint8))
            rec.finish_object_query(1,dict(status='UNCERTAIN'))
            rec._queue.join()
            trace=json.loads((Path(root)/'shots/shot_00000001/trace.json').read_text())
            self.assertFalse(trace['completeness']['trace_complete'])
            self.assertEqual(trace['completeness']['queue_drops'],1)

    def test_crop_launch_is_opt_in_and_forbidden_paths_refused(self):
        with tempfile.TemporaryDirectory(dir='/data/skjutbana') as root:
            manifest,env=prepare_can_play(Path(root)/'run',object_crops=True)
            self.assertTrue(json.loads(manifest.read_text())['object_crops'])
            self.assertEqual(env['SKJUTBANA_PHYSICAL_TRACE_DISABLE'],'1')
            self.assertEqual(env['SKJUTBANA_OBJECT_CROP_ROOT'],str(manifest.parent/'object_crops'))
        for name in ('/data/skjutbana/S03/forbidden','/tmp/not_data'):
            with patch.dict(os.environ,SKJUTBANA_OBJECT_CROP_ROOT=name):
                with self.assertRaises(ValueError):get_object_crop_recorder()

    def test_capture_event_and_pixel_caps(self):
        with tempfile.TemporaryDirectory() as root:
            rec=PhysicalTraceRecorder(root,enabled=True,object_only=True,writer_workers=1)
            self.addCleanup(rec.shutdown)
            for sid in range(1,22):
                rec.start_from_scanner(S(),S(shot_id=sid,peak_ts=2.+sid))
            self.assertEqual(len(rec._active),20)
            self.assertIn(21,rec._refused_events)
            rec._active[1]['object_pixels']=11_000_000
            rec.capture_object_crop(1,'post',3.1,[0,0,1,1],np.zeros((1,1),np.uint8))
            rec.finish_object_query(1,dict(status='UNCERTAIN'))
            rec._queue.join()
            trace=json.loads((Path(root)/'shots/shot_00000001/trace.json').read_text())
            self.assertFalse(trace['completeness']['trace_complete'])
            self.assertEqual(trace['completeness']['errors'][0]['kind'],'object_crop_limit')


if __name__=='__main__':unittest.main()
