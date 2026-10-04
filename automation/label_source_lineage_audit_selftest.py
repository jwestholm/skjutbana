"""Lineage metadata and coordinate mechanics; no physical accuracy claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from automation import physical_ai_dataset as ds
from automation import label_source_lineage_audit as a
from automation.physical_trace_target import camera_to_target,target_to_camera


class LineageTests(unittest.TestCase):
    def fixture(self):
        events=[dict(event=k,session=k.split(':')[0],source_root='/tmp/lineage_fixture/'+k.replace(':','_'),truth='HIT',gt_xy=[100.,200.],indices=[i],candidates=[dict(xy=[90.,210.],id='original')]) for i,k in enumerate(['S01:3','S02:6'])]
        hashes={str(Path(e['source_root'])/f):h*64 for e in events for f,h in [('ground_truth.json','a'),('trace.json','b')]}
        data=dict(events=events,dataset_hash='c'*64,source_hashes=hashes)
        row=dict(event='S01:3',status='LABEL_SOURCE_LINEAGE_UNRESOLVED',label_sha256='a'*64,trace_sha256='b'*64,reason='frame not recorded',report='/data/skjutbana/test_lineage_report.md',annotation_source_frame=None)
        assessment=dict(schema='physical-label-lineage-1',dataset_hash=data['dataset_hash'],events=[row])
        return data,assessment
    def test_sidecar_preserves_all_truth_and_candidates(self):
        data,side=self.fixture();old=copy.deepcopy(data);events=ds.with_label_source_lineage(data,side)
        self.assertEqual(data,old)
        for e,original in zip(events,data['events']):
            del e['label_source_lineage'];self.assertEqual(e,original)
    def test_no_automatic_exclusion(self):
        data,side=self.fixture();events=ds.with_label_source_lineage(data,side)
        self.assertEqual([e['event'] for e in events],['S01:3','S02:6'])
        self.assertEqual([e['truth'] for e in events],['HIT','HIT'])
    def test_frozen_metrics_unaffected_by_lineage_overlay(self):
        data,side=self.fixture()
        outputs={e['event']:dict(proposals=[dict(camera_xy=[100.,200.],confidence=.5)]) for e in data['events']}
        original=a.vision.evaluate_union(data['events'],outputs)
        annotated=a.vision.evaluate_union(ds.with_label_source_lineage(data,side),outputs)
        self.assertEqual(original,annotated)
    def test_missing_audit_is_not_verified(self):
        data,side=self.fixture();p=ds.label_lineage_partition(ds.with_label_source_lineage(data,side))
        self.assertEqual(p['verified'],[]);self.assertEqual(p['unresolved'],['S01:3']);self.assertEqual(p['not_audited'],['S02:6'])
    def test_dataset_hash_binding(self):
        d,s=self.fixture();s['dataset_hash']='wrong'
        with self.assertRaises(ValueError):ds.with_label_source_lineage(d,s)
    def test_native_label_hash_binding(self):
        d,s=self.fixture();s['events'][0]['label_sha256']='d'*64
        with self.assertRaises(ValueError):ds.with_label_source_lineage(d,s)
    def test_trace_hash_binding(self):
        d,s=self.fixture();s['events'][0]['trace_sha256']='d'*64
        with self.assertRaises(ValueError):ds.with_label_source_lineage(d,s)
    def test_duplicate_refused(self):
        d,s=self.fixture();s['events']*=2
        with self.assertRaises(ValueError):ds.with_label_source_lineage(d,s)
    def test_verified_needs_source_image_identity(self):
        d,s=self.fixture();s['events'][0]['status']='LABEL_SOURCE_LINEAGE_VERIFIED'
        with self.assertRaises(ValueError):ds.with_label_source_lineage(d,s)
    def test_explicit_verified_partition_only(self):
        d,s=self.fixture();s['events'][0].update(status='LABEL_SOURCE_LINEAGE_VERIFIED',annotation_source_frame=dict(path='/tmp/lineage_fixture/frame.npy',sha256='e'*64))
        events=ds.with_label_source_lineage(d,s);self.assertEqual(ds.label_lineage_partition(events)['verified'],['S01:3'])
    def test_s03_target_refused(self):
        with self.assertRaises(PermissionError):a.validate_target('S03:1')
    def test_s03_sidecar_refused(self):
        d,s=self.fixture();s['events'][0]['event']='S03:1'
        with self.assertRaises(PermissionError):ds.with_label_source_lineage(d,s)
    def test_s03_report_refused(self):
        d,s=self.fixture();s['events'][0]['report']='/data/skjutbana/S03/report.md'
        with self.assertRaises(PermissionError):ds.with_label_source_lineage(d,s)
    def test_s03_source_refused_even_if_unresolved(self):
        d,s=self.fixture();s['events'][0]['annotation_source_frame']=dict(path='/tmp/S03/never_open.npy')
        with self.assertRaises(PermissionError):ds.with_label_source_lineage(d,s)
    def test_unknown_partition_state_refused(self):
        d,_=self.fixture();d['events'][0]['label_source_lineage']=dict(status='guess')
        with self.assertRaises(ValueError):ds.label_lineage_partition(d['events'])
    def test_current_calibration_records_identity_not_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'settings.json'
            path.write_text(json.dumps(dict(camera_calibration=dict(calibrated_at='fixture',homography=[[1]]))))
            result=a.current_calibration_identity(path)
            self.assertEqual(result['sha256'],ds.digest(path));self.assertEqual(result['calibrated_at'],'fixture')
            self.assertNotIn('homography',result)
    def test_unknown_target_not_discovered(self):
        with self.assertRaises(ValueError):a.validate_target('D02:1')
    def test_source_cannot_escape_event(self):
        with self.assertRaises(ValueError):a.source_path(Path('/tmp/lineage_fixture/event'),'../other/frame.npy')
    def test_default_frame_is_not_verified_annotation(self):
        trace=dict(frames=[dict(kind='post',path='a.npy',timestamp=2.),dict(kind='post',path='b.npy',timestamp=1.)])
        r=a.default_display_entry(trace);self.assertEqual(r['entry']['path'],'a.npy');self.assertEqual(r['association_status'],'UNKNOWN')
    def test_default_frame_does_not_invent_post(self):
        with self.assertRaises(ValueError):a.default_display_entry(dict(frames=[]))
    def test_camera_display_roundtrip(self):
        r=a.transforms((2160,3840),[2286.667,1336.813],[2287,1321],np.array([1521,972]))
        self.assertLess(next(x for x in r if x['test']=='default_raw_display_roundtrip')['error_to_manual_gt'],1e-9)
    def test_existing_homography_helpers_roundtrip(self):
        h=np.array([[.44,-.05,-620],[.003,.64,-650],[-.000001,-.00015,1.]])
        p=(2286.667,1336.813);q=target_to_camera(camera_to_target(p,h),h)
        np.testing.assert_allclose(p,q,atol=1e-9)
    def test_no_transform_fitting(self):
        r=a.transforms((2160,3840),[2286.667,1336.813],[2287,1321],np.array([1521,972]))
        self.assertEqual(next(x for x in r if x['test']=='historical_target_homography')['status'],'UNKNOWN')
    def test_output_restricted(self):
        with self.assertRaises(ValueError):ds.output_root(Path('/tmp/not_a_lab_output'))
    def test_default_assessment_deterministic(self):
        d,s=self.fixture();self.assertEqual(ds.with_label_source_lineage(d,s),ds.with_label_source_lineage(d,s))


if __name__=='__main__':unittest.main()
