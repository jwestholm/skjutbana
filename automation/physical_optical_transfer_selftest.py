"""Optical falsification tests. Generated pixels only; never physical hit labels."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from automation import physical_optical_transfer as lab
from automation import physical_gt_capture_selftest as fixtures
from automation import physical_trace_label_crops as originals


class ArithmeticTests(unittest.TestCase):
    def test_exact_affine_uint8_prediction(self):
        b=np.full((17,19,3),10,np.uint8);w=np.full_like(b,214)
        g=np.full_like(b,112)
        ch,_=lab.measure_triplet(b,w,g,[.5]*3)
        self.assertTrue(all(c['all_camera']['mae_dn']==0 for c in ch))
        self.assertTrue(all(c['held_ratio']['median']==.5 for c in ch))

    def test_nonlinear_middle_is_not_fitted_away(self):
        b=np.full((8,8,3),10,np.uint8);w=np.full_like(b,210);g=np.full_like(b,50)
        ch,_=lab.measure_triplet(b,w,g)
        self.assertGreater(ch[0]['responsive_endpoints']['mae_dn'],59)
        self.assertEqual(ch[0]['coefficient'],127/255)

    def test_endpoint_mask_never_reads_held_gray(self):
        b=np.full((8,8,3),10,np.uint8);w=np.full_like(b,210)
        for value in [0,255,40]:
            ch,_=lab.measure_triplet(b,w,np.full_like(b,value))
            self.assertEqual(ch[0]['responsive_endpoint_fraction'],1.)

    def test_negative_or_clipped_response_not_silently_absolute(self):
        b=np.full((4,4,3),100,np.uint8);w=np.full_like(b,20)
        ch,_=lab.measure_triplet(b,w,b)
        self.assertEqual(ch[0]['nonpositive_span_fraction'],1)
        self.assertEqual(ch[0]['responsive_endpoints']['status'],'UNAVAILABLE')
        self.assertEqual(ch[0]['all_camera']['status'],'AVAILABLE')
        ch,_=lab.measure_triplet(np.zeros_like(b),np.full_like(b,255),b)
        self.assertEqual(ch[0]['endpoint_clipped_fraction'],1)

    def test_fold_excludes_held_response_and_incompatible_stream(self):
        def row(i,v):return dict(shot_id=i,status='AVAILABLE',transfer_context='stream1',
            channels=[dict(held_ratio=dict(status='AVAILABLE',median=v)) for _ in range(3)])
        rows=[row(1,.9),row(2,.2),row(3,.4)]
        coef,train=lab.coefficients_without(1,rows)
        self.assertEqual(train,[2,3]);np.testing.assert_allclose(coef,[.3]*3)
        rows[0]['channels']=[dict(held_ratio=dict(status='UNAVAILABLE'))]*3
        self.assertEqual(lab.coefficients_without(1,rows),(coef,train))
        rows[2]['transfer_context']='stream2'
        self.assertEqual(lab.coefficients_without(1,rows),(None,[]))

    def test_dimension_and_nonfinite_coefficients_rejected(self):
        a=np.zeros((5,5,3),np.uint8)
        for args in [(a,a[:4],a,None),(a,a,a,[float('nan')]*3),(a.astype(float),a,a,None)]:
            with self.assertRaises(ValueError):lab.measure_triplet(*args)

    def test_missing_state_explicit_and_no_pre_substitution(self):
        shot=dict(frames=[dict(crops=[dict(role='PRE',projector_state='BLACK')])])
        self.assertEqual(lab.triplet_identity(lab.neutral_entries(shot)),'MISSING_NEUTRAL_STATE')

    def test_synthetic_no_hardware_and_frozen_transfer_preserves_injection(self):
        with tempfile.TemporaryDirectory(dir='/data/skjutbana/evaluation_runs') as d:
            out=Path(d)/'out'
            result=lab.synthetic_sensitivity(out)
            for row in result['rows']:
                self.assertAlmostEqual(row['perturbation_retained_l1_fraction'],1.,places=6)
                self.assertLessEqual(row['perturbation_identity_max_error'],1e-5)
                self.assertIsNone(row['emitted_coordinate'])
            known=next(r for r in result['rows'] if r['model']=='known_transfer')
            self.assertEqual(known['outside_residual_p95_dn'],0)
            wrong=next(r for r in result['rows'] if r['model']=='one_pixel_alignment_error')
            self.assertGreater(wrong['outside_residual_p95_dn'],known['outside_residual_p95_dn'])
            self.assertFalse(result['live_authority']);self.assertTrue(result['synthetic'])
            again=lab.synthetic_sensitivity(Path(d)/'again');self.assertEqual(result,again)
            with self.assertRaises(FileExistsError):lab.synthetic_sensitivity(out)

    def test_s03_rejected_before_io(self):
        with patch.object(Path,'read_text',side_effect=AssertionError('source read forbidden')):
            with self.assertRaises(ValueError):lab.transfer_audit('/data/skjutbana/S03','/data/skjutbana/evaluation_runs/x')


class SavedTripletTests(unittest.TestCase):
    def setUp(self):
        self.fx=fixtures.ReviewTests();self.fx.setUp();self.addCleanup(self.fx.doCleanups)
        f=self.fx
        for sid in (1,2,3):
            f.event.shot_id=sid
            if sid>1:
                f.add_frame(f.now);f.event.peak_ts=f.now+.7
                f.event.audio_trigger=f.audio_trigger(f.event.peak_ts)
                f.now+=.8;f.add_frame(f.now);f.now+=.05;f.add_frame(f.now)
            f.observe();f.step(.51)
            for _ in range(3):f.step(.61,True)
            f.step(1.01,True)
        f.r.close();self.capture=f.capture
        self.outputs=tempfile.TemporaryDirectory(dir='/data/skjutbana/evaluation_runs')
        self.addCleanup(self.outputs.cleanup);self.output=Path(self.outputs.name)/'transfer'

    def test_original_hash_validation_no_candidates_labels_or_current_calibration(self):
        manifest=originals.load_gt_review(self.capture)
        with patch('cv2.VideoCapture',side_effect=AssertionError('no live camera')):
            result=lab.transfer_audit(self.capture,self.output)
        self.assertEqual(result['available_triplets'],3)
        self.assertEqual(result['emitted_accuracy'],'UNAVAILABLE')
        self.assertFalse(result['physical_labels_used'])
        originals.assert_inputs(manifest)
        self.assertFalse(any('candidate' in p or 'ground_truth' in p for p in result['input_hashes']))
        again=lab.transfer_audit(self.capture,Path(self.outputs.name)/'again')
        for a,b in zip(result['shots'],again['shots']):
            self.assertEqual(a['channels'],b['channels'])
            self.assertEqual(a['other_triplets_coefficient'],b['other_triplets_coefficient'])
        with self.assertRaises(ValueError):lab.transfer_audit(self.capture,self.output)
        with self.assertRaises(ValueError):lab.transfer_audit(self.capture,self.capture/'analysis')

    def test_metadata_hash_tampering_refused(self):
        path=self.capture/'gt_review/shots/shot_00000001/white.json'
        data=json.loads(path.read_text());data['projector_rgb']=[99]*3;path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):lab.transfer_audit(self.capture,self.output)
        self.assertFalse(self.output.exists())

    def test_partial_triplet_unavailable_without_fabricated_images(self):
        manifest=originals.load_gt_review(self.capture)
        manifest['shots'][0]['frames']=[f for f in manifest['shots'][0]['frames']
            if f['crops'][0]['projector_state']!='MID_GRAY']
        with patch.object(originals,'load_gt_review',return_value=manifest):
            result=lab.transfer_audit(self.capture,self.output)
        self.assertEqual(result['shots'][0]['status'],'UNAVAILABLE')
        self.assertEqual(result['shots'][0]['reason'],'MISSING_NEUTRAL_STATE')
        self.assertEqual(result['shots'][1]['other_triplets_coefficient']['status'],'UNAVAILABLE')

    def test_interrupted_audit_keeps_completed_measurements(self):
        original=lab.measure_triplet;calls=0
        def measure(*a,**kw):
            nonlocal calls
            calls+=1
            if calls==2:raise KeyboardInterrupt()
            return original(*a,**kw)
        with patch.object(lab,'measure_triplet',side_effect=measure):
            with self.assertRaises(KeyboardInterrupt):lab.transfer_audit(self.capture,self.output)
        result=json.loads((self.output/'summary.json').read_text())
        self.assertEqual(result['status'],'INTERRUPTED');self.assertEqual(len(result['shots']),1)
        self.assertEqual(result['shots'][0]['shot_id'],1)


if __name__=='__main__':unittest.main()
