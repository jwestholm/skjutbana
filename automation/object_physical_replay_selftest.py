"""Forensic measurement regressions; synthetic examples are not validation.

Optional --capture explicitly replays a CAN_OBJECT root; default tests never
discover or read physical sessions. Tests preserve known live limitations rather
than concealing them with a new acceptance rule.
"""
import argparse
import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from automation import object_physical_replay as research
from automation.object_physical_query_selftest import query, frame, hit_result


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.pre=np.full((64,64),140,np.uint8)
        self.mask=np.zeros((64,64),bool);self.mask[28:33,28:33]=True
        self.surround=~self.mask

    def signature(self,frames):
        return research.change_signature(self.pre,frames,self.mask,self.surround)

    def test_persistent_local_damage_survives_later_frames(self):
        damage=self.pre.copy();damage[self.mask]=90
        rows=self.signature([self.pre,damage,damage,damage])
        self.assertEqual([r['signed_mean'] for r in rows],[0,-50,-50,-50])
        q,pre=query();post=pre.copy();post[89:94,122:127]=90
        self.assertEqual(hit_result(q,post)['status'],'PHYSICAL_HIT')

    def test_false_two_frame_persistence_is_not_descriptive_permanence(self):
        disturbance=self.pre.copy();disturbance[self.mask]=180
        rows=self.signature([disturbance,disturbance,self.pre,self.pre])
        self.assertEqual([r['absolute_mean'] for r in rows],[40,40,0,0])

    def test_live_absolute_persistence_can_confirm_opposite_signs(self):
        # Reproduction of a real limitation, NOT a desired damage classifier.
        q,pre=query();dark=pre.copy();light=pre.copy()
        dark[89:94,122:127]=80;light[89:94,122:127]=200
        q.observe(frame(dark,2.03),delivered_ts=2.032)
        result=q.observe(frame(light,2.065),delivered_ts=2.067)
        h=result['hypotheses'][0]
        self.assertTrue(h['confirmed'])
        self.assertEqual(h['evidence']['dark_fraction'],0)
        self.assertEqual(h['previous_same_xy_evidence']['dark_fraction'],1)
        self.assertFalse(result['global_coordinate_authority'])
        self.assertFalse(result['truth_or_learning_input'])

    def test_coherent_translation_is_shared_then_decays(self):
        pre=np.full((100,240),100,np.uint8)
        for x in (30,90,150,210):pre[30:60,x:x+12]=200
        moving=cv2.warpAffine(pre,np.float32([[1,0,1],[0,1,0]]),(240,100),borderMode=cv2.BORDER_REPLICATE)
        signatures=[]
        for x in (30,90,150,210):
            mask=np.zeros_like(pre,bool);mask[35:55,x:x+1]=True
            surround=np.zeros_like(pre,bool);surround[70:80,:]=True
            rows=research.change_signature(pre,[moving,moving,pre],mask,surround)
            signatures.append([r['signed_mean'] for r in rows])
        self.assertEqual(signatures,[[-100,-100,0]]*4)

    def test_background_brightness_does_not_create_local_step(self):
        rows=self.signature([self.pre+10,self.pre-10])
        self.assertEqual([r['signed_mean'] for r in rows],[0,0])

    def test_reference_already_contains_damage_removes_novelty(self):
        damaged=self.pre.copy();damaged[self.mask]=90
        r=research.change_signature(damaged,[damaged],self.mask,self.surround)
        self.assertEqual(r[0]['absolute_mean'],0)

    def test_empty_context_refused(self):
        with self.assertRaises(ValueError):
            research.change_signature(self.pre,[self.pre],self.mask,np.zeros_like(self.mask))

    def test_path_escape_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.assertEqual(research.safe_file(root,'input.npy'),root/'input.npy')
            with self.assertRaises(ValueError):research.safe_file(root,'../outside')
            (root/'link').symlink_to('/tmp')
            with self.assertRaises(ValueError):research.safe_file(root,'link/file')

    def test_forbidden_session_refused_before_read(self):
        with self.assertRaises(ValueError):research.load_capture('/nonexistent/S03')

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):research.run('/nonexistent',temp)

    def test_object_choice_needs_physical_confirmation_and_exact_collision(self):
        def result(confirmed,inside):
            return dict(status='PHYSICAL_HIT',hypotheses=[dict(confirmed=confirmed,
                regions=[dict(object_id='a',inside_exact_shape=inside)])])
        self.assertIsNone(research.object_choice(result(False,True)))
        self.assertIsNone(research.object_choice(result(True,False)))
        self.assertEqual(research.object_choice(result(True,True)),'a')


def physical_suite(capture):
    shots=research.load_capture(capture)
    if len(shots)!=3:raise ValueError('explicit accepted three-shot regression capture required')
    class PhysicalReplayTests(unittest.TestCase):
        pass
    # These are RECORDED outputs, including the two known failures, not truth.
    for shot,expected in zip(shots,('gold_can','beans_xxl','mini_can')):
        def check(self,shot=shot,expected=expected):
            replay=research.replay(shot)
            self.assertTrue(replay['matches_original_xy'])
            self.assertLessEqual(replay['max_original_score_error'],1e-6)
            self.assertEqual(replay['object_choice'],expected)
        setattr(PhysicalReplayTests,f"test_recorded_shot_{shot['trace']['shot_id']}",check)
    return unittest.defaultTestLoader.loadTestsFromTestCase(PhysicalReplayTests)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path)
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(MeasurementTests)
    if args.capture:suite.addTests(physical_suite(args.capture))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
