"""Evaluation integrity and causal evidence regression tests, not physical validation."""
import copy
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from automation.accuracy_physical_dataset import causal_stages, guard_path, json_read
from automation.accuracy_verifier_research import event_predictions, fit_model, predict, summarize, training_data
from src.engine.offline.accuracy_verifier import EvidenceContext, align_to_reference, features, proposals
from src.engine.offline.physical_event_truth import event_truth, score_event


class AccuracyResearchTests(unittest.TestCase):
    def context(self):
        rng=np.random.default_rng(17)
        pre=cv2.GaussianBlur(rng.uniform(80,160,(96,128)).astype(np.float32),(0,0),1)
        post=np.stack([pre.copy() for _ in range(4)])
        post[:,45:50,61:66]-=30
        return EvidenceContext(pre,np.stack([pre,pre]),post,1.,1.3,(.5,.6),(1.,1.1,1.2,1.3),
                               (100,200),np.ones(pre.shape,np.uint8),[])

    def test_unknown_and_multi_are_never_silently_scored_as_single(self):
        point=dict(camera_x=1,camera_y=2)
        for state in ('UNKNOWN','AMBIGUOUS','MULTI_IMPACT'):
            truth=event_truth(state,[point,point] if state=='MULTI_IMPACT' else [])
            self.assertFalse(score_event(truth,point)['evaluable'])
        with self.assertRaises(ValueError):event_truth('MULTI_IMPACT',[point])

    def test_truth_preserves_caveat_and_impact_array(self):
        point=dict(camera_x=1,camera_y=2)
        truth=event_truth('SINGLE_IMPACT',[point],caveats=['UNCONFIRMED'])
        self.assertEqual(truth['impacts'],[point]);self.assertEqual(truth['caveats'],['UNCONFIRMED'])
        truth['impacts'][0]['camera_x']=3
        self.assertEqual(point['camera_x'],1)

    def test_threshold_boundaries_and_false_rejection(self):
        truth=event_truth('SINGLE_IMPACT',[dict(camera_x=0,camera_y=0)])
        self.assertEqual(score_event(truth,dict(camera_x=42,camera_y=0))['hits'],dict(zip(('5','10','20','42'),(False,False,False,True))))
        self.assertFalse(score_event(truth,dict(camera_x=42.001,camera_y=0))['hits']['42'])
        result=score_event(truth,None)
        self.assertTrue(result['false_rejection']);self.assertFalse(any(result['hits'].values()))

    def test_nonphysical_requires_explicit_state(self):
        with self.assertRaises(ValueError):event_truth('NO_PHYSICAL_SHOT',[dict(camera_x=0,camera_y=0)])
        truth=event_truth('NO_PHYSICAL_SHOT')
        self.assertTrue(score_event(truth,None)['true_negative'])
        self.assertTrue(score_event(truth,dict(camera_x=0,camera_y=0))['false_emission'])

    def test_nonfinite_truth_and_time_rejected(self):
        with self.assertRaises(ValueError):event_truth('SINGLE_IMPACT',[dict(camera_x=float('nan'),camera_y=0)])
        with self.assertRaises(ValueError):replace(self.context(),post_times=(1.,1.1,1.2,float('nan')))

    def test_s03_refused_before_open_including_alias(self):
        with patch.object(Path,'read_text',side_effect=AssertionError('Must not read')):
            with self.assertRaises(PermissionError):json_read('/tmp/session_20260909_S03_held/trace.json')
        with tempfile.TemporaryDirectory() as directory:
            alias=Path(directory)/'innocent';alias.symlink_to(Path(directory)/'session_S03_held')
            with self.assertRaises(PermissionError):guard_path(alias/'trace.json')

    def test_stage_ownership_and_cutoff(self):
        trace=dict(shot_id=2,decision_input=dict(timestamp=5),stages=[
            dict(timestamp=3,candidate_pool_shot_id=1),dict(timestamp=4,candidate_pool_shot_id=2),
            dict(timestamp=6,candidate_pool_shot_id=2)])
        self.assertEqual(causal_stages(trace),[trace['stages'][1]])

    def test_future_post_and_post_as_pre_refused(self):
        with self.assertRaises(ValueError):replace(self.context(),cutoff=1.29)
        with self.assertRaises(ValueError):replace(self.context(),pre_times=(.5,1.))
        with self.assertRaises(ValueError):replace(self.context(),post_times=(.99,1.1,1.2,1.3))

    def test_coordinate_translation_exactly_once(self):
        context=self.context()
        names,vector=features(context,(163,247))
        names2,vector2=features(replace(context,origin=(0,0)),(63,47))
        self.assertEqual(names,names2);np.testing.assert_array_equal(vector,vector2)
        with self.assertRaises(ValueError):features(context,(63,47))

    def test_features_deterministic_and_inputs_immutable(self):
        context=self.context();before=copy.deepcopy(context)
        first=features(context,(163,247));second=features(context,(163,247))
        np.testing.assert_array_equal(first[1],second[1])
        for name in ('pre','post','history','roi'):np.testing.assert_array_equal(getattr(context,name),getattr(before,name))
        self.assertEqual(len(first[0]),72)
        self.assertFalse(any('score' in name or 'source' in name or 'camera' in name for name in first[0]))

    def test_registration_translation_direction(self):
        pre=self.context().pre
        moved=cv2.warpAffine(pre,np.float32([[1,0,2],[0,1,-1]]),(pre.shape[1],pre.shape[0]),borderMode=cv2.BORDER_REFLECT)
        aligned,metadata=align_to_reference(pre,moved)
        self.assertTrue(metadata['applied']);self.assertAlmostEqual(metadata['dx'],2,delta=.4)
        self.assertAlmostEqual(metadata['dy'],-1,delta=.4)
        self.assertLess(np.mean(abs(pre[8:-8,8:-8]-aligned[8:-8,8:-8])),np.mean(abs(pre[8:-8,8:-8]-moved[8:-8,8:-8])))

    def test_proposals_bounded_causal_and_repeatable(self):
        context=self.context();a=proposals(context,8);b=proposals(context,8)
        self.assertEqual(a,b);self.assertLessEqual(len(a),8);self.assertTrue(a)
        self.assertTrue(any(np.hypot(c['camera_x']-163,c['camera_y']-247)<=5 for c in a))
        self.assertTrue(all(c['evidence_timestamp']<=context.cutoff for c in a))
        with self.assertRaises(ValueError):proposals(context,0)
        with self.assertRaises(ValueError):proposals(replace(context,roi=np.zeros(context.pre.shape,np.uint8)))

    def dataset(self):
        truth=event_truth('SINGLE_IMPACT',[dict(camera_x=0,camera_y=0)])
        records=[dict(index=0,event='S01:1',session='S01',pool='current',camera_x=100,camera_y=0,gt_distance=100,ready=True,training_label=0),
                 dict(index=1,event='S01:1',session='S01',pool='training_gt_only',camera_x=0,camera_y=0,gt_distance=0,ready=False,training_label=1)]
        return dict(events=[dict(event='S01:1',session='S01',truth=truth,indices=[0,1])],records=records)

    def test_injected_training_gt_can_never_win_evaluation(self):
        rows=event_predictions(self.dataset(),np.array([.1,.99]),['S01'])
        self.assertEqual(rows[0]['selected']['index'],0)
        self.assertFalse(rows[0]['oracle']['42']);self.assertEqual(summarize(rows)['hits']['42'],0)

    def test_rejections_stay_in_accuracy_denominator(self):
        data=self.dataset();rows=event_predictions(data,np.array([.1,.99]),['S01'],threshold=.5)
        metrics=summarize(rows)
        self.assertEqual(metrics['physical'],1);self.assertEqual(metrics['false_rejections'],1)
        self.assertEqual(metrics['accuracy']['42'],0);self.assertIsNone(metrics['mean_error_px'])

    def test_unready_winner_is_not_replaced_by_oracle(self):
        data=self.dataset();data['records'][0]['ready']=False
        row=event_predictions(data,np.array([.1,.99]),['S01'])[0]
        self.assertIsNone(row['selected']);self.assertTrue(row['unready_at_recorded_decision'])

    def test_no_challenge_training(self):
        for alias in ('S02','S03'):
            with self.assertRaises(ValueError):training_data(self.dataset(),np.ones((2,3)),[alias])

    def test_fitted_models_deterministic(self):
        cv2.setNumThreads(1)
        rng=np.random.default_rng(19);x=rng.normal(size=(100,4)).astype(np.float32)
        y=(x[:,0]+x[:,1]>0).astype(np.int32);w=np.full(len(y),1/len(y))
        for family in ('logistic','forest'):
            a=predict(fit_model(x,y,w,family),x);b=predict(fit_model(x,y,w,family),x)
            np.testing.assert_allclose(a,b,rtol=0,atol=0)
            self.assertGreater(np.mean((a>.5)==y),.85)

    def test_unavailable_rescue_is_not_a_measured_no_impact(self):
        data=self.dataset();data['events'][0]['proposal_unavailable']='missing owned mask'
        self.assertEqual(event_predictions(data,np.ones(2),['S01'],'expanded'),[])
        self.assertEqual(len(event_predictions(data,np.ones(2),['S01'],'union')),1)

    def test_pairwise_training_refuses_challenge(self):
        from automation.accuracy_pairwise_research import fit_pairwise
        with self.assertRaises(ValueError):fit_pairwise(self.dataset(),np.ones((2,3)),['S02'])

    def test_visual_event_gate_does_not_consume_truth_or_training_points(self):
        from automation.accuracy_no_impact_research import EVIDENCE_NAMES,event_matrix,gate,train_event
        data=self.dataset();data['records'][0]['pool']='expanded';data['feature_names']=list(EVIDENCE_NAMES)
        matrix=np.arange(18).reshape(2,9)
        before=event_matrix(data,matrix)
        data['events'][0]['truth']=event_truth('UNKNOWN');data['records'][1]['camera_x']=999
        matrix[1]=999
        np.testing.assert_array_equal(before,event_matrix(data,matrix))
        with self.assertRaises(ValueError):train_event(data,before,['S02'])
        rows=event_predictions(self.dataset(),np.array([.1,.99]),['S01'])
        gated=gate(rows,{'S01:1':.1},.5)
        self.assertEqual(summarize(gated)['false_rejections'],1)
        self.assertIsNotNone(rows[0]['selected'])

    def test_contour_rescue_has_spatial_and_size_bounds(self):
        from automation.accuracy_contour_rescue import contour_rescue
        context=self.context();mask=np.zeros(context.pre.shape,np.uint8)
        cv2.circle(mask,(63,47),3,255,-1);cv2.circle(mask,(2,2),2,255,-1)
        points=contour_rescue(mask,context,1)
        self.assertEqual(len(points),1);self.assertAlmostEqual(points[0]['camera_x'],163)
        self.assertTrue(points[0]['ready']);self.assertLessEqual(points[0]['evidence_timestamp'],context.cutoff)

    def test_frozen_artifact_tampering_is_rejected(self):
        import hashlib
        from automation.accuracy_frozen_replay import checked
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.npz';path.write_bytes(b'original')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(checked(path,digest),path)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):checked(path,digest)

    def test_external_frozen_evaluation_refuses_validation_before_reads(self):
        from automation.accuracy_frozen_replay import evaluate_external
        with patch.object(Path, 'read_text', side_effect=AssertionError('Must not read')):
            with self.assertRaises(PermissionError):
                evaluate_external(Path('/tmp/manifest'), Path('/tmp/dataset'), 'S03', Path('/tmp/output'))

    def test_external_frozen_evaluation_requires_same_reference_and_features(self):
        from automation.accuracy_frozen_replay import evaluate_external
        manifest = dict(dataset='/tmp/original')
        original = dict(reference='snapshot', feature_names=['a'], sessions=['S01'])
        for changed in [dict(reference='history_early', feature_names=['a'], sessions=['D01']),
                        dict(reference='snapshot', feature_names=['other'], sessions=['D01'])]:
            with patch('automation.accuracy_frozen_replay.json_read', side_effect=[changed, manifest, original]):
                with self.assertRaisesRegex(ValueError, 'schema differs'):
                    evaluate_external(Path('/tmp/manifest'), Path('/tmp/dataset'), 'D01', Path('/tmp/output'))

    def test_d01_extraction_is_opt_in(self):
        from automation.accuracy_physical_dataset import COHORT, DEFAULT_SESSIONS
        self.assertIn('D01', COHORT)
        self.assertNotIn('D01', DEFAULT_SESSIONS)
        self.assertEqual(DEFAULT_SESSIONS, ('H10', 'H20', 'POST_FIX', 'S01', 'S02'))


if __name__=='__main__':unittest.main()
