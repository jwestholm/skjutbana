"""Small correctness tests for isolated experiment mechanics."""
import unittest
import copy
import tempfile
from pathlib import Path

from automation.offline_10iter import BASE_CONFIG, guardrails, measure, orders, transformed, trial, write
from src.engine.ai.training_v223.schema import CandidateTrainingRow, ShotTrainingRecord


class ExperimentTests(unittest.TestCase):
    def record(self):
        r=ShotTrainingRecord(session_id="s",shot_id="1",source_kind="test",timestamp=0,
            gt_camera_x=0,gt_camera_y=0,candidates=[
                CandidateTrainingRow("wrong",100,100,{"area":5},baseline_score=2),
                CandidateTrainingRow("right",0,0,{"area":10},baseline_score=1)])
        r.finalize_labels()
        return r

    def test_metrics_and_full_membership(self):
        r=self.record()
        baseline,_=measure([r],[[0,1]])
        candidate,_=measure([r],[[1,0]])
        self.assertEqual(baseline['tolerances']['20']['mrr_all'],.5)
        self.assertEqual(candidate['tolerances']['20']['top1'],1)
        self.assertTrue(guardrails(candidate,baseline))
        with self.assertRaises(ValueError):measure([r],[[1]])

    def test_ranking_count_mismatch_rejected(self):
        r=self.record()
        with self.assertRaises(ValueError):measure([r],[])
        with self.assertRaises(ValueError):measure([r],[[0,1],[0,1]])

    def test_missing_oracle_contributes_zero_mrr(self):
        a,b=self.record(),self.record();b.shot_id='2';b.gt_camera_x=1000;b.finalize_labels()
        metrics,_=measure([a,b],[[1,0],[1,0]])
        self.assertEqual(metrics['tolerances']['20']['mrr_all'],.5)
        self.assertEqual(metrics['tolerances']['20']['mean_rank_when_present'],1)

    def test_transforms_do_not_use_gt_or_mutate(self):
        a=self.record();b=copy.deepcopy(a);b.gt_camera_x=1000;b.finalize_labels()
        for mode in ('identity','signed_log','within_shot_percentile'):
            cfg={**BASE_CONFIG,'transform':mode}
            ta,tb=transformed([a,b],cfg)
            self.assertEqual([c.features for c in ta.candidates],[c.features for c in tb.candidates])
        self.assertEqual(a.candidates[0].features,{'area':5})
        self.assertEqual(orders([a],BASE_CONFIG),[[0,1]])

    def test_immutable_files_and_holdout_seal(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);write(out/'frozen_challenger.json',{})
            with self.assertRaises(FileExistsError):write(out/'frozen_challenger.json',{})
            with self.assertRaises(ValueError):trial(out,1,0,'forbidden',{'method':'learned'})


if __name__=='__main__':unittest.main()
