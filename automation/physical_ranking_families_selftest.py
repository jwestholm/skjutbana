"""Software regressions for cache-only ranking; no physical validation claims."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from automation.physical_ranking_families import family, tied_ranks, weighted_correlation
from automation.physical_family_rank_research import (robust_scores, pareto_counts,
    rank_columns, shape_evidence, motion_residual_evidence)
from automation.physical_ranking_context import spatial_components,ready_observed_ids
from automation.physical_family_refinement import correlation_groups,direct_families,grouped_design


class Tests(unittest.TestCase):
    def test_feature_families_do_not_confuse_noise_with_new_damage(self):
        self.assertEqual(family('piecewise_clean_r4_noise'),'pre_context')
        self.assertEqual(family('pre_centered_clean_r4_inner_ring'),'registered_residual')
        self.assertEqual(family('history_local_fraction'),'history')

    def test_tied_ranks_are_neutral_and_permutation_invariant(self):
        np.testing.assert_array_equal(tied_ranks([7,7,7]),[.5,.5,.5])
        a=np.array([1.,2.,2.,5.]);p=np.array([3,1,0,2])
        np.testing.assert_array_equal(tied_ranks(a)[p],tied_ranks(a[p]))
        np.testing.assert_array_equal(tied_ranks(a),tied_ranks(4*a+20))

    def test_nonfinite_ranks_are_refused(self):
        for a in ([],[1,np.nan],[1,np.inf]):
            with self.assertRaises(ValueError):tied_ranks(a)

    def test_equal_event_weight_resists_duplicate_rows(self):
        x=np.array([[1.,2.],[2.,3.],[4.,0.]])
        c=weighted_correlation(x,np.ones(3))
        duplicated=np.vstack([x,x[0]])
        np.testing.assert_allclose(c,weighted_correlation(duplicated,[.5,1,1,.5]))

    def test_constant_correlations_are_finite(self):
        c=weighted_correlation(np.array([[1.,0.],[1.,2.]]),np.ones(2))
        self.assertTrue(np.isfinite(c).all());self.assertEqual(c[0,0],0.)

    def test_one_unbounded_family_cannot_overwhelm_capped_score(self):
        values=np.array([[1000.,-1.,-1.],[0.,.5,.5]])
        self.assertGreater(robust_scores(values,['a','b','c'],'FAMILY_NORMALIZED')[0],0)
        self.assertGreater(np.diff(robust_scores(values,['a','b','c'],'FAMILY_CAPPED'))[0],0)

    def test_dominance_counts_do_not_discard_ties_or_incomparable_candidates(self):
        v=np.array([[0.,0.],[1.,1.],[1.,0.],[0.,2.],[1.,1.]])
        np.testing.assert_array_equal(pareto_counts(v),[4,0,2,0,0])

    def test_ranking_structures_are_candidate_permutation_equivariant(self):
        rng=np.random.default_rng(8);v=rng.normal(size=(30,6));p=rng.permutation(30)
        names=['localization','morphology','new_change','pre_context','registered_residual','temporal']
        for method in ('FAMILY_NORMALIZED','FAMILY_CAPPED','RANK_MEAN','RANK_MEDIAN','CONSENSUS','PARETO','TWO_STAGE'):
            a=robust_scores(v,names,method,raw_scores=v.sum(axis=1))
            b=robust_scores(v[p],names,method,raw_scores=v[p].sum(axis=1))
            np.testing.assert_allclose(a[p],b)

    def test_temporal_shape_does_not_require_after_audio_onset(self):
        names=[f'r{r}_{key}' for r in (2,4,8,16) for key in ('step_explained','step_persistent_fraction','step_snr','step_onset_ms')]
        a=np.ones((2,len(names)));b=a.copy()
        for j,n in enumerate(names):
            if n.endswith('onset_ms'):a[:,j]=-100;b[:,j]=100
        np.testing.assert_array_equal(shape_evidence(a,names),shape_evidence(b,names))

    def test_motion_context_uses_residual_survival_without_absolute_position(self):
        names=[f'{channel}_clean_r{r}_{key}' for r in (2,4,8) for key in ('mean','inner_ring') for channel in ('piecewise','pre_rms')]
        a=np.zeros((2,len(names)))
        for j,n in enumerate(names):a[:,j]=[1,2] if n.startswith('piecewise') else [2,1]
        np.testing.assert_array_equal(motion_residual_evidence(a,names),[-1,1])

    def test_guard_refuses_capture_reads_and_baseline_writes_before_open(self):
        code='''from pathlib import Path
from automation.physical_ranking_families import install_read_guard
install_read_guard(Path('/tmp/research_output'))
for p,mode in [('/data/skjutbana/physical_traces/never_open','r'),('/tmp/session_S03_fixture/never_open','r'),('/data/skjutbana/baseline_never_write','w')]:
 try: open(p,mode)
 except PermissionError: pass
 else: raise AssertionError(p)
print('GUARD_PASS')
'''
        cp=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(cp.returncode,0,cp.stderr);self.assertIn('GUARD_PASS',cp.stdout)

    def test_context_uses_same_plausible_universe_as_atlas(self):
        records=[dict(selectable=True,ready=True),dict(selectable=True,ready=True),
                 dict(selectable=False,ready=True),dict(selectable=True,ready=False)]
        event=dict(indices=[0,1,2,3],pools=dict(current=[0],raw_plausible=[2,3]))
        # Index 1 is an unfiltered raw speck; index 2 is a GT-only probe.
        self.assertEqual(ready_observed_ids(event,records),[0])

    def test_spatial_clusters_do_not_claim_independent_impacts(self):
        groups=spatial_components(np.array([[0,0],[10,0],[20,0],[100,0]]))
        self.assertEqual(groups,[[0,1,2],[3]])

    def test_exact_duplicate_grouping_preserves_distinct_evidence(self):
        x=np.array([[0.,0.,2.],[1.,1.,0.],[2.,2.,1.]])
        groups,_=correlation_groups(x,['r2_abs_late','r4_abs_late','r8_abs_late'],np.ones(3),'unique')
        self.assertEqual(groups,[[0,1],[2]])

    def test_grouping_keeps_semantic_families_separate(self):
        x=np.column_stack([np.arange(10),-np.arange(10),np.arange(10)])
        groups,_=correlation_groups(x,['r2_abs_late','r4_abs_late','r2_pre_std'],np.ones(10),'grouped')
        self.assertEqual(groups,[[0,1],[2]])

    def test_complete_link_does_not_merge_correlation_chains(self):
        rng=np.random.default_rng(24);a,b=rng.normal(size=(2,10000))
        x=np.column_stack([np.cos(t)*a+np.sin(t)*b for t in (0.,.2,.4)])
        groups,corr=correlation_groups(x,['r2_abs_late','r4_abs_late','r8_abs_late'],np.ones(len(a)),'grouped')
        self.assertGreater(corr[0,1],.95);self.assertLess(corr[0,2],.95)
        self.assertEqual(groups,[[0,1],[2]])

    def test_direct_ranker_ignores_source_gt_and_onset_metadata(self):
        names=[f'r{r}_{key}' for r in (2,4,8) for key in ('abs_late','early_late_agreement','spatial_consistency','center_ring_ratio')]
        names += [f'{channel}_clean_r{r}_inner_ring' for channel in ('persistent','pre_centered','piecewise') for r in (2,4,8)]
        names += ['source_score','ground_truth_x','audio_onset_ms']
        rng=np.random.default_rng(16);x=rng.normal(size=(12,len(names)));altered=x.copy();altered[:,-3:]*=1000
        for pixel in (False,True):
            np.testing.assert_array_equal(direct_families(x,names,pixel)[1],direct_families(altered,names,pixel)[1])

    def test_group_fitting_excludes_the_entire_held_session(self):
        data=dict(records=[dict(ready=True,selectable=True) for _ in range(4)],events=[
            dict(session='D01',pools=dict(current=[0,1])),dict(session='S01',pools=dict(current=[2,3]))])
        x=np.array([[0.,0.],[0.,1.],[1.,2.],[3.,4.]])
        a,meta=grouped_design(data,x,['r2_abs_late','r4_abs_late'],['S01'],'grouped')
        changed=x.copy();changed[:2]=[[1000,0],[-1000,2000]]
        b,other=grouped_design(data,changed,['r2_abs_late','r4_abs_late'],['S01'],'grouped')
        self.assertEqual(meta,other);np.testing.assert_array_equal(a[2:],b[2:])


if __name__=='__main__':unittest.main()
