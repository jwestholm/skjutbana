"""Generated-only evidence mechanics; these are not physical acceptance tests."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from automation import object_region_research as lab
from automation import object_hit_replay as replay
from automation.object_hit_replay_selftest import capture
from automation.object_physical_query_selftest import snapshot
from src.engine.camera.analysis_geometry_v2221 import AnalysisGeometryV2221
from src.engine.input.hit_regions import HitRegion, object_hit_regions
from src.engine.input.object_hit_v2223 import CameraHitRegionV240


def geometry(x=100,y=200,w=80,h=80):
    return AnalysisGeometryV2221(500,600,x,y,x+w,y+h,np.empty((0,2)),np.empty((0,2)),
        np.ones((h,w),np.uint8),'test',0,8)


class GeometryTests(unittest.TestCase):
    def test_subset_is_original_pixels_with_native_origin(self):
        a=np.arange(120*130).reshape(120,130)
        saved=replay.SavedCrops((500,600),{(90,190,220,310):a})
        g=geometry();b=lab.subset_crop(g,saved)
        np.testing.assert_array_equal(b,a[10:90,10:90])
        self.assertTrue(np.shares_memory(a,b));self.assertEqual(g.local_to_camera(5,7),(105.,207.))

    def test_missing_not_padded(self):
        with self.assertRaisesRegex(ValueError,'UNAVAILABLE'):
            lab.subset_crop(geometry(),replay.SavedCrops((500,600),{}))

    def test_overlapping_conflicting_tiles_rejected(self):
        s=replay.SavedCrops((500,600),{(100,200,180,280):np.zeros((80,80)),
            (90,190,190,290):np.ones((100,100))})
        with self.assertRaisesRegex(ValueError,'conflicting'):lab.subset_crop(geometry(),s)

    def test_exact_shape_and_margin_separate(self):
        shape=dict(kind='ellipse',x=110,y=210,width=40,height=40)
        r=HitRegion('enemy',110,210,40,40,metadata={'v250_object_snapshot':{'shape':shape}})
        mask=lab.exact_mask(r,geometry(),lambda x,y:(x,y))
        self.assertTrue(mask[30,30]);self.assertFalse(mask[10,10]);self.assertFalse(mask[0,0])
        self.assertEqual(mask[30,10],True)  # exact ellipse boundary

    def test_plain_rectangle_boundary_matches_live(self):
        r=HitRegion('enemy',110,210,40,40)
        self.assertTrue(lab.inside(r,(110,210)));self.assertFalse(lab.inside(r,(150,210)))

    def test_snapshot_selects_one_and_does_not_mutate(self):
        s=snapshot(regions=[CameraHitRegionV240('a',10,10,20,20),CameraHitRegionV240('b',50,50,20,20)],
            game=object_hit_regions([HitRegion('a',10,10,20,20),HitRegion('b',50,50,20,20)]))
        v=lab.one_snapshot(s,'a');self.assertEqual(len(v.camera_regions),1)
        self.assertEqual(len(s.camera_regions),2)
        v.game_regions[0].metadata['audit']='new';self.assertNotIn('audit',s.game_regions[0].metadata)
        with self.assertRaises(ValueError):lab.one_snapshot(s,'absent')


class EvidenceTests(unittest.TestCase):
    def arrays(self):
        rng=np.random.default_rng(731)
        pre=cv2.GaussianBlur(rng.normal(120,30,(128,128)).astype(np.float32),(0,0),1)
        mask=np.ones_like(pre,bool);mask[:8]=mask[-8:]=False;mask[:,:8]=mask[:,-8:]=False
        exact=np.zeros_like(mask);exact[45:83,45:83]=True
        return pre,mask,exact

    def test_zero_change_no_invented_residual(self):
        pre=np.full((128,128),120,np.float32);_,mask,exact=self.arrays()
        maps,f=lab.residual_maps([pre,pre],pre,mask,exact)
        self.assertEqual(np.max(np.abs(maps['registered_residual'])),0.)
        self.assertFalse(f[0]['values']['physical_cause_certified'])

    def test_bounded_motion_leaves_new_local_residual(self):
        pre,mask,exact=self.arrays()
        moved=cv2.warpAffine(pre,np.float32([[1,0,2],[0,1,1]]),(128,128),borderMode=cv2.BORDER_REFLECT)
        hole=moved.copy();hole[60:65,60:65]-=70
        maps,f=lab.residual_maps([pre,pre],hole,mask,exact)
        reg=f[1]['values']['registration'];self.assertTrue(reg['applied'])
        self.assertLess(np.hypot(reg['dx'],reg['dy']),4.01)
        self.assertGreater(np.abs(maps['registered_residual'][60:65,60:65]).mean(),55)
        self.assertLess(f[1]['values']['residual_margin']['abs_mean'],f[1]['values']['raw_margin']['abs_mean'])
        self.assertFalse(f[1]['values']['projector_vs_surface_separated'])

    def test_outside_change_is_not_inside_mass(self):
        pre=np.full((128,128),120,np.float32);_,mask,exact=self.arrays()
        post=pre.copy();post[25:30,25:30]=30
        maps,_=lab.residual_maps([pre,pre],post,mask,exact)
        f=lab.residual_structure(maps['registered_residual'],maps['valid']>0,exact,[],geometry(w=128,h=128))
        self.assertEqual(f['values']['inside_absolute_mass'],0)
        self.assertGreater(f['values']['outside_absolute_mass'],0)

    def test_one_pre_has_no_fake_envelope(self):
        pre,mask,exact=self.arrays();maps,f=lab.residual_maps([pre],pre,mask,exact)
        self.assertNotIn('observed_envelope_residual',maps)
        self.assertIsNone(f[0]['values']['observed_envelope_inside'])

    def test_no_context_no_fitted_explanation(self):
        pre,mask,exact=self.arrays();maps,f=lab.residual_maps([pre,pre],pre,mask,mask)
        self.assertNotIn('registered_residual',maps);self.assertEqual(f[1]['availability'],'UNAVAILABLE')

    def test_shape_mismatch_rejected(self):
        pre,mask,exact=self.arrays()
        with self.assertRaises(ValueError):lab.residual_maps([pre],pre[:-1],mask,exact)

    def test_exact_xy_no_source_bonus_and_no_secondary_veto(self):
        region=HitRegion('a',100,200,80,80)
        def c(x,y):return dict(camera_xy=[x,y],confirmed=True,producer='ANY',event_freshness={})
        original=dict(object_decision={'state':'OBJECT_HIT','reason':'unique_supported_object'},
            per_object=[{'candidates':[c(110.25,220.75),c(130,230),c(90,220)]}])
        before=deepcopy(original);r=lab.region_result(original,region,lambda x,y:(x,y))
        self.assertEqual(r['state'],'UNKNOWN');self.assertIsNone(r['camera_xy'])
        self.assertEqual(r['diagnostic_xy_status'],'MULTIPLE_UNRESOLVED')
        self.assertEqual(r['qualified_by_existing_query_only'][0]['camera_xy'],[110.25,220.75])
        self.assertEqual(r['qualified_by_existing_query_only'][0]['region_local_xy'],[10.25,20.75])
        self.assertIsNone(r['qualified_by_existing_query_only'][0]['object_local_xy'])
        self.assertEqual(before,original)

    def test_no_support_does_not_prove_miss_or_no_shot(self):
        r=lab.region_result({},HitRegion('a',0,0,10,10),lambda x,y:(x,y))
        self.assertEqual(r['state'],'UNKNOWN');self.assertFalse(r['no_physical_shot_established'])


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory(prefix='single_region_test_',dir='/data/skjutbana/evaluation_runs')
        self.addCleanup(self.t.cleanup);self.root=Path(self.t.name);self.source=self.root/'capture'
        capture(self.source);_,self.shots,self.hashes=replay.load(self.source)

    def test_same_query_on_single_region_retains_original_coordinates(self):
        baseline,_=replay.replay(self.shots[0]);row,maps=lab.analyze_region(self.shots[0],'can')
        self.assertEqual(replay.parity_view(baseline['replay']),replay.parity_view(row['query_result']))
        self.assertTrue(all(p.startswith('f') or p.endswith('mask') for p in maps))
        self.assertEqual(row['result']['state'],'UNKNOWN')
        self.assertTrue(all(replay.digest(p)==h for p,h in self.hashes.items()))

    def test_deterministic_no_candidate_or_truth_fitting(self):
        a,ma=lab.analyze_region(self.shots[0],'can');b,mb=lab.analyze_region(self.shots[0],'can')
        self.assertEqual(lab.semantic(a),lab.semantic(b))
        for k in ma:np.testing.assert_array_equal(ma[k],mb[k])
        self.shots[0]['log']['intended_object']='NOT_TRUTH'
        c,_=lab.analyze_region(self.shots[0],'can');self.assertEqual(lab.semantic(a),lab.semantic(c))
        projector=[f for x in c['frames'] for f in x.get('families',[]) if f['family']=='PROJECTOR_EXPLAINABILITY']
        self.assertTrue(all(f['availability']=='UNAVAILABLE' for f in projector))

    def test_late_missing_crop_explicit(self):
        self.shots[0]['posts'][-1].frame_bgr.crops={}
        row,_=lab.analyze_region(self.shots[0],'can')
        self.assertEqual(row['frames'][-1]['status'],'UNAVAILABLE')

    def test_later_shadow_frame_never_reopens_query(self):
        from types import SimpleNamespace
        self.shots[0]['trace']['object_query_diagnostic']['decision_ts']+=.02
        before,_=lab.analyze_region(self.shots[0],'can')
        last=self.shots[0]['posts'][-1]
        frame=SimpleNamespace(timestamp=last.timestamp+.01,frame_bgr=replay.SavedCrops(
            last.frame_bgr.shape,{box:np.zeros_like(a) for box,a in last.frame_bgr.crops.items()}))
        self.shots[0]['posts'].append(frame)
        after,_=lab.analyze_region(self.shots[0],'can')
        self.assertEqual(lab.semantic(before['query_result']),lab.semantic(after['query_result']))
        self.assertTrue(after['frames'][-1]['shadow_after_query_closed'])

    def test_post_outside_captured_cutoff_is_rejected(self):
        self.shots[0]['posts'][-1].timestamp+=1
        with self.assertRaisesRegex(ValueError,'CAUSAL_WINDOW'):lab.analyze_region(self.shots[0],'can')

    def test_source_oracle_and_game_points_do_not_enter_measurements(self):
        before,_=lab.analyze_region(self.shots[0],'can')
        self.shots[0]['snapshot'].game_regions[0].metadata.update(points=999,known_hole=True,tape=True)
        after,_=lab.analyze_region(self.shots[0],'can')
        self.assertEqual(lab.semantic(before['frames']),lab.semantic(after['frames']))
        self.assertEqual(lab.semantic(before['result']),lab.semantic(after['result']))

    def test_run_immutable_and_source_hashes_unchanged(self):
        with redirect_stdout(io.StringIO()):rows=lab.run(self.source,self.root/'out',object_id='can')
        m=json.loads((self.root/'out/manifest.json').read_text());self.assertEqual(m['status'],'COMPLETE')
        self.assertEqual(rows[0]['research_state'],'UNKNOWN')
        with self.assertRaises(ValueError):lab.run(self.source,self.root/'out',object_id='can')
        self.assertTrue(all(replay.digest(p)==h for p,h in self.hashes.items()))

    def test_interrupted_run_preserves_checkpoint(self):
        with patch.object(lab,'analyze_region',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):lab.run(self.source,self.root/'out')
        self.assertEqual(json.loads((self.root/'out/manifest.json').read_text())['status'],'INTERRUPTED')
        self.assertTrue((self.root/'out/summary.json').exists())

    def test_missing_frame_event_unavailable(self):
        (self.source/'object_crops/shots/shot_00000001/trace.json').unlink()
        with redirect_stdout(io.StringIO()):rows=lab.run(self.source,self.root/'out')
        self.assertEqual(rows[0]['status'],'UNAVAILABLE');self.assertEqual(rows[0]['research_state'],'UNKNOWN')

    def test_partial_log_only_sixth_event_stays_unavailable(self):
        with (self.source/'runtime.log').open('a') as f:f.write('[V2.22.3 SHOT] shot=6 partial\n')
        with redirect_stdout(io.StringIO()):rows=lab.run(self.source,self.root/'out')
        self.assertEqual(rows[-1]['shot_id'],6);self.assertEqual(rows[-1]['status'],'UNAVAILABLE')
        self.assertEqual(rows[-1]['reason'],'NO_SAVED_QUERY_GEOMETRY_OR_CROPS')

    def test_forbidden_source_before_read_and_output_guards(self):
        with patch.object(Path,'read_text',side_effect=AssertionError('must not read')):
            with self.assertRaises(ValueError):lab.run('/data/skjutbana/S03/forbidden',self.root/'out')
        with self.assertRaises(ValueError):lab.run(self.source,self.source/'out')
        with self.assertRaises(ValueError):lab.run(self.source,'/tmp/unallowed')

    def test_default_cli_unchanged(self):
        with patch('sys.argv',['replay','--capture','unused','--output','unused']),patch.object(replay,'run') as run:
            replay.main();run.assert_called_once()


if __name__=='__main__':unittest.main()
