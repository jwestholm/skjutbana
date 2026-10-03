"""Retention/measurement contracts; synthetic fixtures do not measure accuracy."""
import copy
import unittest
from automation.exact_witness_research import exact_supported_keep
from automation.chronological_pre_selftest import candidate
from automation.chronological_pre_research import bounded_keep,temporal

class Tests(unittest.TestCase):
    def test_supported_exact_points_not_overwritten_by_producer(self):
        a=candidate(1,born=0,x=100,evidence=99);a['temporal']=dict(supported=True,support=8.)
        b=candidate(2,born=1,x=102,evidence=1);b['temporal']=dict(supported=True,support=10.)
        cs=[a,b];before=copy.deepcopy(cs)
        self.assertEqual([c['camera_x'] for c in bounded_keep(cs,1)[0]],[100.])
        self.assertEqual([c['camera_x'] for c in exact_supported_keep(cs,1)[0]],[102.,100.])
        self.assertEqual(cs,before)
    def test_pending_cannot_replace_support(self):
        a=candidate(1,born=0,x=100,evidence=0,supported=True)
        b=candidate(9,born=3,x=102,evidence=100)
        self.assertEqual(exact_supported_keep([a,b],3)[0],[a])
    def test_pending_dedupe_unchanged(self):
        cs=[candidate(i,x=100+i,born=0) for i in range(3)]
        self.assertEqual(exact_supported_keep(cs,1),bounded_keep(cs,1))
    def test_hard_bound_deterministic_and_no_mutation(self):
        cs=[candidate(i,born=0,x=i,supported=True) for i in range(32)];before=copy.deepcopy(cs)
        a=exact_supported_keep(cs,49)
        self.assertEqual(len(a[0]),24);self.assertEqual(len(a[1]),8)
        self.assertEqual(a,exact_supported_keep(cs,49));self.assertEqual(cs,before)
    def test_oversized_archive_or_capacity_refused(self):
        with self.assertRaises(ValueError):exact_supported_keep([candidate(i) for i in range(33)],1)
        with self.assertRaises(ValueError):exact_supported_keep([],1,capacity=25)
    def test_reversion_revokes_and_retires(self):
        c=candidate(1,born=3,x=100);signal=[0.]*6+[8.]*3
        c['temporal']=temporal(signal);self.assertTrue(exact_supported_keep([c],8)[0])
        c['temporal']=temporal(signal+[0.]);self.assertFalse(exact_supported_keep([c],9)[0])
    def test_no_support_future_or_old_hole(self):
        for sig in ([0.]*12,[-8.]*12,[0.]*6+[-8.]*2):self.assertFalse(temporal(sig)['supported'])

class InformationTests(unittest.TestCase):
    def test_duplicate_operators_not_independent_neighbors(self):
        from automation.physical_structure_information import spatial_values
        cs=[dict(camera_xy=[0.,0.],source_name=s) for s in ('subtract','absdiff','blackhat')]
        out=spatial_values((0,0),cs)
        self.assertEqual(out['raw_proposal_count_r12'],3)
        self.assertEqual(out['unique_xy_count_r12'],1)
        self.assertEqual(out['neighbor_count_r3'],0)
        self.assertEqual(out['unique_producer_count_r12'],3)
        self.assertEqual(out['unique_source_family_count_r12'],1)
        self.assertIsNone(out['linearity'])
        unknown=spatial_values((0,0),[dict(camera_xy=[0,0])])
        self.assertIsNone(unknown['unique_producer_count_r12'])
    def test_line_square_and_distance_units(self):
        from automation.physical_structure_information import spatial_values
        line=[dict(camera_xy=[x,0]) for x in (-6,-3,0,3,6)]
        square=[dict(camera_xy=[x,y]) for x,y in ((-2,-2),(-2,2),(2,-2),(2,2))]
        a=spatial_values((0,0),line);b=spatial_values((0,0),square)
        self.assertEqual([a[f'neighbor_count_r{r}'] for r in (3,6,12)],[2,4,4])
        self.assertAlmostEqual(a['linearity'],1);self.assertAlmostEqual(b['isotropy'],1)
        self.assertEqual(a,spatial_values((0,0),line));self.assertEqual(a['compactness'],0.)
    def test_information_does_not_mutate_pixels_or_proposals(self):
        import numpy as np
        from automation.physical_structure_information import spatial_values
        im=np.arange(729,dtype=np.float32).reshape(27,27);original=im.copy()
        cs=[dict(camera_xy=[0,0])];before=copy.deepcopy(cs);spatial_values((0,0),cs,im)
        np.testing.assert_array_equal(im,original);self.assertEqual(cs,before)
    def test_inventory_extractability_not_damage_label(self):
        import numpy as np
        from automation.physical_source_inventory import valid_patch_footprint
        image=np.zeros((80,80),np.uint8)
        self.assertTrue(valid_patch_footprint(image,[40.,40.]))
        for p in ([10.,40.],[40.,79.],[float('nan'),40.]):self.assertFalse(valid_patch_footprint(image,p))
        self.assertFalse(valid_patch_footprint(image[:,:,None],[40.,40.]))
    def test_custom_recorded_crop_preserves_camera_xy(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        import numpy as np
        from automation import chronological_pre_research as c
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pre=[]
            for i in range(6):
                p=root/f'{i}.npy';np.save(p,np.zeros((128,128),np.uint8));pre.append(dict(path=p,ordinal=i,timestamp=None,sequence=None))
            mask=root/'mask.npy';np.save(mask,np.full((96,96),255,np.uint8));out=root/'out';out.mkdir()
            proposal=dict(camera_x=40.,camera_y=40.,score=1.,v251_region_evidence=1.)
            e=dict(id='FIXTURE:1',sid=1,pre=pre,post=[],mask=mask,crop=(10,20,96,96))
            with patch.object(c.region,'_region_physical_candidates',return_value=([proposal],{})),patch.object(c.region,'_balance_merged_candidates',side_effect=lambda cs,*args,**kw:cs):
                r=c.run_event(e,out,keep_fn=exact_supported_keep)
            self.assertEqual(r['retained'][0]['camera_xy'],[50.,60.]);self.assertIsNone(r['selected'])
            self.assertEqual(r['retained'][0]['observations'],3)
    def test_forbidden_path_denied_before_file_access(self):
        import subprocess,sys
        code="from pathlib import Path; from automation.causal_pre_onset_research import read_guard; read_guard(Path('/tmp'), []); open('/tmp/S03/does_not_exist')"
        r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertIn('S03 prohibited',r.stderr)


if __name__=='__main__':unittest.main()
