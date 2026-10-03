"""Demo coordinate-plane regressions; software fixtures, no physical validation."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.engine.camera import demo_object_zone_extension as demo
from src.engine.input.object_hit_v2223 import CameraHitRegionV240


class Tests(unittest.TestCase):
    def fixture(self, scale=1.):
        geometry=SimpleNamespace(frame_height=2160,frame_width=3840,crop_x0=1500,crop_y0=900,
            crop_width=200,crop_height=120,mode='analysis_crop')
        scanner=SimpleNamespace(_v2221_active_geometry=geometry)
        gray=np.zeros((int(120*scale),int(200*scale)),np.uint8)
        region=CameraHitRegionV240('can',1540,930,60,40,
            metadata={'v250_object_snapshot':{'tags':['shoot_stuff']}})
        return scanner,gray,region

    def test_nonzero_crop_and_scale_use_existing_mapping(self):
        for scale in (1.,.5):
            scanner,gray,region=self.fixture(scale)
            work=demo._regions_in_work_space((region,),scanner,gray.shape)[0]
            self.assertEqual((work.x,work.y,work.width,work.height),(40*scale,30*scale,60*scale,40*scale))
            self.assertEqual(region.x,1540)
            self.assertIs(work.metadata,region.metadata)

    def test_cropped_region_reads_correct_physical_patch(self):
        scanner,gray,region=self.fixture()
        values=np.zeros((110,190),np.float32)
        # bbox origin is another (10,10) offset within the working image.
        values[30:50,45:65]=10.
        context=dict(bbox=(10,10,200,120),absdiff=values,darkening=values,zscore=values,
                     valid=np.ones_like(values,bool),frame_ts=2.1)
        self.assertEqual(demo._metrics_for_regions(regions=(region,),context=context,candidates=[]),{})
        work=demo._regions_in_work_space((region,),scanner,gray.shape)
        result=demo._metrics_for_regions(regions=work,context=context,candidates=[])
        self.assertGreater(result['can']['physical_signal'],.5)

    def test_full_camera_identity(self):
        region=CameraHitRegionV240('can',40,30,60,40)
        self.assertEqual(demo._regions_in_work_space((region,),SimpleNamespace(),(120,200))[0],region)

    def test_wrapper_keeps_canonical_result_and_arrays_unchanged(self):
        scanner,gray,region=self.fixture()
        values=np.full((120,200),5.,np.float32)
        context=dict(bbox=(0,0,200,120),absdiff=values,darkening=values,zscore=values,
                     valid=np.ones_like(values,bool),frame_ts=2.1)
        result=SimpleNamespace(telemetry={'shot_id':7,'frame_ts':2.1},candidates=[{'camera_x':70.,'camera_y':50.,'score':99.}])
        original=copy.deepcopy(result)
        class Generator:
            def generate(self,**kwargs):
                self._v24_last_context=context
                return result
        demo.patch_candidate_generator_for_demo_object_zone(Generator)
        registry=demo.DemoObjectZoneRegistry()
        snapshot=SimpleNamespace(camera_regions=(region,),peak_ts=2.)
        with patch.object(demo.object_hit_registry_v2223,'snapshot_for_shot',return_value=snapshot),patch.object(demo,'demo_object_zone_registry',registry):
            returned=Generator().generate(scanner=scanner,gray=gray,frame_ts=2.1,legacy_candidates=[])
        self.assertIs(returned,result)
        self.assertEqual(result.__dict__,original.__dict__)
        np.testing.assert_array_equal(values,np.full((120,200),5.,np.float32))
        self.assertIsNotNone(registry.result_for_shot(7))
        self.assertIsNone(registry.result_for_shot(8))


if __name__=='__main__':unittest.main()
