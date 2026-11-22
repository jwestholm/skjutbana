"""Render-preserving material safety, not a physical accuracy test."""
import unittest
import numpy as np
from PIL import Image
from automation.light_room_palette import protect_render,hit_render,camera_level,EXPORT_FLOOR

class RenderSafetyTests(unittest.TestCase):
    def test_floor_alpha_input_and_highlights(self):
        rng=np.random.default_rng(41)
        a=rng.integers(0,256,(25,32,4),dtype=np.uint8);original=a.copy()
        protected=np.array(protect_render(Image.fromarray(a)))
        self.assertTrue(np.array_equal(a,original))
        self.assertTrue(np.array_equal(protected[:,:,3],a[:,:,3]))
        self.assertGreaterEqual(float(camera_level(protected[:,:,:3]).min()),EXPORT_FLOOR)
        already=camera_level(a[:,:,:3])>=EXPORT_FLOOR
        self.assertTrue(np.array_equal(protected[:,:,:3][already],a[:,:,:3][already]))
        self.assertTrue(np.all(protected[:,:,:3]>=a[:,:,:3]))
    def test_hit_material_keeps_alpha_and_brightness(self):
        a=np.full((20,20,4),255,np.uint8);a[:,:,:3]=(30,220,175);a[2:6,2:6,3]=0
        hit=np.asarray(hit_render(Image.fromarray(a),'vase'))
        self.assertTrue(np.array_equal(hit[:,:,3],a[:,:,3]))
        self.assertGreaterEqual(float(camera_level(hit[:,:,:3]).min()),EXPORT_FLOOR)
        self.assertGreater(float(np.abs(hit[:,:,:3].astype(float)-a[:,:,:3]).mean()),20)
    def test_broad_shading_is_retained_without_added_texture(self):
        ramp=np.linspace(40,250,100).astype(np.uint8)
        a=np.stack((ramp,ramp,ramp,np.full(100,255,np.uint8)),axis=1)[None,:,:]
        b=np.array(protect_render(Image.fromarray(a)))
        self.assertTrue(np.all(np.diff(b[0,:,:3].astype(int),axis=0)>=0))
        self.assertGreater(float(camera_level(b[0,:,:3]).max()-camera_level(b[0,:,:3]).min()),60)
        self.assertEqual(len(np.unique(b[0,:,0])),len(np.unique(b[0,:,1])))

if __name__=='__main__':unittest.main()
