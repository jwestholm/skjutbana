"""Deterministic camera-coordinate and physically interpretable residual tests."""
import unittest
import cv2,numpy as np
from src.engine.offline.registered_impact import prepare_pair,extract


class Tests(unittest.TestCase):
    def base(self):
        yy,xx=np.mgrid[:128,:160]
        return (140+30*np.sin(xx/13)+20*np.cos(yy/17)).astype(np.uint8)
    def dot(self,pre,xy=(80,64),delta=25,radius=3):
        post=pre.copy();mask=np.zeros_like(pre);cv2.circle(mask,xy,radius,1,-1)
        post[mask>0]=np.maximum(post[mask>0].astype(float)-delta,0).astype(np.uint8);return post
    def features(self,pre,post,**kw):return extract(prepare_pair(pre,post,**kw),(80,64))
    def test_perfect_alignment_and_unchanged(self):
        pre=self.base();p=prepare_pair(pre,pre)
        self.assertAlmostEqual(p.registration['dx'],0,places=5)
        self.assertEqual(extract(p,(80,64))['registered_center_abs'],0)
    def test_nonzero_crop_and_camera_coordinates(self):
        pre=self.base();post=self.dot(pre)
        full=extract(prepare_pair(pre,post,translation=(0,0)),(80,64))
        cropped=extract(prepare_pair(pre[10:110,20:140],post[10:110,20:140],origin=(20,10),translation=(0,0)),(80,64))
        self.assertAlmostEqual(full['ring_affine_center_dark'],cropped['ring_affine_center_dark'])
    def test_known_translation(self):
        pre=self.base();post=cv2.warpAffine(pre,np.float32([[1,0,2],[0,1,-1]]),(160,128),borderMode=cv2.BORDER_REPLICATE)
        f=self.features(pre,post,translation=(2,-1));self.assertEqual(f['registered_center_abs'],0)
        self.assertGreater(f['raw_center_abs'],1)
    def test_subpixel_translation(self):
        pre=self.base().astype(np.float32);post=cv2.warpAffine(pre,np.float32([[1,0,.75],[0,1,-.5]]),(160,128),borderMode=cv2.BORDER_REPLICATE)
        f=self.features(pre,post,translation=(.75,-.5))
        self.assertLess(f['registered_center_abs'],f['raw_center_abs']/2)
    def test_global_brightness(self):
        pre=self.base();post=pre+8;f=self.features(pre,post)
        self.assertAlmostEqual(f['global_median_center_abs'],0,places=4)
        self.assertAlmostEqual(f['ring_affine_center_abs'],0,places=4)
    def test_local_brightness(self):
        pre=self.base();post=pre.copy();post[35:95,50:110]+=8
        f=self.features(pre,post,translation=(0,0))
        self.assertGreater(f['raw_center_abs'],7)
        self.assertLess(f['ring_median_center_abs'],.01)
    def test_affine_exposure(self):
        pre=self.base();post=pre.astype(np.float32)*1.05+3;f=self.features(pre,post,translation=(0,0))
        self.assertLess(f['ring_affine_center_abs'],1e-4)
    def test_new_dark_spot(self):
        pre=self.base();f=self.features(pre,self.dot(pre),translation=(0,0))
        self.assertGreater(f['ring_affine_center_dark'],10)
        self.assertGreater(f['ring_affine_component_area'],10)
    def test_new_spot_near_old_spot(self):
        pre=self.dot(self.base(),xy=(72,64));post=self.dot(pre);f=self.features(pre,post,translation=(0,0))
        self.assertGreater(f['ring_affine_center_dark'],10)
    def test_printed_line(self):
        pre=self.base();cv2.line(pre,(80,0),(80,127),20,2)
        self.assertEqual(self.features(pre,pre)['ring_affine_center_abs'],0)
        self.assertGreater(self.features(pre,self.dot(pre),translation=(0,0))['ring_affine_center_dark'],5)
    def test_low_contrast_new_spot(self):
        pre=self.base();f=self.features(pre,self.dot(pre,delta=4),translation=(0,0))
        self.assertGreater(f['ring_affine_center_dark'],2)
    def test_bright_tear_around_dark_center(self):
        pre=self.base();post=self.dot(pre);mask=np.zeros_like(pre)
        cv2.circle(mask,(80,64),4,1,1);post[mask>0]+=12
        f=self.features(pre,post,translation=(0,0))
        self.assertGreater(f['ring_affine_center_dark'],5)
        self.assertGreater(f['ring_affine_center_bright'],1)

if __name__=='__main__':unittest.main()
