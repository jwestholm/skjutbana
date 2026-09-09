import unittest
import numpy as np
from automation.motion_residual_audit import measure

class MotionResidualTests(unittest.TestCase):
    def test_unchanged_patch(self):
        p=np.full((33,33),150,np.float32);m=measure(p,p,(16,16));self.assertLess(m['raw_energy'],.1)
    def test_new_dark_patch_survives_alignment(self):
        p=np.full((33,33),150,np.float32);q=p.copy();q[14:19,14:19]=80;m=measure(p,q,(16,16));self.assertGreater(m['raw_center'],20)
    def test_illumination_is_measured(self):
        p=np.full((33,33),150,np.float32);q=np.full((33,33),160,np.float32);m=measure(p,q,(16,16));self.assertGreater(m['raw_energy'],0)
    def test_signed_edge_translation_has_residual(self):
        p=np.zeros((33,33),np.float32);p[:,16:]=180;q=np.zeros_like(p);q[:,17:]=180;m=measure(p,q,(16,16));self.assertGreater(m['gradient_energy'],0)

if __name__=='__main__':unittest.main()
