import unittest
import numpy as np
from src.engine.offline.common_verifier import CommonFrameContext,extract_patch_context,patch_representation,CommonVerifier

class Disabled(CommonVerifier):
    def score_track(self,track,context):return {'score':1}

class CommonVerifierTests(unittest.TestCase):
    def setUp(self):
        self.pre=np.full((40,40),100,np.uint8);self.post=self.pre.copy();self.post[18:23,18:23]=50
        self.ctx=CommonFrameContext(self.pre,(self.post,self.post),1.5,(1.2,1.8))
    def test_cutoff_and_patch_shapes(self):
        b=extract_patch_context(self.ctx,(20,20));self.assertEqual(len(b['pairs']),1);self.assertEqual(b['pairs'][0]['signed'].shape,(33,33))
    def test_representations_deterministic(self):
        b=extract_patch_context(self.ctx,(20,20));
        for k in ('difference','gradient','pca','texture'):self.assertTrue(np.array_equal(patch_representation(b,k),patch_representation(b,k)))
    def test_disabled_verifier_is_empty(self):
        self.assertEqual(Disabled().verify_event([],self.ctx),[])

if __name__=='__main__':unittest.main()
