import unittest
from automation.physical_patch_verifier import FEATURES

class PatchVerifierTests(unittest.TestCase):
    def test_features_are_source_independent(self):
        self.assertNotIn('source',FEATURES)
        self.assertNotIn('best_score',FEATURES)
    def test_feature_set_is_small(self):
        self.assertLessEqual(len(FEATURES),8)

if __name__=='__main__':unittest.main()
