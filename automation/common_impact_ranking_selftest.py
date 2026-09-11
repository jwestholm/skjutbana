import unittest
from automation.common_impact_ranking import methods

class CommonImpactTests(unittest.TestCase):
    def test_methods_are_source_independent(self):
        x={'track':{'best_score':9.,'source':'FAST'},'image_features':{'ring_affine_compact':2.,'ring_affine_dark_contrast':1.,'ring_affine_center_dark':3.,'ring_affine_concentration':.4}}
        y=dict(x); y['track']=dict(x['track'],source='V1')
        self.assertEqual(methods(x),methods(y))
    def test_missing_feature_is_rankable_as_missing(self):
        x={'track':{'best_score':1.},'image_features':{}}
        self.assertEqual(methods(x)['raw'],1.)

if __name__=='__main__': unittest.main()
