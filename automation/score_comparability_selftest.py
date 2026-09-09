import unittest
from automation.score_comparability_audit import pct

class ScoreComparabilityTests(unittest.TestCase):
    def test_percentile_ties_are_deterministic(self):
        self.assertEqual(pct(2,[1,2,3]),.5)
        self.assertEqual(pct(2,[2,2,3]),1/3)
    def test_empty_source_is_safe(self):
        self.assertEqual(pct(4,[]),0.0)

if __name__=='__main__': unittest.main()
