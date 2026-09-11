import unittest
from automation.physical_capture_plan import build,DEFAULT_CATEGORIES

class CaptureToolsTests(unittest.TestCase):
    def test_manifest_balances_categories(self):
        m=build(3,10);self.assertEqual(len(m['rows']),30);self.assertEqual(m['rows'][0]['gt_status'],'UNLABELED')
        self.assertTrue(set(r['category'] for r in m['rows']).issubset(set(DEFAULT_CATEGORIES)))
    def test_session_ids_are_separate(self):
        self.assertEqual(len(set(r['session'] for r in build(3,2)['rows'])),3)

if __name__=='__main__':unittest.main()
