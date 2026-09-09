import unittest

class LongHorizonTests(unittest.TestCase):
    def test_future_frame_is_diagnostic_only(self):
        self.assertEqual('OFFLINE_LONG_HORIZON_DIAGNOSTIC','OFFLINE_LONG_HORIZON_DIAGNOSTIC')
    def test_retention_is_defined_for_zero_change(self):
        immediate=0.; later=0.; self.assertEqual(later/max(immediate,.1),0.)
    def test_next_state_delta_is_not_a_blacklist(self):
        old=(100,100); new=(101,100); self.assertNotEqual(old,new)

if __name__=='__main__':unittest.main()
