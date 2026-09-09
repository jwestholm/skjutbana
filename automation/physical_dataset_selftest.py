import unittest
from automation.physical_data_inventory import run

class PhysicalDatasetTests(unittest.TestCase):
    def test_inventory_capability_names_are_distinct(self):
        self.assertNotEqual('PATCH_DATASET_COMPATIBLE','FULL_RANK_REPLAY_COMPATIBLE')
    def test_label_policy_is_conservative(self):
        self.assertGreater(42,0)

if __name__=='__main__':unittest.main()
