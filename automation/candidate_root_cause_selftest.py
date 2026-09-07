import json
import tempfile
import unittest
from pathlib import Path

from automation.candidate_root_cause import diagnose


class RootCauseTests(unittest.TestCase):
    def test_missing_temporal_data_is_not_misclassified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "shot_000001.json").write_text(json.dumps({
                "shot_id": "1", "source_kind": "f2_projected", "gt_camera_x": 10, "gt_camera_y": 10,
                "candidates": [{"camera_x": 100, "camera_y": 100}],
            }))
            report = diagnose(root)
            self.assertEqual(report["missing_at_20"], 1)
            self.assertEqual(report["failure_categories"], {"TEMPORAL_DATA_INSUFFICIENT": 1})
            self.assertEqual(report["shots"][0]["evidence"]["pre_post_signal_near_gt"], "UNAVAILABLE")
            self.assertFalse(report["interpretation"]["candidate_generation_testable"])
            self.assertIn("does not mean the live detector has a temporal failure", report["interpretation"]["classification_note"])

    def test_oracle_and_offsets_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "shot_000001.json").write_text(json.dumps({
                "shot_id": "1", "source_kind": "f2_projected", "gt_camera_x": 10, "gt_camera_y": 10,
                "candidates": [{"camera_x": 40, "camera_y": 50}],
            }))
            report = diagnose(root)
            self.assertEqual(report["oracle_coverage"]["5"], 0)
            self.assertEqual(report["offset_px"]["dx_mean"], 30)
            self.assertEqual(report["offset_px"]["dy_mean"], 40)


if __name__ == "__main__":
    unittest.main()
