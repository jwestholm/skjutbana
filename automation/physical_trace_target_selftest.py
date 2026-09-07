"""Headless tests for physical evidence target-space helpers."""
import unittest

import numpy as np

from automation.physical_trace_target import (
    TargetEvidenceError,
    camera_to_target,
    enhanced_difference,
    homography_from_calibration,
    register_pre_post,
    target_evidence_view,
    target_to_camera,
    warp_to_target,
)


class TargetTests(unittest.TestCase):
    def test_forward_inverse_round_trip(self):
        h = np.array([[2.0, 0.1, 10.0], [0.05, 1.5, 20.0], [0.001, 0.002, 1.0]])
        point = (123.4, 56.7)
        target = camera_to_target(point, h)
        self.assertAlmostEqual(target_to_camera(target, h)[0], point[0], places=6)
        self.assertAlmostEqual(target_to_camera(target, h)[1], point[1], places=6)

    def test_warp_dimensions_and_missing_calibration(self):
        h = np.eye(3)
        image = np.arange(24, dtype=np.uint8).reshape(4, 6)
        warped = warp_to_target(image, h, (12, 8))
        self.assertEqual(warped.shape, (8, 12))
        self.assertIsNone(homography_from_calibration(None))
        self.assertIsNone(homography_from_calibration({"homography": [[1, 2], [3, 4]]}))

    def test_registration_and_enhancement_show_change(self):
        pre = np.full((64, 64), 180, dtype=np.uint8)
        post = pre.copy()
        post[30:34, 31:35] = 20
        shift, _ = register_pre_post(pre, post)
        self.assertTrue(np.isfinite(shift).all())
        evidence = enhanced_difference(pre, post, register=False)
        self.assertGreater(int(evidence.max()), 0)
        base, overlay = target_evidence_view(pre, post, np.eye(3), (64, 64))
        self.assertEqual(base.shape, (64, 64, 3))
        self.assertEqual(overlay.shape, (64, 64, 3))
        self.assertGreater(int(np.abs(overlay.astype(int) - base.astype(int)).max()), 0)

    def test_dimension_mismatch_is_clean(self):
        with self.assertRaises(TargetEvidenceError):
            enhanced_difference(np.zeros((4, 4), np.uint8), np.zeros((3, 4), np.uint8))


if __name__ == "__main__":
    unittest.main()
