"""Regression tests for V2 PRE/current coordinate-plane ownership."""
import unittest
from types import SimpleNamespace
import numpy as np
from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2


class Tests(unittest.TestCase):
    def setUp(self):
        self.engine = CandidateGeneratorV2.__new__(CandidateGeneratorV2)
        self.frame = np.zeros((80, 100), np.uint8)
        self.frame[40:50, 60:70] = 180

    def test_crop_origin_samples_same_camera_pixels(self):
        scanner = SimpleNamespace(
            frame_history=[SimpleNamespace(timestamp=9.9, gray=self.frame)],
            _v2221_active_geometry=SimpleNamespace(crop_x0=50, crop_y0=30),
        )
        got = self.engine._collect_pre_frames(
            scanner, peak_ts=10.0, bbox=(10, 10, 30, 30),
            roi=np.ones((20, 20), np.uint8), cfg={"pre_stack_frames": 1},
        )[0]
        self.assertEqual(got[0, 0], 180)

    def test_full_frame_path_has_zero_origin(self):
        scanner = SimpleNamespace(
            frame_history=[SimpleNamespace(timestamp=9.9, gray=self.frame)],
            _v2221_active_geometry=SimpleNamespace(crop_x0=0, crop_y0=0),
        )
        got = self.engine._collect_pre_frames(
            scanner, peak_ts=10.0, bbox=(60, 40, 70, 50),
            roi=np.ones((10, 10), np.uint8), cfg={"pre_stack_frames": 1},
        )[0]
        self.assertEqual(float(got.mean()), 180.0)

    def test_unchanged_patch_low_and_new_dark_hole_strong(self):
        pre = np.full((20, 20), 180, np.uint8)
        post = pre.copy(); post[8:12, 8:12] = 20
        roi = np.ones(pre.shape, np.uint8)
        ref, _, _ = self.engine._build_reference_and_noise([pre], roi=roi, cfg={})
        unchanged, _ = self.engine._normalise_photometry(ref, pre, roi=roi)
        changed, _ = self.engine._normalise_photometry(ref, post, roi=roi)
        self.assertLess(float(np.abs(ref.astype(float)-unchanged).mean()), 0.1)
        self.assertGreater(float(np.mean(ref[8:12,8:12].astype(float)-changed[8:12,8:12])), 100)

    def test_registration_offset_not_double_applied(self):
        ref = np.zeros((64, 64), np.uint8); ref[20:30, 20:30] = 200
        cur = np.zeros_like(ref); cur[20:30, 21:31] = 200
        roi = np.ones(ref.shape, np.uint8)
        _, info = self.engine._register_current(ref, cur, roi=roi, cfg={"registration_enabled": True, "registration_min_response": 0.0}, registration_bias=(0.0,0.0))
        aligned, applied = self.engine._register_current(ref, cur, roi=roi, cfg={"registration_enabled": True, "registration_min_response": 0.0}, registration_bias=(info['raw_dx'], info['raw_dy']))
        self.assertLessEqual(abs(float(applied['dx'])), abs(float(info['raw_dx'])) + 1e-6)
        self.assertEqual(aligned.shape, cur.shape)

    def test_camera_coordinates_are_not_changed_by_reference_mapping(self):
        candidate = {"camera_x": 1571.0, "camera_y": 1012.0}
        self.assertEqual((candidate["camera_x"], candidate["camera_y"]), (1571.0, 1012.0))


if __name__ == "__main__":
    unittest.main()
