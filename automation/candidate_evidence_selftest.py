"""Compatibility fixtures for the flat candidate fields consumed by replay exports."""
import os
import unittest
from unittest.mock import patch

import numpy as np

from src.engine import hole_morphology_evidence as morphology
from src.engine.candidate_evidence import CandidateEvidence
from src.engine.winner_selection_flags import attach_h11


class CandidateEvidenceTests(unittest.TestCase):
    def test_contract_represents_zero_missing_and_optional_context(self):
        zero = CandidateEvidence("sample", "v1", value=0.0, window_s=(2.0, 3.0), cost_ms=0.25)
        self.assertTrue(zero.valid)
        self.assertEqual((zero.window_s, zero.cost_ms), ((2.0, 3.0), 0.25))
        missing = CandidateEvidence("sample", "v1", missing_reason="no_data")
        self.assertFalse(missing.valid)
        self.assertEqual((missing.value, missing.window_s, missing.cost_ms), (None, None, None))
        for value, reason in ((None, None), (0.0, "no_data")):
            with self.assertRaises(ValueError):
                CandidateEvidence("sample", "v1", value, reason)

    def test_patch_measured_row_and_batch_cost(self):
        frame = np.zeros((65, 65), np.uint8)
        row = {"camera_x": 32, "camera_y": 32, "score": 7.0,
               "ev_hole_morph_patch_missing": "no_pre"}
        with patch.object(morphology.time, "perf_counter", side_effect=[1.0, 1.125]):
            info = morphology.measure(frame, frame, [row])
        self.assertEqual(info, {"ms": 125.0, "measured": 1.0})
        self.assertEqual(list(row.items()), [
            ("camera_x", 32), ("camera_y", 32), ("score", 7.0),
            ("ev_hole_morph_patch_version", "hole_morph_mf_v2"),
            ("ev_hole_morph_patch_snr", 0.0), ("ev_hole_morph_patch_valid", 1.0),
        ])

    def test_patch_missing_reasons_preserve_prior_value(self):
        frame = np.zeros((65, 65), np.uint8)
        for pre, cur, xy, reason in (
            (None, frame, {}, "no_pre"),
            (frame, frame[:5], {}, "shape"),
            (frame, frame, {}, "no_xy"),
            (frame, frame, {"camera_x": 30, "camera_y": 32}, "border"),
        ):
            with self.subTest(reason=reason):
                row = dict(xy, ev_hole_morph_patch_snr=12.0)
                self.assertEqual(morphology.measure(pre, cur, [row])["measured"], 0.0)
                self.assertEqual(list(row.items()), list(xy.items()) + [
                    ("ev_hole_morph_patch_snr", 12.0),
                    ("ev_hole_morph_patch_version", "hole_morph_mf_v2"),
                    ("ev_hole_morph_patch_valid", 0.0),
                    ("ev_hole_morph_patch_missing", reason),
                ])

    def test_map_rows_and_disabled_passthrough(self):
        snr = np.arange(25, dtype=np.float32).reshape(5, 5)
        row = {"camera_x": 12, "camera_y": 22, "score": 1.0, "ev_hole_morph_missing": "old"}
        previous_fields = list(row.items())
        outside = {"camera_x": 80, "camera_y": 80, "ev_hole_morph_snr": 99.0}
        invalid = {"camera_x": "invalid"}
        rows = [row, outside, invalid]
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(attach_h11(rows, None, None, 10, 20, snr=snr))
            self.assertEqual(list(row.items()), previous_fields)
        with patch.dict(os.environ, {"SKJUTBANA_H11_FULL_ROI_MAP": "1"}, clear=True):
            self.assertIsNone(attach_h11(rows, None, None, 10, 20, snr=snr))
        self.assertEqual(list(row.items()), previous_fields + [
            ("ev_hole_morph_snr", 24.0), ("ev_hole_morph_valid", 1.0),
            ("ev_hole_morph_version", "hole_morph_v2map_v1"),
        ])
        self.assertEqual(outside, {"camera_x": 80, "camera_y": 80, "ev_hole_morph_snr": 99.0})
        self.assertEqual(invalid, {"camera_x": "invalid"})


if __name__ == "__main__":
    unittest.main()
