"""Stage 7 characterization of the scanner's projector-response artifact mask.

Synthetic only. This pins the current hard-mask behavior; it does not validate
that low projector response means tape or that an impact happened there.
Run: python3 -m unittest automation.stage07_board_evidence_selftest -v
"""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.engine.camera.hit_scanner import HitScanner


class ArtifactMaskCharacterization(unittest.TestCase):
    def test_low_response_patch_erases_local_change_map(self) -> None:
        scanner = HitScanner.__new__(HitScanner)
        scanner.scene_reference_gray = np.full((40, 40), 120, dtype=np.uint8)
        scanner.surface_reference_gray = np.full((40, 40), 80, dtype=np.uint8)
        # Synthetic low-response patch: endpoint difference is below the
        # current hard threshold of 12. Surrounding surface remains active.
        scanner.scene_reference_gray[10:30, 10:30] = 100
        scanner.surface_reference_gray[10:30, 10:30] = 108
        scanner.debug_frames = {}

        scanner._rebuild_artifact_mask()
        mask = scanner.artifact_suppression_mask
        self.assertIsNotNone(mask)

        change = np.zeros((40, 40), dtype=np.uint8)
        change[17:23, 17:23] = 180
        filtered = change & mask

        self.assertEqual(int(np.count_nonzero(filtered[17:23, 17:23])), 0)
        self.assertEqual(int(np.count_nonzero(mask[17:23, 17:23])), 0)
        self.assertEqual(int(mask[5, 5]), 255)

        # Follow the same hard-mask boundary used by HitScanner before the
        # threshold maps are formed. The interior patch must remain absent
        # from both the combined and change maps, and therefore from candidate
        # pixels after thresholding and closing.
        combined = change.copy()
        combined_delta = change.copy()
        combined &= mask
        combined_delta &= mask
        self.assertEqual(int(np.count_nonzero(combined[17:23, 17:23])), 0)
        self.assertEqual(int(np.count_nonzero(combined_delta[17:23, 17:23])), 0)

        roi_mask = np.full((40, 40), 255, dtype=np.uint8)
        threshold = 12
        merged = ((combined >= threshold) | (combined_delta >= threshold)).astype(np.uint8) * 255
        merged &= roi_mask
        merged = cv2.morphologyEx(
            merged,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        self.assertEqual(int(np.count_nonzero(merged[17:23, 17:23])), 0)


if __name__ == "__main__":
    unittest.main()
