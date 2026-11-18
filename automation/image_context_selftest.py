import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from automation.bench.image_context import image_context, main
from automation.bench.train import _feature_names


class ImageContextTests(unittest.TestCase):
    def test_context_fields_stay_out_of_training(self):
        fixture = {"sample": 7, "ctx_brightness": 80.0, "ctx_texture": 2.0, "ctx_motion": 0.0}
        self.assertEqual(_feature_names([{"candidate": fixture}]), ["sample"])

    def test_partial_cells_cover_each_pixel_once(self):
        ramp = np.tile(np.arange(5, dtype=np.uint8), (3, 1))
        result = image_context([ramp], rows=2, columns=3)
        coverage = np.zeros_like(ramp)
        self.assertEqual(len(result["regions"]), 6)
        self.assertEqual(result["motion_missing_reason"], "single_frame")
        for cell in result["regions"]:
            left, top, right, bottom = cell["xyxy"]
            coverage[top:bottom, left:right] += 1
            self.assertEqual(cell["ctx_brightness"], ramp[top:bottom, left:right].mean())
            self.assertIsNone(cell["ctx_motion"])
        np.testing.assert_array_equal(coverage, np.ones_like(ramp))
        self.assertEqual(result["regions"][1]["ctx_texture"], 1.0)

    def test_population_variation_and_constant_texture(self):
        first = np.full((3, 5), 20, np.uint8)
        last = np.full_like(first, 40)
        result = image_context([first, last], rows=1, columns=1)
        cell = result["regions"][0]
        self.assertIsNone(result["motion_missing_reason"])
        self.assertEqual(cell["ctx_motion"], 10.0)
        self.assertEqual(cell["ctx_texture"], 0.0)
        self.assertEqual(cell["ctx_brightness"], 40.0)
        self.assertGreaterEqual(result["cost_ms"], 0)

    def test_identical_series_has_measured_zero_variation(self):
        image = np.arange(12, dtype=np.uint8).reshape(3, 4)
        result = image_context([image, image.copy()], rows=2, columns=3)
        self.assertTrue(all(cell["ctx_motion"] == 0 for cell in result["regions"]))

    def test_invalid_shapes_or_grid(self):
        gray = np.zeros((2, 2), np.uint8)
        for frames, rows, cols in (([], 1, 1), ([gray], 0, 1), ([gray], 1, 3),
                                   ([gray, gray[:1]], 1, 1), ([gray.astype(float)], 1, 1)):
            with self.subTest(rows=rows, columns=cols), self.assertRaises(ValueError):
                image_context(frames, rows=rows, columns=cols)

    def test_file_cli(self):
        with TemporaryDirectory() as tmp, redirect_stdout(StringIO()) as output:
            path = Path(tmp) / "frame.png"
            self.assertTrue(cv2.imwrite(str(path), np.full((4, 6), 10, np.uint8)))
            self.assertEqual(main([str(path), "--grid", "1", "1"]), 0)
        self.assertEqual(json.loads(output.getvalue())["regions"][0]["ctx_brightness"], 10.0)


if __name__ == "__main__":
    unittest.main()
