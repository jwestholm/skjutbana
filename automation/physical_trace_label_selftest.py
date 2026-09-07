"""Headless tests for physical trace click-labeling helpers."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from automation.physical_trace_label import (
    AnnotationDataError,
    display_to_camera,
    discover_shots,
    fit_display,
    load_frame,
    load_existing_annotation,
    save_annotation,
    save_skip,
)
from src.engine.physical_trace import TRACE_SCHEMA


class LabelTests(unittest.TestCase):
    def make_session(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        for sid in (1, 2):
            shot = root / "shots" / f"shot_{sid:08d}"
            (shot / "frames").mkdir(parents=True)
            np.save(shot / "frames/post_000000.npy", np.arange(24, dtype=np.uint8).reshape(4, 6))
            np.save(shot / "frames/post_000001.npy", np.full((4, 6), sid, dtype=np.uint8))
            trace = {"schema_version": TRACE_SCHEMA, "shot_id": sid, "frames": [
                {"kind": "post", "timestamp": 2.0, "path": "frames/post_000000.npy"},
                {"kind": "post", "timestamp": 3.0, "path": "frames/post_000001.npy"}],
                "stages": [{"candidates": [{"camera_x": 1, "camera_y": 1}]}]}
            (shot / "trace.json").write_text(json.dumps(trace))
        return temp, root

    def test_scaling_maps_exact_original_coordinates(self):
        scale, size, origin = fit_display((100, 200), (100, 100))
        self.assertEqual((scale, size, origin), ((0.5, 0.5), (100, 50), (0, 25)))
        self.assertEqual(display_to_camera((50, 50), origin, scale, (100, 200)), (100.0, 50.0))
        self.assertIsNone(display_to_camera((150, 50), origin, scale, (100, 200)))
        uneven, uneven_size, uneven_origin = fit_display((3, 7), (100, 100))
        self.assertEqual(uneven_size, (100, 43))
        mapped = display_to_camera((uneven_origin[0] + 70, uneven_origin[1] + 28), uneven_origin, uneven, (3, 7))
        self.assertAlmostEqual(mapped[0], 4.9, places=6)
        self.assertAlmostEqual(mapped[1], 1.953488, places=5)

    def test_existing_ground_truth_and_relabel_are_separate_from_predictions(self):
        temp, root = self.make_session()
        with temp:
            shot = root / "shots/shot_00000001"
            save_annotation(shot, 1, (4.25, 2.5))
            annotation = load_existing_annotation(shot)
            self.assertEqual((annotation["camera_x"], annotation["camera_y"]), (4.25, 2.5))
            self.assertEqual(json.loads((shot / "trace.json").read_text())["stages"][0]["candidates"][0], {"camera_x": 1, "camera_y": 1})
            save_annotation(shot, 1, (5.5, 3.5))
            self.assertEqual(load_existing_annotation(shot)["camera_x"], 5.5)

    def test_skip_resume_and_multiple_post_frame_selection(self):
        temp, root = self.make_session()
        with temp:
            shots = discover_shots(root)
            self.assertEqual([s["trace"]["shot_id"] for s in shots], [1, 2])
            image, entry, index = load_frame(shots[0], -1)
            self.assertEqual(index, 1)
            self.assertEqual(entry["timestamp"], 3.0)
            self.assertEqual(int(image[0, 0]), 1)
            save_skip(shots[0]["shot_dir"], 1)
            self.assertEqual([s["trace"]["shot_id"] for s in discover_shots(root)], [2])
            self.assertEqual([s["trace"]["shot_id"] for s in discover_shots(root, include_skipped=True)], [1, 2])

    def test_missing_or_malformed_image_is_clean_error(self):
        temp, root = self.make_session()
        with temp:
            path = root / "shots/shot_00000001/frames/post_000001.npy"
            path.unlink()
            shot = discover_shots(root)[0]
            with self.assertRaises(AnnotationDataError):
                load_frame(shot, -1)
            (shot["shot_dir"] / "trace.json").write_text("not-json")
            with self.assertRaises(AnnotationDataError):
                discover_shots(root)


if __name__ == "__main__":
    unittest.main()
