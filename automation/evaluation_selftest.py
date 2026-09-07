"""Measurement tests; no detector imports or hardware required."""
import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.engine.offline.evaluation import STAGES, encoded, evaluate, provenance


def shot():
    p = {"camera_x": 0, "camera_y": 0}
    return {"session_id": "s", "shot_id": "1", "coordinate_space": "camera",
            "ground_truth": p, **{s: [p.copy()] for s in STAGES}}


def score(rows):
    return evaluate(rows, mode="offline_candidate")


class EvaluationTests(unittest.TestCase):
    def test_tolerance_boundaries(self):
        for radius in (5, 10, 20, 42):
            s = shot()
            s["emitted"] = [{"camera_x": radius * .6, "camera_y": radius * .8}]
            self.assertEqual(score([s])["tolerances_camera_px"][str(radius)]["stages"]["emitted"]["correct"], 1)
            s["emitted"][0]["camera_x"] += .001
            self.assertEqual(score([s])["tolerances_camera_px"][str(radius)]["stages"]["emitted"]["correct"], 0)

    def test_all_first_losses(self):
        from src.engine.offline.evaluation import LOSSES
        for stage, loss in zip(STAGES, LOSSES):
            s = shot()
            s[stage] = []
            self.assertEqual(score([s])["tolerances_camera_px"]["10"]["first_loss_counts"], {loss: 1})

    def test_missing_is_not_empty(self):
        s = shot()
        s["raw"] = None
        s["filtered"] = []
        report = score([s])["tolerances_camera_px"]["10"]
        self.assertEqual(report["first_loss_counts"], {"not_evaluated": 1})
        self.assertIsNone(report["stages"]["raw"]["accuracy_percent"])
        self.assertEqual(report["stages"]["filtered"]["accuracy_percent"], 0)
        s["ground_truth"] = None
        self.assertEqual(score([s])["labelled_shots"], 0)

    def test_emissions_ranking_latency(self):
        s = shot()
        p = s["ground_truth"]
        far = {"camera_x": 100, "camera_y": 100}
        s.update(emitted=[p, p, far], ranked=[far, p], latency_ms=25, rescue_used=True)
        r = score([s])
        t = r["tolerances_camera_px"]["10"]
        self.assertEqual(t["false_emissions"]["count"], 1)
        self.assertEqual(t["duplicates"]["count"], 1)
        self.assertEqual(t["stages"]["top_1"]["correct"], 0)
        self.assertEqual(t["stages"]["top_3"]["correct"], 1)
        self.assertEqual(r["latency_ms"]["p95"], 25)
        self.assertEqual(r["rescue"], {"used": 1, "evaluated": 1})

    def test_deterministic_and_no_mutation(self):
        rows = [shot()]
        before = copy.deepcopy(rows)
        self.assertEqual(encoded(score(rows)), encoded(score(rows)))
        self.assertEqual(rows, before)

    def test_invalid_input_rejected(self):
        for change in ({"coordinate_space": "screen"}, {"latency_ms": -1},
                       {"raw": [{"camera_x": float("nan"), "camera_y": 0}]}, {"rescue_used": "false"}):
            s = shot()
            s.update(change)
            with self.assertRaises(ValueError):
                score([s])
        with self.assertRaises(ValueError):
            score([shot(), shot()])

    def test_coverage_and_empty_emission(self):
        a, b = shot(), shot()
        b["shot_id"] = "2"
        b["emitted"] = None
        a["emitted"] = []
        t = score([a, b])["tolerances_camera_px"]["10"]
        self.assertEqual(t["stages"]["emitted"], {
            "correct": 0, "evaluated": 1, "unavailable": 1, "accuracy_percent": 0})
        self.assertEqual(t["false_emissions"], {"count": 0, "evaluated_shots": 1})

    def test_framepack_adapter_and_cli_preserves_output(self):
        import numpy as np
        from automation.evaluate_pipeline import framepack_observations
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "captures" / "session"
            session.mkdir(parents=True)
            meta = session / "shot_000001.json"
            meta.write_text(json.dumps({"session_id": "s", "shot_id": "1",
                "gt_camera_xy": [0, 0], "source_kind": "projected_test",
                "current_candidates": [{"camera_x": 0, "camera_y": 0}]}))
            np.savez(meta.with_suffix(".npz"), pre_gray=np.zeros((4, 4), dtype=np.uint8),
                     post_frames=np.zeros((1, 4, 4), dtype=np.uint8), post_timestamps=[1.0])
            shots, paths = framepack_observations(root / "captures")
            self.assertEqual(len(paths), 2)
            t = score(shots)["tolerances_camera_px"]["10"]["stages"]
            self.assertEqual(t["saved_pool"]["correct"], 1)
            self.assertIsNone(t["raw"]["accuracy_percent"])
            self.assertIsNone(t["top_1"]["accuracy_percent"])
            out = root / "run"
            command = ["python3", "-m", "automation.evaluate_pipeline", "--framepacks",
                       str(root / "captures"), "--dataset-id", "fixture", "--output", str(out)]
            subprocess.run(command, check=True, capture_output=True)
            original = (out / "report.json").read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual((out / "report.json").read_bytes(), original)
            meta.with_suffix(".npz").unlink()
            with self.assertRaises(ValueError):
                framepack_observations(root / "captures")

    def test_latency_without_emission_rejected(self):
        s = shot()
        s.update(emitted=[], latency_ms=25)
        with self.assertRaises(ValueError):
            score([s])
        s["latency_ms"] = None
        self.assertEqual(score([s])["latency_ms"]["evaluated"], 0)

    def test_source_hash_independent_of_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                subprocess.run(["git", "-C", directory, *args], check=True, capture_output=True)
            git("init")
            (root / "a.py").write_text("a")
            git("add", "a.py")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
            (root / "z.py").write_text("z")
            before = provenance(root, [], "test", {})
            git("add", "z.py")
            after = provenance(root, [], "test", {})
            self.assertEqual(before["source_manifest"], after["source_manifest"])
            self.assertEqual(before["source_sha256"], after["source_sha256"])

    def test_provenance_hashes_dirty_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                subprocess.run(["git", "-C", directory, *args], check=True, capture_output=True)
            git("init")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-m", "fixture")
            runtime = {"effective_detector_settings": {"example": 1}, "models": [], "calibration": None}
            clean = provenance(root, [], "test", runtime)
            self.assertFalse(clean["working_tree_dirty"])
            p = root / "fixture.py"
            p.write_text("a")
            first = provenance(root, [p], "test", runtime)
            self.assertTrue(first["working_tree_dirty"])
            self.assertEqual(first["producer_runtime"], runtime)
            p.write_text("b")
            second = provenance(root, [p], "test", runtime)
            self.assertNotEqual(first["dataset_sha256"], second["dataset_sha256"])
            self.assertNotEqual(first["source_sha256"], second["source_sha256"])
            json.loads(encoded(second))


if __name__ == "__main__":
    unittest.main()
