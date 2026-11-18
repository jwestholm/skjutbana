"""Selftest: calibration saves keep a backup and never silently drop/overwrite a lens model.

  python3 -m automation.calibration_save_selftest
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class CalibrationSaveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.settings = root / "settings.json"
        self.settings.write_text(json.dumps({"camera_calibration": {"homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                                             "other": 1}))
        self.backups = root / "backups"
        self.env = mock.patch.dict(os.environ, {"SKJUTBANA_BACKUP_DIR": str(self.backups)})
        self.env.start()
        import src.engine.settings as st
        self.st = st
        self.path = mock.patch.object(st, "_settings_path", return_value=self.settings)
        self.path.start()

    def tearDown(self):
        self.path.stop(); self.env.stop(); self.tmp.cleanup()

    def test_backup_before_overwrite(self):
        self.st.save_camera_calibration({"homography": [[2, 0, 0], [0, 2, 0], [0, 0, 1]]})
        files = list(self.backups.glob("*_calibration.json"))
        self.assertEqual(len(files), 1)
        self.assertEqual(json.loads(files[0].read_text())["camera_calibration"]["homography"][0][0], 1)
        self.assertEqual(json.loads(self.settings.read_text())["other"], 1)

    def test_no_backup_no_overwrite(self):
        with mock.patch.object(Path, "write_text", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.st.save_camera_calibration({"homography": "new"})
        self.assertEqual(json.loads(self.settings.read_text())["camera_calibration"]["homography"][0][0], 1)

    def test_pending_lens_model_only_for_unchanged_calibration(self):
        from src.engine import lens_model
        cal = self.st.load_camera_calibration()
        fp = lens_model.calibration_fingerprint(cal)
        lens_model._PENDING.clear()
        lens_model._PENDING.append((fp, {"model": "radial2_v1"}))
        with mock.patch("src.engine.input.hit_input.hit_input.reload_calibration") as reload:
            self.assertTrue(lens_model.apply_pending(log=lambda *_: None))
            reload.assert_called_once()
        self.assertEqual(self.st.load_camera_calibration()["camera_distortion"]["model"], "radial2_v1")
        # a model fitted for an older calibration is discarded
        self.st.save_camera_calibration({"homography": "changed"})
        lens_model._PENDING.append((fp, {"model": "radial2_v1"}))
        self.assertFalse(lens_model.apply_pending(log=lambda *_: None))
        self.assertNotIn("camera_distortion", self.st.load_camera_calibration())


if __name__ == "__main__":
    unittest.main(verbosity=1)
