"""Saved-settings trace capture: fresh unique session per process, atomic claim, archived roots read-only,
session byte cap and free-space floor.  Run: PYTHONPATH=. python3 automation/physical_trace_session_guard_selftest.py
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.engine import physical_trace as pt
from src.engine.physical_trace import PhysicalTraceRecorder, is_archived_session


# Synthetic captures use controlled free space; live disk policy is unchanged.
from automation.trace_test_support import setUpModule, tearDownModule


class Event:
    def __init__(self, shot_id):
        self.shot_id = shot_id
        self.peak_ts = 1.0
        self.audio_trigger = None


def start(rec, shot_id):
    with mock.patch("src.engine.camera.aruco_calibrator.geometry_context", return_value={}, create=True), \
         mock.patch.object(rec, "_provenance", return_value={}):
        try:
            rec.start_from_scanner(object(), Event(shot_id), {})
        except Exception:
            pass   # later context steps may need a real scanner; the claim/refusal happens before them


class SessionGuard(unittest.TestCase):
    def setUp(self):
        os.environ.pop("SKJUTBANA_PHYSICAL_CAPTURE_BINDING", None)
        os.environ.pop("SKJUTBANA_PHYSICAL_TRACE_DISABLE", None)
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name) / "physical_traces"
        self.base.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def rec(self, settings):
        r = PhysicalTraceRecorder(self.base, writer_workers=1)
        r.configure(dict(settings, physical_trace_capture_enabled=True))
        self.addCleanup(r.stop_for_exit)
        return r

    def test_old_session_path_is_never_reused(self):
        old = self.base / "session_20260909_194947_D01_60806ff2"
        (old / "shots").mkdir(parents=True)
        r = self.rec(dict(physical_trace_root=str(old)))
        self.assertNotEqual(r.root, old)
        self.assertEqual(r.root.parent, self.base)
        self.assertIn("_auto_", r.root.name)
        r.configure(dict(physical_trace_capture_enabled=True, physical_trace_root=str(old)))
        self.assertEqual(r.root, r._settings_root_cache[1], "resolved once per process")
        start(r, 1)
        self.assertTrue((r.root / "runtime_claim.json").exists())
        self.assertEqual(list((old / "shots").iterdir()), [], "old session untouched")

    def test_deleted_archived_session_is_not_recreated(self):
        old = self.base / "session_X_D01"
        (self.base / "session_X_D01.tar.zst").write_bytes(b"x")
        self.assertTrue(is_archived_session(old))
        r = self.rec(dict(physical_trace_root=str(old)))
        start(r, 1)
        self.assertFalse(old.exists(), "archived session must not be recreated")

    def test_two_instances_get_distinct_sessions(self):
        a = self.rec(dict(physical_trace_root=str(self.base)))
        b = self.rec(dict(physical_trace_root=str(self.base)))
        self.assertNotEqual(a.root, b.root)

    def test_same_session_claim_is_exclusive(self):
        a = self.rec(dict(physical_trace_root=str(self.base)))
        b = PhysicalTraceRecorder(self.base, writer_workers=1)
        b.enabled = True; b.root = a.root; b._claim_required.add(a.root.absolute())
        self.assertIsNone(a._claim_session(a.root))
        self.assertIn("already claimed", b._claim_session(b.root))

    def test_prepared_session_requires_explicit_mode_and_matching_record(self):
        for record, key in (("collection_binding.json", "trace_root"), ("test_setup.json", "root")):
            prep = self.base / f"session_20261201_120000_{record[:4]}_abcd1234"
            prep.mkdir()
            self.assertIn("no matching", self.rec(dict(physical_trace_root=str(prep),
                          physical_trace_session_mode="prepared"))._claim_session(prep) or "", "no record -> refused")
            (prep / record).write_text(json.dumps({key: str(self.base / "elsewhere")}))
            self.assertIn("no matching", self.rec(dict(physical_trace_root=str(prep),
                          physical_trace_session_mode="prepared"))._claim_session(prep) or "", "other root -> refused")
            (prep / record).write_text(json.dumps({key: str(prep)}))
            self.assertNotEqual(self.rec(dict(physical_trace_root=str(prep))).root, prep, "default mode is new")
            r = self.rec(dict(physical_trace_root=str(prep), physical_trace_session_mode="prepared"))
            self.assertEqual(r.root, prep)
            self.assertIsNone(r._claim_session(prep))
            r2 = self.rec(dict(physical_trace_root=str(prep), physical_trace_session_mode="prepared"))
            self.assertIn("already launched", r2._claim_session(r2.root) or "",
                          "explicit 'prepared' fails closed on a launched session")

    def test_enable_without_root_uses_new_session_under_base(self):
        r = self.rec({})
        self.assertEqual(r.root.parent, self.base)
        self.assertIn("_auto_", r.root.name)

    def test_byte_cap_is_reserved_before_writing(self):
        r = self.rec(dict(physical_trace_max_session_gb=0.000012))    # 12 kB; each job reserves 1000+4096
        # Exercise reservation before any writer consumes these placeholder jobs.
        with mock.patch.object(pt, "_disk_guard_log"), mock.patch.object(r._queue, "put_nowait") as put:
            ok = [r._enqueue(dict(kind="array", path=r.root / "shots/x/a.npy", value=b"", nbytes=1000))
                  for _ in range(4)]
        self.assertEqual(ok, [True, True, False, False])
        self.assertEqual(put.call_count, 2)
        self.assertEqual(r._session_bytes, 2 * 5096)
        self.assertTrue(r._budget_stop and "cap" in r._budget_stop)

    def test_metadata_never_refused_and_root_sticky_after_claim(self):
        r = self.rec(dict(physical_trace_max_session_gb=0.000001))
        with mock.patch.object(pt, "_disk_guard_log"):
            self.assertTrue(r._enqueue(dict(kind="json", path=r.root / "m.json", value={}, nbytes=0)))
        first = r.root
        self.assertIsNone(r._claim_session(first))
        r.configure(dict(physical_trace_capture_enabled=True, physical_trace_root=str(self.base / "other")))
        self.assertEqual(r.root, first, "claimed session never moves within a process")

    def test_failed_write_keeps_reservation_for_leftover(self):
        r = self.rec({})
        job = dict(reserved_bytes=10000)
        r._session_bytes = 10000
        r._release_reservation(job, keep=2500)
        self.assertEqual(r._session_bytes, 2500)

    def test_invalid_limits_fall_back_to_defaults(self):
        r = self.rec(dict(physical_trace_max_session_gb="nan", physical_trace_min_free_gb=-3))
        self.assertEqual(r.max_session_bytes, int(pt.DEFAULT_MAX_SESSION_GB * 1e9))
        self.assertEqual(r.min_free_gb, pt.DEFAULT_MIN_FREE_GB)

    def test_writer_fails_closed_when_disk_check_fails(self):
        r = self.rec({})
        with mock.patch.object(pt, "_free_gb", side_effect=OSError("boom")), mock.patch.object(pt, "_disk_guard_log"):
            self.assertFalse(r._writer_disk_ok())
            self.assertIn("free-space", r._budget_stop)

    def test_ground_truth_refused_on_archived_session(self):
        s = self.base / "session_A"
        (s / "shots").mkdir(parents=True)
        (s / "shots" / "_PACKED_SEE_ARCHIVE.txt").write_text("x")
        r = PhysicalTraceRecorder(s, writer_workers=1)
        with self.assertRaises(ValueError):
            r.attach_ground_truth(1, (1.0, 2.0))

    def test_archived_marker_variants(self):
        s = self.base / "session_Y"
        (s / "shots").mkdir(parents=True)
        self.assertFalse(is_archived_session(s))
        (s / "shots" / "_PACKED_SEE_ARCHIVE.txt").write_text("x")
        self.assertTrue(is_archived_session(s))

    def test_budget_cap_and_free_floor(self):
        r = self.rec(dict(physical_trace_root=str(self.base), physical_trace_max_session_gb=0.000001))
        with mock.patch.object(pt, "_disk_guard_log"):
            self.assertIsNone(r._budget_problem())
            r._session_bytes = 2000
            self.assertIn("cap", r._budget_problem())
            self.assertIn("cap", r._budget_problem(), "stays stopped")
        r2 = self.rec(dict(physical_trace_root=str(self.base), physical_trace_min_free_gb=1e9))
        with mock.patch.object(pt, "_disk_guard_log") as log:
            self.assertIn("free space", r2._budget_problem())
            log.assert_called_once()
            self.assertFalse(r2._writer_disk_ok())

    def test_disabled_settings_have_no_side_effects(self):
        r = PhysicalTraceRecorder(self.base, writer_workers=1)
        r.configure(dict(physical_trace_capture_enabled=False, physical_trace_root="stale"))
        self.assertEqual(r.root, Path("stale"))
        self.assertEqual(r._claim_required, set())

    def test_launch_binding_refuses_archived_root(self):
        from automation import physical_collection as pc
        root = (self.base / "session_Z_D02_x").resolve()
        root.mkdir()
        binding = self.base / "binding.json"
        binding.write_text(json.dumps(dict(planned_session_id="D02", capture_configuration="PROCESS_ONLY",
                                           session_class="DEVELOPMENT", trace_root=str(root))))
        with self.assertRaises(Exception) as before:
            pc.verify_capture_launch(binding)
        self.assertNotIn("archived", str(before.exception))
        (root.parent / "session_Z_D02_x.tar.zst").write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "archived"):
            pc.verify_capture_launch(binding)


if __name__ == "__main__":
    unittest.main()
