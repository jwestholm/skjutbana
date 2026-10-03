"""Synthetic, camera-free tests for fail-open physical trace capture."""
import json
import queue
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from automation.physical_trace_export import export
from src.engine.physical_trace import PhysicalTraceRecorder, TRACE_SCHEMA, validate_capture_history


class TraceTests(unittest.TestCase):
    def test_six_completed_windows_checkpoint_without_shutdown_or_atexit(self):
        # Reproduce the observed failure shape: all detector events and all raw
        # writes complete, then the process exits without running finalizers.
        script = '''
import os, sys, signal, resource
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from automation.physical_trace_selftest import TraceTests
from src.engine.physical_trace import PhysicalTraceRecorder
r=PhysicalTraceRecorder(sys.argv[1],enabled=True,capture_history=dict(pre_seconds=4.,post_seconds=5.,pre_max_frames=240,post_max_frames=300,max_events=16))
s=TraceTests().scanner();pixels=np.zeros((4,5),np.uint8)
frame=lambda t:SimpleNamespace(timestamp=t,gray=pixels)
for sid in range(1,7):
    peak=10.*sid
    s.frame_history=[frame(peak-5),frame(peak-1)]
    e=SimpleNamespace(shot_id=sid,peak_ts=peak,state='matched',emitted=True)
    s.audio_events=[e];r.observe_scanner(s)
    r._queue.join()
    s.audio_events=[];s.frame_history=[frame(peak+1),frame(peak+5),frame(peak+6)]
    r.observe_scanner(s);r._queue.join()
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
os.kill(os.getpid(),signal.SIGSEGV)
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(['python3','-c',script,directory],capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,-11,result.stderr)
            paths=sorted(Path(directory).glob('shots/*/trace.json'))
            self.assertEqual(len(paths),6)
            for sid,path in enumerate(paths,1):
                trace=json.loads(path.read_text())
                self.assertEqual(trace['shot_id'],sid)
                self.assertEqual(trace['outcome']['status'],'matched')
                self.assertTrue(trace['completeness']['trace_complete'])
                self.assertEqual(trace['capture_history']['coverage']['post']['saved_frames'],2)

    def test_metadata_failure_is_visible_and_does_not_skip_other_shots(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);recorder=PhysicalTraceRecorder(root,enabled=True)
            self._shot(recorder,self.scanner(),1,'matched',emitted=True)
            self._shot(recorder,self.scanner(),2,'missed')
            original=recorder._finalize_trace
            def fail_one(sid,active,*,timed_out):
                if sid==1:raise ValueError('fixture metadata failure')
                return original(sid,active,timed_out=timed_out)
            with patch.object(recorder,'_finalize_trace',side_effect=fail_one):
                self.assertFalse(recorder.flush())
            self.assertFalse((root/'shots/shot_00000001/trace.json').exists())
            error=json.loads((root/'shots/shot_00000001/trace_error.json').read_text())
            self.assertEqual(error['message'],'fixture metadata failure')
            self.assertTrue((root/'shots/shot_00000002/trace.json').exists())
            self.assertEqual(recorder.stats()['metadata_failures'],1)

    def test_missing_metadata_health_preserves_and_reports_available_artifacts(self):
        from automation.physical_test import health
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shot=root/'shots/shot_00000001'
            (shot/'frames').mkdir(parents=True);(shot/'maps').mkdir()
            for kind in ('pre_snapshot','pre_history','post'):
                np.save(shot/'frames'/f'{kind}_000000.npy',np.zeros((2,3),np.uint8))
            np.save(shot/'maps/000000_change_map.npy',np.ones((2,3),np.uint8))
            (shot/'ground_truth.json').write_text('{"shot_id":1,"camera_x":2,"camera_y":3}')
            before={p:p.read_bytes() for p in shot.rglob('*') if p.is_file()}
            result=health(root);evidence=result['shots'][0]['evidence']
            self.assertFalse(result['healthy']);self.assertTrue(result['shots'][0]['labeled'])
            self.assertEqual(evidence['saved_frames'],dict(pre_snapshot=1,pre_history=1,post=1))
            self.assertEqual(evidence['evidence_maps'],1);self.assertEqual(evidence['problems'],[])
            self.assertIn('candidate_ledger',evidence['unavailable'])
            self.assertEqual(before,{p:p.read_bytes() for p in shot.rglob('*') if p.is_file()})

    def test_extended_history_bound_coverage_and_terminal_retention(self):
        import copy
        from unittest.mock import patch
        from src.engine.camera.camera_manager import camera_manager
        config=dict(pre_seconds=4.,post_seconds=5.,pre_max_frames=3,post_max_frames=2,max_events=2)
        with tempfile.TemporaryDirectory() as directory:
            recorder=PhysicalTraceRecorder(directory,enabled=True,capture_history=config)
            scanner=self.scanner();pixels=np.arange(20,dtype=np.uint8).reshape(4,5)
            frames=lambda times:[SimpleNamespace(timestamp=t,gray=pixels) for t in times]
            scanner.frame_history=frames([5.,6.,7.,8.,9.,10.])
            event=SimpleNamespace(shot_id=1,peak_ts=10.,state='pending',emitted=False)
            scanner.audio_events=[event];before=copy.deepcopy(scanner.last_candidates)
            with patch.object(camera_manager,'provenance_for_timestamp',side_effect=lambda t:dict(stream_id='fixture',sequence=int(t))):
                recorder.start_from_scanner(scanner,event)
                scanner.frame_history=frames([10.,11.,12.]);recorder.observe_scanner(scanner)
                event.state='missed';recorder.observe_scanner(scanner)
                terminal=copy.deepcopy(recorder._active[1]['trace']['outcome'])
                stages=len(recorder._active[1]['trace']['stages'])
                scanner.audio_events=[];scanner.frame_history=frames([12.,14.,15.,16.])
                recorder.observe_scanner(scanner);recorder.observe_scanner(scanner)
                self.assertTrue(recorder.flush())
            trace=recorder._active[1]['trace'];coverage=trace['capture_history']['coverage']
            self.assertEqual(trace['outcome'],terminal)
            self.assertEqual(len(trace['stages']),stages)
            self.assertEqual(scanner.last_candidates,before);self.assertFalse(event.emitted)
            np.testing.assert_array_equal(pixels,np.arange(20,dtype=np.uint8).reshape(4,5))
            self.assertEqual([f['timestamp'] for f in trace['frames'] if f['kind']=='pre_history'],[8.,9.,10.])
            self.assertEqual([f['timestamp'] for f in trace['frames'] if f['kind']=='post'],[11.,12.])
            self.assertEqual(coverage['pre_history']['available_frames'],5)
            self.assertEqual(coverage['pre_history']['saved_frames'],3)
            self.assertEqual(coverage['pre_history']['frame_limit_drops'],2)
            self.assertEqual(coverage['post']['available_frames'],4)
            self.assertEqual(coverage['post']['saved_frames'],2)
            self.assertEqual(coverage['post']['frame_limit_drops'],2)
            self.assertEqual(coverage['post']['camera_sequence_gaps'],1)
            self.assertTrue(coverage['post']['right_boundary_observed'])
            self.assertTrue(trace['capture_history']['post_window_closed'])
            self.assertLessEqual(len(recorder._active[1]['frame_ts']),2)

    def test_extended_short_history_early_shutdown_and_writer_loss_are_explicit(self):
        from unittest.mock import patch
        config=dict(pre_seconds=4.,post_seconds=5.,pre_max_frames=240,post_max_frames=300,max_events=1)
        with tempfile.TemporaryDirectory() as directory:
            recorder=PhysicalTraceRecorder(directory,enabled=True,capture_history=config)
            scanner=self.scanner();event=SimpleNamespace(shot_id=1,peak_ts=2.,state='missed',emitted=False)
            scanner.audio_events=[event]
            with patch.object(recorder,'_enqueue',return_value=False):
                recorder.start_from_scanner(scanner,event)
            recorder.observe_scanner(scanner)
            recorder.start_from_scanner(scanner,SimpleNamespace(shot_id=2,peak_ts=3.,state='pending'))
            self.assertEqual(len(recorder._active),1)
            self.assertTrue(recorder.flush())
            coverage=recorder._active[1]['trace']['capture_history']['coverage']
            self.assertFalse(coverage['pre_history']['left_boundary_observed'])
            self.assertEqual(coverage['pre_history']['persistence_losses'],1)
            self.assertEqual(coverage['pre_history']['unknown_acquisition_frames'],1)
            self.assertFalse(coverage['post']['right_boundary_observed'])
            self.assertFalse(recorder._active[1]['trace']['completeness']['trace_complete'])

    def test_extended_configuration_validation_and_process_override(self):
        from unittest.mock import patch
        config=dict(pre_seconds=4.,post_seconds=5.,pre_max_frames=240,post_max_frames=300,max_events=16)
        for key,value in [('pre_seconds',float('nan')),('post_seconds',11),('pre_max_frames',361),('max_events',0),('post_max_frames',3.5)]:
            with self.assertRaises(ValueError):validate_capture_history({**config,key:value})
        with tempfile.TemporaryDirectory() as directory:
            recorder=PhysicalTraceRecorder(directory)
            settings={'physical_trace_capture_enabled':False,'physical_trace_root':'stale','mode':'advisory'}
            original=dict(settings)
            with patch.dict('os.environ',SKJUTBANA_PHYSICAL_CAPTURE_BINDING='fixture',SKJUTBANA_PHYSICAL_TRACE_ROOT=directory,
                            SKJUTBANA_PHYSICAL_CAPTURE_HISTORY=json.dumps(config)):
                recorder.configure(settings)
                self.assertEqual(recorder.capture_history,config)
                self.assertEqual(recorder._provenance(settings)['capture_history_requested'],config)
                self.assertEqual(settings,original)
            self.assertEqual(recorder._queue.maxsize,128)
            self.assertEqual(recorder._max_queued_bytes,1024**3)

    def test_extended_post_survives_retirement_without_future_or_duplicate_frames(self):
        config=dict(pre_seconds=2.,post_seconds=3.,pre_max_frames=60,post_max_frames=90,max_events=2)
        with tempfile.TemporaryDirectory() as directory:
            recorder=PhysicalTraceRecorder(directory,enabled=True,capture_history=config)
            scanner=self.scanner();pixels=np.zeros((4,5),np.uint8)
            frame=lambda t:SimpleNamespace(timestamp=t,gray=pixels)
            scanner.frame_history=[frame(0.),frame(1.),frame(2.)]
            event=SimpleNamespace(shot_id=1,peak_ts=2.,state='missed',emitted=False)
            scanner.audio_events=[event];recorder.observe_scanner(scanner)
            self.assertTrue(recorder.flush())
            self.assertFalse(recorder._active[1]['trace']['completeness']['trace_complete'])
            scanner.audio_events=[]
            scanner.frame_history=[frame(4.),frame(3.),frame(4.),frame(5.),frame(6.)]
            recorder.observe_scanner(scanner);recorder.observe_scanner(scanner)
            self.assertTrue(recorder.flush())
            trace=recorder._active[1]['trace'];coverage=trace['capture_history']['coverage']
            self.assertEqual([f['timestamp'] for f in trace['frames'] if f['kind']=='post'],[3.,4.,5.])
            self.assertEqual([f['timestamp'] for f in coverage['post']['available_identities']],[3.,4.,5.])
            self.assertTrue(trace['completeness']['trace_complete'])
            self.assertEqual(trace['outcome']['status'],'missed')
            self.assertFalse(trace['outcome']['emitted'])

    def test_legacy_pre_post_frame_limits_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder=PhysicalTraceRecorder(directory,enabled=True)
            scanner=self.scanner();pixels=np.zeros((4,5),np.uint8)
            scanner.frame_history=[SimpleNamespace(timestamp=float(t),gray=pixels) for t in range(50)]
            event=SimpleNamespace(shot_id=1,peak_ts=50.,state='missed',emitted=False)
            scanner.audio_events=[event];recorder.start_from_scanner(scanner,event)
            scanner.frame_history=[SimpleNamespace(timestamp=float(t),gray=pixels) for t in range(51,121)]
            recorder.observe_scanner(scanner);self.assertTrue(recorder.flush())
            trace=recorder._active[1]['trace']
            self.assertEqual(sum(f['kind']=='pre_history' for f in trace['frames']),24)
            self.assertEqual(sum(f['kind']=='post' for f in trace['frames']),64)
            self.assertNotIn('capture_history',trace)

    def test_extended_native_history_exceeds_legacy_limits_without_resampling(self):
        config=dict(pre_seconds=4.,post_seconds=5.,pre_max_frames=240,post_max_frames=300,max_events=16)
        with tempfile.TemporaryDirectory() as directory:
            recorder=PhysicalTraceRecorder(directory,enabled=True,capture_history=config)
            scanner=self.scanner();pixels=np.zeros((4,5),np.uint8)
            scanner.frame_history=[SimpleNamespace(timestamp=i/20,gray=pixels) for i in range(101)]
            event=SimpleNamespace(shot_id=1,peak_ts=5.,state='pending',emitted=False)
            recorder.start_from_scanner(scanner,event);self.assertTrue(recorder.flush())
            scanner.audio_events=[event]
            scanner.frame_history=[SimpleNamespace(timestamp=i/20,gray=pixels) for i in range(101,201)]
            event.state='missed';recorder.observe_scanner(scanner);self.assertTrue(recorder.flush())
            trace=recorder._active[1]['trace']
            self.assertEqual([f['timestamp'] for f in trace['frames'] if f['kind']=='pre_history'],[i/20 for i in range(20,101)])
            self.assertEqual([f['timestamp'] for f in trace['frames'] if f['kind']=='post'],[i/20 for i in range(101,201)])
            self.assertTrue(trace['completeness']['trace_complete'])

    def test_object_crops_separate_from_global_frames_with_hash_and_bbox(self):
        import hashlib
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);recorder=PhysicalTraceRecorder(root,enabled=True)
            scanner=self.scanner();event=SimpleNamespace(shot_id=1,peak_ts=2.,state='pending')
            recorder.start_from_scanner(scanner,event)
            pixels=np.full((10,12),80,np.uint8)
            recorder.capture_object_crop(1,'pre',1.94,(100,50,112,60),pixels)
            recorder.capture_object_crop(1,'post',2.03,(100,50,112,60),pixels)
            self.assertTrue(recorder.flush())
            active=recorder._active[1]['trace'];crops=active['object_evidence_crops']
            self.assertEqual(len(crops),2)
            self.assertFalse(any(f['artifact']=='object_crop' for f in active['frames']))
            for crop in crops:
                self.assertEqual(crop['camera_bbox'],[100,50,112,60])
                self.assertEqual(crop['sha256'],hashlib.sha256(pixels.tobytes()).hexdigest())
                np.testing.assert_array_equal(np.load(root/'shots/shot_00000001'/crop['path']),pixels)
            with patch.dict('os.environ',SKJUTBANA_PHYSICAL_CAPTURE_BINDING='fixture',SKJUTBANA_PHYSICAL_TRACE_ROOT=str(root)):
                recorder.configure({'physical_trace_capture_enabled':False,'physical_trace_root':'unread-stale-root'})
                self.assertTrue(recorder.enabled);self.assertEqual(recorder.root,root)
                recorder.start_from_scanner(scanner,SimpleNamespace(shot_id=2,peak_ts=3.,state='pending'),
                    {'physical_trace_session_id':'stale_saved_session'})
                self.assertEqual(recorder._active[2]['trace']['session_id'],root.name)
                self.assertEqual(recorder._active[2]['trace']['provenance']['capture_binding'],'fixture')
            self.assertTrue(recorder.flush())

    def test_restart_refuses_existing_shot_and_preserves_all_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = PhysicalTraceRecorder(root, enabled=True)
            self._shot(first, self.scanner(), 1, "matched", emitted=True)
            self.assertTrue(first.flush())
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            second = PhysicalTraceRecorder(root, enabled=True)
            scanner = self.scanner()
            event = SimpleNamespace(shot_id=1, peak_ts=50., state="pending")
            second.start_from_scanner(scanner, event)
            scanner.audio_events = [event]
            second.observe_scanner(scanner)
            second.capture_decision(scanner, SimpleNamespace(), event)
            self.assertTrue(second.flush())
            with self.assertRaises(ValueError):
                second.attach_ground_truth(1, (99, 99))
            self.assertEqual(second.stats()['refused_existing_shots'], 1)
            self.assertEqual(second.stats()['persisted_items'], 0)
            second.start_from_scanner(scanner, SimpleNamespace(shot_id=99, peak_ts=60., state='pending'))
            self.assertEqual(second.stats()['refused_capture_roots'], 1)
            self.assertEqual(before, {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()})
            self.assertEqual(scanner.last_candidates, self.scanner().last_candidates)

    def test_empty_claimed_directory_is_also_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'shots/shot_00000001').mkdir(parents=True)
            recorder = PhysicalTraceRecorder(root, enabled=True)
            recorder.start_from_scanner(self.scanner(), SimpleNamespace(shot_id=1, peak_ts=2.))
            self.assertEqual(recorder.stats()['refused_existing_shots'], 1)
            self.assertFalse(list((root / 'shots/shot_00000001').iterdir()))

    def scanner(self):
        pre = np.zeros((4, 5), dtype=np.uint8)
        post = np.ones((4, 5), dtype=np.uint8)
        return SimpleNamespace(
            frame_history=[SimpleNamespace(timestamp=1.0, gray=pre)], pre_shot_snapshot=pre,
            pre_shot_snapshot_ts=0.9, audio_events=[],
            last_candidates=[{"camera_x": 2.0, "camera_y": 3.0, "score": 4.0}],
            last_stable_tracks=[], last_threshold_value=5.0, last_change_threshold_value=3.0,
            last_vote_threshold_value=0.0, last_window_debug={"raw_blobs_total": 1},
            debug_frames={"change_map": post, "camera_gray": post},
            last_best_candidate={"camera_x": 2.0, "camera_y": 3.0},
            last_trace_pipeline={"rejected_blobs": [{"reason": "area"}], "quota_limited": False},
        )

    def _shot(self, recorder, scanner, shot_id, state, emitted=False, rescue=False):
        event = SimpleNamespace(shot_id=shot_id, peak_ts=2.0, state="pending",
                                emitted=False, confidence=0.0, note="")
        recorder.start_from_scanner(scanner, event, {"physical_trace_capture_enabled": True,
                                                       "physical_trace_session_id": "synthetic"})
        scanner.frame_history = [SimpleNamespace(timestamp=1.0, gray=np.zeros((4, 5), dtype=np.uint8)),
                                 SimpleNamespace(timestamp=2.1, gray=np.ones((4, 5), dtype=np.uint8)),
                                 SimpleNamespace(timestamp=2.2, gray=np.full((4, 5), 2, dtype=np.uint8))]
        scanner.audio_events = [event]
        scanner.last_window_debug = {"rescue_used": rescue}
        recorder.observe_scanner(scanner, settings={"physical_trace_capture_enabled": True})
        event.state, event.emitted = state, emitted
        event.note = "timeout" if state == "missed" else "emitted"
        recorder.observe_scanner(scanner, settings={"physical_trace_capture_enabled": True})
        recorder.attach_ground_truth(shot_id, (2, 3))

    def test_synthetic_end_to_end_and_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder, scanner = PhysicalTraceRecorder(root, enabled=True), self.scanner()
            self._shot(recorder, scanner, 1, "matched", emitted=True)
            self._shot(recorder, scanner, 2, "missed")
            self._shot(recorder, scanner, 3, "matched", emitted=True, rescue=True)
            self.assertTrue(recorder.flush())
            trace = json.loads((root / "shots/shot_00000001/trace.json").read_text())
            self.assertEqual(trace["schema_version"], TRACE_SCHEMA)
            self.assertEqual([f["kind"] for f in trace["frames"]], ["pre_snapshot", "pre_history", "post", "post"])
            self.assertEqual(trace["stages"][0]["pipeline"]["rejected_blobs"][0]["reason"], "area")
            self.assertEqual(trace["outcome"]["status"], "matched")
            timeout_trace = json.loads((root / "shots/shot_00000002/trace.json").read_text())
            rescue_trace = json.loads((root / "shots/shot_00000003/trace.json").read_text())
            self.assertTrue(timeout_trace["outcome"]["timeout"])
            self.assertTrue(rescue_trace["outcome"]["rescue_used"])
            gt = json.loads((root / "shots/shot_00000001/ground_truth.json").read_text())
            self.assertEqual(gt["camera_x"], 2)
            output = root / "evaluation_trace.json"
            export(root, output)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["mode"], "live_path_replay")
            self.assertEqual(payload["shots"][0]["ground_truth"]["camera_x"], 2)
            report_dir = root / "report"
            subprocess.run(["python3", "-m", "automation.evaluate_pipeline", "--trace", str(output),
                            "--dataset-id", "synthetic-physical-trace", "--output", str(report_dir)],
                           check=True, capture_output=True, text=True)
            report = json.loads((report_dir / "report.json").read_text())
            self.assertEqual(report["total_shots"], 3)
            self.assertEqual(report["mode"], "live_path_replay")
            self.assertEqual(report["rescue"]["used"], 1)
            self.assertEqual(report["tolerances_camera_px"]["20"]["stages"]["raw"]["unavailable"], 3)

    def test_disabled_enabled_fail_open_and_ordering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scanner, event = self.scanner(), SimpleNamespace(shot_id=9, peak_ts=1.0, state="pending")
            disabled = PhysicalTraceRecorder(root / "disabled", enabled=False)
            before = list(scanner.last_candidates)
            disabled.start_from_scanner(scanner, event)
            disabled.observe_scanner(scanner, event)
            self.assertEqual(before, scanner.last_candidates)
            self.assertFalse(list((root / "disabled").rglob("*")))
            enabled = PhysicalTraceRecorder(root / "enabled", enabled=True)
            enabled.start_from_scanner(scanner, event, {"physical_trace_capture_enabled": True})
            scanner.audio_events = [event]
            enabled.observe_scanner(scanner, settings={"physical_trace_capture_enabled": True})
            self.assertEqual(before, scanner.last_candidates)
            self.assertTrue(enabled.flush())
            frames = json.loads((root / "enabled/shots/shot_00000009/trace.json").read_text())["frames"]
            self.assertEqual([f["timestamp"] for f in frames], [0.9, 1.0])

    def test_missing_data_rejection_corruption_and_queue_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = PhysicalTraceRecorder(root, enabled=True)
            recorder._queue = queue.Queue(maxsize=1)
            job = {"kind": "json", "path": root / "a.json", "value": {"x": 1}, "meta": {}, "shot_id": None}
            self.assertTrue(recorder._enqueue(job))
            self.assertFalse(recorder._enqueue({**job, "path": root / "b.json"}))
            self.assertGreaterEqual(recorder._error_count, 1)
            writer_error = PhysicalTraceRecorder(root / "writer", enabled=True)
            (root / "writer").mkdir(parents=True, exist_ok=True)
            bad_parent = root / "writer" / "not-a-directory"
            bad_parent.write_text("x")
            self.assertTrue(writer_error._enqueue({"kind": "array", "path": bad_parent / "frame.npy", "value": np.ones((2, 2)), "meta": {}, "shot_id": None}))
            self.assertTrue(writer_error.flush())
            self.assertGreaterEqual(writer_error._error_count, 1)
            (root / "shots/shot_00000001").mkdir(parents=True)
            (root / "shots/shot_00000001/trace.json").write_text("{broken")
            with self.assertRaises(ValueError):
                export(root, root / "bad-export.json")

    def test_provenance_and_performance_sanity(self):
        with tempfile.TemporaryDirectory() as directory:
            scanner, event = self.scanner(), SimpleNamespace(shot_id=4, peak_ts=1.0, state="pending")
            off = PhysicalTraceRecorder(Path(directory) / "off", enabled=False)
            start = time.perf_counter()
            for _ in range(200):
                off.observe_scanner(scanner, event)
            disabled_ms = (time.perf_counter() - start) * 1000
            on = PhysicalTraceRecorder(Path(directory) / "on", enabled=True)
            on.start_from_scanner(scanner, event, {"physical_trace_capture_enabled": True})
            start = time.perf_counter()
            for _ in range(200):
                on.observe_scanner(scanner, event, {"physical_trace_capture_enabled": True})
            enabled_ms = (time.perf_counter() - start) * 1000
            self.assertLess(disabled_ms, 1000)
            self.assertLess(enabled_ms, 5000)
            self.assertIn("git_commit", on._provenance({}))
            self.assertIn("git_branch", on._provenance({}))
            self.assertIn("python_version", on._provenance({}))
            print(f"PERF_SANITY disabled_ms={disabled_ms:.2f} enabled_ms={enabled_ms:.2f}")
            self.assertTrue(on.flush())


if __name__ == "__main__":
    unittest.main()
