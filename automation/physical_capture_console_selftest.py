"""Capture presentation and real pipe/SIGINT tests; never starts camera/game code."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from automation.physical_capture_console import CaptureConsole
from automation.physical_collection import _run_capture, prepare_can_play, play_can


def object_line(sid=1, *, current=False, payload_size=100000):
    body=dict(shot_id=sid, status='UNKNOWN', peak_ts=2.+sid,
        object_id=None, supported_object_ids=['a','b'], reason='multiple_supported_objects',
        per_object=[{'candidates':[{'diagnostic_payload':'Z'*payload_size}]}],
        timing={'audio_to_decision_ms':337.},
        exact_coordinate_result={'state':'missed' if current else 'pending'})
    return ('[OBJECT-HIT-CURRENT] ' if current else '[OBJECT-HIT] ')+json.dumps(body)+'\n'


def fake_child(root, lines, *, compact=True, verbose=False, env=None):
    """Use a real child pipe, but replace main.py with a fixture byte emitter."""
    source=root/'emission.bin';source.write_bytes(lines)
    command=[sys.executable,'-u','-c',
             'import pathlib,sys; sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())',str(source)]
    popen=subprocess.Popen
    out=io.StringIO()
    with patch('automation.physical_collection.subprocess.Popen',side_effect=lambda *a,**kw:popen(command,**kw)),redirect_stdout(out):
        result=_run_capture(root/'play.json',env or dict(os.environ),compact=compact,verbose_console=verbose)
    return result,out.getvalue()


class ConsoleTests(unittest.TestCase):
    def test_large_payload_warning_scan_is_bounded(self):
        view=CaptureConsole('/unused',io.StringIO())
        line=object_line(payload_size=1000000)
        started=time.monotonic()
        with redirect_stdout(io.StringIO()):view.observe(line,0,len(line))
        self.assertLess(time.monotonic()-started,2.)

    def test_default_quiet_full_raw_persisted_and_compact_references(self):
        original=(b'[DETECT] huge debug\n'+object_line().encode()+
                  b'[OBJECT-MATURE-CAPTURE] shot=1 stage=WORKER_DELIVERY candidates=33 copy_enqueue_ms=3.1\n'+
                  object_line(current=True).encode())
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);code,console=fake_child(root,original)
            self.assertEqual(code,0);self.assertEqual((root/'runtime.log').read_bytes(),original)
            self.assertNotIn('diagnostic_payload',console);self.assertNotIn('[DETECT]',console)
            shots=[l for l in console.splitlines() if l.startswith('[SHOT ')]
            self.assertEqual(len(shots),1);self.assertLessEqual(len(shots[0]),160)
            records=[json.loads(l) for l in (root/'logs/event_summary.jsonl').read_text().splitlines()]
            self.assertEqual([r['shot_id'] for r in records],[1,1,1])
            self.assertEqual(records[-1]['exact_state'],'missed')
            self.assertLess((root/'logs/event_summary.jsonl').stat().st_size,4096)
            for r in records:
                ref=r['raw'];raw=original[ref['byte_offset']:ref['byte_offset']+ref['byte_length']]
                self.assertTrue(raw.startswith(('['+r['source']+']').encode()))
            status=json.loads((root/'runtime_exit.json').read_text())
            self.assertTrue(status['log_drain_complete']);self.assertEqual(status['logging_cost']['bytes'],len(original))

    def test_verbose_and_non_play_keep_detailed_echo(self):
        raw=object_line(payload_size=100).encode()
        for compact,verbose in ((True,True),(False,False)):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);code,console=fake_child(root,raw,compact=compact,verbose=verbose)
                self.assertEqual(code,0);self.assertIn(raw.decode(),console)
                self.assertEqual((root/'runtime.log').read_bytes(),raw)

    def test_multiple_out_of_order_updates_do_not_reassign_shot_ids(self):
        raw=(object_line(1)+object_line(2)+object_line(1,current=True)+object_line(2,current=True)).encode()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,console=fake_child(root,raw)
            records=[json.loads(l) for l in (root/'logs/event_summary.jsonl').read_text().splitlines()]
            self.assertEqual([r['shot_id'] for r in records],[1,2,1,2])
            self.assertEqual(sum(l.startswith('[SHOT ') for l in console.splitlines()),2)

    def test_warnings_errors_and_counter_losses_visible_without_giant_json(self):
        raw=('WARNING: camera startup degraded\n'
             '[OBJECT-CROPS] unavailable: ValueError: invalid dimensions\n'
             '[OBJECT-MATURE-CAPTURE] unavailable: OSError: disk full\n'
             "[PHYSICAL-TRACE] SHUTDOWN {'writer_errors': 2, 'queue_drops': 1}\n"
             'Traceback (most recent call last):\n'
             'ValueError: fixture failure\n'+object_line()).encode()
        with tempfile.TemporaryDirectory() as d:
            _,console=fake_child(Path(d),raw)
            for phrase in ('startup degraded','invalid dimensions','disk full','writer_errors','Traceback','fixture failure'):
                self.assertIn(phrase,console)
            self.assertNotIn('diagnostic_payload',console)

    def test_interleaved_and_unkeyed_text_kept_no_guessed_shot(self):
        raw=(b'[V2.22.3 SHOT] shot=7 dispatch=1ms objects=7\n'+
             b'[OBJECT-HIT] {"shot_id":7,"status":"UNKNOWN",[CV] interleaved\n'+
             b'not utf8: \xff\n[OBJECT-HIT] {"shot_id":null}\n')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,console=fake_child(root,raw)
            self.assertEqual((root/'runtime.log').read_bytes(),raw)
            self.assertIn('partially parsed',console);self.assertIn('not associated by guess',console)
            records=[json.loads(l) for l in (root/'logs/event_summary.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['shot_id'] in (7,None) for r in records))

    def test_audio_without_object_decision_still_announced_at_exit(self):
        with tempfile.TemporaryDirectory() as d:
            _,console=fake_child(Path(d),b'[V2.22.3 SHOT] shot=4 dispatch=3ms objects=0\n')
            self.assertIn('[SHOT 4] OBJECT_HIT=UNAVAILABLE',console)
            self.assertIn('no completed OBJECT_HIT log',console)

    def test_missing_incomplete_and_limited_artifacts_are_visible(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'object_crops/shots/shot_00000002';folder.mkdir(parents=True)
            (folder/'trace.json').write_text(json.dumps({'completeness':{'trace_complete':False,'queue_drops':1}}))
            (folder/'mature_candidates').mkdir();(folder/'mature_candidates/LIMIT.json').write_text('{}')
            env=dict(os.environ,SKJUTBANA_OBJECT_CROP_ROOT=str(root/'object_crops'),SKJUTBANA_OBJECT_MATURE_CAPTURE='1')
            _,console=fake_child(root,(object_line(1)+object_line(2)).encode(),env=env)
            for text in ('missing/unreadable trace.json','trace incomplete','mature candidate artifacts unavailable','export limit reached'):
                self.assertIn(text,console)

    def test_summary_write_failure_visible_and_never_successful_capture(self):
        with tempfile.TemporaryDirectory() as d:
            err=io.StringIO()
            with patch.object(CaptureConsole,'_append',side_effect=OSError('fixture ENOSPC')),redirect_stderr(err):
                code,console=fake_child(Path(d),object_line().encode())
            status=json.loads((Path(d)/'runtime_exit.json').read_text())
            self.assertNotEqual(code,0);self.assertFalse(status['log_drain_complete'])
            self.assertIn('Log drain/write failed',err.getvalue())

    def test_truncated_mature_pool_visible_even_if_trace_persisted_completely(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'object_crops/shots/shot_00000001';folder.mkdir(parents=True)
            (folder/'trace.json').write_text(json.dumps({'completeness':{'trace_complete':True}}))
            (folder/'mature_candidates').mkdir()
            (folder/'mature_candidates/pool_0000.json').write_text(json.dumps(
                dict(shot_id=1,complete=False,unavailable_reason='pool_byte_limit')))
            env=dict(os.environ,SKJUTBANA_OBJECT_CROP_ROOT=str(root/'object_crops'),SKJUTBANA_OBJECT_MATURE_CAPTURE='1')
            _,console=fake_child(root,object_line().encode(),env=env)
            self.assertIn('pool_0000.json incomplete: pool_byte_limit',console)

    def test_launcher_flag_only_changes_presentation_and_manifest(self):
        settings=Path('content/ai/settings.json').read_bytes()
        with tempfile.TemporaryDirectory(dir='/data/skjutbana') as d:
            root=Path(d)
            a,env_a=prepare_can_play(root/'a');b,env_b=prepare_can_play(root/'b',verbose_console=True)
            self.assertEqual(env_a,env_b)
            self.assertEqual(json.loads(a.read_text())['console_mode'],'operator')
            self.assertEqual(json.loads(b.read_text())['console_mode'],'verbose')
            with patch('automation.physical_collection._run_capture',return_value=0) as run,redirect_stdout(io.StringIO()):
                play_can(root/'c',verbose_console=True)
                self.assertTrue(run.call_args.kwargs['verbose_console'])
        self.assertEqual(settings,Path('content/ai/settings.json').read_bytes())

    def test_reconciliation_reads_identical_raw_capture_after_quiet_routing(self):
        from automation.object_hit_replay_selftest import capture
        from automation.object_candidate_reconciliation import gameplay
        from automation.object_candidate_reconciliation_selftest import packet
        with tempfile.TemporaryDirectory(dir='/data/skjutbana') as d:
            source=Path(d)/'source';trace=capture(source)
            folder=source/'object_crops/shots/shot_00000001/mature_candidates';folder.mkdir()
            p=packet();p['available_at']=trace['object_query_diagnostic']['decision_ts']-.001
            p['frame_ts']=p['candidates'][0]['timestamp']=2.065
            artifact=folder/'pool_0000.json';artifact.write_text(json.dumps(p));before=artifact.read_bytes()
            expected=gameplay(source,{})
            self.assertEqual(expected['rows'][0]['mature_at_cutoff']['state'],'OBJECT_HIT')
            raw=(source/'runtime.log').read_bytes();(source/'runtime.log').unlink()  # generated fixture only
            _,console=fake_child(source,raw)
            self.assertEqual(gameplay(source,{}),expected)
            self.assertEqual((source/'runtime.log').read_bytes(),raw)
            self.assertEqual(artifact.read_bytes(),before)

    def test_sigint_drains_child_shutdown_and_preserves_completed_updates(self):
        with tempfile.TemporaryDirectory(dir='/data/skjutbana') as d:
            root=Path(d)
            child=root/'child.py';child.write_text(
                'import signal,time,sys\n'
                'def stop(*args):\n'
                ' print('+repr(object_line(1,current=True).rstrip())+',flush=True)\n'
                ' print("WARNING: shutdown fixture completed",flush=True)\n'
                ' sys.exit(0)\n'
                'signal.signal(signal.SIGINT,stop)\n'
                'print('+repr(object_line().rstrip())+',flush=True)\n'
                'while True: time.sleep(.1)\n')
            harness=root/'runner.py';harness.write_text(
                'import pathlib,subprocess,sys,os\nfrom unittest.mock import patch\n'
                'from automation.physical_collection import _run_capture\n'
                'popen=subprocess.Popen\n'
                'with patch("automation.physical_collection.subprocess.Popen",side_effect=lambda *a,**kw:popen([sys.executable,"-u",sys.argv[2]],**kw)):\n'
                ' sys.exit(_run_capture(pathlib.Path(sys.argv[1])/"play.json",dict(os.environ),compact=True))\n')
            process=subprocess.Popen([sys.executable,str(harness),str(root),str(child)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONPATH=str(Path.cwd())),start_new_session=True)
            try:
                summary=root/'logs/event_summary.jsonl';deadline=time.monotonic()+8
                while not summary.exists() or not summary.stat().st_size:
                    if process.poll() is not None or time.monotonic()>deadline:self.fail('fixture did not complete initial log')
                    time.sleep(.02)
                completed=summary.read_bytes();process.send_signal(signal.SIGINT)
                console=process.communicate(timeout=8)[0].decode()
                self.assertEqual(process.returncode,0,console)
                self.assertTrue(summary.read_bytes().startswith(completed))
                self.assertIn('shutdown fixture completed',(root/'runtime.log').read_text())
                records=[json.loads(l) for l in summary.read_text().splitlines()]
                self.assertTrue(any(r['source']=='OBJECT-HIT-CURRENT' and r['exact_state']=='missed' for r in records))
                state=json.loads((root/'runtime_exit.json').read_text())
                self.assertTrue(state['interrupted']);self.assertTrue(state['log_drain_complete'])
                self.assertEqual(sum(l.startswith('[SHOT ') for l in console.splitlines()),1)
            finally:
                if process.poll() is None:process.kill();process.wait()
                process.stdout.close()


if __name__=='__main__':unittest.main()
