"""Destructive-boundary tests using disposable sessions, never physical data."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from automation.physical_label_reset import label_files, reset_labels, resolve_session


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


class ResetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'session_test'
        for event in (1, 2):
            shot = self.root / 'shots' / f'shot_{event:08d}'
            (shot / 'frames').mkdir(parents=True)
            (shot / 'trace.json').write_text(json.dumps({
                'schema_version': 'physical-shot-trace-1', 'shot_id': event,
                'completeness': {'trace_complete': True}, 'outcome': {'status': 'matched'},
                'decision_input': {'retained_candidates': [1]}, 'selectors': {'CURRENT': 1}}))
            for name in ('pre.npy', 'post.npy', 'candidate_map.npy'):
                (shot / 'frames' / name).write_bytes(b'evidence\x00\xff')
            (shot / 'ground_truth.json').write_text('{"human": true}\n')
            (shot / 'ground_truth_status.json').write_text('{"status": "unresolved"}\n')
            (shot / 'ground_truth_extra.json').write_text('unknown file; preserve')
        (self.root / 'physical_assignments.json').write_text('{"2": {"state": "NO_PHYSICAL_SHOT"}}')
        (self.root / 'runtime.log').write_bytes(b'runtime evidence')
        (self.root / 'test_setup.json').write_text('{"provenance": true}')
        (self.root / 'finalized.json').write_text('{"historical_report": true}')
        self.binding = self.base / 'binding.json'
        self.meta = {'trace_root': str(self.root), 'collection_plan_id': 'plan',
                     'planned_session_id': 'S02', 'session_class': 'DEVELOPMENT'}
        self.write_binding()
        self.backups = self.base / 'backups'
        self.before = hashes(self.root)
        self.closed = patch('automation.physical_label_reset._ensure_application_closed')
        self.closed.start()
        self.addCleanup(self.closed.stop)

    def write_binding(self):
        self.binding.write_text(json.dumps(self.meta))
        (self.root / 'collection_binding.json').write_text(json.dumps(self.meta))

    def reset(self, **kwargs):
        return reset_labels(binding=self.binding, backup_root=self.backups, **kwargs)

    def test_default_preview_changes_nothing(self):
        self.assertEqual(self.reset()['files'], 5)
        self.assertEqual(hashes(self.root), self.before)
        self.assertFalse(self.backups.exists())

    def test_apply_removes_only_allowlist_and_backs_up_exact_bytes(self):
        allowed = {str(p.relative_to(self.root)) for p in label_files(self.root)}
        result = self.reset(apply=True)
        self.assertEqual(result['status'], 'RESET_COMPLETE')
        self.assertEqual(hashes(self.root), {k: v for k, v in self.before.items() if k not in allowed})
        backup = Path(result['backup'])
        for name in allowed:
            self.assertEqual(hashlib.sha256((backup / name).read_bytes()).hexdigest(), self.before[name])
        self.assertEqual(self.reset(apply=True)['status'], 'ALREADY_UNLABELED')

    def test_symlinked_storage_root_supported(self):
        link = self.base / 'physical_traces_link'
        link.symlink_to(self.root, target_is_directory=True)
        self.assertEqual(resolve_session(root=link)[0], self.root)

    def test_wrong_binding_session_and_plan_refused(self):
        with self.assertRaisesRegex(ValueError, 'requested session'):
            self.reset(session='S01', apply=True)
        self.meta['collection_plan_id'] = 'wrong'
        self.binding.write_text(json.dumps(self.meta))
        with self.assertRaisesRegex(ValueError, 'Binding mismatch'):
            self.reset(apply=True)
        self.assertEqual(hashes(self.root), self.before)

    def test_untouched_validation_refused(self):
        self.meta.update(planned_session_id='S03', session_class='VALIDATION_UNTOUCHED')
        self.write_binding()
        before = hashes(self.root)
        with self.assertRaisesRegex(ValueError, 'validation'):
            self.reset(apply=True)
        self.assertEqual(hashes(self.root), before)

    def test_symlink_label_refused_before_any_removal(self):
        label = self.root / 'shots/shot_00000002/ground_truth.json'
        label.unlink()
        label.symlink_to(self.root / 'runtime.log')
        before = hashes(self.root)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.reset(apply=True)
        self.assertEqual(hashes(self.root), before)
        self.assertFalse(self.backups.exists())

    def test_symlink_shot_directory_refused(self):
        (self.root / 'shots/shot_00000003').symlink_to(self.root / 'shots/shot_00000001', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Unexpected shot'):
            self.reset(apply=True)

    def test_incomplete_or_invalid_identity_refused(self):
        trace = self.root / 'shots/shot_00000002/trace.json'
        data = json.loads(trace.read_text())
        data['completeness']['trace_complete'] = False
        trace.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.reset(apply=True)
        data['completeness']['trace_complete'] = True
        data['shot_id'] = 1
        trace.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.reset(apply=True)
        self.assertEqual(len(list(self.root.glob('shots/*/ground_truth.json'))), 2)

    def test_backup_inside_session_refused(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            reset_labels(root=self.root, apply=True, backup_root=self.root / 'backup')
        self.assertEqual(hashes(self.root), self.before)

    def test_backup_failure_preserves_labels(self):
        with patch('automation.physical_label_reset.os.fsync', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.reset(apply=True)
        self.assertEqual(hashes(self.root), self.before)

    def test_concurrent_label_change_refused(self):
        calls = []
        def inspect(root):
            calls.append(1)
            if len(calls) == 2:
                (root / 'shots/shot_00000001/ground_truth.json').write_text('new label')
            return label_files(root)
        with patch('automation.physical_label_reset.label_files', side_effect=inspect):
            with self.assertRaisesRegex(ValueError, 'changed during backup'):
                self.reset(apply=True)
        self.assertEqual(len(label_files(self.root)), 5)

    def test_running_application_refused(self):
        self.closed.stop()
        with patch('automation.physical_label_reset.socket.create_connection'):
            with self.assertRaisesRegex(ValueError, 'Close the application'):
                self.reset(apply=True)
        self.assertEqual(hashes(self.root), self.before)

    def test_real_cli_binding_preview_and_refusal(self):
        command = [sys.executable, '-m', 'automation.physical_collection', 'reset-labels',
                   '--binding', str(self.binding), '--session', 'S02']
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PREVIEW ONLY', result.stdout)
        self.assertEqual(hashes(self.root), self.before)
        result = subprocess.run(command[:-1] + ['S01', '--apply'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('RESET REFUSED', result.stderr)
        self.assertEqual(hashes(self.root), self.before)

    def test_bound_labeler_ignores_stale_active_session(self):
        from automation import physical_test
        stale = self.base / 'stale_active.json'
        stale.write_text('invalid stale lifecycle metadata')
        with patch.object(physical_test, 'ACTIVE', stale), \
             patch.object(sys, 'argv', ['physical_test', 'label', '--binding', str(self.binding)]), \
             patch.object(physical_test.subprocess, 'run') as run:
            physical_test.main()
        self.assertEqual(run.call_args.args[0], [sys.executable, '-m', 'automation.physical_trace_label', '--root', str(self.root)])
        self.assertEqual(hashes(self.root), self.before)

    def test_bound_start_cannot_override_prepared_collection(self):
        result = subprocess.run([sys.executable, '-m', 'automation.physical_test', 'start',
                                 '--binding', str(self.binding), '--prepare-only'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('already prepared', result.stderr)
        self.assertEqual(hashes(self.root), self.before)


if __name__ == '__main__':
    unittest.main()
