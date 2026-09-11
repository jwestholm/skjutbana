"""Production CLI regressions for aggregate finalization; fixtures are not accuracy truth."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from automation.physical_finalize_manifest import build_manifest, read_json


class ManifestTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root = self.base / 'session_D01'
        self.plan = dict(collection_plan_id='development-test', rows=[
            dict(session='D01', session_class='DEVELOPMENT', category=str(i),
                 planned_physical_shot=i, actual_event_id=None) for i in range(1, 7)])
        self.meta = dict(collection_plan_id='development-test', planned_session_id='D01',
                         session_class='DEVELOPMENT', trace_root=str(self.root), rows=self.plan['rows'])
        self.binding = self.base / 'binding.json'
        for i in range(1, 7):
            self.event(i)
        self.save()

    def event(self, sid, physical=True):
        directory = self.root / 'shots' / f'shot_{sid:08d}'
        directory.mkdir(parents=True)
        trace = dict(schema_version='physical-shot-trace-1', shot_id=sid, peak_ts=1., created_at=2.,
                     completeness={'trace_complete': True}, outcome={'status': 'matched'},
                     decision_input={'complete_track_audit': {'complete': True}}, selectors={'CURRENT': {}},
                     frames=[dict(kind=k, timestamp=t, path=k+'.npy', shape=[8, 8], dtype='uint8')
                             for k, t in [('pre_snapshot', .9), ('post', 1.1)]])
        self.write(directory / 'trace.json', trace)
        for f in trace['frames']:
            np.save(directory / f['path'], np.zeros((8, 8), np.uint8))
        if physical:
            self.write(directory / 'ground_truth.json', dict(schema_version='physical-shot-trace-1',
                shot_id=sid, space='camera', camera_x=3., camera_y=4., attached_at=3., quality='precise'))
        return directory

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def save(self):
        self.write(self.base / 'plan.json', self.plan)
        self.write(self.binding, self.meta)
        self.write(self.root / 'collection_binding.json', self.meta)

    def build(self, **kwargs):
        return build_manifest(self.plan, 'D01', self.binding, **kwargs)

    def cli(self, *extra, module='automation.physical_collection'):
        return subprocess.run([sys.executable, '-m', module,
            *(['finalize'] if module.endswith('physical_collection') else []),
            '--plan', str(self.base / 'plan.json'), '--session', 'D01',
            '--binding', str(self.binding), *extra], capture_output=True, text=True)

    def hashes(self):
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.root.rglob('*') if p.is_file()}

    def test_d01_six_saved_labels_finalize_without_manual_json(self):
        before = self.hashes()
        out = self.base / 'finalized.json'
        result = self.cli('--in-capture-order', '--output', str(out))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = read_json(out)
        self.assertEqual(report['physical_shots'], 6)
        self.assertEqual(report['non_physical_events'], 0)
        self.assertEqual([r['event_id'] for r in report['label_manifest']['labels']], list(range(1, 7)))
        self.assertEqual(report['mapping_basis'], 'operator_confirmed_capture_order')
        self.assertEqual(before, self.hashes())
        self.assertFalse((self.base / 'labels.json').exists())

    def test_missing_mapping_does_not_guess(self):
        with self.assertRaisesRegex(ValueError, 'mapping is unresolved'):
            self.build()

    def test_explicit_mapping_out_of_order(self):
        manifest, _, _ = self.build(shot_map=['1=6', '2=5', '3=4', '4=3', '5=2', '6=1'])
        self.assertEqual(manifest['labels'][0]['planned_physical_shot'], 6)

    def test_existing_binding_mappings_need_no_order_flag(self):
        for i, row in enumerate(self.plan['rows'], 1):
            row['actual_event_id'] = i
        self.save()
        self.assertEqual(len(self.build()[0]['labels']), 6)

    def test_order_cannot_override_explicit_mapping(self):
        self.plan['rows'][0]['actual_event_id'] = 2
        self.save()
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            self.build(in_capture_order=True)

    def test_reject_duplicate_missing_unknown_and_malformed_mapping(self):
        for mapping in [['1=1', '1=1'], ['1=1', '2=1'], ['1=9'], ['7=1'],
                        ['1=1'], ['True=1'], ['1=0'], ['1=1=2']]:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                self.build(shot_map=mapping)

    def test_no_impact_extra_does_not_consume_physical_ordinal(self):
        directory = self.event(7, False)
        self.write(directory / 'ground_truth_status.json', {'status': 'unresolved'})
        self.write(self.root / 'physical_assignments.json', {'7':
            dict(state='NO_PHYSICAL_SHOT', label_shot_id=None, reason='Human observed no impact')})
        before = self.hashes()
        manifest, quality, _ = self.build(in_capture_order=True)
        self.assertEqual(manifest['labels'][-1], dict(event_id=7, planned_physical_shot=None, status='NO_PHYSICAL_SHOT'))
        self.assertEqual(quality['label_completeness'], 'PASS')
        self.assertEqual(before, self.hashes())

    def test_no_impact_without_reason_or_with_coordinate_refused(self):
        for assignment in [dict(state='NO_PHYSICAL_SHOT'),
                           dict(state='NO_PHYSICAL_SHOT', reason='human observed'),
                           dict(state='NO_PHYSICAL_SHOT', reason='human', planned_physical_shot=1)]:
            self.write(self.root / 'physical_assignments.json', {'1': assignment})
            with self.subTest(assignment=assignment), self.assertRaises(ValueError):
                self.build(in_capture_order=True)

    def test_unknown_ambiguous_and_skip_refused(self):
        for state in ['UNKNOWN', 'AMBIGUOUS', 'SKIP']:
            self.write(self.root / 'physical_assignments.json', {'1': {'state': state}})
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, 'UNKNOWN/AMBIGUOUS'):
                self.build(in_capture_order=True)

    def test_unresolved_marker_and_missing_ground_truth_refused(self):
        shot = self.root / 'shots/shot_00000001'
        self.write(shot / 'ground_truth_status.json', {'status': 'unresolved'})
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            self.build(in_capture_order=True)
        (shot / 'ground_truth_status.json').unlink()
        (shot / 'ground_truth.json').unlink()
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            self.build(in_capture_order=True)

    def test_duplicate_coordinate_label_assignment_refused(self):
        self.write(self.root / 'physical_assignments.json', {'2': {'label_shot_id': 1}})
        with self.assertRaisesRegex(ValueError, 'reused'):
            self.build(in_capture_order=True)

    def test_duplicate_json_and_noncanonical_assignment_keys_refused(self):
        path = self.root / 'physical_assignments.json'
        path.write_text('{"1":{},"1":{}}')
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON'):
            self.build(in_capture_order=True)
        path.write_text('{"01":{}}')
        with self.assertRaisesRegex(ValueError, 'noncanonical'):
            self.build(in_capture_order=True)

    def test_wrong_plan_session_and_class_refused(self):
        for key, value in [('collection_plan_id', 'wrong'), ('planned_session_id', 'S01'),
                           ('session_class', 'other')]:
            meta = dict(self.meta, **{key: value})
            self.write(self.binding, meta)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.build(in_capture_order=True)

    def test_binding_rows_and_embedded_root_mismatch_refused(self):
        meta = dict(self.meta, rows=self.meta['rows'][:-1])
        self.write(self.binding, meta)
        with self.assertRaisesRegex(ValueError, 'rows'):
            self.build(in_capture_order=True)
        self.save()
        self.write(self.root / 'collection_binding.json', dict(self.meta, trace_root=str(self.base)))
        with self.assertRaisesRegex(ValueError, 'another session'):
            self.build(in_capture_order=True)

    def test_recovered_binding_without_embedded_metadata_is_supported(self):
        (self.root / 'collection_binding.json').unlink()
        self.meta['recovered'] = True
        self.write(self.binding, self.meta)
        self.assertTrue(self.build(in_capture_order=True)[0]['recovered'])

    def test_external_assignments_supported_without_writing_session(self):
        self.event(7, False)
        mapping = self.base / 'external.json'
        self.write(mapping, {'7': dict(state='NO_PHYSICAL_SHOT', reason='Human confirmed')})
        before = self.hashes()
        self.assertEqual(len(self.build(mapping=mapping, in_capture_order=True)[0]['labels']), 7)
        self.assertEqual(before, self.hashes())

    def test_preview_does_not_write(self):
        output = self.base / 'preview.json'
        result = self.cli('--in-capture-order', '--preview', '--output', str(output))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PREVIEW: no files written', result.stdout)
        self.assertFalse(output.exists())

    def test_existing_output_refused_and_preserved(self):
        output = self.base / 'existing.json'
        output.write_text('baseline')
        result = self.cli('--in-capture-order', '--output', str(output))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Output exists', result.stderr)
        self.assertEqual(output.read_text(), 'baseline')

    def test_output_inside_session_refused(self):
        result = self.cli('--in-capture-order', '--output', str(self.root / 'report.json'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('outside the captured session', result.stderr)

    def test_fresh_quality_cannot_be_bypassed_by_saved_pass(self):
        _, quality, _ = self.build(in_capture_order=True)
        self.write(self.base / 'quality.json', [quality])
        (self.root / 'shots/shot_00000001/post.npy').unlink()
        result = self.cli('--in-capture-order', '--quality', str(self.base / 'quality.json'), '--preview')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('not complete', result.stderr)

    def test_wrong_quality_root_refused(self):
        _, quality, _ = self.build(in_capture_order=True)
        quality['trace_root'] = str(self.base / 'different')
        self.write(self.base / 'quality.json', [quality])
        result = self.cli('--in-capture-order', '--quality', str(self.base / 'quality.json'), '--preview')
        self.assertNotEqual(result.returncode, 0)

    def test_sibling_never_opened(self):
        sibling = self.base / 'session_validation'
        sibling.mkdir()
        (sibling / 'trace.json').write_text('invalid; must never be read')
        self.assertEqual(len(self.build(in_capture_order=True)[0]['labels']), 6)

    def test_symlinked_internal_evidence_refused(self):
        (self.root / 'extra').symlink_to(self.base / 'plan.json')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            self.build(in_capture_order=True)

    def test_escaping_frame_reference_refused_before_quality_read(self):
        path = self.root / 'shots/shot_00000001/trace.json'
        trace = read_json(path)
        trace['frames'][0]['path'] = '../../../outside.npy'
        self.write(path, trace)
        with self.assertRaisesRegex(ValueError, 'escapes captured event'):
            self.build(in_capture_order=True)

    def test_duplicate_gt_json_keys_refused(self):
        path = self.root / 'shots/shot_00000001/ground_truth.json'
        path.write_text('{"shot_id":1,"shot_id":2}')
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON'):
            self.build(in_capture_order=True)

    def test_nonfinite_label_timestamp_refused(self):
        path = self.root / 'shots/shot_00000001/ground_truth.json'
        gt = read_json(path)
        gt['attached_at'] = float('nan')
        self.write(path, gt)
        with self.assertRaisesRegex(ValueError, 'Nonfinite JSON'):
            self.build(in_capture_order=True)

    def test_validation_path_refused_before_any_trace_read(self):
        self.meta['trace_root'] = str(self.base / 'session_S03_protected')
        self.write(self.binding, self.meta)
        with self.assertRaisesRegex(ValueError, 'validation trace root refused'):
            self.build(in_capture_order=True)

    def test_noninteger_planned_ordinal_refused(self):
        self.plan['rows'][0]['planned_physical_shot'] = True
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            self.build(in_capture_order=True)

    def test_legacy_labels_cli_still_works(self):
        manifest, quality, _ = self.build(in_capture_order=True)
        self.write(self.base / 'labels.json', manifest)
        self.write(self.base / 'quality.json', quality)
        for module in ['automation.physical_collection', 'automation.physical_finalize']:
            output = self.base / (module + '.json')
            result = subprocess.run([sys.executable, '-m', module,
                *(['finalize'] if module.endswith('physical_collection') else []),
                '--plan', str(self.base / 'plan.json'), '--session', 'D01',
                '--labels', str(self.base / 'labels.json'), '--quality', str(self.base / 'quality.json'),
                '--output', str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(read_json(output)['physical_shots'], 6)


if __name__ == '__main__':
    unittest.main()
