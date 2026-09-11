"""Regressions for misleading quality/finalization passes and session leakage."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from automation.physical_capture_plan import build
from automation.physical_finalize import validate
from automation.physical_trace_quality import inspect_one, inspect_session


class PostflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'session_test'
        self.shot = self.root / 'shots/shot_00000001'
        self.shot.mkdir(parents=True)
        self.trace = dict(schema_version='physical-shot-trace-1', shot_id=1, peak_ts=1,
                          created_at=2, completeness={'trace_complete': True},
                          decision_input={'complete_track_audit': {'complete': True}}, selectors={'CURRENT': {}},
                          frames=[dict(kind=kind, path=kind+'.npy', timestamp=ts, shape=[8, 8], dtype='uint8')
                                  for kind, ts in [('pre_snapshot', .9), ('post', 1.1)]])
        self.write_trace()
        for frame in self.trace['frames']:
            np.save(self.shot / frame['path'], np.zeros((8, 8), np.uint8))
        self.gt = dict(schema_version='physical-shot-trace-1', shot_id=1, space='camera',
                       camera_x=3., camera_y=4., quality='precise', attached_at=3.)
        self.write_gt()
        self.plan = build(3, 1)
        self.labels = dict(collection_plan_id=self.plan['collection_plan_id'], planned_session_id='S02',
                           trace_root=str(self.root), labels=[dict(event_id=1, planned_physical_shot=1, status='PHYSICAL')])
        self.quality = inspect_session(self.root)

    def write_trace(self):
        (self.shot / 'trace.json').write_text(json.dumps(self.trace))

    def write_gt(self):
        (self.shot / 'ground_truth.json').write_text(json.dumps(self.gt))

    def finalize(self, quality=None):
        label_path, quality_path = self.base / 'labels.json', self.base / 'quality.json'
        label_path.write_text(json.dumps(self.labels))
        quality_path.write_text(json.dumps(self.quality if quality is None else quality))
        return validate(self.plan, 'S02', label_path, quality_path)

    def test_complete_session_passes(self):
        for field in ('trace_completeness', 'frame_completeness', 'label_completeness', 'full_replay_ready'):
            self.assertEqual(self.quality[field], 'PASS')
        self.assertEqual(self.finalize([self.quality])['physical_shots'], 1)

    def test_missing_or_wrong_shape_frame_fails(self):
        (self.shot / 'post.npy').unlink()
        self.assertEqual(inspect_session(self.root)['frame_completeness'], 'FAIL')
        np.save(self.shot / 'post.npy', np.zeros((3, 3), np.uint8))
        self.assertEqual(inspect_session(self.root)['patch_dataset_ready'], 'FAIL')

    def test_empty_session_never_passes(self):
        empty = self.base / 'session_empty'
        empty.mkdir()
        quality = inspect_session(empty)
        self.assertEqual(quality['frame_completeness'], 'FAIL')
        self.assertEqual(quality['trace_completeness'], 'FAIL')

    def test_incomplete_producer_keeps_frames_but_blocks_finalization(self):
        self.trace['completeness']['trace_complete'] = False
        self.write_trace()
        quality = inspect_session(self.root)
        self.assertEqual(quality['frame_completeness'], 'PASS')
        with self.assertRaisesRegex(ValueError, 'trace_completeness'):
            self.finalize(quality)

    def test_skip_requires_explicit_human_assignment(self):
        (self.shot / 'ground_truth.json').unlink()
        (self.shot / 'ground_truth_status.json').write_text('{"status":"unresolved"}')
        self.assertEqual(inspect_session(self.root)['label_completeness'], 'WARN')
        mapping = {'1': {'state': 'NO_PHYSICAL_SHOT', 'reason': 'Human confirmed no impact'}}
        self.assertEqual(inspect_session(self.root, mapping)['label_completeness'], 'PASS')
        self.assertTrue((self.shot / 'ground_truth_status.json').exists())
        self.assertFalse((self.root / 'physical_assignments.json').exists())

    def test_unknown_assignment_and_conflicting_label_fail(self):
        for mapping in ({'2': {'state': 'NO_PHYSICAL_SHOT'}}, {'1': {'state': 'NO_PHYSICAL_SHOT', 'reason': 'x'}}):
            self.assertEqual(inspect_session(self.root, mapping)['label_completeness'], 'WARN')

    def test_invalid_label_identity_bounds_and_timing(self):
        for key, value in [('shot_id', 2), ('camera_x', 99), ('attached_at', .5)]:
            with self.subTest(key=key):
                previous = self.gt[key]
                self.gt[key] = value
                self.write_gt()
                self.assertEqual(inspect_session(self.root)['label_completeness'], 'WARN')
                self.gt[key] = previous

    def test_single_session_does_not_inspect_siblings(self):
        sibling = self.base / 'session_validation'
        (sibling / 'shots/shot_00000001').mkdir(parents=True)
        (sibling / 'shots/shot_00000001/trace.json').write_text('must never be read')
        self.assertEqual(len(inspect_one(self.root)), 1)

    def test_truthy_fail_and_missing_quality_fields_refused(self):
        for field in ('frame_completeness', 'trace_completeness', 'label_completeness'):
            with self.subTest(field=field):
                quality = dict(self.quality, **{field: 'FAIL'})
                with self.assertRaisesRegex(ValueError, field):
                    self.finalize(quality)
                quality.pop(field)
                with self.assertRaisesRegex(ValueError, field):
                    self.finalize(quality)

    def test_wrong_quality_session_never_falls_back_to_first(self):
        wrong = dict(self.quality, trace_root=str(self.base / 'another_session'))
        with self.assertRaisesRegex(ValueError, 'exactly the labeled session'):
            self.finalize([wrong])
        with self.assertRaisesRegex(ValueError, 'differs from labels'):
            self.finalize(wrong)

    def test_missing_physical_shot_or_extra_event_refused(self):
        self.labels['labels'] = []
        with self.assertRaisesRegex(ValueError, 'every planned shot'):
            self.finalize()
        self.labels['labels'] = [dict(event_id=1, planned_physical_shot=1, status='PHYSICAL'),
                                 dict(event_id=2, planned_physical_shot=None, status='NO_PHYSICAL_SHOT')]
        with self.assertRaisesRegex(ValueError, 'captured event ids'):
            self.finalize()

    def test_unknown_status_and_nonphysical_planned_ordinal_refused(self):
        self.labels['labels'][0]['status'] = 'SKIP'
        with self.assertRaisesRegex(ValueError, 'unknown'):
            self.finalize()
        self.labels['labels'][0]['status'] = 'PHYSICAL'
        self.labels['labels'].append(dict(event_id=2, planned_physical_shot=2, status='NO_PHYSICAL_SHOT'))
        with self.assertRaisesRegex(ValueError, 'nonphysical events'):
            self.finalize()


if __name__ == '__main__':
    unittest.main()
