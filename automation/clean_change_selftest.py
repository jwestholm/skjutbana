"""Numerical/integrity regressions only; constructed fixtures are not physical validation."""
import inspect
import unittest

import cv2
import numpy as np

from src.engine.offline.accuracy_verifier import EvidenceContext
from src.engine.offline.clean_physical_change import (
    VARIANTS, change_maps, clean_features, map_proposals, map_metrics, motion_compensate)


class CleanChangeTests(unittest.TestCase):
    def context(self, pre=None, history=None, post=None, origin=(0, 0)):
        pre = np.full((96, 128), 100, np.float32) if pre is None else pre
        history = np.repeat(pre[None], 4, axis=0) if history is None else history
        post = np.repeat(pre[None], 4, axis=0) if post is None else post
        return EvidenceContext(pre, history, post, 1., 1.5,
            (.5, .6, .7, .8), (1.05, 1.15, 1.25, 1.35), origin,
            np.full(pre.shape, 255, np.uint8), [])

    def setUp(self):
        cv2.setNumThreads(1)

    def test_unchanged_frames_have_no_proposals(self):
        ctx = self.context()
        maps, aux = change_maps(ctx)
        for name in VARIANTS:
            self.assertFalse(np.any(maps[name]))
            self.assertEqual(map_proposals(ctx, maps[name])[0], [])

    def test_inputs_and_roi_remain_immutable(self):
        ctx = self.context()
        ctx.post[:, 40:44, 60:64] -= 3
        originals = [arr.copy() for arr in (ctx.pre, ctx.history, ctx.post, ctx.roi)]
        maps, _ = change_maps(ctx)
        map_proposals(ctx, maps['pre_variability'])
        for arr, original in zip((ctx.pre, ctx.history, ctx.post, ctx.roi), originals):
            np.testing.assert_array_equal(arr, original)

    def test_weak_persistent_dark_and_bright_changes_survive(self):
        for sign in (-1, 1):
            ctx = self.context()
            ctx.post[:, 40:44, 60:64] += sign * 1.5
            maps, _ = change_maps(ctx)
            self.assertGreater(maps['pre_variability'][41, 61], .2)
            candidates, _ = map_proposals(ctx, maps['pre_variability'])
            self.assertTrue(any(np.hypot(c['camera_x']-61, c['camera_y']-41) < 5 for c in candidates))

    def test_pre_fluctuation_attenuates_same_post_change(self):
        stable, unstable = self.context(), self.context()
        for ctx in (stable, unstable):
            ctx.post[:, 40:44, 60:64] -= 4
        unstable.history[::2, 40:44, 60:64] -= 6
        stable_maps, _ = change_maps(stable)
        unstable_maps, _ = change_maps(unstable)
        self.assertLess(unstable_maps['pre_variability'][41, 61], stable_maps['pre_variability'][41, 61])
        self.assertGreater(unstable_maps['pre_variability'][41, 61], 0)

    def test_transient_is_attenuated_by_persistence(self):
        ctx = self.context()
        ctx.post[0, 40:44, 60:64] -= 8
        maps, _ = change_maps(ctx)
        self.assertGreater(maps['raw'][41, 61], 0)
        self.assertEqual(maps['persistent'][41, 61], 0)

    def test_no_wholesale_edge_veto(self):
        pre = np.full((96, 128), 100, np.float32)
        pre[:, 64:] = 150
        ctx = self.context(pre)
        ctx.post[:, 40:44, 62:66] -= 3
        maps, _ = change_maps(ctx)
        self.assertGreater(maps['pre_variability'][41, 63], 0)

    def test_global_brightness_drift_is_removed_locally(self):
        ctx = self.context()
        ctx.post -= 5
        maps, _ = change_maps(ctx)
        self.assertGreater(maps['raw'].mean(), 4)
        self.assertLess(maps['local_background'].mean(), 1e-4)

    def test_causal_context_refuses_future_and_pre_after_peak(self):
        ctx = self.context()
        for key, times in [('post_times', (1.05, 1.15, 1.25, 2.)), ('pre_times', (.5, .6, .7, 1.01))]:
            fields = vars(ctx).copy()
            fields[key] = times
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Noncausal'):
                EvidenceContext(**fields)

    def test_proposals_bounded_deterministic_and_canonical(self):
        ctx = self.context(origin=(100, 200))
        magnitude = np.random.default_rng(99).uniform(0, 10, ctx.pre.shape).astype(np.float32)
        first, _ = map_proposals(ctx, magnitude, 17)
        second, _ = map_proposals(ctx, magnitude, 17)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 17)
        self.assertTrue(all(c['camera_x'] >= 125 and c['camera_y'] >= 225 for c in first))
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                map_proposals(ctx, magnitude, invalid)

    def test_roi_excludes_only_outside_points(self):
        ctx = self.context()
        ctx.roi[:, :64] = 0
        ctx.post[:, 40:44, 40:44] -= 4
        ctx.post[:, 40:44, 80:84] -= 4
        maps, _ = change_maps(ctx)
        candidates, _ = map_proposals(ctx, maps['pre_variability'])
        self.assertFalse(np.any(maps['pre_variability'][:, :64]))
        self.assertTrue(candidates)
        self.assertTrue(all(c['camera_x'] >= 64 for c in candidates))

    def test_same_local_evidence_same_features_at_different_origin(self):
        a, b = self.context(), self.context(origin=(100, 200))
        for ctx in (a, b):
            ctx.post[:, 40:44, 60:64] -= 3
        ma, aa = change_maps(a)
        mb, ab = change_maps(b)
        names, fa = clean_features(a, ma['pre_variability'], aa, (61, 41))
        names_b, fb = clean_features(b, mb['pre_variability'], ab, (161, 241))
        self.assertEqual(names, names_b)
        np.testing.assert_array_equal(fa, fb)

    def test_gt_metrics_do_not_change_maps_or_proposals(self):
        ctx = self.context()
        ctx.post[:, 40:44, 60:64] -= 3
        maps, _ = change_maps(ctx)
        first = map_proposals(ctx, maps['pre_variability'])
        for gt in [dict(camera_x=61, camera_y=41), dict(camera_x=80, camera_y=50)]:
            metrics = map_metrics(ctx, maps['pre_variability'], maps['raw'], gt)
            self.assertTrue(np.isfinite(metrics['retained_mass']))
        self.assertEqual(first, map_proposals(ctx, maps['pre_variability']))
        self.assertNotIn('gt', inspect.signature(change_maps).parameters)
        self.assertNotIn('gt', inspect.signature(map_proposals).parameters)

    def test_piecewise_motion_reduces_broad_shift_residual(self):
        pre = np.random.default_rng(123).uniform(0, 255, (160, 256)).astype(np.float32)
        frame = cv2.warpAffine(pre, np.float32([[1, 0, 1], [0, 1, 0]]), (256, 160), borderMode=cv2.BORDER_REPLICATE)
        post, _ = motion_compensate(pre, frame[None], 'piecewise')
        self.assertLess(np.abs(pre[8:-8, 8:-8]-post[0, 8:-8, 8:-8]).mean(),
                        np.abs(pre[8:-8, 8:-8]-frame[8:-8, 8:-8]).mean())

    def test_roi_normalization_is_explicit_and_reversible(self):
        from automation.clean_change_analysis import roi_target
        row = dict(context={'crop': [100, 200, 400, 300]},
                   truth={'impacts': [dict(camera_x=220, camera_y=320)]})
        normalized = roi_target(row)
        np.testing.assert_allclose(normalized, [.3, .4])
        np.testing.assert_allclose(normalized * [400, 300] + [100, 200], [220, 320])
        row['truth']['impacts'][0]['camera_x'] = 99
        with self.assertRaisesRegex(ValueError, 'outside'):
            roi_target(row)

    def test_motion_features_cannot_read_gt(self):
        from automation.clean_change_analysis import motion_features
        row = dict(context={'crop': [100, 200, 400, 300], 'registrations': [dict(dx=.1, dy=.2)]},
                   motion={'piecewise': [[dict(dx=.2, dy=.3), dict(dx=.1, dy=.1)]]})
        before = motion_features(row)
        row['truth'] = {'impacts': [dict(camera_x=99999, camera_y=-1)]}
        np.testing.assert_array_equal(before, motion_features(row))

    def test_coarse_motion_prediction_is_deterministic(self):
        from automation.clean_change_analysis import ridge_predictions
        rng = np.random.default_rng(9)
        x, y, test = rng.normal(size=(10, 3)), rng.uniform(size=(10, 2)), rng.normal(size=(4, 3))
        np.testing.assert_array_equal(ridge_predictions(x, y, test), ridge_predictions(x, y, test))


if __name__ == '__main__':
    unittest.main()
