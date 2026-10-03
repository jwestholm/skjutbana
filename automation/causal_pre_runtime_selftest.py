"""Exact semantics and bounded ownership for the causal temporal prototype."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from automation.causal_pre_onset_research import infer, sample
from src.engine.offline.clean_physical_change import (
    causal_local_evidence, causal_local_onset, causal_local_onset_batch,
    local_contrast, local_contrast_batch)
from src.engine.offline.common_verifier import CommonTemporalHistory, camera_local_patches


class TemporalRuntimeTests(unittest.TestCase):
    def history(self, values, **kwargs):
        h = CommonTemporalHistory(**kwargs)
        images = []
        for i, value in enumerate(values):
            gray = np.full((100, 100), 100, np.uint8)
            gray[49:52, 49:52] = value
            h.append(gray, timestamp=i*.08, frame_id=str(i), stream_id='fixture', camera_sequence=i,
                     provenance={'receipt':i*.08, 'exposure':None})
            images.append(gray)
        return h, images

    def test_incremental_equals_batch_contaminated_pre(self):
        h, frames = self.history([100]*4+[50]*4)
        points = [(50.1, 50.3), (60., 60.)]
        pixels = np.stack([sample(f, points) for f in frames])
        expected = infer([i*.08 for i in range(8)], pixels, ({'timestamp':.48}, pixels[6]), .56, [.0,.08])
        actual, meta = h.causal_query(list(map(str,range(8))), points, cutoff=.56, snapshot_frame_id='6', early_times=[.0,.08])
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0]['pre_status'], 'PRE_CONTAMINATED')
        self.assertEqual(meta['frames'][3]['camera_sequence'], 3)

    def test_ambiguous_and_no_onset_match(self):
        signals = np.array([[0]*4+[8]*4+[16]*4, [0]*12, [0]*4+[8]*4+[0]*4]).T
        t = np.arange(12)*.08
        result = causal_local_onset_batch(signals, t, cutoff=t[-1])
        self.assertEqual(result, [causal_local_onset(x,t,cutoff=t[-1]) for x in signals.T])
        self.assertEqual([r['state'] for r in result], ['ONSET_AMBIGUOUS','NO_SUPPORTED_ONSET','ONSET_AMBIGUOUS'])
        self.assertTrue(all(r['physical_state']=='UNCERTAIN' for r in result))

    def test_random_signal_equivalence(self):
        rng = np.random.default_rng(9025)
        for length in (2, 5, 6, 8, 35):
            signals = rng.normal(0, 5, (length, 80));t = np.arange(length)*.08
            self.assertEqual(causal_local_onset_batch(signals,t,cutoff=t[-1]),
                             [causal_local_onset(x,t,cutoff=t[-1]) for x in signals.T])

    def test_exact_float_reduction_and_sampler(self):
        rng = np.random.default_rng(2525)
        gray = rng.integers(0,256,(100,100),dtype=np.uint8)
        points = [(50.01,50.99),(60.3,61.4)]
        np.testing.assert_array_equal(sample(gray,points), camera_local_patches(gray,points))
        pixels = rng.uniform(0,255,(7,5,49,49)).astype(np.float32)
        expected = np.stack([local_contrast(pixels[:,j]) for j in range(5)],axis=1)
        np.testing.assert_array_equal(local_contrast_batch(pixels), expected)

    def test_input_and_metadata_ownership(self):
        h, images = self.history([100]*8)
        images[0][:] = 0
        frame = h.select(['0'],cutoff=0)[0]
        self.assertEqual(frame.gray[0,0],100)
        self.assertTrue(images[0].flags.writeable)
        with self.assertRaises(ValueError): frame.gray[0,0] = 5
        frame.provenance['exposure']=9
        self.assertIsNone(h.select(['0'],cutoff=0)[0].provenance['exposure'])

    def test_bounded_history_requires_complete_requested_interval(self):
        for limits in ({'max_frames':3}, {'max_bytes':30000}):
            h,_ = self.history([100]*8,**limits)
            self.assertEqual(len(h),3);self.assertEqual(h.bytes_used,30000)
            with self.assertRaises(LookupError): h.select(['0','7'],cutoff=1)
            self.assertEqual(len(h.select(['5','6','7'],cutoff=1)),3)

    def test_frame_order_and_sequence(self):
        h, frames = self.history([100]*2)
        for ts,seq in ((.08,2),(.04,2),(.16,1),(.16,None)):
            with self.assertRaises(ValueError):
                h.append(frames[0],timestamp=ts,camera_sequence=seq,frame_id='2',stream_id='fixture')
        with self.assertRaises(ValueError): h.select(['1','0'],cutoff=1)
        with self.assertRaises(ValueError): h.select(['0','0'],cutoff=1)
        with self.assertRaises(ValueError):
            h.append(frames[0],timestamp=.16,camera_sequence=2,frame_id='2',stream_id='other')

    def test_no_future_leakage(self):
        h,_ = self.history([100]*4+[50]*4)
        with self.assertRaises(ValueError): h.select(['0','7'],cutoff=.48)
        with self.assertRaises(ValueError):
            h.causal_query(['0','1','2'],[(50,50)],cutoff=.16,snapshot_frame_id='7')
        r,_ = h.causal_query(['0','1','2','3'],[(50,50)],cutoff=.24)
        self.assertEqual(r[0]['state'],'INSUFFICIENT_HISTORY')

    def test_multi_candidate_reuse_repeatability_and_no_mutation(self):
        h,frames = self.history([100]*4+[50]*4)
        ids = list(map(str,range(8)));points=[(50,50),(60,60),(50,50)]
        before = [f.copy() for f in frames];memory=h.bytes_used
        a,_=h.causal_query(ids,points,cutoff=.56)
        b,_=h.causal_query(ids,points,cutoff=.56)
        self.assertEqual(a,b);self.assertEqual(a[0],a[2]);self.assertEqual(h.bytes_used,memory)
        for x,y in zip(before,frames):np.testing.assert_array_equal(x,y)

    def test_preexisting_hole_remains_unsupported(self):
        h,_=self.history([50]*8)
        r,_=h.causal_query(list(map(str,range(8))),[(50,50)],cutoff=.56)
        self.assertEqual(r[0]['state'],'NO_SUPPORTED_ONSET')
        self.assertEqual(r[0]['physical_state'],'UNCERTAIN')

    def test_sparse_history_and_invalid_metadata(self):
        x=np.zeros((8,2));t=np.arange(8)*.08;t[4:]+=.3
        self.assertTrue(all(r['state']=='INSUFFICIENT_HISTORY' for r in causal_local_onset_batch(x,t,cutoff=t[-1])))
        for times,cutoff in (([0,0],1),([1,0],1),([0,1],.5)):
            with self.assertRaises(ValueError): causal_local_onset_batch(np.zeros((2,2)),times,cutoff=cutoff)
        with self.assertRaises(ValueError): CommonTemporalHistory(max_frames=0)


class GrayscaleReuseTests(unittest.TestCase):
    def test_latest_frame_converted_only_once_preserving_isolation(self):
        import cv2
        from src.engine.camera.hit_scanner import HitScanner
        import importlib
        module=importlib.import_module('src.engine.camera.hit_scanner')
        scanner=HitScanner();scanner.enabled=True;scanner.state=scanner.STATE_ARMING;scanner.arm_until_ts=float('inf')
        frames=[SimpleNamespace(timestamp=float(i),frame_bgr=np.full((20,30,3),i*20,np.uint8)) for i in (1,2,3)]
        original=cv2.cvtColor
        with patch.object(module.camera_manager,'get_new_frames_since_last_pickup',return_value=frames), \
             patch.object(module.cv2,'cvtColor',wraps=original) as convert:
            scanner.update(.01)
        self.assertEqual(convert.call_count,len(frames))
        for source,entry in zip(frames,scanner.frame_history):
            self.assertEqual(entry.timestamp,source.timestamp)
            np.testing.assert_array_equal(entry.gray,original(source.frame_bgr,cv2.COLOR_BGR2GRAY))
            self.assertTrue(np.all(source.frame_bgr==source.timestamp*20))
        self.assertFalse(np.shares_memory(scanner.debug_frames['camera_gray'],scanner.frame_history[-1].gray))


if __name__=='__main__': unittest.main()
