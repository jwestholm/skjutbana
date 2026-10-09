"""Synthetic crop replay/batching regressions. No physical sessions discovered."""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from automation import object_hit_replay as replay
from automation import object_physical_query_selftest as fx
from src.engine.camera.object_physical_query import ObjectPhysicalQuery
from src.engine.input.hit_regions import HitRegion,object_hit_regions


def capture(root):
    root.mkdir();shot=root/'object_crops/shots/shot_00000001';shot.mkdir(parents=True)
    entries=[]
    def crop(sid,kind,ts,bbox,pixels):
        arr=pixels.astype(np.uint8);name=f'{len(entries)}.npy';np.save(shot/name,arr)
        entries.append(dict(seq=len(entries),path=name,kind=kind,timestamp=ts,camera_bbox=list(bbox),
            shape=list(arr.shape),dtype=str(arr.dtype),sha256=hashlib.sha256(arr.tobytes()).hexdigest()))
    regions=object_hit_regions([HitRegion('can',100,60,60,80)])
    snap=fx.snapshot(game=regions);snap.transform_method='homography';snap.viewport_xywh=(0,0,360,240)
    pre=np.full((240,360),140,np.uint8);post=pre.copy();post[89:94,122:127]=30
    q=ObjectPhysicalQuery(snap,[fx.frame(pre,t) for t in (1.90,1.94)],camera_to_game=lambda x,y:(x,y),capture_crop=crop)
    fx.hit_result(q,post)
    trace=dict(shot_id=1,peak_ts=2.,completeness={'trace_complete':True},object_query_diagnostic=q.result,
        object_evidence_crops=entries,context=dict(runtime_geometry=dict(runtime_homography=np.eye(3).tolist(),
            calibration={'camera_frame_size':[360,240]}),shot_game_context=dict(transform_available=True,
            transform_method='homography',viewport_xywh=[0,0,360,240],
            game_hit_regions=[asdict(r) for r in regions],camera_hit_regions=[asdict(r) for r in snap.camera_regions])))
    replay.save(shot/'trace.json',trace)
    replay.save(root/'play.json',dict(mode='OBJECT_HIT_GAMEPLAY_ACCEPTANCE',full_frame_capture=False))
    replay.save(root/'event_notes.json',dict(intended_object='WRONG_INTENT_NOT_TRUTH'))
    (root/'runtime.log').write_text('[OBJECT-HIT] '+json.dumps(q.compact_record())+'\n')
    return trace


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/data/skjutbana');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'source';capture(self.source)

    def test_saved_pixels_geometry_evidence_and_decision_exact(self):
        _,shots,hashes=replay.load(self.source);r,_=replay.replay(shots[0])
        self.assertTrue(all(c['equal'] for c in r['recorded_ledger_comparison']))
        self.assertEqual(r['replay']['object_decision']['object_id'],'can')
        self.assertEqual(r['mapping']['max_game_error'],0.)
        self.assertTrue(all(replay.digest(p)==h for p,h in hashes.items()))

    def test_scalar_and_batch_full_candidate_parity(self):
        _,shots,_=replay.load(self.source)
        a,_=replay.replay(shots[0]);b,_=replay.replay(shots[0],scalar=True)
        self.assertEqual(replay.parity_view(a['all_candidates']),replay.parity_view(b['all_candidates']))
        self.assertEqual(replay.parity_view(a['replay']),replay.parity_view(b['replay']))

    def test_resume_does_not_replay_completed_shot(self):
        output=self.root/'out'
        with redirect_stdout(io.StringIO()):replay.run(self.source,output,compare_scalar=True)
        with patch.object(replay,'replay',side_effect=AssertionError('must reuse checkpoint')):
            with redirect_stdout(io.StringIO()):replay.run(self.source,output,resume=True,compare_scalar=True)
        self.assertTrue((output/'objects.csv').is_file())
        self.assertTrue(json.loads((output/'summary.json').read_text())[0]['scalar_comparison']['exact_equal'])

    def test_resume_refuses_changed_inputs_and_existing_output_preserved(self):
        output=self.root/'out'
        with redirect_stdout(io.StringIO()):replay.run(self.source,output)
        before=(output/'shot_1.json').read_bytes()
        with self.assertRaises(FileExistsError):replay.run(self.source,output)
        (self.source/'event_notes.json').write_text('{}')
        with self.assertRaises(ValueError):replay.run(self.source,output,resume=True)
        self.assertEqual(before,(output/'shot_1.json').read_bytes())

    def test_hash_mismatch_refused(self):
        p=self.source/'object_crops/shots/shot_00000001/0.npy'
        a=np.load(p);a[0,0]+=1;np.save(p,a)
        with self.assertRaisesRegex(ValueError,'hash'):replay.load(self.source)

    def test_missing_pixels_never_filled_with_current_or_zero_image(self):
        _,shots,_=replay.load(self.source);shots[0]['posts'][-1].frame_bgr.crops.clear()
        with self.assertRaisesRegex(ValueError,'missing saved crop'):replay.replay(shots[0])

    def test_saved_mapping_mismatch_refused(self):
        p=self.source/'object_crops/shots/shot_00000001/trace.json'
        t=json.loads(p.read_text());t['context']['runtime_geometry']['runtime_homography'][0][2]=10
        replay.save(p,t)
        with self.assertRaisesRegex(ValueError,'mapping disagrees'):replay.load(self.source)

    def test_forbidden_paths_rejected_before_source_read(self):
        with patch.object(Path,'read_text',side_effect=AssertionError('must not read')):
            with self.assertRaises(ValueError):replay.load('/data/skjutbana/S03/forbidden')

    def test_output_cannot_modify_capture_or_leave_data_root(self):
        for path in (self.source/'results',Path('/tmp/forbidden_output')):
            with self.assertRaises(ValueError):replay.run(self.source,path)

    def test_compact_only_capture_has_no_invented_replay(self):
        (self.source/'object_crops/shots/shot_00000001/trace.json').unlink()
        out=self.root/'out'
        with redirect_stdout(io.StringIO()):replay.run(self.source,out)
        result=json.loads((out/'summary.json').read_text())[0]
        self.assertEqual(result['mode'],'COMPACT_LOG_ONLY');self.assertIsNone(result['replay_decision'])


class BatchedMeasurements(unittest.TestCase):
    def test_scalar_values_and_qualification_bit_exact(self):
        rng=np.random.default_rng(227)
        for dtype in (np.float32,np.float64):
            for count in (1,3,24,128):
                signed=rng.normal(0,5,(80,90)).astype(dtype)
                absolute=np.abs(signed);noise=rng.uniform(0,8,(80,90)).astype(dtype)
                points=list(zip(rng.integers(8,82,count),rng.integers(8,72,count)))
                scalar=[ObjectPhysicalQuery._evidence(absolute,noise,signed,x,y) for x,y in points]
                self.assertEqual(scalar,ObjectPhysicalQuery._evidence_many(absolute,noise,signed,points))

    def test_clipped_edges_preserve_scalar_path(self):
        a=np.arange(900,dtype=np.float32).reshape(30,30)/10;n=np.ones_like(a)
        points=[(0,0),(2,2),(29,29),(8,8),(20,20)]
        self.assertEqual([ObjectPhysicalQuery._evidence(a,n,-a,x,y) for x,y in points],
            ObjectPhysicalQuery._evidence_many(a,n,-a,points))
        self.assertEqual(ObjectPhysicalQuery._evidence_many(a,n,-a,[]),[])


if __name__=='__main__':unittest.main()
