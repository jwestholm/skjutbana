"""Batching contracts; synthetic fixtures are not physical validation."""
import copy
import unittest
import numpy as np
from src.engine.offline.clean_physical_change import temporal_step_evidence,temporal_step_evidence_batch,local_contrast
from src.engine.offline.common_verifier import (query_physical_witness,query_physical_witnesses,
    CommonTemporalHistory,camera_local_contrasts,camera_local_patches)


def frames(n,clocked=True):
    return [dict(frame_id=str(i),ordinal=i,receipt_timestamp=i*.08 if clocked else None,
        camera_sequence=i if clocked else None) for i in range(n)]

def candidate(i,previous=None):
    return dict(candidate_id=str(i),coordinate_version='v1',camera_xy=[40.+i,40.],
        source_provenance={'producer':'fixture','born_ordinal':0},previous=previous)

class Tests(unittest.TestCase):
    def test_arithmetic_each_prefix_and_coordinate(self):
        rng=np.random.default_rng(23)
        values=rng.normal(size=(50,32));values[8:,0]+=12;values[14:,1]-=8;values[18:22,2]+=9
        values[:,3]=0;values[10:,4]+=8;values[30:,4]+=8
        for n in range(1,51):
            for clocked in (False,True):
                t=np.arange(n)*.08 if clocked else None
                batch=temporal_step_evidence_batch(values[:n],t)
                self.assertEqual(batch,[temporal_step_evidence(values[:n,j],t) for j in range(32)])
    def test_full_witness_lifecycle_exact(self):
        signals=np.asarray([[0]*6+[8]*3+[0]*3,[0]*6+[8]*6,[0]*12],float).T
        for clocked in (False,True):
            old=[None]*3;states=[]
            for n in range(1,13):
                f=frames(n,clocked);cs=[candidate(i,old[i]) for i in range(3)];args=dict(event_id='event',cutoff_frame_id=f[-1]['frame_id'],cutoff_ts=f[-1]['receipt_timestamp'])
                result=query_physical_witnesses(signals[:n],f,candidates=cs,**args)
                expected=[query_physical_witness(signals[:n,i],f,**c,**args) for i,c in enumerate(cs)]
                self.assertEqual([b.witnesses for b in result],[b.witnesses for b in expected])
                old=[b.witnesses[0] for b in result];states.append(old[0]['state'])
            self.assertEqual(states[8:11],['SUPPORTED','CONTRADICTED','UNKNOWN'])
    def test_no_alias_or_input_mutation(self):
        x=np.zeros((9,2));x[6:]=8;f=frames(9);cs=[candidate(0),candidate(1)];before=copy.deepcopy((f,cs));saved=x.copy()
        out=query_physical_witnesses(x,f,event_id='e',candidates=cs,cutoff_frame_id='8',cutoff_ts=.64)
        self.assertEqual((f,cs),before);np.testing.assert_array_equal(x,saved)
        out[0].witnesses[0]['frames'][0]['frame_id']='changed'
        self.assertEqual(out[1].witnesses[0]['frames'][0]['frame_id'],'0');self.assertEqual(f[0]['frame_id'],'0')
    def test_order_identity_and_bounds(self):
        f=frames(6);x=np.zeros((6,2));args=dict(event_id='e',cutoff_frame_id='5',cutoff_ts=.4)
        with self.assertRaises(ValueError):query_physical_witnesses(x,f,candidates=[candidate(0)]*2,**args)
        with self.assertRaises(ValueError):query_physical_witnesses(np.zeros((6,33)),f,candidates=[candidate(i) for i in range(33)],**args)
        with self.assertRaises(ValueError):query_physical_witnesses(x,f,candidates=[candidate(0),candidate(1)],**dict(args,cutoff_ts=.2))
        f[0]['ordinal']=9
        with self.assertRaises(ValueError):query_physical_witnesses(x,f,candidates=[candidate(0),candidate(1)],**args)
    def test_no_foreign_previous_or_rewritten_signal(self):
        f=frames(9);x=np.zeros((9,1));x[6:]=8;args=dict(event_id='e',cutoff_frame_id='8',cutoff_ts=.64)
        old=query_physical_witnesses(x,f,candidates=[candidate(0)],**args)[0].witnesses[0]
        c=candidate(0,old);c['camera_xy'][0]+=.1
        with self.assertRaises(ValueError):query_physical_witnesses(x,f,candidates=[c],**args)
        x[0]=1
        with self.assertRaises(ValueError):query_physical_witnesses(x,f,candidates=[candidate(0,old)],**args)
    def test_history_adapter_equals_scalar(self):
        h=CommonTemporalHistory(max_frames=12);points=[candidate(i) for i in range(3)]
        for i in range(12):
            im=np.full((96,96),150,np.uint8)
            if i>=6:im[37:44,37:44]=110
            h.append(im,timestamp=i*.08,frame_id=str(i),stream_id='s',camera_sequence=i)
        ids=[str(i) for i in range(12)]
        batch=h.witness_queries(ids,cutoff=.88,event_id='e',candidates=points)
        scalar=[h.witness_query(ids,cutoff=.88,event_id='e',**c) for c in points]
        self.assertEqual([b.witnesses for b in batch],[b.witnesses for b in scalar])
        h.append(im,timestamp=.96,frame_id='12',stream_id='s',camera_sequence=12)
        with self.assertRaises(LookupError):h.witness_queries(ids,cutoff=.96,event_id='e',candidates=points)
    def test_local_reductions_preserve_exact_sampler(self):
        rng=np.random.default_rng(9);image=rng.integers(0,256,(160,160),dtype=np.uint8)
        points=[(60+i/7,70+i/5) for i in range(32)]
        expected=[local_contrast(camera_local_patches(image,[p]))[0] for p in points]
        np.testing.assert_array_equal(camera_local_contrasts(image,points),expected)
        np.testing.assert_array_equal(camera_local_contrasts(image,np.asarray(points)),expected)
    def test_sparse_and_unknown_clocks(self):
        x=np.zeros((9,2));x[6:]=8;t=np.arange(9)*.08;t[6:]+=1
        self.assertEqual(temporal_step_evidence_batch(x,t),[temporal_step_evidence(x[:,j],t) for j in range(2)])
        out=temporal_step_evidence_batch(x)
        self.assertTrue(out[0]['supported']);self.assertEqual(out[0]['certification'],'ORDINAL_ONLY_TIMING_UNKNOWN')
    def test_empty_candidate_bank(self):
        result=query_physical_witnesses(np.empty((6,0)),frames(6),event_id='e',candidates=[],cutoff_frame_id='5',cutoff_ts=.4)
        self.assertEqual(result,())
        bad=frames(6);bad[1]['receipt_timestamp']=0.
        with self.assertRaises(ValueError):
            query_physical_witnesses(np.empty((6,0)),bad,event_id='e',candidates=[],cutoff_frame_id='5',cutoff_ts=.4)

if __name__=='__main__':unittest.main()
