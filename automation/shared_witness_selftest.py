"""Shared proof contracts; tests confer no physical accuracy."""
import copy,unittest
import numpy as np
from src.engine.offline.common_verifier import query_physical_witness,CommonTemporalHistory
from src.engine.offline.clean_physical_change import temporal_step_evidence
from automation.real_hole_donors import damage_layer,qualify
from automation.real_hole_v3 import compose,split_sources


def query(values,previous=None,clocked=True,xy=(40.,40.),version='v1'):
    fs=[dict(frame_id=str(i),ordinal=i,receipt_timestamp=i*.08 if clocked else None,camera_sequence=i if clocked else None) for i in range(len(values))]
    return query_physical_witness(values,fs,event_id='test',candidate_id='c1',coordinate_version=version,camera_xy=xy,
        source_provenance={'producer':'fixture'},cutoff_frame_id=fs[-1]['frame_id'],cutoff_ts=fs[-1]['receipt_timestamp'],previous=previous).witnesses[0]

class Tests(unittest.TestCase):
    def test_support_revoke_and_no_hit_claim(self):
        a=query([0.]*6+[8.]*3);self.assertEqual(a['state'],'SUPPORTED');self.assertEqual(a['physical_state'],'UNKNOWN')
        b=query([0.]*6+[8.]*3+[0.],a);self.assertEqual(b['state'],'CONTRADICTED')
        self.assertEqual(b['revocation']['reason'],'PERSISTENCE_REVERTED');self.assertEqual(b['physical_state'],'UNKNOWN')
        c=query([0.]*6+[8.]*3+[0.,0.],b);self.assertEqual(c['state'],'UNKNOWN');self.assertEqual(c['revocation']['last_revocation_reason'],'PERSISTENCE_REVERTED')
    def test_ordinal_is_not_a_clock(self):
        a=query([0.]*6+[8.]*3,clocked=False)
        self.assertIsNone(a['availability']['receipt_cutoff']);self.assertEqual(a['availability']['certification'],'ORDINAL_ONLY_TIMING_UNKNOWN')
    def test_exact_reference_arithmetic(self):
        for signal in ([0.]*9,[0.]*6+[8.]*3,[0.]*6+[8.]*3+[0.],[0.]*6+[8.]*3+[16.]*3):
            self.assertEqual(query(signal)['measurements'],temporal_step_evidence(signal,np.arange(len(signal))*.08,np.arange(len(signal))))
    def test_identity_changes_cannot_inherit(self):
        old=query([0.]*6+[8.]*3)
        with self.assertRaises(ValueError):query([0.]*6+[8.]*3,old,xy=(40.1,40.))
        with self.assertRaises(ValueError):query([0.]*6+[8.]*3,old,version='v2')
    def test_no_mutation_aliasing(self):
        values=[0.]*6+[8.]*3;old=query(values);snapshot=copy.deepcopy(old);new=query(values+[8.],old)
        new['frames'][0]['frame_id']='mutated';new['source_provenance']['producer']='mutated'
        self.assertEqual(old,snapshot);self.assertEqual(values,[0.]*6+[8.]*3)
    def test_prefix_values_and_metadata_are_immutable(self):
        old=query([0.]*6+[8.]*3)
        with self.assertRaises(ValueError):query([1.]+[0.]*5+[8.]*3,old)
        old['frames'][0]['receipt_timestamp']=-1.
        with self.assertRaises(ValueError):query([0.]*6+[8.]*3,old)
    def test_candidate_cannot_arrive_from_future(self):
        fs=[dict(frame_id='0',ordinal=0,receipt_timestamp=None)]
        with self.assertRaises(ValueError):
            query_physical_witness([0],fs,event_id='a',candidate_id='b',coordinate_version='c',
                camera_xy=(40,40),source_provenance={'born_ordinal':1},cutoff_frame_id='0')
    def test_bound_and_ambiguity(self):
        with self.assertRaises(ValueError):query([0.]*51)
        self.assertEqual(query([0.]*6+[8.]*3+[16.]*3)['state'],'UNKNOWN')
        self.assertEqual(query([0.]*12)['state'],'UNKNOWN')
    def test_no_future_and_frame_order(self):
        fs=[dict(frame_id=str(i),ordinal=i,receipt_timestamp=i*.08) for i in range(6)]
        args=dict(event_id='a',candidate_id='b',coordinate_version='c',camera_xy=(40,40),source_provenance={},cutoff_frame_id='5')
        with self.assertRaises(ValueError):query_physical_witness([0]*6,fs,cutoff_ts=.1,**args)
        fs[2]['ordinal']=0
        with self.assertRaises(ValueError):query_physical_witness([0]*6,fs,cutoff_ts=.4,**args)
    def test_history_query_and_eviction(self):
        h=CommonTemporalHistory(max_frames=9);frames=[]
        for i in range(9):
            im=np.full((96,96),150,np.uint8)
            if i>=6:im[37:44,37:44]=120
            h.append(im,timestamp=i*.08,frame_id=str(i),stream_id='x',camera_sequence=i)
            frames.append(str(i))
        b=h.witness_query(frames,cutoff=.64,camera_xy=(40,40),event_id='a',candidate_id='b',coordinate_version='c',source_provenance={})
        self.assertEqual(b.witnesses[0]['state'],'SUPPORTED');self.assertIsNone(b.fused)
        h.append(im,timestamp=.72,frame_id='9',stream_id='x',camera_sequence=9)
        with self.assertRaises(LookupError):h.witness_query(frames,cutoff=.72,camera_xy=(40,40),event_id='a',candidate_id='b',coordinate_version='c',source_provenance={})
    def test_donor_old_hole_not_new_damage(self):
        p=np.full((49,49),180,np.float32);p[22:27,22:27]=70
        arrays,meta=damage_layer(p,p);self.assertEqual(meta['mask_area'],0)
        r,_=qualify([0.]*12,[dict(timestamp=i*.08,kind='post' if i>=6 else 'pre_history') for i in range(12)],
            cutoff_known=True,complete=True,lineage_unique=True,caveats=None,patches=[p]*12)
        self.assertEqual(r['status'],'PARTIAL');self.assertFalse(r['pre_demonstrably_before_local_step'])
    def test_donor_local_step_and_partial_timing(self):
        pre=np.full((49,49),180,np.float32);post=pre.copy();post[22:27,22:27]=70
        signal=[0.]*6+[-20.]*6;frames=[dict(timestamp=i*.08,kind='post' if i>=6 else 'pre_history') for i in range(12)]
        r,a=qualify(signal,frames,cutoff_known=True,complete=True,lineage_unique=True,caveats=None,patches=[pre]*6+[post]*6)
        self.assertEqual(r['status'],'QUALIFIED');self.assertEqual(a['damage_mask'].sum(),25)
        for f in frames:f['timestamp']=None
        r,_=qualify(signal,frames,cutoff_known=False,complete=True,lineage_unique=True,caveats=None,patches=[pre]*6+[post]*6)
        self.assertEqual(r['status'],'PARTIAL');self.assertIn('FRAME_TIMING_UNAVAILABLE',r['reasons'])

class DatasetTests(unittest.TestCase):
    def test_session_split_before_augmentation(self):
        rows=[dict(source_session=f'session_{s}',source_hole_id=f'{s}:{i}',result={'status':'QUALIFIED'})
              for s in range(6) for i in range(4)]
        rows.append(dict(source_session='partial',source_hole_id='partial:1',result={'status':'PARTIAL'}))
        before=copy.deepcopy(rows);a=split_sources(rows,20260927)
        self.assertEqual(a,split_sources(rows,20260927));self.assertEqual(rows,before)
        self.assertNotIn('partial',a);self.assertEqual(set(a.values()),{'train','validation','test'})
        self.assertEqual(len(a),6)
        with self.assertRaises(ValueError):split_sources(rows+[rows[0]],20260927)
    def test_real_delta_composition_deterministic_and_local(self):
        pre=np.full((49,49),150,np.float32);post=pre.copy();post[22:27,22:27]=80
        donor,_=damage_layer(pre,post);snapshot={k:v.copy() for k,v in donor.items()}
        for method in ('additive_delta','relative_response'):
            a=compose(donor,pre,method=method,shift=(.25,-.25))
            b=compose(donor,pre,method=method,shift=(.25,-.25))
            for x,y in zip(a[:3],b[:3]):np.testing.assert_array_equal(x,y)
            self.assertEqual(a[3],b[3]);self.assertEqual(a[3]['outside_mask_mae'],0.)
            self.assertGreater(a[3]['masked_mae'],0.)
        for k in snapshot:np.testing.assert_array_equal(donor[k],snapshot[k])
        self.assertTrue(np.all(pre==150))
    def test_same_background_native_composition_reconstructs_local_delta(self):
        pre=np.full((49,49),150,np.float32);post=pre.copy();post[22:27,22:27]=80
        donor,_=damage_layer(pre,post)
        for method in ('additive_delta','relative_response'):
            _,generated,_,_=compose(donor,pre,method=method)
            np.testing.assert_allclose(generated,post,atol=1e-5)
    def test_unsupported_composition_refused(self):
        p=np.full((49,49),150,np.float32);donor,_=damage_layer(p,p)
        with self.assertRaises(ValueError):compose(donor,p,method='invented_blob')

if __name__=='__main__':unittest.main()
