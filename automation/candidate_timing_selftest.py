"""Timing provenance and deterministic one-worker scheduling; no accuracy tuning."""
from copy import deepcopy
import unittest
from automation.candidate_timing_research import reconstruct, serialize_jobs, simulate


class TimingTests(unittest.TestCase):
    def fixture(self, native=True, moved=False):
        point=[51.,50.] if moved else [50.,50.]
        batches=[]
        for n,(ts,wall) in enumerate(((1.05,1.20),(1.30,1.36))):
            candidate=dict(camera_x=50.,camera_y=50.,timestamp=ts,v2225_local_confirm=n)
            batches.append(dict(dispatch_owner=1,frame_ts=ts,observed_at=wall if native else 9999.,
                records=[dict(track_id=1,candidate=candidate,after={'xy':point if n else [50.,50.]},reason='later_frame_nearest_track' if n else 'no_track_within_merge_radius')]))
        snapshot=dict(event={'peak_ts':1.,'shot_id':1},decision_observed_at=1.40 if native else 9999.,associations=batches,
            tracks=[dict(track_id=1,camera_x=point[0],camera_y=point[1],eligible=True,final_rank=1,first_seen_ts=1.05,last_seen_ts=1.30,
                         hits=2,readiness_inputs={'required_hits':3,'required_span_s':.09})])
        trace=dict(decision_input={'complete_track_audit':snapshot} if native else {},stages=[
            dict(timestamp=1.25,event={'shot_id':1,'created_at':1.01},candidate_pool_shot_id=1,candidates=[batches[0]['records'][0]['candidate']],tracks=[{'last_seen_ts':1.05}]),
            dict(timestamp=1.39,event={'shot_id':1,'created_at':1.01},candidate_pool_shot_id=1,candidates=[batches[1]['records'][0]['candidate']],tracks=[{'last_seen_ts':1.30}])])
        return snapshot,trace

    def test_native_observation_is_not_delivery_and_confirmation_is_separate(self):
        s,t=self.fixture();r=reconstruct(s,t,1.41);c=r['candidates'][0]
        self.assertEqual(c['first_observation']['value'],1.05)
        self.assertEqual(c['exact_xy_available_by']['value'],1.20)
        self.assertIsNone(c['first_delivery']['value'])
        self.assertEqual(c['confirmation_available_by']['value'],1.36)
        self.assertEqual(c['readiness_derived']['value'],1.39)

    def test_reconstructed_wall_time_is_rejected(self):
        s,t=self.fixture(False);r=reconstruct(s,t,1.41)
        self.assertFalse(r['native']);self.assertIsNone(r['selector'])
        self.assertEqual(r['candidates'][0]['exact_xy_available_by']['value'],1.25)
        self.assertNotEqual(r['batches'][0]['available_by']['value'],9999.)

    def test_changed_xy_cannot_inherit_early_track_delivery(self):
        s,t=self.fixture(moved=True);c=reconstruct(s,t,1.41)['candidates'][0]
        self.assertEqual(c['first_observation']['value'],1.05)
        self.assertEqual(c['exact_xy_first_observation']['value'],1.30)
        self.assertEqual(c['exact_xy_available_by']['value'],1.39)
        self.assertGreater(c['exact_xy_available_by']['value'],c['confirmation_available_by']['value'])

    def test_absent_provenance_stays_unknown(self):
        s,_=self.fixture();c=reconstruct(s,None,1.41)['candidates'][0]
        self.assertEqual(c['exact_xy_available_by']['status'],'UNKNOWN')
        self.assertIsNone(c['readiness_derived']['value'])

    def test_future_snapshot_not_used_for_delivery(self):
        s,t=self.fixture(False);t['stages'][0]['timestamp']=2.
        r=reconstruct(s,t,1.41)
        self.assertEqual(r['batches'][0]['available_by']['value'],1.41)
        self.assertEqual(r['batches'][0]['available_by']['status'],'DERIVED')

    def test_timing_reconstruction_is_deterministic_and_immutable(self):
        s,t=self.fixture();old=deepcopy((s,t));a=reconstruct(s,t,1.41)
        self.assertEqual(a,reconstruct(s,t,1.41));self.assertEqual((s,t),old)

    def test_interleaved_jobs_one_worker_and_stable_ties(self):
        jobs=[dict(name='B',release=2.,cost_ms=20),dict(name='A',release=1.,cost_ms=30),
              dict(name='C',release=1.,cost_ms=10),dict(name='D',release=0.,cost_ms=10,depends=['B','C'])]
        out=serialize_jobs(jobs)
        self.assertEqual([x['name'] for x in out],['A','C','B','D'])
        self.assertTrue(all(a['end']<=b['start'] for a,b in zip(out,out[1:])))
        self.assertTrue(all(x['start']>=x['release'] for x in out))

    def test_busy_window_is_not_free_parallel_cpu(self):
        out=serialize_jobs([dict(name='a',release=1.,cost_ms=50)],busy=[(1.02,1.2),(1.22,1.4)])
        self.assertAlmostEqual(out[0]['start'],1.4)

    def test_missing_and_cyclic_dependencies_fail(self):
        for deps in (['missing'],['a']):
            with self.assertRaises(ValueError):serialize_jobs([dict(name='a',release=0.,cost_ms=1,depends=deps)])
        with self.assertRaises(ValueError):serialize_jobs([dict(name='a',release=0.,cost_ms=-1)])

    def test_candidate_and_post_delivery_late(self):
        for name in ('candidate','post'):
            jobs=[dict(name='candidate',release=2. if name=='candidate' else 1.,cost_ms=1),
                  dict(name='post',release=2. if name=='post' else 1.,cost_ms=1),
                  dict(name='final',release=0.,cost_ms=5,depends=['candidate','post'])]
            out=serialize_jobs(jobs);self.assertGreater(out[-1]['end'],1.5)

    def test_superseded_coordinate_cpu_is_not_erased_by_final_pool(self):
        costs=dict(append_ms=1.,setup_ms=1.,patch_ms=4.,inference_ms=3.)
        a=simulate([0.,.1],{(0,0):.05},{(0,0)},.3,costs)
        b=simulate([0.,.1],{(0,0):.05,(1,1):.08},{(0,0)},.3,costs)
        self.assertAlmostEqual(b['total_cpu_ms']-a['total_cpu_ms'],4.)

    def test_repeated_schedule_no_mutation(self):
        jobs=[dict(name='a',release=1.,cost_ms=1),dict(name='b',release=1.,cost_ms=3,depends=['a'])]
        before=deepcopy(jobs);self.assertEqual(serialize_jobs(jobs),serialize_jobs(jobs));self.assertEqual(jobs,before)

    def test_future_or_unordered_evidence_is_rejected(self):
        costs=dict(append_ms=1.,setup_ms=1.,patch_ms=4.,inference_ms=3.)
        for frames in ([0.,.4],[.1,0.],[0.,0.]):
            with self.assertRaises(ValueError):simulate(frames,{(0,0):0.},{(0,0)},.3,costs)


if __name__=='__main__': unittest.main()
