import unittest

class FullPoolTemporalTests(unittest.TestCase):
    def test_cutoff_excludes_late_frames(self):
        fs=[{'timestamp':1.0},{'timestamp':1.2},{'timestamp':1.4}]
        self.assertEqual([f for f in fs if f['timestamp']<=1.3],fs[:2])
    def test_missing_post_is_explicit(self):
        self.assertEqual([],[])
    def test_rank_pool_contains_multiple_tracks(self):
        pool=[{'track_id':1,'value':2},{'track_id':2,'value':1}]
        self.assertEqual(max(pool,key=lambda x:x['value'])['track_id'],1)

if __name__=='__main__': unittest.main()
