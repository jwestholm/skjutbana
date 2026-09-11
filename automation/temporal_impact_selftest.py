"""Deterministic tests for research temporal residual measurements."""
import unittest
import numpy as np
from src.engine.offline.registered_impact import prepare_pair, extract

def measure(pre, frames):
    out=[]
    for f in frames:
        p=prepare_pair(pre,f,translation=(0,0)); v=extract(p,(16,16))
        out.append(v['ring_affine_center_dark'])
    return out

class TemporalImpactTests(unittest.TestCase):
    def setUp(self):
        self.pre=np.full((33,33),180,np.uint8)
    def test_unchanged_is_stable(self):
        vals=measure(self.pre,[self.pre.copy() for _ in range(5)])
        self.assertLess(max(vals),.1)
    def test_persistent_dark_impact_has_after_signal(self):
        x=self.pre.copy(); yy,xx=np.ogrid[:33,:33];x[(xx-16)**2+(yy-16)**2<9]=130
        vals=measure(self.pre,[self.pre.copy(),x,x,x])
        self.assertGreater(vals[1],20); self.assertGreater(vals[-1],20)
    def test_transient_change_does_not_persist(self):
        x=self.pre.copy();x[16,16]=80
        vals=measure(self.pre,[self.pre.copy(),x,self.pre.copy(),self.pre.copy()])
        self.assertGreater(vals[1],0); self.assertLess(vals[-1],.1)
    def test_global_brightness_is_not_localized(self):
        vals=measure(self.pre,[np.full((33,33),190,np.uint8) for _ in range(3)])
        self.assertLess(max(vals),.1)
    def test_nearby_second_impact_remains_possible(self):
        old=self.pre.copy(); yy,xx=np.ogrid[:33,:33];old[(xx-14)**2+(yy-16)**2<9]=120
        new=old.copy();new[(xx-19)**2+(yy-16)**2<9]=100
        vals=measure(old,[old,new,new])
        self.assertGreater(vals[1],5)

if __name__=='__main__': unittest.main()
