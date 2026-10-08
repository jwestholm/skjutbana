"""Locality measurements are descriptive, bounded, and independent of truth."""
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from automation import physical_ai_dataset as ds
from automation import physical_ai_locality_audit as audit


class LocalityTests(unittest.TestCase):
    def patches(self):
        return np.full((6,49,49),100.,np.float32)

    def test_core_localized_change(self):
        p=self.patches();p[3:,audit.MASKS['core']]-=6
        r=audit.measure(p)
        self.assertEqual(r['concentration'],1.)
        self.assertEqual(r['regions']['core']['signed_change'],-6.)
        self.assertEqual(r['contrast']['signed_change'],-6.)
        self.assertEqual(r['physical_state'],'UNKNOWN')
        self.assertFalse(r['clean_pre_certified'])

    def test_uniform_change_is_area_normalized(self):
        p=self.patches();p[3:]-=6;r=audit.measure(p)
        self.assertNotEqual(audit.MASKS['core'].sum(),audit.MASKS['ring'].sum())
        self.assertEqual(r['concentration'],0.)
        self.assertEqual(r['contrast']['signed_change'],0.)
        self.assertEqual(r['regions']['ring']['pixel_absolute_change'],6.)

    def test_ring_change_and_sign(self):
        p=self.patches();p[3:,audit.MASKS['ring']]+=4
        r=audit.measure(p);self.assertEqual(r['concentration'],-1.)
        self.assertEqual(r['contrast']['signed_change'],-4.)

    def test_unsigned_not_absolute_region_mean(self):
        p=self.patches();yy,xx=np.nonzero(audit.MASKS['core'])
        p[3:,yy[0],xx[0]]+=5;p[3:,yy[1],xx[1]]-=5
        r=audit.measure(p)['regions']['core']
        self.assertEqual(r['pixel_signed_change'],0.)
        self.assertGreater(r['pixel_absolute_change'],0.)

    def test_explicit_missing_or_unstable_ratio(self):
        self.assertIsNone(audit.measure(self.patches())['concentration'])
        self.assertIsNone(audit.measure(self.patches())['stable_signed_core_ring_ratio'])
        short=audit.measure(self.patches()[:5]);self.assertFalse(short['available'])
        self.assertIsNone(short['concentration'])

    def test_bounded_finite_inputs(self):
        self.assertFalse(audit.measure(np.ones((51,49,49)))['available'])
        with self.assertRaises(ValueError):audit.measure(np.ones((6,48,49)))
        p=self.patches();p[0,0,0]=np.nan
        with self.assertRaises(ValueError):audit.measure(p)

    def test_deterministic_no_truth_source_xy_inputs(self):
        self.assertEqual(list(inspect.signature(audit.measure).parameters),['patches'])
        p=self.patches();p[3:,audit.MASKS['core']]-=6
        self.assertEqual(audit.measure(p),audit.measure(p.copy()))
        a=audit.measure(p);b=audit.measure(p+20)
        self.assertEqual(a['concentration'],b['concentration'])
        self.assertEqual(a['contrast']['signed_change'],b['contrast']['signed_change'])

    def test_tail_reversal_not_certified_persistence(self):
        p=self.patches();p[3:,audit.MASKS['core']]-=6;p[4,audit.MASKS['core']]+=12
        self.assertEqual(audit.measure(p)['regions']['core']['tail_coherent_strength'],0.)

    def test_s03_and_output_refusal(self):
        with self.assertRaises(PermissionError):audit.bank('/data/skjutbana/S03/forbidden')
        with self.assertRaises(ValueError):audit.pair('/tmp/not-a-data-output')

    def test_event_checkpoint_resume_preserves_completed_record(self):
        data=dict(dataset_hash='fixed',events=[dict(event='KOMP:1',candidates=[]),dict(event='D02:1',candidates=[])])
        calls=[]
        def extract(d,e,c):
            calls.append(e['event'])
            if calls==['KOMP:1','D02:1']:raise InterruptedError('stop after one durable event')
            return dict(event=e['event'],rows=[],source_hashes={},frames=[])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'pair').mkdir()
            ds.atomic_json(root/'pair/pair.json',dict(identity=dict(contract=audit.CONTRACT)))
            ds.atomic_json(root/'generality_plan.json',dict(frozen=True))
            with patch.object(ds,'output_root',return_value=root),patch.object(audit.fusion,'load',return_value=(data,None)),patch.object(audit,'extract',side_effect=extract):
                with self.assertRaises(InterruptedError):audit.bank(root)
                old=ds.digest(root/'bank/KOMP_1.json')
                audit.bank(root)
                self.assertEqual(old,ds.digest(root/'bank/KOMP_1.json'))
                self.assertEqual(calls,['KOMP:1','D02:1','D02:1'])


if __name__=='__main__':unittest.main()
