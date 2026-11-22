"""Bounded cache/training and nightly failure contracts, without hardware or /data writes."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from automation.bench import train as t
from automation.bench.candidate_cache import Export, selection
from automation.bench.dataset_manifest import validate
from automation.bench import nightly


def fixture(root, name, seed):
    d=root/name; d.mkdir(); rng=np.random.default_rng(seed); rows=[]
    for g in range(8):
        rows.append(dict(kind='shot', group=str(g), scenario='test'))
        if g == 7: continue  # absence must remain a miss
        for j in range(12):
            rows.append(dict(kind='candidate', group=str(g), producer_has_impact=g!=6,
                label=g<3 and j==0, candidate=dict(score=float(rng.integers(0,10)),
                    signal=float(20 if g<3 and j==0 else rng.integers(0,4)),
                    missing=None if j%3 else 2.0, camera_x=3, timestamp=10)))
    (d/'candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)); return d

class TrainingTests(unittest.TestCase):
    def test_equivalence_sampling_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp); a=fixture(r,'a',1); b=fixture(r,'b',2)
            e=Export(a,r/'cache'); self.assertTrue(np.isnan(e.X).any())
            self.assertNotIn('camera_x',e.manifest['features'])
            ix=selection(e,negative_cap=2); self.assertTrue(np.array_equal(ix,selection(e,negative_cap=2)))
            for g in range(3): self.assertEqual(sum(e.meta['group'][ix]==g),12)
            for g in range(3,6): self.assertEqual(sum(e.meta['group'][ix]==g),2)
            ts,tr,_=t._load([str(a)]); hs,ho,_=t._load([str(b)])
            tr=[x for x in tr if x['producer_has_impact']]; names=t._feature_names(tr)
            X,med=t._matrix(tr,names); mu=X.mean(0); sd=X.std(0); sd=np.where(sd>1e-9,sd,1)
            w,bias=t._fit_logistic((X-mu)/sd,np.array([float(x['label']) for x in tr]))
            H,_=t._matrix(ho,names,med); expected=t._top1(hs,ho,(H-mu)/sd@w+bias)
            got=t.train([str(a)],[str(b)],r/'out',cache_root=r/'cache')['metrics']
            self.assertEqual(got['holdout_top1']['model'],expected)
            self.assertEqual(got['holdout']['shots'],8)
            self.assertEqual(got['holdout']['candidates'],84)
            self.assertFalse(got['resources']['evaluation_sampling'])
            # Same group names in two sources must never cross-match.
            self.assertEqual(t._cached_top1([e,Export(r/'b',r/'cache')],np.zeros(168))['shots'],16)
    def test_cross_source_group_collision_is_not_a_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp); exports=[]
            for name, label, score in [('a',True,1),('b',False,2)]:
                d=r/name; d.mkdir()
                rr=[dict(kind='shot',group='same',scenario='case'),dict(kind='candidate',group='same',label=label,
                    producer_has_impact=True,candidate=dict(score=score))]
                (d/'candidates.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rr))
                exports.append(Export(d,r/'cache'))
            scores=np.array([1,2])
            self.assertEqual(t._cached_top1(exports,scores)['hits'],1)
            self.assertEqual(t._cached_top1(exports,scores,source_scoped=False)['hits'],0)
    def test_training_child_kill_publishes_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp); m=dict(train=[dict(export='a')],holdout=[dict(export='b')],regression=[dict(name='defaults',args=[],env={},baseline='old')])
            with patch.object(nightly,'validate',return_value=(m,dict(input_bytes=1))), patch.object(nightly,'_game_running',return_value=False), patch.object(nightly,'_mem_available_gb',return_value=100), patch.object(nightly.shutil,'disk_usage',return_value=type('D',(),{'free':10**12})()), patch.object(nightly,'_run',side_effect=[0,-9]), patch('automation.bench.__main__.compare',return_value=dict(overall=dict(verdict='NO_SIGNIFICANT_DIFFERENCE'))):
                self.assertEqual(nightly.main(['--root',str(r)]),1)
            text=(r/'LATEST.md').read_text()
            self.assertIn('FAILED',text); self.assertIn('exit=-9',text)
    def test_regression_requires_ai_settings_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp); settings=r/'settings.json'; settings.write_text('{}')
            ai=r/'ai.json'; ai.write_text('{}'); base=r/'base'; base.mkdir()
            from automation.bench.candidate_cache import file_hash
            m=dict(settings_sha256=file_hash(settings),ai_settings_path=str(ai),background=[])
            (base/'results.json').write_text(json.dumps(dict(manifest=m)))
            manifest=dict(regression=[dict(name='a',baseline=str(base),baseline_sha256='present',args=['--settings',str(settings)])])
            self.assertIn('a',nightly._regression_preflight(manifest))
            m['ai_settings_sha256']=file_hash(ai)
            (base/'results.json').write_text(json.dumps(dict(manifest=m)))
            self.assertEqual(nightly._regression_preflight(manifest),{})
            ai.write_text('{"changed":true}')
            self.assertIn('a',nightly._regression_preflight(manifest))
    def test_manifest_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp); a=fixture(r,'a',1); b=fixture(r,'b',2)
            from automation.bench.candidate_cache import file_hash
            m=dict(version=1,train=[dict(export=str(a),sha256=file_hash(a/'candidates.jsonl'))],
                holdout=[dict(export=str(b),sha256=file_hash(b/'candidates.jsonl'))])
            p=r/'manifest.json'; p.write_text(json.dumps(m)); validate(p)
            (a/'candidates.jsonl').write_text('{}')
            with self.assertRaises(ValueError): validate(p)
            with self.assertRaises(ValueError): t.train([str(b)],[str(b)],r/'bad')
    def test_nightly_failure_is_latest_and_dry_run_no_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp); m=dict(train=[dict(export='a')],holdout=[dict(export='b')],regression=[])
            info=dict(exports=2,input_bytes=100,manifest_sha256='x')
            with patch.object(nightly,'validate',return_value=(m,info)):
                self.assertEqual(nightly.main(['--root',str(r/'dry'),'--dry-run']),0)
                self.assertFalse((r/'dry').exists())
                with patch.object(nightly,'_game_running',return_value=False), patch.object(nightly,'_mem_available_gb',return_value=100), patch.object(nightly.shutil,'disk_usage',return_value=type('D',(),{'free':10**12})()):
                    self.assertEqual(nightly.main(['--root',str(r/'run')]),1)
                self.assertIn('FAILED',(r/'run/LATEST.md').read_text())
                self.assertIn('No fixed regression suite',(r/'run/LATEST.md').read_text())
    def test_not_comparable_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp); m=dict(train=[],holdout=[],regression=[dict(name='defaults',args=[],env={},baseline='old')])
            with patch.object(nightly,'validate',return_value=(m,dict(input_bytes=1))), patch.object(nightly,'_game_running',return_value=False), patch.object(nightly,'_mem_available_gb',return_value=100), patch.object(nightly.shutil,'disk_usage',return_value=type('D',(),{'free':10**12})()), patch.object(nightly,'_run',return_value=0), patch('automation.bench.__main__.compare',side_effect=SystemExit('settings differ')):
                self.assertEqual(nightly.main(['--root',str(r)]),1)
            self.assertIn('NOT_COMPARABLE',(r/'LATEST.md').read_text())

if __name__=='__main__': unittest.main()
