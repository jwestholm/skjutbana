"""Regression: cleanup trace capture must preserve rejection evidence without policy changes."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.engine.camera import hit_scanner_v2222 as cleanup
from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2


class ProposalTraceTests(unittest.TestCase):
    def scanner(self, tracing, write_ledger=True, full_frame=False):
        def detect(scanner, gray, timestamp):
            if write_ledger:
                scanner.last_trace_pipeline=dict(rejected_blobs=[dict(cx=4,cy=5,reason='area')],
                    pre_limit_candidates=[dict(camera_x=4,camera_y=5,score=20)])
            return [dict(camera_x=104.,camera_y=205.,score=20.,pre_shot_change=.5),
                    dict(camera_x=150.,camera_y=250.,score=10.,pre_shot_change=10.),
                    dict(camera_x=180.,camera_y=280.,score=9.,pre_shot_change=8.)]
        class Scanner:
            _detect_frame_candidates=detect
            _on_audio_peak=disable=_frame_roi_mask=_resolve_audio_events=update=lambda *args:None
        module=SimpleNamespace(HitScanner=Scanner,camera_manager=SimpleNamespace())
        with patch('importlib.import_module',return_value=module),patch.object(cleanup,'_INSTALLED',False):
            cleanup.install_v2222_hit_scanner_patch()
        scanner=Scanner()
        scanner.physical_trace_capture_enabled=tracing
        scanner.last_trace_pipeline=dict(previous_event_secret='must not carry forward')
        scanner.last_window_debug=dict(v2221_crop_x0=100,v2221_crop_y0=200,
                                      v2221_geometry_mode='full_frame_fallback' if full_frame else 'homography')
        scanner.known_holes=[];scanner.candidate_limit=2
        return scanner

    def detect(self, scanner):
        with patch.object(cleanup,'_runtime_settings',return_value=dict(analysis_horizontal_ridge_filter_v2222_enabled=False,
                                                                       analysis_v2222_log=False)):
            return scanner._detect_frame_candidates(np.zeros((2,2),np.uint8),12.)

    def test_capture_does_not_change_candidates_or_scores(self):
        off=self.scanner(False);on=self.scanner(True)
        self.assertEqual(self.detect(off),self.detect(on))
        self.assertEqual(off.last_window_debug['v2222_novelty_demoted'],on.last_window_debug['v2222_novelty_demoted'])

    def test_upstream_ledger_and_cleanup_boundaries_survive(self):
        scanner=self.scanner(True);result=self.detect(scanner);pipeline=scanner.last_trace_pipeline
        self.assertEqual(pipeline['upstream']['pipeline']['rejected_blobs'][0]['reason'],'area')
        self.assertEqual(pipeline['upstream']['crop_origin_camera'],[100,200])
        self.assertEqual(pipeline['cleanup_stages']['input'][0]['score'],20)
        demoted=next(c for c in pipeline['cleanup_stages']['after_novelty'] if c['camera_x']==104)
        self.assertEqual(demoted['score'],2)
        self.assertEqual(len(pipeline['cleanup_stages']['after_ridge']),3)
        self.assertEqual(len(pipeline['retained_candidates']),2)
        frozen=copy.deepcopy(pipeline);result[0]['score']=999
        self.assertEqual(pipeline,frozen)
        self.assertEqual(pipeline['source_frame_ts'],12.)
        self.assertEqual(pipeline['raw_candidates'],'UNAVAILABLE')

    def test_early_return_does_not_borrow_previous_ledger(self):
        scanner=self.scanner(True,write_ledger=False);self.detect(scanner)
        self.assertEqual(scanner.last_trace_pipeline['upstream']['pipeline'],'UNAVAILABLE')

    def test_full_frame_fallback_does_not_reuse_stale_crop_origin(self):
        scanner=self.scanner(True,full_frame=True);self.detect(scanner)
        self.assertEqual(scanner.last_trace_pipeline['upstream']['crop_origin_camera'],[0,0])


class HybridTraceTests(unittest.TestCase):
    def merge(self, legacy, v2, tracing=True, **config):
        scanner = SimpleNamespace(candidate_limit=3, physical_trace_capture_enabled=tracing,
                                  last_trace_pipeline={'legacy_marker': True})
        result = CandidateGeneratorV2.__new__(CandidateGeneratorV2)._merge_hybrid(
            scanner=scanner, legacy=legacy, v2=v2,
            cfg=dict(v2_reserved_slots=1, legacy_reserved_slots=2, **config))
        return result, scanner.last_trace_pipeline

    def test_exact_cap_reason_and_observational_equivalence(self):
        legacy = [dict(camera_x=i*20., camera_y=0., score=4.-i) for i in range(3)]
        v2 = [dict(camera_x=100., camera_y=0., score=10.)]
        before = copy.deepcopy((legacy, v2))
        on, ledger = self.merge(legacy, v2)
        off, _ = self.merge(legacy, v2, False)
        self.assertEqual(on, off)
        self.assertEqual((legacy, v2), before)
        row = ledger['hybrid_merge']['records'][2]
        self.assertEqual(row['input_id'], 'legacy:2')
        self.assertFalse(row['retained'])
        self.assertEqual(row['operations'], [])
        self.assertEqual(row['retention_operation'], 'hybrid_capacity_exhausted')
        self.assertEqual(row['merged_rank'], 4)
        on[0]['score'] = -999
        self.assertEqual(ledger['hybrid_merge']['records'][3]['output_candidate']['score'], 10.)

    def test_geometry_replacement_has_both_input_identities(self):
        legacy = [dict(camera_x=10., camera_y=20., score=1.)]
        v2 = [dict(camera_x=11., camera_y=20., score=8.)]
        on, ledger = self.merge(legacy, v2)
        self.assertEqual(on, self.merge(legacy, v2, False)[0])
        for row in ledger['hybrid_merge']['records']:
            self.assertEqual(row['operations'][0]['geometry_from'], 'v2')
            self.assertEqual(row['output_candidate']['camera_x'], 11.)
            self.assertTrue(row['retained'])

    def test_v2_neighbors_do_not_create_false_legacy_agreement(self):
        v2 = [dict(camera_x=x, camera_y=20., score=8.) for x in (10., 11.)]
        result, ledger = self.merge([], v2)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(not r['operations'] for r in ledger['hybrid_merge']['records']))

    def test_disabled_hybrid_records_explicit_bypass(self):
        c = dict(camera_x=10., camera_y=20., score=1.)
        result, ledger = self.merge([c], [c], hybrid_with_legacy=False)
        self.assertEqual(result, [c])
        self.assertEqual(ledger['hybrid_merge']['records'][0]['retention_operation'], 'legacy_disabled')


if __name__=='__main__':unittest.main()
