"""Exact D01 decision replay and per-event proposal/rank forensics. Read-only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from automation.accuracy_physical_dataset import COHORT, REPO, causal_stages, json_read
from automation.accuracy_proposal_forensics import raw_contours, nearest, region_stats
from automation.physical_score_audit import source
from src.engine.offline.causal_candidates import analyze_trace
from src.engine.offline.track_replay import current_exact_replay

RADII = (5, 10, 20, 42)


def distance(candidate, gt):
    return float(np.hypot(candidate['camera_x']-gt['camera_x'], candidate['camera_y']-gt['camera_y']))


def translated(values, origin):
    return [dict(c, camera_x=c.get('camera_x', c.get('cx', 0))+origin[0],
                   camera_y=c.get('camera_y', c.get('cy', 0))+origin[1]) for c in values]


def analyze(output):
    output.mkdir(parents=True, exist_ok=False)
    root = REPO / 'content/ai/physical_traces' / COHORT['D01'][0]
    rows = []
    for sid in range(1, 7):
        path = root / f'shots/shot_{sid:08d}/trace.json'
        trace, gt = json_read(path), json_read(path.parent / 'ground_truth.json')
        decision = trace['decision_input']
        snapshot = decision['complete_track_audit']
        winner = current_exact_replay(snapshot, decision['deterministic_selection'])
        emission = trace['outcome']['final_camera_xy']
        if not all(winner[k] == emission[k] for k in ('camera_x', 'camera_y')):
            raise ValueError('Recorded emission differs from selected point; investigate remapping')
        eligible = [t for t in snapshot['tracks'] if t['eligible']]
        causal = analyze_trace(trace, trace.get('next_audio_peak_ts'), gt)
        stages = []
        for stage in causal_stages(trace):
            debug, pipeline = stage['window_debug'], stage.get('pipeline')
            if not isinstance(pipeline, dict):
                continue
            stamp = pipeline.get('source_frame_ts', debug.get('frame_ts'))
            if stamp is None or not trace['peak_ts'] <= stamp <= decision['timestamp']:
                continue
            origin = (debug.get('v2221_crop_x0', 0), debug.get('v2221_crop_y0', 0))
            maps = {k: np.load(path.parent / v['path'], allow_pickle=False)
                    for k, v in stage['evidence_maps'].items()}
            raw = raw_contours(maps['candidate_mask'], origin) if 'candidate_mask' in maps else []
            upstream = pipeline.get('upstream', {})
            up = upstream.get('pipeline')
            ledgers = {}
            if isinstance(up, dict):
                org = upstream['crop_origin_camera']
                for name in ('pre_limit_candidates', 'retained_candidates', 'rejected_blobs'):
                    values = up.get(name)
                    if isinstance(values, list):
                        values = translated(values, org)
                        if name == 'pre_limit_candidates':
                            values = [dict(c, pre_limit_rank=i) for i, c in enumerate(values, 1)]
                        ledgers[name] = dict(count=len(values), nearest=nearest(values, gt),
                            within42=[c for c in values if distance(c, gt) <= 42])
            for name, values in pipeline.get('cleanup_stages', {}).items():
                ledgers['cleanup_'+name] = dict(count=len(values), nearest=nearest(values, gt),
                    within42=[c for c in values if distance(c, gt) <= 42])
            kept = pipeline.get('retained_candidates', [])
            local = (gt['camera_x']-origin[0], gt['camera_y']-origin[1])
            stages.append(dict(observed_at=stage['timestamp'], source_frame_ts=stamp,
                crop_origin=origin, raw_count=len(raw), nearest_raw=nearest(raw, gt),
                nearest_geometric=nearest([c for c in raw if c['area'] >= 2 and 1 <= c['radius'] <= 24], gt),
                boundaries=ledgers, nearest_retained=nearest(kept, gt),
                source_nearest={s: nearest([c for c in kept if source(c) == s], gt)
                                for s in sorted({source(c) for c in kept})},
                gt_map_stats={k: region_stats(v, local) for k, v in maps.items()
                              if k in ('pre_shot_delta', 'change_map', 'candidate_mask')},
                unavailable=dict(full_hybrid_raw=pipeline.get('raw_candidates', 'UNAVAILABLE'),
                                 upstream=up if not isinstance(up, dict) else None)))
        associations = [r for batch in snapshot['associations'] for r in batch['records']
                        if distance(r['candidate'], gt) <= 10]
        rows.append(dict(event=sid, current_distance=distance(winner, gt), emission_distance=distance(emission, gt),
            current_track_id=winner['track_id'], current_source=winner.get('source'),
            current_exact_replay='MATCH', gt=gt, eligible_count=len(eligible), nearest_eligible=nearest(eligible, gt),
            positive_ranks={str(r): min((t['final_rank'] for t in eligible if distance(t, gt) <= r), default=None) for r in RADII},
            eligible_oracle={str(r): any(distance(t, gt) <= r for t in eligible) for r in RADII},
            useful_tracks=[dict(track_id=t['track_id'], distance=distance(t, gt), rank=t['final_rank'],
                                source=t.get('source'), last_candidate=t.get('last_candidate'),
                                locally_confirmed=(t.get('last_candidate') or {}).get('v2225_local_confirm'))
                           for t in eligible if distance(t, gt) <= 42],
            causal_oracle=causal['causal_oracle'], stages=stages,
            near_gt_associations=associations,
            audio=trace['audio_trigger'], trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    compact = [dict(event=r['event'], track=r['current_track_id'], distance=r['current_distance']) for r in rows]
    report = dict(session='D01_DEVELOPMENT', events=6, physical=6, no_physical=0,
        current_hits={str(r): sum(e['emission_distance'] <= r for e in rows) for r in RADII},
        eligible_oracle={str(r): sum(e['eligible_oracle'][str(r)] for e in rows) for r in RADII},
        causal_oracle={str(r): sum(e['causal_oracle']['hits'][str(r)] for e in rows) for r in RADII},
        exact_replay_matches=len(rows), decision_sha256=hashlib.sha256(json.dumps(compact, sort_keys=True).encode()).hexdigest(),
        rows=rows, limitations=['Recorded selector-input replay, not regenerated detector replay',
            'Full hybrid RAW and discarded FAST/V2 coordinates remain unavailable',
            'No physical label changed and no no-impact event inferred'])
    (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    analyze(parser.parse_args().output)
