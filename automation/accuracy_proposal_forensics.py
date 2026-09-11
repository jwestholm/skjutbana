"""Read-only S02 proposal-miss audit. Reconstructed contours are diagnostic only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from automation.accuracy_physical_dataset import COHORT, REPO, causal_stages, json_read
from automation.physical_score_audit import source


def raw_contours(mask, origin=(0, 0)):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        area = float(cv2.contourArea(contour))
        perimeter = cv2.arcLength(contour, True)
        result.append(dict(camera_x=x+origin[0], camera_y=y+origin[1], area=area,
                           radius=radius, circularity=4*np.pi*area/perimeter**2 if perimeter else 0))
    return result


def nearest(candidates, gt):
    if not candidates:
        return None
    distances = [(float(np.hypot(c['camera_x']-gt['camera_x'], c['camera_y']-gt['camera_y'])), c) for c in candidates]
    d, candidate = min(distances, key=lambda pair:pair[0])
    return dict(distance_px=d, candidate=candidate)


def region_stats(array, local_xy, radius=4):
    x, y = local_xy
    yy, xx = np.ogrid[:array.shape[0], :array.shape[1]]
    selected = (xx-x)**2+(yy-y)**2 <= radius**2
    values = array[selected]
    return dict(mean=float(values.mean()), max=float(values.max()), nonzero=int(np.count_nonzero(values)), pixels=len(values))


def audit(output):
    output.mkdir(parents=True, exist_ok=False)
    root = REPO/'content/ai/physical_traces'/COHORT['S02'][0]
    rows = []
    for sid in (4, 5, 6, 7, 13):
        path = root/f'shots/shot_{sid:08d}/trace.json'
        trace, gt = json_read(path), json_read(path.parent/'ground_truth.json')
        stages = []
        for stage in causal_stages(trace):
            debug = stage.get('window_debug', {})
            origin = (debug.get('v2221_crop_x0', 0), debug.get('v2221_crop_y0', 0))
            local = (gt['camera_x']-origin[0], gt['camera_y']-origin[1])
            maps = {k:np.load(path.parent/ref['path'], allow_pickle=False) for k,ref in stage['evidence_maps'].items()}
            raw = raw_contours(maps['candidate_mask'], origin)
            geometric = [c for c in raw if c['area']>=2 and 1<=c['radius']<=24]
            pipeline = stage.get('pipeline', {})
            retained = pipeline.get('retained_candidates', [])
            source_pools = {name:[c for c in retained if source(c)==name]
                            for name in sorted({source(c) for c in retained})}
            stages.append(dict(timestamp=stage['timestamp'], source_frame_ts=debug.get('frame_ts'),
                source_frame_after_peak=debug.get('frame_ts', trace['peak_ts'])-trace['peak_ts'],
                crop_origin=origin, inside_roi=bool(maps['roi_polygon'][round(local[1]),round(local[0])]),
                map_stats={k:region_stats(v,local) for k,v in maps.items() if k in
                           ('pre_shot_delta','change_map','combined_map','candidate_mask','blackhat_map','whitehat_map')},
                raw_contours_reconstructed=len(raw), nearest_raw_contour=nearest(raw,gt),
                nearest_broad_geometric_contour=nearest(geometric,gt),
                raw_candidate_ledger=pipeline.get('raw_candidates','UNAVAILABLE'),
                retained_nearest=nearest(retained,gt),
                source_nearest={k:nearest(v,gt) for k,v in source_pools.items()},
                generation_and_filter_counters={k:v for k,v in debug.items() if k.startswith(('rej_','raw_blobs','v2222_','v2_','v2225_')) or k in ('candidates_generated','candidates_kept')},
                thresholds=stage['thresholds']))
        rows.append(dict(event=sid, physical_shot={4:3,5:4,6:5,7:6,13:10}[sid],
                         ground_truth=gt, stages=stages,
                         limitations=['Saved candidate mask reconstructs raw contour geometry, not pre-limit scores or discarded V2/FAST candidates',
                                      'Aggregate filter counts cannot identify why an individual contour disappeared',
                                      'Map values near a label are diagnostic; nonzero change alone does not prove a projectile']))
    (output/'forensics.json').write_text(json.dumps(dict(session='S02_DEVELOPMENT_USED',shots=rows),indent=2)+'\n')
    for row in rows:
        s=row['stages'][-1]
        print(row['event'],'ROI',s['inside_roi'],'raw',s['nearest_raw_contour'],
              'broad',s['nearest_broad_geometric_contour'],'delta',s['map_stats']['pre_shot_delta'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    audit(parser.parse_args().output)
