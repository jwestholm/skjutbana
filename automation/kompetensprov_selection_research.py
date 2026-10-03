"""Bounded partial-capture diagnosis, NOT a live-path replay or new detector.

Only explicitly bound Kompetensprov inputs are read. Proposals are frozen before
labels are opened. Missing history/timestamps are never filled with synthetic
camera timestamps. The observed-step diagnostic deliberately cannot certify a
causal onset: its adjacency/gap/deadline provenance is unavailable.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import patch as mock

import cv2
import numpy as np

from automation.causal_pre_onset_research import digest, read_guard, sample, write
from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2
from src.engine.camera.analysis_geometry_v2221 import build_perspective_geometry_v2221
from src.engine.offline.clean_physical_change import local_contrast
from src.engine import shot_region_proposal_v251 as region
from src.engine import shot_region_freshness_v252 as fresh
from src.engine import shot_cross_thread_novelty_v253 as novelty
from src.engine.shot_fast_v2225 import local_confirm_candidates_v2225

BASE = Path('/data/skjutbana/evaluation_runs')
RUN = BASE/'kompetensprov_capture_20260926_215855'
SESSION = Path('/data/skjutbana/physical_traces/session_20260926_215856_KOMPETENSPROV_AK_906234cc')
CROP = (1541, 987, 1483, 711)
RADII = (5, 10, 20, 42)


def ordinal_steps(signal):
    """Compatibility name for the shared ordinal-only measurement."""
    from src.engine.offline.clean_physical_change import observed_local_step
    return observed_local_step(signal)


def crop(im):
    x, y, w, h = CROP
    return im[y:y+h, x:x+w]


def xy(c):
    return [float(c['camera_x'])+CROP[0], float(c['camera_y'])+CROP[1]]


def clean(c):
    # API timestamps are rounded RECORDED log values, but reconstructed
    # confirmation used a different, un-timestamped frame: do not export them.
    return {k: v for k, v in c.items() if k != 'timestamp' and not k.endswith('_ts')}


class MapsReady(Exception):
    pass


def prepare_maps(pre_frames, current, mask, sid, frame_ts, peak_ts):
    """Stop the existing generator before hybrid/banking/debug side effects."""
    engine = CandidateGeneratorV2()
    scanner = SimpleNamespace(audio_events=[SimpleNamespace(state='pending', shot_id=sid, peak_ts=peak_ts)],
        _frame_roi_mask=lambda shape: mask, artifact_suppression_mask=None,
        _v244_roi_diag=dict(crop=CROP, scale=(1.,1.), work_shape=(CROP[3],CROP[2])))
    def pre_override(_scanner, *, bbox, **kwargs):
        x0, y0, x1, y1 = bbox
        return [im[y0:y1, x0:x1] for im in pre_frames]
    engine._collect_pre_frames = pre_override
    maps = {}
    def intercept(**kwargs):
        maps.update(kwargs)
        raise MapsReady
    engine._extract_candidates = intercept
    try:
        engine.generate(scanner, current, frame_ts, [])
    except MapsReady:
        pass
    if not maps:
        raise RuntimeError('Generator did not reach proposal stage')
    return engine, scanner, maps


def context(engine, maps, sid, group):
    return fresh.RegisteredFrameV252(sid, maps['frame_ts'], maps['bbox'],
        maps['absdiff'], maps['zscore'], maps['darkening'],
        engine._shot_models[sid]['temporal_noise'], (group,))


def stages(candidates, ctx, scanner, sid, peak, pre, post):
    annotated = fresh._normalise_groups([fresh._annotate_candidate(c, ctx) for c in candidates])
    # Original cross-shot confirmed history did not survive. Explicit missing-
    # history conditional calculation; NEVER described as recorded final rank.
    with mock.object(novelty.shot_bridge_v253, 'history', return_value=[]):
        annotated = novelty.annotate_cross_shot_novelty_v253(scanner, sid, peak, annotated)
    confirmed, telemetry = local_confirm_candidates_v2225(pre, post, annotated, frame_ts=ctx.frame_ts)
    balanced = novelty.balance_confirmed_v253(confirmed)
    return dict(proposal=[clean(c) for c in candidates], freshness=[clean(c) for c in annotated],
        confirmed=[clean(c) for c in confirmed], balanced=[clean(c) for c in balanced],
        tested=int(telemetry['tested']))


def metrics(candidates, gt):
    if not candidates:
        return dict(count=0, nearest=None, oracle={str(r): False for r in RADII})
    distances = [float(np.linalg.norm(np.asarray(xy(c))-gt)) for c in candidates]
    i = int(np.argmin(distances))
    return dict(count=len(candidates), nearest=distances[i], nearest_xy=xy(candidates[i]),
        nearest_list_rank=i+1, oracle={str(r): distances[i] <= r for r in RADII},
        first_xy=xy(candidates[0]), first_error=distances[0])


def evidence(paths, point, snapshot):
    values = np.array([local_contrast(sample(np.load(p, mmap_mode='r'), [point]))[0] for p in paths])
    pre = values[:50]; post = values[50:]
    step = ordinal_steps(pre)
    result = dict(pre_signal=pre.tolist(), post_signal=post.tolist(), ordinal_step=step,
        pre_range=float(np.ptp(pre)), snapshot_contrast=float(local_contrast(sample(snapshot, [point]))[0]),
        early3_to_late3_signed=float(np.median(pre[-3:])-np.median(pre[:3])),
        snapshot_to_first_post=float(post[0]-pre[45]),
        snapshot_to_last3post=float(np.median(post[-3:])-pre[45]),
        post_range=float(np.ptp(post)), post_mad=float(np.median(np.abs(post-np.median(post)))))
    if step['state'] == 'UNIQUE_PERSISTENT_OBSERVED_STEP':
        c = step['candidates'][0]; k = c['index']
        result['all_post_support_same_step'] = float(np.min(np.sign(c['signed_step'])*(post-pre[k-1]))/c['noise'])
        result['snapshot_after_step'] = k <= 45
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); out = args.output.resolve()
    if not out.is_relative_to(BASE) or out.is_relative_to(RUN) or 'S03' in str(out).upper():
        raise ValueError('Use a fresh evaluation output outside the original capture')
    out.mkdir(parents=True, exist_ok=False)
    accesses = read_guard(out, [SESSION])
    cv2.setNumThreads(1)
    log = (RUN/'runtime.log').read_text()
    configuration = json.loads((RUN/'saved_configuration.json').read_text())
    g = configuration['geometry']
    geometry = build_perspective_geometry_v2221((2160,3840), g['camera_calibration']['inverse_homography'], g['viewport'])
    assert (geometry.crop_x0, geometry.crop_y0, geometry.crop_width, geometry.crop_height) == CROP
    global_mask = geometry.safe_mask_local
    protocol = dict(status='OFFLINE_RECONSTRUCTION_NOT_LIVE_REPLAY', caps=[8,16,32,64],
        cap64='offline only; production setting is clamped to 32',
        reference='last three saved PRE-history frames; true live stack identity unavailable',
        proposal_frame='last saved blackhat map matched by exact recomputation to a saved POST',
        confirmation='next saved POST ordinal, actual live confirmation frame unavailable',
        history='cross-shot novelty history unavailable; conditional ranks assume empty history',
        other_missing='known-hole state, artifact suppression, hybrid/bank, cleanup, track ledger',
        configuration='current source/config hashes; no claim of complete capture-time runtime settings',
        effective_detector_config=CandidateGeneratorV2().config.snapshot(),
        global_mask_fraction=float(np.mean(global_mask > 0)), recorded_global_fraction_rounded=.796,
        global_mask='saved geometry via existing helper; does not exactly reproduce logged global area',
        final_live_ranks='UNAVAILABLE', authority_changed=False,
        ordinal_step='amplitude-only frozen onset criteria; no gap/deadline or acquisition-time certification',
        joint_comparison='fixed existing V252 metric quality, region_evidence=0; no region/source bonus; no fit',
        gt='opened only after all coordinate pools are frozen')
    write(out/'protocol.json', protocol)
    pools = []; inputs = {str(RUN/'runtime.log'), str(RUN/'saved_configuration.json')}
    settings = json.loads(Path('content/ai/settings.json').read_text())
    with ExitStack() as stack:
        for module in (region, fresh, novelty):
            stack.enter_context(mock.object(module, '_runtime_settings', return_value=settings))
        for sid in range(1,7):
            shot = SESSION/'shots'/f'shot_{sid:08d}'
            prepaths = sorted((shot/'frames').glob('pre_history_*.npy'))
            postpaths = sorted((shot/'frames').glob('post_*.npy'))
            snapshot_path = next((shot/'frames').glob('pre_snapshot*.npy'))
            snapshot = np.load(snapshot_path, mmap_mode='r')
            assert len(prepaths) == 50 and len(postpaths) == 62
            snapshot_matches = [i for i,p in enumerate(prepaths) if np.array_equal(np.load(p, mmap_mode='r'), snapshot)]
            map_path = sorted((shot/'maps').glob('*_blackhat_map.npy'))[-1]
            target = np.load(map_path)
            matches = []
            for i,p in enumerate(postpaths):
                bh = cv2.morphologyEx(cv2.GaussianBlur(crop(np.load(p, mmap_mode='r')), (5,5),0),
                     cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
                if np.array_equal(bh, target): matches.append(i)
            if len(matches) != 1: raise RuntimeError('Ambiguous map/frame identity')
            idx = matches[0]
            post = crop(np.load(postpaths[idx], mmap_mode='r'))
            confirm = crop(np.load(postpaths[idx+1], mmap_mode='r'))
            pre = [crop(np.load(p, mmap_mode='r')) for p in prepaths[-3:]]
            localmask_path = sorted((shot/'maps').glob('*_roi_polygon.npy'))[-1]
            localmask = np.load(localmask_path)
            bx, by, bw, bh = cv2.boundingRect(localmask)
            group = region.RegionGroupV251('kompetensprov_hf2020', ('kompetensprov_hf2020',), (), bx, by, bx+bw-1, by+bh-1)
            match = re.search(rf'READY-BRIDGE\] shot={sid} source=worker frame=([\d.]+) peak=([\d.]+)', log)
            frame_ts, peak = map(float, match.groups())
            engine, scanner, maps = prepare_maps(pre, post, localmask, sid, frame_ts, peak)
            ctx = context(engine, maps, sid, group)
            valid = region._bbox_region_mask(maps['saliency'].shape, maps['valid'], maps['bbox'], group)
            threshold, _ = engine._robust_threshold(maps['saliency'], valid=valid, cfg=maps['cfg'])
            cases = {}
            for cap in (8,16,32,64):
                with mock.object(region, '_per_region_proposals', return_value=cap):
                    candidates, telemetry = region._region_physical_candidates(engine, scanner, group,
                        **{k: maps[k] for k in ('saliency','absdiff','darkening','dog','zscore','bbox','frame_ts','cfg')},
                        region_valid=valid, region_threshold=threshold, shot_id=sid)
                    # No legacy hybrid is invented. Rebalancing still applies
                    # to the registered proposals available in this reconstruction.
                    candidates = region._balance_merged_candidates(candidates, (group,), shot_id=sid)
                cases[str(cap)] = stages(candidates, ctx, scanner, sid, peak, crop(snapshot), confirm)
                cases[str(cap)]['proposal_telemetry'] = telemetry
                if cap == 32 and len(candidates) < cap:
                    break  # candidate pool exhausted; 64 cannot add anything
            ge, gs, gm = prepare_maps(pre, post, global_mask, sid, frame_ts, peak)
            global_candidates = CandidateGeneratorV2._extract_candidates(ge, **gm)
            globalgroup = region.RegionGroupV251('global', ('global',), (), 0,0,CROP[2]-1,CROP[3]-1)
            gc = context(ge, gm, sid, globalgroup)
            # Compare union on ONE global reference/context, not scores from
            # differently normalized regional and global proposal channels.
            union = {tuple(xy(c)):dict(c) for c in global_candidates + cases['8']['proposal']}
            joint = []
            for c in union.values():
                m = fresh._best_metrics_for_candidate(c, gc)
                if m is not None and fresh._fresh_gate(m):
                    joint.append(dict(c, joint_physical_score=fresh._metric_quality(m, 0.)))
            joint.sort(key=lambda c:c['joint_physical_score'], reverse=True)
            row = dict(shot_id=sid, snapshot_pre_ordinals=snapshot_matches, proposal_post_ordinal=idx,
                proposal_map=str(map_path), confirmation_post_ordinal=idx+1,
                local_mask_bounds=[bx,by,bx+bw-1,by+bh-1], cases=cases,
                global_candidates=[clean(c) for c in global_candidates], joint=[clean(c) for c in joint])
            pools.append(row)
            inputs.update(str(p) for p in prepaths+postpaths+[snapshot_path, map_path, localmask_path])
            print(f'Frozen shot {sid}: caps '+str({k:len(v['proposal']) for k,v in cases.items()})+f', global {len(global_candidates)}', flush=True)
    write(out/'frozen_candidate_pools.json', pools)
    results = []
    for pool in pools:
        sid = pool['shot_id']; shot = SESSION/'shots'/f'shot_{sid:08d}'
        label_path = shot/'ground_truth.json'; label = json.loads(label_path.read_text()); inputs.add(str(label_path))
        gt = [label['camera_x'],label['camera_y']]
        emitted = next(json.loads(line.split('] ',1)[1]) for line in log.splitlines()
                       if line.startswith('[KOMPETENSPROV-PHYSICAL] ') and json.loads(line.split('] ',1)[1])['shot_id'] == sid)
        winner = [emitted['camera_x'],emitted['camera_y']]
        paths = sorted((shot/'frames').glob('pre_history_*.npy'))+sorted((shot/'frames').glob('post_*.npy'))
        snapshot = np.load(next((shot/'frames').glob('pre_snapshot*.npy')), mmap_mode='r')
        row = dict(shot_id=sid, gt=gt, emitted=winner, recorded_error=float(np.linalg.norm(np.array(gt)-winner)),
            local_caps={k:{s:metrics(v[s],gt) for s in ('proposal','freshness','confirmed','balanced')} for k,v in pool['cases'].items()},
            global_pool=metrics(pool['global_candidates'],gt), joint=metrics(pool['joint'],gt),
            gt_evidence=evidence(paths,gt,snapshot), winner_evidence=evidence(paths,winner,snapshot))
        # Frozen candidate witnesses, not GT-centered generated coordinates.
        points = sorted(set(tuple(xy(c)) for case in pool['cases'].values() for c in case['proposal']) |
                        set(tuple(xy(c)) for c in pool['global_candidates']))
        signals = np.array([local_contrast(sample(np.load(p,mmap_mode='r'),points)) for p in paths[:50]])
        row['candidate_steps'] = [dict(xy=list(p), error=float(np.linalg.norm(np.array(p)-gt)), **ordinal_steps(signals[:,j]))
                                  for j,p in enumerate(points)]
        # Registered evidence at the two diagnostic locations uses the same
        # global context, last-three PRE convention and matched proposal frame.
        pre = [crop(np.load(p,mmap_mode='r')) for p in paths[47:50]]
        post = crop(np.load(paths[50+pool['proposal_post_ordinal']],mmap_mode='r'))
        match = re.search(rf'READY-BRIDGE\] shot={sid} source=worker frame=([\d.]+) peak=([\d.]+)',log)
        ge,gs,gm = prepare_maps(pre,post,global_mask,sid,*map(float,match.groups()))
        gc = context(ge,gm,sid,region.RegionGroupV251('global',('global',),(),0,0,CROP[2]-1,CROP[3]-1))
        for name,p in [('gt',gt),('winner',winner)]:
            m = fresh._best_metrics_for_candidate(dict(camera_x=p[0]-CROP[0],camera_y=p[1]-CROP[1]),gc)
            row[name+'_registered'] = dict(metrics=asdict(m), quality=fresh._metric_quality(m,0.), fresh=fresh._fresh_gate(m)) if m else None
        results.append(row)
    write(out/'results.json',results)
    # Exact duplicate maps from earlier shots are reported; they are not
    # treated as current evidence merely because a writer saved them here.
    owners = {}; map_provenance=[]
    for sid in range(1,7):
        for p in sorted((SESSION/'shots'/f'shot_{sid:08d}'/'maps').glob('*.npy')):
            h=digest(p); earlier=owners.get((p.name.split('_',1)[1],h))
            map_provenance.append(dict(path=str(p),sha256=h,identical_earlier=earlier))
            owners.setdefault((p.name.split('_',1)[1],h),str(p));inputs.add(str(p))
    write(out/'map_provenance.json',map_provenance)
    sources = [Path(__file__), Path('content/ai/detector_v2.json'), Path('content/ai/settings.json')]
    sources += [Path(m.__file__) for m in (region,fresh,novelty)]
    sources += [Path('src/engine/camera/candidate_generator_v2.py'),Path('src/engine/shot_fast_v2225.py'),
                Path('src/engine/camera/analysis_geometry_v2221.py'),Path('src/engine/offline/accuracy_verifier.py'),
                Path('src/engine/offline/clean_physical_change.py'),Path('automation/causal_pre_onset_research.py')]
    write(out/'hashes.json',dict(inputs={p:digest(p) for p in sorted(inputs)},sources={str(p):digest(p) for p in sources}))
    write(out/'read_accesses.json',sorted(accesses))


if __name__ == '__main__':
    main()
