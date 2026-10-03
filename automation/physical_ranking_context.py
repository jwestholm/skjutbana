"""Whole-event and temporal forensics using accepted derived caches only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np

from automation.physical_ranking_families import (install_read_guard,load_inputs,
    frozen_contributions,family_totals,family,tied_ranks)
from automation.physical_family_rank_research import rank_columns,shape_evidence,motion_residual_evidence
from automation.physical_counterfactual_forensics import auc,distribution,decode
from automation.physical_counterfactual_models import universe
from automation.physical_counterfactual_research import SESSIONS,RADII,write_json
from automation.physical_hypothesis_research import digest


def ready_observed_ids(event, records):
    """Same plausible observed universe as the atlas, not every raw mask speck."""
    return [i for i in universe(event, records) if records[i]['ready']]


def spatial_components(points,radius=12.):
    """Fixed association-scale connected components; descriptive, not impacts."""
    points=np.asarray(points);remaining=set(range(len(points)));groups=[]
    distance=np.sum((points[:,None]-points[None])**2,axis=2)
    while remaining:
        todo=[min(remaining)];remaining.remove(todo[0]);group=[]
        while todo:
            i=todo.pop();group.append(i)
            neighbors=sorted(j for j in remaining if distance[i,j]<=radius**2)
            remaining.difference_update(neighbors);todo.extend(neighbors)
        groups.append(sorted(group))
    return groups


def event_descriptors(points,families,family_values,shape,motion,crop):
    """Pure GT-free summaries. Spatial proximity never vetoes or supplies truth."""
    ranks=rank_columns(family_values)
    physical=[families.index(g) for g in ('new_change','temporal','localization','morphology')]
    support=(ranks[:,physical]>=.75).sum(axis=1)
    evidence=np.mean(ranks[:,physical],axis=1)
    components=spatial_components(points)
    cluster=np.array([max(evidence[g]) for g in components])
    weight=cluster**2;weight/=weight.sum()
    entropy=-np.sum(weight*np.log(weight))/max(1.,math.log(len(weight)))
    point_weight=evidence**2;point_weight/=point_weight.sum()
    center=np.sum(np.asarray(points)*point_weight[:,None],axis=0)
    dispersion=np.sqrt(np.sum(point_weight*np.sum((points-center)**2,axis=1)))/math.hypot(*crop[2:])
    best=int(np.argmax(evidence));ordered=np.sort(cluster)
    return dict(candidate_count=len(points),spatial_components=len(components),
        supported_components=sum(any(support[j]>=3 for j in g) for g in components),
        best_family_agreement=int(np.max(support)),best_physical_rank=float(evidence[best]),
        best_family_median=float(np.median(ranks[best])),
        best_cluster_mass_fraction=float(max(weight)),cluster_entropy=float(entropy),
        evidence_dispersion_crop=float(dispersion),cluster_margin=float(ordered[-1]-ordered[-2]) if len(ordered)>1 else 0.,
        best_temporal_shape=float(np.mean(shape[best])),max_temporal_shape=float(np.max(np.mean(shape,axis=1))),
        best_motion_residual=float(motion[best]),median_motion_residual=float(np.median(motion)))


def descriptive_separability(rows):
    names=list(rows[0]['features']);descriptions=[];held=[]
    for name in names:
        positives=[r['features'][name] for r in rows if r['physical']]
        negatives=[r['features'][name] for r in rows if not r['physical']]
        descriptions.append(dict(feature=name,positive=distribution(positives),negative=distribution(negatives),
            auc_positive_higher=auc(positives,negatives)))
        for session in SESSIONS:
            train=[r for r in rows if r['session']!=session]
            pos=[r['features'][name] for r in train if r['physical']]
            neg=[r['features'][name] for r in train if not r['physical']]
            a=auc(pos,neg);direction=1 if a>=.5 else -1
            test=[r for r in rows if r['session']==session]
            p=[direction*r['features'][name] for r in test if r['physical']]
            n=[direction*r['features'][name] for r in test if not r['physical']]
            # This is a descriptive interval-overlap probe, not an installed gate.
            boundary=max(direction*v for v in neg)
            held.append(dict(feature=name,held_out=session,training_sessions=[s for s in SESSIONS if s!=session],
                direction_from_training=direction,held_auc=auc(p,n),training_negative_upper=boundary,
                hypothetical_false_rejections=[r['event'] for r in test if r['physical'] and direction*r['features'][name]<=boundary],
                hypothetical_false_acceptances=[r['event'] for r in test if not r['physical'] and direction*r['features'][name]>boundary]))
    return dict(distributions=descriptions,held_session_descriptive_bounds=held,
        limitation='Only four negatives from two sessions; directions/bounds are diagnostic and no live threshold is proposed')


def run(base,prior,ranking,output):
    output.mkdir(parents=True,exist_ok=False);accessed=install_read_guard(output)
    data,matrix,details,models,hashes=load_inputs(base,prior);records=data['records'];names=data['feature_names']
    protocol=json.loads((ranking/'protocol.json').read_text())
    for name,expected in protocol['temporal_cache']['input_hashes'].items():
        if digest(Path(name))!=expected:raise ValueError('Derived temporal cache changed')
    steps=np.load(base/'change_step/step_features.npy',allow_pickle=False)
    step_names=json.loads((base/'change_step/protocol.json').read_text())['feature_names']
    shape=shape_evidence(steps,step_names);motion=motion_residual_evidence(matrix,names)
    atlas=[];events=[];temporal=[];motion_pairs=[];contribution_audit=[];coordinate_evidence=[]
    for e in data['events']:
        model=models[e['session']];contrib=frozen_contributions(matrix,model)
        families,values=family_totals(contrib,[names[j] for j in model['meta']['feature_indices']])
        ids=[i for i in e['pools']['current'] if records[i]['ready']]
        features=event_descriptors(np.array([[records[i]['camera_x'],records[i]['camera_y']] for i in ids]),
            families,values[ids],shape[ids],motion[ids],e['context']['crop'])
        registration=e['context']['registrations'];half=e['motion']['piecewise']
        features['global_registration_mean']=float(np.mean([math.hypot(r['dx'],r['dy']) for r in registration]))
        features['crop_half_disagreement']=float(np.mean([math.hypot(a['dx']-b['dx'],a['dy']-b['dy']) for a,b in half]))
        magnitudes=[np.mean([math.hypot(r['dx'],r['dy']) for r in pair]) for pair in half]
        features['crop_motion_late_minus_early']=float(np.mean(magnitudes[len(magnitudes)//2:])-np.mean(magnitudes[:max(1,len(magnitudes)//2)]))
        for channel in ('persistent','pre_rms','pre_centered','piecewise'):
            features['global_'+channel+'_mass_retention']=e['clean_metrics'][channel]['retained_mass']
        events.append(dict(event=e['event'],session=e['session'],physical=e['truth']['state']=='SINGLE_IMPACT',features=features,
            caveats=e['truth']['caveats'],motion_semantics='Image crop halves, not calibrated physical panels',
            confirmation_count=sum(bool(records[i]['local_confirmation']) for i in ids)))
        if e['truth']['state']!='SINGLE_IMPACT':continue
        observed=ready_observed_ids(e,records)
        good=min(observed,key=lambda i:records[i]['gt_distance']);winner=e['current_id']
        probes=[i for i in e['indices'] if records[i]['tags']==['diagnostic_gt_only'] and not records[i]['selectable']]
        if probes:
            gt=probes[0]
            coordinate_evidence.append(dict(event=e['event'],nearest_observed_id=good,
                nearest_observed_error=records[good]['gt_distance'],diagnostic_gt_probe_id=gt,
                gt_probe_is_never_selectable=True,
                nearest_features=dict(zip(names,decode(matrix[good]).tolist())),
                gt_probe_features=dict(zip(names,decode(matrix[gt]).tolist())),
                nearest_step=dict(zip(step_names,decode(steps[good]).tolist())),
                gt_probe_step=dict(zip(step_names,decode(steps[gt]).tolist())),
                gt_probe_frozen_score_rank=1+sum(contrib[i].sum()>contrib[gt].sum() for i in ids),
                interpretation='GT-local evidence diagnostic, never a candidate, oracle or selected accuracy result'))
        good_tracks=[r for r in e['representations'] if records[r['current']]['gt_distance']<=42]
        winner_track=next(r for r in e['representations'] if r['track_id']==e['current_track_id'])
        if records[winner]['gt_distance']<=42:category='current_success'
        elif good_tracks:category='producer_best_score_wins_tied_onset'
        elif records[good]['gt_distance']<=42:category='useful_observation_absent_from_eligible_representatives'
        else:category='no_ready_observed_coordinate_within42'
        rank_pair=None
        if good_tracks:
            target=min(good_tracks,key=lambda r:r['final_rank'])
            rank_pair=dict(good=target['rank_key'],winner=winner_track['rank_key'],good_track=target['track_id'],
                first_different_component=next((j for j,(a,b) in enumerate(zip(target['rank_key'],winner_track['rank_key'])) if a!=b),None))
        atlas.append(dict(event=e['event'],category=category,current_error=records[winner]['gt_distance'],
            nearest_ready_error=records[good]['gt_distance'],rank_pair=rank_pair,
            current_source=next(t['source'] for t in details[e['event']]['snapshot']['tracks'] if t['track_id']==e['current_track_id'])))
        temporal.append(dict(event=e['event'],good_id=good,winner_id=winner,good_error=records[good]['gt_distance'],
            winner_error=records[winner]['gt_distance'],
            good_step=dict(zip(step_names,decode(steps[good]).tolist())),
            winner_step=dict(zip(step_names,decode(steps[winner]).tolist())),
            timestamp_semantics=e['context']['timestamp_semantics'],
            all_frames_at_or_before_cutoff=max(e['context']['pre_times']+e['context']['post_times'])<=e['context']['cutoff']))
        motion_pairs.append(dict(event=e['event'],good_id=good,winner_id=winner,
            good_error=records[good]['gt_distance'],good_residual_survival=float(motion[good]),
            winner_residual_survival=float(motion[winner]),difference=float(motion[good]-motion[winner]),
            global_registration=features['global_registration_mean'],crop_half_disagreement=features['crop_half_disagreement']))
        # Inspect all available correct tolerances; ranking did not use this GT join.
        for radius in RADII:
            candidates=[i for i in observed if records[i]['gt_distance']<=radius]
            if not candidates:continue
            target=max(candidates,key=lambda i:(contrib[i].sum(),-i))
            contribution_audit.append(dict(event=e['event'],radius=radius,target_id=target,
                target_record=records[target],target_features=dict(zip(names,matrix[target].tolist())),
                current_family_delta=dict(zip(families,(values[target]-values[winner]).tolist())),
                target_model_score=float(contrib[target].sum()),current_model_score=float(contrib[winner].sum())))
    write_json(output/'atlas_classification.json',dict(events=atlas,counts=dict(Counter(r['category'] for r in atlas)),
        classification_basis='Derived after inspecting exact CURRENT rank keys and available coordinates; model family attributions are separate'))
    write_json(output/'tolerance_pairs.json',contribution_audit)
    write_json(output/'temporal_context.json',dict(events=temporal,semantics='Cached local step may begin before audio; no PRE/timing veto'))
    write_json(output/'motion_context.json',dict(events=motion_pairs,semantics='Relative residual after image crop-half compensation; no calibrated seam/board attribution'))
    write_json(output/'coordinate_evidence_gap.json',coordinate_evidence)
    write_json(output/'no_impact.json',dict(events=events,**descriptive_separability(events)))
    write_json(output/'protocol.json',dict(status='DESCRIPTIVE_ONLY',command=sys.argv,input_hashes=hashes,
        source_hash=digest(Path(__file__)),constants=dict(cluster_radius_px=12,top_quartile=.75),
        inputs='Accepted original feature/step caches only; no new pixels or future frames'))
    write_json(output/'input_accesses.json',sorted(accessed))
    print('ATLAS',dict(Counter(r['category'] for r in atlas)))
    for d in descriptive_separability(events)['distributions']:
        if abs(d['auc_positive_higher']-.5)>.2:print('DESCRIPTIVE',d['feature'],round(d['auc_positive_higher'],3))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('base','prior','ranking','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.base,a.prior,a.ranking,a.output)
