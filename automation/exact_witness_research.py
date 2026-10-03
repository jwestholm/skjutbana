"""Bounded OFFLINE retention intervention; no live hook or engine changes.

One comparison only. Spatial structure is not used in this selector. Labels enter
only after all event outputs are frozen. Receipt/ordinal semantics stay unchanged.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch
import cv2
from automation import chronological_pre_research as chrono
from automation.causal_pre_onset_research import digest, read_guard, write


def exact_supported_keep(candidates, ordinal, capacity=chrono.CAPACITY):
    """Compatibility adapter for the shared bounded exact-observation keeper."""
    from src.engine.offline.observation_hypotheses import retain_exact_physical_witnesses
    return retain_exact_physical_witnesses(candidates,ordinal,capacity)


def evaluate(events,out,baseline):
    recovered=json.loads((chrono.BASE/'kompetensprov_recovery_20260927_01/recovered_results.json').read_text())['events']
    assignments=json.loads((chrono.D02/'physical_assignments.json').read_text())
    rows=[]
    for e in events:
        key=e['id'].replace(':','_');p=out/(key+'_frozen.json');r=json.loads(p.read_text())
        oldpath=baseline/(key+'_frozen.json')
        if not oldpath.exists():oldpath=out.parent/'baseline_added'/(key+'_frozen.json')
        old=json.loads(oldpath.read_text())
        truth=json.loads((e['root']/'ground_truth.json').read_text()) if e['id'].startswith('KOMP') or e['sid'] in (1,3) else None
        gt=[truth['camera_x'],truth['camera_y']] if truth else None
        if not gt:assert assignments[str(e['sid'])]['state']=='NO_PHYSICAL_SHOT'
        transitions=[json.loads(l) for l in (out/(key+'_transitions.jsonl')).read_text().splitlines()]
        generated=[c for t in transitions for c in t['generated']]
        witnesses=[dict(c,observed_at=t['ordinal'],retained_now=c['uid'] in t['retained']) for t in transitions for c in t['evaluated'] if c['temporal']['supported']]
        def error(c):return math.dist(c['camera_xy'],gt) if c and gt else None
        def nearest(cs):return min((error(c) for c in cs),default=None) if gt else None
        current=(recovered[e['sid']-1]['error_px'] if e['id'].startswith('KOMP') else None)
        rows.append(dict(event=e['id'],gt=gt,current_error=current,baseline_xy=old['selected']['camera_xy'] if old['selected'] else None,
            baseline_error=error(old['selected']),selected=r['selected'],selected_error=error(r['selected']),
            generated_nearest=nearest(generated),supported_nearest=nearest(witnesses),final_supported_nearest=nearest(r['supported']),
            supported_unique_xy=len({tuple(c['camera_xy']) for c in witnesses}),final_supported=len(r['supported']),
            any_supported_prefix=bool(witnesses),max_retained=r['max_retained'],max_transient=r['max_transient'],
            witnesses=witnesses,final_ranked=r['supported'],timings=r['timings'],total_ms=r['total_pre_work_ms']))
    def summary(group):
        rs=[r for r in rows if r['event'].startswith(group)];real=[r for r in rs if r['gt']];neg=[r for r in rs if not r['gt']]
        return dict(physical=len(real),selected_at={str(d):sum(r['selected_error'] is not None and r['selected_error']<=d for r in real) for d in chrono.RADII},
            oracle_at={str(d):sum(r['generated_nearest'] is not None and r['generated_nearest']<=d for r in real) for d in chrono.RADII},
            witness_at={str(d):sum(r['supported_nearest'] is not None and r['supported_nearest']<=d for r in real) for d in chrono.RADII},
            physical_uncertain=sum(r['selected'] is None for r in real),negative_final_witnesses=sum(r['final_supported']>0 for r in neg),
            negative_transient_witnesses=sum(r['any_supported_prefix'] for r in neg),no_physical_hit_declarations=0)
    write(out/'evaluation.json',dict(summary={g:summary(g) for g in ('KOMP','D02')},events=rows))
    return {g:summary(g) for g in ('KOMP','D02')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    accesses=read_guard(out.parent,[chrono.SESSION,chrono.D02]);cv2.setNumThreads(1)
    events=chrono.descriptors(d02_sids=range(1,7));settings=json.loads(Path('content/ai/settings.json').read_text())
    sources=[Path(__file__),Path(chrono.__file__),Path(chrono.region.__file__),Path('src/engine/offline/clean_physical_change.py'),Path('content/ai/settings.json'),Path('content/ai/detector_v2.json')]
    write(out/'sources.json',{str(p):digest(p) for p in sources})
    write(out/'inputs.json',[dict(event=e['id'],pre=[{**f,'path':str(f['path'])} for f in e['pre']],mask=str(e['mask'])) for e in events])
    added=out.parent/'baseline_added';added.mkdir(exist_ok=False)
    with patch.object(chrono.region,'_runtime_settings',return_value=settings):
        assert chrono.region._per_region_proposals()==8
        for e in events:
            if e['id'] in ('D02:1','D02:3'):chrono.run_event(e,added)
            r=chrono.run_event(e,out,keep_fn=exact_supported_keep)
            print(e['id'],'supported',len(r['supported']),'selected',r['selected']['camera_xy'] if r['selected'] else None,flush=True)
    print(json.dumps(evaluate(events,out,chrono.BASE/'chronological_pre_20260927_01/final')),flush=True)
    inputs={str(f['path']) for e in events for f in e['pre']+e['post']}
    inputs.update(str(e['mask']) for e in events);inputs.update(str(e['trace']) for e in events if e['trace'])
    inputs.update(str(e['root']/'ground_truth.json') for e in events if e['id'].startswith('KOMP') or e['sid'] in (1,3))
    inputs.add(str(chrono.D02/'physical_assignments.json'))
    write(out/'input_hashes.json',{p:digest(p) for p in sorted(inputs)})
    write(out/'read_accesses.json',sorted(accesses))

def historical_regression(output):
    """Same last-PRE reconstruction, paired original/exact retention on older data.

    This does not replace the older PRE+POST causal experiment or claim live parity.
    Missing D01 pixels are never opened. All labels joined after paired inference.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cache=chrono.BASE/'D01_counterfactual_20260910_152433/cache/dataset.json'
    data=json.loads(cache.read_text());allowed=[e for e in data['events'] if e['session'] in ('S01','S02','POST_FIX')]
    roots={Path(e['trace_path']).parents[2] for e in allowed};accesses=read_guard(output,roots);cv2.setNumThreads(1)
    events=[];unavailable=[dict(event=f'D01:{i}',reason='Original raw pixels overwritten; path not accessed') for i in range(1,7)]
    unavailable.append(dict(event='S01:11',reason='No complete terminal cutoff; excluded from paired selection'))
    for e in allowed:
        path=Path(e['trace_path']);assert digest(path)==e['trace_sha256']
        t=json.loads(path.read_text());root=path.parent;cutoff=min(e['context']['cutoff'],t.get('next_audio_peak_ts') or math.inf)
        pre=sorted((f for f in t['frames'] if f['kind']=='pre_history' and f['timestamp']<t['peak_ts']),key=lambda f:f['timestamp'])
        stage=next(s for s in reversed(t['stages']) if s['timestamp']<=cutoff and s.get('evidence_maps',{}).get('roi_polygon'))
        events.append(dict(id=e['event'],sid=t['shot_id'],root=root,pre=[dict(path=root/f['path'],ordinal=i,timestamp=f['timestamp'],sequence=f.get('acquisition',{}).get('sequence')) for i,f in enumerate(pre)],
            post=[],mask=root/stage['evidence_maps']['roi_polygon']['path'],trace=path,cutoff=cutoff,crop=e['context']['crop']))
    write(output/'inputs.json',[dict(event=e['id'],crop=e['crop'],pre=[{**f,'path':str(f['path'])} for f in e['pre']],mask=str(e['mask']),cutoff=e['cutoff']) for e in events])
    results={};settings=json.loads(Path('content/ai/settings.json').read_text())
    for mode in ('original','exact'):(output/mode).mkdir()
    with patch.object(chrono.region,'_runtime_settings',return_value=settings):
        for e in events:
            results[e['id']]={mode:chrono.run_event(e,output/mode,keep_fn=fn) for mode,fn in [('original',chrono.bounded_keep),('exact',exact_supported_keep)]}
            print(e['id'],{m:(r['selected']['camera_xy'] if r['selected'] else None) for m,r in results[e['id']].items()},flush=True)
    rows=[]
    for e in allowed:
        gt=e['truth']['impacts'][0] if e['truth']['state']=='SINGLE_IMPACT' else None
        xy=[gt['camera_x'],gt['camera_y']] if gt else None
        row=dict(event=e['event'],truth=e['truth']['state'],gt=xy,current_error=data['records'][e['current_id']]['gt_distance'] if gt else None)
        for mode,result in results[e['event']].items():
            key=e['event'].replace(':','_');ts=[json.loads(l) for l in (output/mode/(key+'_transitions.jsonl')).read_text().splitlines()]
            props=[c for t in ts for c in t['generated']];sup=[c for t in ts for c in t['evaluated'] if c['temporal']['supported']]
            chosen=result['selected'];dist=lambda cs:min((math.dist(c['camera_xy'],xy) for c in cs),default=None) if xy else None
            row[mode]=dict(selected_xy=chosen['camera_xy'] if chosen else None,error=math.dist(chosen['camera_xy'],xy) if chosen and xy else None,
                oracle=dist(props),witness=dist(sup),final_supported=len(result['supported']),any_supported=bool(sup))
        rows.append(row)
    summary={}
    for cohort in ('S01','S02','POST_FIX'):
        rs=[r for r in rows if r['event'].startswith(cohort+':')];physical=[r for r in rs if r['gt']];neg=[r for r in rs if not r['gt']]
        summary[cohort]={mode:dict(selected_at={str(d):sum(r[mode]['error'] is not None and r[mode]['error']<=d for r in physical) for d in chrono.RADII},
            oracle_at={str(d):sum(r[mode]['oracle'] is not None and r[mode]['oracle']<=d for r in physical) for d in chrono.RADII},
            witness_at={str(d):sum(r[mode]['witness'] is not None and r[mode]['witness']<=d for r in physical) for d in chrono.RADII},
            uncertain=sum(r[mode]['selected_xy'] is None for r in physical),false_final=sum(r[mode]['final_supported']>0 for r in neg),false_prefix=sum(r[mode]['any_supported'] for r in neg)) for mode in ('original','exact')}
    write(output/'evaluation.json',dict(summary=summary,events=rows,unavailable=unavailable))
    inputs={str(f['path']) for e in events for f in e['pre']};inputs.update(str(e['mask']) for e in events);inputs.update(str(e['trace']) for e in events)
    write(output/'input_hashes.json',{p:digest(p) for p in sorted(inputs)});write(output/'read_accesses.json',sorted(accesses))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
