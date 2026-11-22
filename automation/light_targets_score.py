"""Score Ljusjakten against independent operator truth; UNKNOWN never counts as a verified miss.

python3 -m automation.light_targets_score RUN.log truth.jsonl
Truth: {"shot_id": 123, "expected":"HIT"|"MISS"|"UNKNOWN", "owner":"optional if only one owner"}
Use shot_id=null and a unique attempt_id for a real shot with no audio event.
MISS means independently verified outside-target/no-impact control, not merely no game award.
"""
import argparse
import json
from pathlib import Path
from src.engine.offline.paired_stats import wilson_interval


def score(log, truth):
    results = {}
    for line in Path(log).read_text().splitlines():
        if '[LAMP-RESULT] ' not in line: continue
        r = json.loads(line.split('[LAMP-RESULT] ',1)[1])
        if r['source'] != 'OBJECT': continue
        key = (r['owner'], r['shot_id'])
        if key in results: raise ValueError('Duplicate gameplay result: '+str(key))
        results[key] = r
    owners = {k[0] for k in results}
    rows = []; seen = set(); truth_seen = set(); excluded = 0
    for line in Path(truth).read_text().splitlines():
        if not line.strip(): continue
        t = json.loads(line); expected=t['expected']
        if expected not in ('HIT','MISS','UNKNOWN'): raise ValueError('Invalid truth state')
        owner=t.get('owner')
        if owner is None:
            if len(owners)!=1: raise ValueError('Truth needs owner when log has zero/multiple owners')
            owner=next(iter(owners))
        sid=t.get('shot_id')
        key=(owner,int(sid)) if sid is not None else None
        if key is None and t.get('attempt_id') is None: raise ValueError('Missing audio event needs attempt_id')
        identity=key if key is not None else (owner,'attempt',str(t['attempt_id']))
        if identity in truth_seen: raise ValueError('Duplicate truth')
        truth_seen.add(identity)
        if key is not None: seen.add(key)
        if expected=='UNKNOWN': excluded+=1; continue
        r=results.get(key); actual=r['outcome'] if r else 'NO_RESULT'
        rows.append(dict(shot_id=int(sid) if sid is not None else None,attempt_id=t.get('attempt_id'),owner=owner,expected=expected,actual=actual,correct=actual==expected))
    n=len(rows); correct=sum(r['correct'] for r in rows)
    hits=[r for r in rows if r['expected']=='HIT']; misses=[r for r in rows if r['expected']=='MISS']
    unlabelled=len(set(results)-seen)
    false_awards=sum(r['actual']=='HIT' for r in misses)
    return dict(evidence_level='OPERATOR_LABELLED_PHYSICAL_ONLY_IF_TRUTH_IS_PHYSICAL', cases=n,correct=correct,
        accuracy=correct/n if n else None, ci95=wilson_interval(correct,n) if n else None,
        true_hit_correct=sum(r['correct'] for r in hits),true_hits=len(hits),
        hit_recall=sum(r['correct'] for r in hits)/len(hits) if hits else None,
        negative_no_award_rate=1-false_awards/len(misses) if misses else None,
        false_awards=sum(r['actual']=='HIT' for r in misses),negative_controls=len(misses),
        abstentions=sum(r['actual'] in ('UNKNOWN','NO_RESULT') for r in rows),excluded_unknown_truth=excluded,
        unlabelled_results=unlabelled,rows=rows,minimum_pilot_size_met=len(hits)>=60 and len(misses)>=20,
        point_target_met=bool(n and correct/n>=.95 and not false_awards and not unlabelled and not excluded))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('log');p.add_argument('truth');a=p.parse_args()
    print(json.dumps(score(a.log,a.truth),indent=2,ensure_ascii=False))

if __name__=='__main__': main()
