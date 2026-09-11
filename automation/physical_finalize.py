"""Postflight/label consistency gate for a planned research session."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(plan, session, labels, quality):
    return validate_data(plan, session, json.loads(Path(labels).read_text()),
                         json.loads(Path(quality).read_text()))


def validate_data(plan, session, ls, qs):
    """Validate an in-memory manifest; the historical path-based API is retained."""
    rows = [r for r in plan['rows'] if r['session'] == session]
    if not rows:
        raise ValueError('unknown planned session')
    if isinstance(qs, list):
        def matches(q):
            if ls.get('trace_root') and q.get('trace_root'):
                return Path(ls['trace_root']).resolve() == Path(q['trace_root']).resolve()
            return q.get('session') == session
        found = [q for q in qs if matches(q)]
        if len(found) != 1:
            raise ValueError('quality must identify exactly the labeled session; no fallback to another report')
        qs = found[0]
    if ls.get('collection_plan_id') != plan.get('collection_plan_id'):
        raise ValueError('collection plan mismatch; data is safe, use the original plan')
    if ls.get('planned_session_id', session) != session:
        raise ValueError('label manifest identifies another planned session')
    if ls.get('trace_root') and qs.get('trace_root') and Path(ls['trace_root']).resolve() != Path(qs['trace_root']).resolve():
        raise ValueError('quality trace root differs from labels')
    expected = [r['planned_physical_shot'] for r in rows]
    if len(set(expected)) != len(expected):
        raise ValueError('duplicate planned shots')
    labels_list = ls.get('labels', [])
    ids = [x.get('event_id') for x in labels_list]
    if any(type(i) is not int or i < 1 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('invalid or duplicate label assignment')
    if any(x.get('status') not in ('PHYSICAL', 'NO_PHYSICAL_SHOT') for x in labels_list):
        raise ValueError('unresolved, unknown or AMBIGUOUS labels remain')
    planned = [x.get('planned_physical_shot') for x in labels_list if x['status'] == 'PHYSICAL']
    if any(type(i) is not int or i < 1 for i in expected + planned) or len(set(planned)) != len(planned) or set(planned) != set(expected):
        raise ValueError('physical mapping must cover every planned shot exactly once')
    if any(x.get('planned_physical_shot') is not None for x in labels_list if x['status'] == 'NO_PHYSICAL_SHOT'):
        raise ValueError('nonphysical events must not consume planned physical shots')
    if 'event_ids' in qs and set(ids) != set(qs['event_ids']):
        raise ValueError('label mapping does not cover the captured event ids exactly')
    if qs.get('events', len(ids)) != len(ids):
        raise ValueError('label count differs from captured event count')
    # Strings such as FAIL are truthy in Python: accept only explicit pass values.
    for name in ('frame_completeness', 'trace_completeness', 'label_completeness'):
        value = qs.get(name)
        if not (value is True or value == 'PASS'):
            raise ValueError(f'trace quality not complete: {name}={value!r}')
    return dict(status='FINALIZED', session=session, labels=len(labels_list),
                physical_shots=len(planned), non_physical_events=len(ids) - len(planned),
                session_class=rows[0]['session_class'], trace_root=ls.get('trace_root'),
                recovered=bool(ls.get('recovered')),
                mapping=[{k: x.get(k) for k in ('planned_physical_shot', 'event_id', 'status')} for x in labels_list])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from automation.physical_finalize_manifest import add_arguments, finalize
    add_arguments(parser)
    args = parser.parse_args()
    try:
        finalize(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f'FINALIZE REFUSED: {exc}\n')


if __name__ == '__main__':
    main()
