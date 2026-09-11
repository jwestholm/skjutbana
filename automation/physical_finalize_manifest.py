"""Build an auditable finalization manifest from a bound, human-labeled session.

No event ordering implies a physical ordinal unless the operator explicitly
confirms it. This module never writes inside captured sessions or changes labels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from automation.physical_finalize import validate, validate_data
from automation.physical_label_reset import REPO, resolve_session
from automation.physical_trace_quality import inspect_session


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key {key!r} in {path}')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError(f'Nonfinite JSON number {value} in {path}')
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=nonfinite)


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def identity_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Binding/plan has no rows')
    result = {}
    for row in rows:
        ordinal = positive_int(row['planned_physical_shot'], 'planned_physical_shot')
        if ordinal in result:
            raise ValueError('Duplicate planned shots')
        result[ordinal] = tuple(row.get(k) for k in ('session', 'session_class', 'category'))
    return result


def inventory(root, extras):
    """Hash all artifacts of this exact session, including absence/added files."""
    paths = list(extras)
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Symlink inside captured session refused: {path}')
        if path.is_file():
            paths.append(path)
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def build_manifest(plan, session, binding, *, mapping=None, shot_map=(), in_capture_order=False):
    meta = read_json(binding)
    rows = [r for r in plan['rows'] if r['session'] == session]
    expected = identity_rows(rows)
    if not plan.get('collection_plan_id') or meta.get('collection_plan_id') != plan['collection_plan_id']:
        raise ValueError('Binding collection_plan_id differs from plan')
    if meta.get('planned_session_id') != session:
        raise ValueError('Binding identifies another planned session')
    if session == 'S03' or meta.get('session_class') == 'VALIDATION_UNTOUCHED':
        raise ValueError('Untouched validation session refused; no trace opened')
    bound_root = Path(meta['trace_root'])
    if not bound_root.is_absolute():
        bound_root = REPO / bound_root
    if any(re.search(r'(^|_)S03(_|$)', part, re.IGNORECASE)
           for part in (*bound_root.parts, *bound_root.resolve().parts)):
        raise ValueError('Untouched validation trace root refused; no trace opened')
    if any(r['session_class'] != meta.get('session_class') for r in rows) or identity_rows(meta.get('rows')) != expected:
        raise ValueError('Binding rows/session class differ from plan')
    root, embedded = resolve_session(binding=binding, session=session)
    if (root / 'collection_binding.json').exists():
        read_json(root / 'collection_binding.json')  # Reject duplicate identity keys.
    if embedded.get('rows') is not None and identity_rows(embedded['rows']) != expected:
        raise ValueError('Embedded binding rows differ from plan')
    extras = [Path(binding)] + ([Path(mapping)] if mapping else [])
    before = inventory(root, extras)
    native = root / 'physical_assignments.json'
    assignments = read_json(mapping) if mapping else read_json(native) if native.exists() else {}
    if not isinstance(assignments, dict):
        raise ValueError('Assignments must be an event-id object')
    directories = sorted((root / 'shots').iterdir())
    events, used_labels = {}, set()
    for directory in directories:
        if not directory.is_dir() or not re.fullmatch(r'shot_\d{8}', directory.name):
            raise ValueError(f'Unexpected captured event entry: {directory}')
        trace = read_json(directory / 'trace.json')
        if trace.get('schema_version') != 'physical-shot-trace-1':
            raise ValueError('Unsupported physical trace schema')
        references = list(trace.get('frames', []))
        for stage in trace.get('stages', []):
            references.extend(stage.get('evidence_maps', {}).values())
        for ref in references:
            if not (directory / ref['path']).resolve().is_relative_to(directory.resolve()):
                raise ValueError('Evidence path escapes captured event')
        for name in ('ground_truth.json', 'ground_truth_status.json'):
            if (directory / name).exists():
                read_json(directory / name)
        sid = positive_int(trace['shot_id'], 'event id')
        if sid != int(directory.name[5:]) or sid in events:
            raise ValueError('Invalid/duplicate captured event identity')
        assignment = assignments.get(str(sid), {})
        if not isinstance(assignment, dict):
            raise ValueError(f'Event {sid}: assignment must be an object')
        state = assignment.get('state')
        if state not in (None, 'precise', 'approximate', 'NO_PHYSICAL_SHOT'):
            raise ValueError(f'Event {sid}: unresolved/UNKNOWN/AMBIGUOUS assignment')
        if state == 'NO_PHYSICAL_SHOT':
            if not isinstance(assignment.get('reason'), str) or not assignment['reason'].strip():
                raise ValueError('Nonphysical assignment needs an explicit human evidence reason')
            if assignment.get('planned_physical_shot') is not None:
                raise ValueError('Nonphysical event consumes a planned shot')
            events[sid] = dict(event_id=sid, planned_physical_shot=None, status=state)
        else:
            label_id = assignment.get('label_shot_id', sid)
            if isinstance(label_id, str) and re.fullmatch(r'[1-9]\d*', label_id):
                label_id = int(label_id)
            positive_int(label_id, 'label_shot_id')
            if label_id in used_labels:
                raise ValueError('Duplicate physical label mapping: a coordinate label was reused')
            used_labels.add(label_id)
            events[sid] = dict(event_id=sid, planned_physical_shot=None, status='PHYSICAL')
    if set(assignments) - {str(sid) for sid in events}:
        raise ValueError('Assignments reference unknown/noncanonical event ids')
    if used_labels - set(events):
        raise ValueError('Physical assignment references an unknown captured label event')
    # Validate actual coordinates, unresolved markers, nonphysical reasons, frame
    # headers and trace completeness. A previously saved PASS cannot bypass this.
    quality = inspect_session(root, assignments)
    if quality['label_completeness'] != 'PASS':
        raise ValueError('Saved labels are unresolved or invalid: ' + '; '.join(quality['reasons']))
    physical = [sid for sid in events if events[sid]['status'] == 'PHYSICAL']
    mapped = {}

    def assign(ordinal, sid):
        positive_int(ordinal, 'planned_physical_shot')
        positive_int(sid, 'event id')
        if ordinal not in expected or sid not in physical:
            raise ValueError('Mapping references an unknown planned shot or nonphysical/missing event')
        if ordinal in mapped and mapped[ordinal] != sid:
            raise ValueError('Conflicting planned physical shot mapping')
        if sid in mapped.values() and mapped.get(ordinal) != sid:
            raise ValueError('Duplicate physical event mapping')
        mapped[ordinal] = sid

    for source_rows in (rows, meta['rows'], embedded.get('rows', [])):
        for row in source_rows:
            if row.get('actual_event_id') is not None:
                assign(row['planned_physical_shot'], row['actual_event_id'])
    for key, assignment in assignments.items():
        if assignment.get('planned_physical_shot') is not None:
            assign(assignment['planned_physical_shot'], int(key))
    explicit = set()
    for pair in shot_map:
        if not re.fullmatch(r'[1-9]\d*=[1-9]\d*', pair):
            raise ValueError('--shot-map requires PLANNED=EVENT, for example --shot-map 2=3')
        ordinal, sid = map(int, pair.split('='))
        if ordinal in explicit:
            raise ValueError('Duplicate --shot-map planned shot')
        explicit.add(ordinal)
        assign(ordinal, sid)
    if in_capture_order:
        if len(physical) != len(expected):
            raise ValueError('Capture-order mapping requires exactly the planned physical count')
        for ordinal, sid in zip(sorted(expected), physical):
            assign(ordinal, sid)
    if set(mapped) != set(expected):
        raise ValueError('Planned-shot mapping is unresolved. Physical event ids: '
                         f'{physical}. Supply --shot-map PLANNED=EVENT for each shot, '
                         'or --in-capture-order only if you confirm that sequence.')
    for ordinal, sid in mapped.items():
        events[sid]['planned_physical_shot'] = ordinal
    manifest = dict(collection_plan_id=plan['collection_plan_id'], planned_session_id=session,
                    session_class=meta['session_class'], trace_root=str(root),
                    recovered=bool(meta.get('recovered') or embedded.get('recovered')),
                    labels=list(events.values()))
    validate_data(plan, session, manifest, quality)
    if inventory(root, extras) != before:
        raise ValueError('Session/binding/assignments changed during inspection; retry after capture/labeling closes')
    return manifest, quality, before


def add_arguments(parser):
    parser.add_argument('--plan', type=Path, required=True, help='Original development capture plan')
    parser.add_argument('--session', required=True, help='Planned session id, for example D01')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--binding', type=Path, help='Build manifest from the exact bound trace root and saved labels')
    source.add_argument('--labels', type=Path, help='Historical aggregate manifest (requires --quality)')
    parser.add_argument('--mapping', type=Path, help='External physical assignments for --binding; otherwise use the saved assignments')
    order = parser.add_mutually_exclusive_group()
    order.add_argument('--in-capture-order', action='store_true', help='Explicitly confirm sorted physical event ids follow planned shot order; excludes NO_PHYSICAL_SHOT')
    order.add_argument('--shot-map', action='append', default=[], metavar='PLANNED=EVENT', help='Explicit ordinal mapping; repeat for each physical shot')
    parser.add_argument('--quality', type=Path, help='Saved quality report to cross-check; binding mode always checks current artifacts too')
    parser.add_argument('--preview', action='store_true', help='Print verified manifest and quality without writing output')
    parser.add_argument('--output', type=Path, help='New finalization report; required unless --preview; existing files refused')


def finalize(args):
    if not args.preview and args.output is None:
        raise ValueError('--output is required unless --preview')
    if args.output and (args.output.exists() or args.output.is_symlink()):
        raise FileExistsError(f'Output exists: {args.output}; preserve it and choose a new path')
    plan_bytes = args.plan.read_bytes()
    plan = read_json(args.plan)
    if args.binding:
        manifest, quality, hashes = build_manifest(plan, args.session, args.binding,
            mapping=args.mapping, shot_map=args.shot_map, in_capture_order=args.in_capture_order)
        root = Path(manifest['trace_root'])
        if args.output and args.output.resolve().is_relative_to(root):
            raise ValueError('Finalization output must be outside the captured session')
        if args.quality:
            validate_data(plan, args.session, manifest, read_json(args.quality))
        if args.plan.read_bytes() != plan_bytes:
            raise ValueError('Plan changed during inspection')
        hashes[str(args.plan.resolve())] = hashlib.sha256(plan_bytes).hexdigest()
        result = validate_data(plan, args.session, manifest, quality)
        result.update(label_manifest=manifest, quality=quality, input_sha256=hashes,
                      mapping_basis='operator_confirmed_capture_order' if args.in_capture_order else 'explicit_mapping')
        print(f'SESSION: {args.session}\nTRACE_ROOT: {root}')
        for row in manifest['labels']:
            print(f"  event {row['event_id']} -> planned {row['planned_physical_shot']} ({row['status']})")
        print('QUALITY: trace/frame/label PASS; readiness is structural, run exact replay separately')
    else:
        if args.mapping or args.shot_map or args.in_capture_order:
            raise ValueError('Mapping options require --binding')
        if not args.quality:
            raise ValueError('Historical --labels workflow requires --quality')
        result = validate(plan, args.session, args.labels, args.quality)
    if args.preview:
        print('PREVIEW: no files written')
        print(json.dumps(result, indent=2))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(json.dumps(result, indent=2) + '\n')
        print(f'FINALIZED {args.session}: {args.output}')
    return result
