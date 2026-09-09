"""Read-only physical quality checks with explicit session and assignment scope."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from automation.physical_trace_label import load_existing_annotation


def inspect_session(root, mapping=None):
    root = Path(root)
    from automation.physical_test import health
    health_report = health(root)  # Load actual frame/map headers, not just JSON references.
    paths = sorted(root.glob('shots/*/trace.json'))
    if mapping is None:
        assignment = root / 'physical_assignments.json'
        mapping = json.loads(assignment.read_text()) if assignment.exists() else {}
    if not isinstance(mapping, dict):
        raise ValueError('Assignments must be an event-id object.')
    reasons, frame_problems, label_problems = [], [], []
    events, resolved, complete, replay, temporal = [], [], [], [], []
    for row in health_report['shots']:
        for problem in row['problems']:
            reasons.append(f'{row["shot"]}:{problem}')
            if problem != 'producer reports incomplete trace':
                frame_problems.append(problem)
    for path in paths:
        trace = json.loads(path.read_text())
        sid = trace['shot_id']
        if sid != int(path.parent.name.removeprefix('shot_')) or sid in events:
            raise ValueError(f'Invalid or duplicate event identity: {path}')
        events.append(sid)
        complete.append(bool(trace.get('completeness', {}).get('trace_complete')))
        decision = trace.get('decision_input') or {}
        snapshot = decision.get('complete_track_audit') or {}
        replay.append(bool(snapshot.get('complete') and trace.get('selectors')))
        posts = [f for f in trace.get('frames', []) if f.get('kind') in ('post', 'post_snapshot')]
        temporal.append(bool(posts))
        try:
            assignment = mapping.get(str(sid), {})
            if not isinstance(assignment, dict):
                raise ValueError('assignment must be an object')
            if assignment.get('state') == 'NO_PHYSICAL_SHOT':
                if (path.parent / 'ground_truth.json').exists() or assignment.get('label_shot_id') is not None:
                    raise ValueError('nonphysical assignment conflicts with a coordinate label')
                if not assignment.get('reason'):
                    raise ValueError('nonphysical assignment needs human/timing evidence in reason')
                resolved.append(sid)
                continue  # Explicit human evidence resolves a saved S/unresolved marker.
            if assignment.get('state') not in (None, 'precise', 'approximate'):
                raise ValueError('unknown assignment state')
            label_id = assignment.get('label_shot_id', sid)
            if type(label_id) not in (str, int) or not str(label_id).isdigit():
                raise ValueError('invalid label event id')
            directory = root / 'shots' / f'shot_{int(label_id):08d}'
            gt = load_existing_annotation(directory)
            if gt is None:
                raise ValueError('unresolved: missing physical label or explicit nonphysical assignment')
            if gt.get('shot_id') != int(label_id):
                raise ValueError('label shot_id differs from its file location')
            if (directory / 'ground_truth_status.json').exists():
                raise ValueError('coordinate label also has an unresolved status')
            if gt.get('attached_at', 0) <= trace.get('created_at', 0):
                raise ValueError('label timestamp does not follow capture')
            shape = next((f['shape'] for f in trace.get('frames', []) if f.get('kind') == 'pre_snapshot'), None)
            if shape and not (0 <= gt['camera_x'] < shape[1] and 0 <= gt['camera_y'] < shape[0]):
                raise ValueError('ground truth lies outside the captured camera frame')
            resolved.append(sid)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            label_problems.append(f'event {sid}:{exc}')
    unknown = set(mapping) - {str(sid) for sid in events}
    if unknown:
        label_problems.append(f'assignments reference unknown events: {sorted(unknown)}')
    reasons.extend(label_problems)
    frames_ok = bool(events) and not frame_problems
    return dict(session=root.name, trace_root=str(root.resolve()), events=len(events), event_ids=events,
                resolved_event_ids=resolved,
                trace_completeness='PASS' if events and all(complete) and health_report['healthy'] else 'FAIL',
                label_completeness='PASS' if events and len(resolved) == len(events) and not label_problems else 'WARN',
                frame_completeness='PASS' if frames_ok else 'FAIL',
                full_replay_ready='PASS' if events and all(replay) else 'WARN',
                patch_dataset_ready='PASS' if frames_ok else 'FAIL',
                temporal_ready='PASS' if frames_ok and all(temporal) else 'FAIL', reasons=reasons,
                replay_semantics='Complete snapshot presence only; run CURRENT_EXACT_REPLAY to verify predicates, ranks and winner.')


def inspect(root):
    return [inspect_session(session) for session in sorted(Path(root).glob('session_*'))]


def inspect_one(root):
    root = Path(root)
    return [inspect_session(root)] if (root / 'shots').is_dir() else inspect(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('content/ai/physical_traces'))
    parser.add_argument('--session-root', type=Path)
    parser.add_argument('--mapping', type=Path, help='External human assignments; requires --session-root')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; preserve the baseline and choose a new report path.')
    if args.mapping and not args.session_root:
        parser.error('--mapping requires --session-root; no broad assignment reuse')
    mapping = json.loads(args.mapping.read_text()) if args.mapping else None
    payload = [inspect_session(args.session_root, mapping)] if args.session_root else inspect(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(payload, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
