"""Preview/archive/reset only the explicitly named physical labeling artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import tempfile

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BACKUPS = Path('/data/skjutbana/label_resets')
SHOT_LABELS = ('ground_truth.json', 'ground_truth_status.json')


def resolve_session(*, binding=None, root=None, session=None):
    """Use an explicit root or binding, never latest-session discovery."""
    if (binding is None) == (root is None):
        raise ValueError('Specify exactly one of --binding or --root.')
    metadata = None
    if binding is not None:
        metadata = json.loads(Path(binding).read_text())
        root = Path(metadata['trace_root'])
        if not root.is_absolute():
            root = REPO / root  # Existing collection bindings are repo-relative.
    root = Path(root).resolve(strict=True)  # The physical_traces symlink is supported.
    if not root.is_dir() or not (root / 'shots').is_dir():
        raise ValueError(f'Not a captured session: {root}')
    embedded = root / 'collection_binding.json'
    if embedded.exists():
        recorded = json.loads(embedded.read_text())
        if metadata is not None:
            for key in ('collection_plan_id', 'planned_session_id', 'session_class'):
                if metadata.get(key) != recorded.get(key):
                    raise ValueError(f'Binding mismatch: {key}; no labels changed.')
        recorded_root = Path(recorded['trace_root'])
        if not recorded_root.is_absolute():
            recorded_root = REPO / recorded_root
        if recorded_root.resolve() != root:
            raise ValueError('Embedded binding points to another session.')
        metadata = recorded
    if session is not None and (metadata or {}).get('planned_session_id') != session:
        raise ValueError(f'Binding does not identify requested session {session}.')
    return root, metadata or {}


def _regular(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f'Refusing symlink or non-regular file: {path}')


def label_files(root):
    """No recursive deletion, globbed label names, or artifact paths from JSON."""
    shots = root / 'shots'
    if shots.is_symlink():
        raise ValueError('Refusing a symlinked shots directory.')
    directories = sorted(shots.iterdir())
    if not directories:
        raise ValueError('No captured events; no labels changed.')
    labels = []
    for directory in directories:
        if directory.is_symlink() or not directory.is_dir() or not re.fullmatch(r'shot_\d{8}', directory.name):
            raise ValueError(f'Unexpected shot entry: {directory}')
        trace = directory / 'trace.json'
        _regular(trace)
        data = json.loads(trace.read_text())
        if data.get('schema_version') != 'physical-shot-trace-1' or data.get('shot_id') != int(directory.name[5:]):
            raise ValueError(f'Invalid trace identity: {trace}')
        if not data.get('completeness', {}).get('trace_complete') or (data.get('outcome') or {}).get('status') in (None, 'pending'):
            raise ValueError(f'Capture is incomplete or still running: {trace}; no labels changed.')
        for name in SHOT_LABELS:
            path = directory / name
            if path.exists() or path.is_symlink():
                _regular(path)
                labels.append(path)
    assignment = root / 'physical_assignments.json'
    if assignment.exists() or assignment.is_symlink():
        _regular(assignment)
        labels.append(assignment)
    return labels


def _ensure_application_closed():
    try:
        connection = socket.create_connection(('127.0.0.1', 8765), timeout=0.3)
    except OSError:
        return
    connection.close()
    raise ValueError('Close the application and labeler before resetting labels; automation port is active.')


def reset_labels(*, binding=None, root=None, session=None, apply=False, backup_root=DEFAULT_BACKUPS):
    root, metadata = resolve_session(binding=binding, root=root, session=session)
    if metadata.get('session_class') == 'VALIDATION_UNTOUCHED' or metadata.get('planned_session_id') == 'S03':
        raise ValueError('Untouched validation session: label reset refused.')
    if apply:
        _ensure_application_closed()
    paths = label_files(root)  # Validate the whole operation before changing any file.
    print(f'SESSION: {metadata.get("planned_session_id", root.name)}\nTRACE_ROOT: {root}')
    print(f'LABEL FILES: {len(paths)}')
    for path in paths:
        print(f'  {path.relative_to(root)}')
    print('Preserved: PRE/POST frames, candidates, selectors, trace metadata, runtime data and external reports.')
    if not apply:
        print('PREVIEW ONLY: no files changed. Close the application and labeler, then repeat with --apply.')
        return {'status': 'PREVIEW', 'files': len(paths), 'root': str(root)}
    if not paths:
        print('ALREADY UNLABELED: nothing to reset.')
        return {'status': 'ALREADY_UNLABELED', 'files': 0, 'root': str(root)}
    backup_root = Path(backup_root).resolve()
    if backup_root == root or backup_root.is_relative_to(root):
        raise ValueError('Label backup must be outside the captured session.')
    originals = {path: path.read_bytes() for path in paths}
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=root.name + '_', dir=backup_root))
    manifest = {'session_root': str(root), 'status': 'BACKED_UP', 'files': {}}
    for path, content in originals.items():
        relative = path.relative_to(root)
        target = backup / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256(content).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise OSError(f'Backup verification failed: {target}; no labels removed.')
        manifest['files'][str(relative)] = digest
    manifest_path = backup / 'reset_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'LABEL BACKUP: {backup}')
    # Refuse concurrent labeling/capture changes before the first unlink.
    if label_files(root) != paths or any(path.read_bytes() != content for path, content in originals.items()):
        raise ValueError('Label files changed during backup; no labels removed. Close the labeler and retry.')
    for path in paths:
        path.unlink()
    manifest['status'] = 'RESET_COMPLETE'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'RESET COMPLETE: {len(paths)} label files removed; captured evidence preserved.')
    print('Previous evaluation/finalization reports describe the old labels. Use new report paths after relabeling.')
    return {'status': 'RESET_COMPLETE', 'files': len(paths), 'root': str(root), 'backup': str(backup)}


def add_reset_arguments(parser):
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--binding', type=Path)
    choice.add_argument('--root', type=Path)
    parser.add_argument('--session', help='Optional expected bound session id, e.g. S02')
    parser.add_argument('--apply', action='store_true', help='Archive and remove labels (default: preview only)')
    parser.add_argument('--backup-root', type=Path, default=DEFAULT_BACKUPS)
