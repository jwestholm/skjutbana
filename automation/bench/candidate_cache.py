"""Content-addressed, loss-aware numeric cache of an immutable candidate export.

No detector or label changes. NaN means missing; shot rows (including empty
candidate groups) remain in shots.jsonl. Each export retains its source hash.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np

VERSION = 1

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def numeric(candidate):
    from .train import EXCLUDED_EXACT, EXCLUDED_SUBSTRINGS
    return {k: float(v) for k, v in candidate.items()
            if k not in EXCLUDED_EXACT and not any(s in k for s in EXCLUDED_SUBSTRINGS)
            and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}

def rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get('kind') not in ('shot', 'candidate'):
                    raise ValueError(f'Unknown export row kind: {r.get("kind")}')
                yield r

def build(directory, root):
    source = Path(directory).resolve() / 'candidates.jsonl'
    sha = file_hash(source)
    dest = Path(root) / f'v{VERSION}_{sha}'
    if (dest / 'manifest.json').exists():
        return dest
    Path(root).mkdir(parents=True, exist_ok=True)
    names = set(); n = 0; shots = []; groups = {}
    for r in rows(source):
        group = str(r['group'])
        groups.setdefault(group, len(groups))
        if r['kind'] == 'shot':
            shots.append(r)
        else:
            names.update(numeric(r['candidate'])); n += 1
    names = sorted(names)
    # A private staging directory makes concurrent cache creation harmless.
    stage = Path(tempfile.mkdtemp(prefix='.building_', dir=root))
    X = np.lib.format.open_memmap(stage / 'features.npy', mode='w+', dtype='float32', shape=(n, len(names)))
    X[:] = np.nan
    meta = np.lib.format.open_memmap(stage / 'rows.npy', mode='w+',
        dtype=[('group', '<i8'), ('label', '?'), ('impact', '?'), ('live_score', '<f8')], shape=(n,))
    counts = np.zeros(len(names), dtype=np.int64); impact_n = 0; j = 0
    cols = {k: i for i, k in enumerate(names)}
    for r in rows(source):
        if r['kind'] != 'candidate': continue
        vals = numeric(r['candidate']); impact = bool(r['producer_has_impact'])
        for k, v in vals.items():
            X[j, cols[k]] = v
            if impact: counts[cols[k]] += 1
        meta[j] = (groups[str(r['group'])], bool(r['label']), impact, float(r['candidate'].get('score', 0) or 0))
        impact_n += impact; j += 1
    X.flush(); meta.flush(); del X, meta
    if file_hash(source) != sha:
        raise ValueError(f'Export changed while caching: {source}')
    (stage / 'shots.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in shots))
    manifest = dict(version=VERSION, source=str(source), sha256=sha, features=names,
                    groups=groups, candidates=n, impact_candidates=impact_n,
                    impact_presence=counts.tolist(), missing='NaN; no imputation in cache',
                    dtype='float32', label_provenance='unchanged export label and producer_has_impact')
    (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    try: os.rename(stage, dest)
    except OSError:
        if not (dest / 'manifest.json').exists(): raise
        import shutil
        shutil.rmtree(stage)  # only our unpublished temporary cache
    return dest

class Export:
    def __init__(self, directory, cache_root):
        self.source = str(Path(directory).resolve())
        self.path = build(directory, cache_root)
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        self.X = np.load(self.path / 'features.npy', mmap_mode='r')
        self.meta = np.load(self.path / 'rows.npy', mmap_mode='r')
        self.shots = [json.loads(l) for l in (self.path / 'shots.jsonl').read_text().splitlines()]

def selection(export, *, seed=20261122, negative_cap=None):
    """Keep whole positive groups; seed-cap negatives in every other impact group.

    Evaluation never calls this. Selection identity is export hash + row index.
    """
    m = export.meta
    impact = np.flatnonzero(m['impact'])
    if negative_cap is None: return impact
    if negative_cap < 1: raise ValueError('negative_cap must be positive')
    positive = set(m['group'][m['label'] & m['impact']].tolist())
    kept = []
    for g in np.unique(m['group'][impact]):
        idx = impact[m['group'][impact] == g]
        if int(g) not in positive and len(idx) > negative_cap:
            key = f'{seed}:{export.manifest["sha256"]}:{int(g)}'
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little'))
            idx = rng.choice(idx, negative_cap, replace=False)
        kept.extend(idx.tolist())
    return np.array(sorted(kept), dtype=np.int64)
