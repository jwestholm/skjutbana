"""Explicit frozen export selection; no directory discovery during nightly."""
from pathlib import Path
import json
from .candidate_cache import file_hash


def freeze(model_path, output):
    model_path = Path(model_path).resolve()
    model = json.loads(model_path.read_text())
    result = dict(version=1, reference_model=str(model_path), reference_model_sha256=file_hash(model_path),
                  note='Inspected historical evaluation set; not an untouched physical holdout.', train=[], holdout=[], regression=[])
    for split, field in (('train', 'training_data'), ('holdout', 'holdout_data')):
        for record in model[field]:
            path, sha = record.rsplit(':', 1)
            result[split].append(dict(export=str(Path(path).parent.resolve()), sha256=sha))
    for arm in ('defaults', 'pilot'):
        baseline = model_path.parent.parent / 'regression' / arm
        path = baseline / 'results.json'
        if not path.exists(): continue
        m = json.loads(path.read_text())['manifest']
        seeds = m['seeds']
        if seeds != list(range(seeds[0], seeds[0]+len(seeds))): raise ValueError('Noncontiguous regression seeds')
        args = ['--background', ','.join(m['background']), '--seed-start', str(seeds[0]), '--seeds', str(len(seeds)),
                '--scenarios', ','.join(m['scenarios']), '--physics', json.dumps(m['physics']),
                '--tolerance-mm', str(m['tolerance_mm']), '--settings', m['settings_path'], '--variant', arm]
        result['regression'].append(dict(name=arm, baseline=str(baseline), baseline_sha256=file_hash(path), args=args, env=m['env']))
    with Path(output).open('x') as f: json.dump(result, f, indent=2)
    return result


def validate(path, *, max_exports=100, max_input_gb=6):
    path = Path(path).resolve(); m = json.loads(path.read_text())
    if m.get('version') != 1: raise ValueError('Unsupported dataset manifest')
    seen = set(); hashes = set(); total = 0
    for split in ('train', 'holdout'):
        if not m.get(split): raise ValueError(f'Empty {split}')
        for e in m[split]:
            p = Path(e['export']).resolve() / 'candidates.jsonl'
            if str(p) in seen or e['sha256'] in hashes: raise ValueError('Duplicate/overlapping dataset export')
            seen.add(str(p)); hashes.add(e['sha256']); total += p.stat().st_size
            if file_hash(p) != e['sha256']: raise ValueError(f'Export changed: {p}')
    if len(seen) > max_exports or total > max_input_gb * 1e9: raise ValueError('Dataset input budget exceeded')
    for r in m.get('regression', []):
        p = Path(r['baseline']) / 'results.json'
        if file_hash(p) != r['baseline_sha256']: raise ValueError(f'Regression baseline changed: {p}')
    return m, dict(exports=len(seen), input_bytes=total, manifest_sha256=file_hash(path))
