"""Bounded SHADOW training and fixed regression suite from an explicit manifest.

No automatic export discovery or synthetic training-data growth. Cron remains an
operator decision. --dry-run validates inputs and prints the plan without writes.
"""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .dataset_manifest import freeze, validate

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = dict(min_free_gb=20, min_available_mem_gb=8, max_input_gb=6,
    max_exports=100, max_matrix_gb=2, train_address_gb=6, max_run_disk_gb=16,
    negative_cap=64, sample_seed=20261122, timeout_s=14400)


def _mem_available_gb():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'): return int(line.split()[1]) / 1e6
    return 0


def _game_running():
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = p.read_bytes().split(b'\0')
            if any(Path(a.decode(errors='replace')).name == 'main.py' for a in args): return True
        except (OSError, ValueError): pass
    return False


def _atomic(path, text):
    tmp = path.with_name(path.name + '.tmp'); tmp.write_text(text); os.replace(tmp, path)


def _publish(root, night, summary):
    _atomic(night / 'summary.json', json.dumps(summary, indent=2) + '\n')
    lines = [f'# Nightly {night.name}', '', f"Status: {summary['status']}",
             'OFFLINE_REPLAY_ONLY_UNVALIDATED; live_authority=false.', '',
             f"Stage: {summary.get('stage')}"]
    if summary.get('error'): lines += ['', f"ERROR: {summary['error']}"]
    for arm, r in summary.get('regression', {}).items(): lines.append(f"- {arm}: {r['verdict']}")
    if 'model' in summary:
        t = summary['model']['metrics']['holdout_top1']
        lines.append(f"- Holdout: model {t['model']['hits']}/{t['model']['shots']}; live score {t['live_score']['hits']}/{t['live_score']['shots']}")
    lines += ['', 'RUNNING is not success: if the supervisor is killed, this record remains visibly unfinished.']
    report = '\n'.join(lines) + '\n'
    _atomic(night / 'REPORT.md', report); _atomic(root / 'LATEST.md', report)


def _run(cmd, env_extra, log, timeout):
    env = {k: v for k, v in os.environ.items() if not k.startswith('SKJUTBANA_')}
    env.update(SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env.update(env_extra)
    with log.open('w') as f:
        return subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode


def _regression_preflight(manifest):
    from .candidate_cache import file_hash
    errors = {}
    for r in manifest.get('regression', []):
        if not r.get('baseline_sha256'): continue
        baseline = json.loads((Path(r['baseline']) / 'results.json').read_text())['manifest']
        args = r['args']
        settings = Path(args[args.index('--settings') + 1])
        if not settings.is_absolute(): settings = REPO / settings
        if not settings.exists() or file_hash(settings) != baseline['settings_sha256']:
            errors[r['name']] = dict(verdict='NOT_COMPARABLE', reason='settings snapshot differs from frozen regression baseline')
        ai_path = Path(baseline.get('ai_settings_path', 'content/ai/settings.json'))
        if not ai_path.is_absolute(): ai_path = REPO / ai_path
        if not baseline.get('ai_settings_sha256') or not ai_path.exists() or file_hash(ai_path) != baseline['ai_settings_sha256']:
            reason = errors.get(r['name'], {}).get('reason', '')
            errors[r['name']] = dict(verdict='NOT_COMPARABLE', reason=(reason + '; AI settings provenance missing or changed').lstrip('; '))
        for bg in baseline.get('background', []):
            p = Path(bg)
            if not p.is_absolute(): p = REPO / p
            if not p.exists() or (p.is_dir() and not (p / 'camera.bin').exists()):
                errors[r['name']] = dict(verdict='NOT_COMPARABLE', reason=f'raw regression background unavailable: {p}; packed inputs unsupported')
    return errors


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/data/skjutbana/nightly')
    p.add_argument('--manifest'); p.add_argument('--config'); p.add_argument('--output-root')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--freeze-from-model', help='create a manifest from an explicitly chosen historical model')
    a = p.parse_args(argv)
    if a.freeze_from_model:
        if not a.manifest: p.error('--manifest is required with --freeze-from-model')
        freeze(a.freeze_from_model, a.manifest); return 0
    root = Path(a.root); manifest_path = Path(a.manifest or root / 'dataset_manifest.json')
    cfg = dict(DEFAULT_CONFIG)
    if a.config: cfg.update(json.loads(Path(a.config).read_text()))
    for k in DEFAULT_CONFIG:
        v = cfg[k]
        if k == 'negative_cap' and v is None: continue
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0: raise ValueError(f'Invalid budget: {k}')
        if k in ('max_exports', 'negative_cap', 'sample_seed') and int(v) != v: raise ValueError(f'Integer budget required: {k}')
    if a.dry_run:
        m, info = validate(manifest_path, max_exports=cfg['max_exports'], max_input_gb=cfg['max_input_gb'])
        errors = _regression_preflight(m)
        disk_path = root
        while not disk_path.exists(): disk_path = disk_path.parent
        preconditions = dict(game_running=_game_running(), available_mem_gb=_mem_available_gb(), free_disk_gb=shutil.disk_usage(disk_path).free / 1e9)
        print(json.dumps(dict(mode='DRY_RUN_NO_WRITES', preconditions=preconditions, regression_errors=errors, inputs=info, budgets=cfg, regression=m.get('regression', []),
            train_exports=len(m['train']), holdout_exports=len(m['holdout']), new_training_synthetic_jobs=0,
            automatic_session_exports=0, packed_camera_reading=False), indent=2)); return 2 if errors else 0
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return 0
        night = Path(a.output_root or root) / (dt.datetime.now().strftime('%Y-%m-%d_%H%M%S_%f'))
        night.mkdir(parents=True, exist_ok=False); (night / 'logs').mkdir()
        s = dict(status='RUNNING', stage='preflight', started=time.time(), manifest=str(manifest_path))
        _publish(root, night, s)  # before hashing/loading/training, never stale success
        try:
            if _game_running(): raise RuntimeError('Game/main.py is running')
            if _mem_available_gb() < cfg['min_available_mem_gb']: raise RuntimeError('Insufficient available RAM')
            if shutil.disk_usage(night).free / 1e9 < cfg['min_free_gb'] + cfg['max_run_disk_gb']:
                raise RuntimeError('Insufficient disk reserve including run budget')
            m, info = validate(manifest_path, max_exports=cfg['max_exports'], max_input_gb=cfg['max_input_gb'])
            s['inputs'] = info
            errors = _regression_preflight(m)
            if errors:
                s['regression'] = errors
                raise RuntimeError('NOT_COMPARABLE: regression preflight failed; explicitly freeze a compatible baseline')
            _atomic(night / 'dataset_manifest.json', json.dumps(m, indent=2) + '\n')
            cache_root = root / 'candidate_cache'
            # Conservative bound: JSON input * 2 covers cache plus allowed work matrices;
            # use the explicit matrix cap as well. Refuse instead of deleting old data.
            if info['input_bytes'] * 2 + cfg['max_matrix_gb'] * 1e9 > cfg['max_run_disk_gb'] * 1e9:
                raise RuntimeError('Conservative cache/work disk estimate exceeds max_run_disk_gb; review budget explicitly')
            s['stage'] = 'regression'; s['regression'] = {}; _publish(root, night, s)
            if not m.get('regression'): raise RuntimeError('No fixed regression suite in manifest')
            from .__main__ import compare
            for r in m['regression']:
                out = night / 'regression' / r['name']; out.parent.mkdir(exist_ok=True)
                code = _run([sys.executable, '-m', 'automation.bench', 'synth', *r['args'], '--output', str(out)],
                            r['env'], night / 'logs' / f"{r['name']}.log", cfg['timeout_s'])
                if code: raise RuntimeError(f"Regression {r['name']} exit={code}")
                try: result = compare(Path(r['baseline']), out, quiet=True)['overall']
                except (Exception, SystemExit) as exc:
                    s['regression'][r['name']] = dict(verdict='NOT_COMPARABLE', reason=str(exc))
                    raise RuntimeError(f"NOT_COMPARABLE: {r['name']}: {exc}") from exc
                s['regression'][r['name']] = result
                _publish(root, night, s)
                if result['verdict'] == 'WORSE': raise RuntimeError(f"Regression WORSE: {r['name']}")
            s['stage'] = 'training'; _publish(root, night, s)
            cmd = [sys.executable, '-m', 'automation.bench.train', '--train', *[e['export'] for e in m['train']],
                   '--holdout', *[e['export'] for e in m['holdout']], '--output', str(night / 'model'),
                   '--cache-root', str(cache_root),
                   '--sample-seed', str(int(cfg['sample_seed'])), '--max-matrix-gb', str(cfg['max_matrix_gb']),
                   '--address-limit-gb', str(cfg['train_address_gb'])]
            if cfg['negative_cap'] is not None: cmd += ['--negative-cap', str(int(cfg['negative_cap']))]
            code = _run(cmd, {}, night / 'logs' / 'train.log', cfg['timeout_s'])
            if code: raise RuntimeError(f'Training failed exit={code}; see logs/train.log')
            s['model'] = dict(metrics=json.loads((night / 'model/metrics.json').read_text()),
                              sha256=json.loads((night / 'model/model.json').read_text())['sha256'])
            s.update(status='COMPLETE', stage='finished', finished=time.time())
        except (Exception, SystemExit) as exc:
            s.update(status='FAILED', error=f'{type(exc).__name__}: {exc}', finished=time.time())
            _publish(root, night, s); return 1
        _publish(root, night, s)
        with (root / 'history.jsonl').open('a') as f: f.write(json.dumps(s) + '\n')
        return 0

if __name__ == '__main__': raise SystemExit(main())
