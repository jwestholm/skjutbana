from __future__ import annotations
import argparse
from src.engine.ai.training_v223.heatmap_v2236 import prepare_heatmap_sessions


def main() -> int:
    ap = argparse.ArgumentParser(description='Prepare V2.23.6 registered direct-heatmap caches')
    ap.add_argument('--session', default='latest')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    report = prepare_heatmap_sessions(session=args.session, force=args.force)
    print('\nV2.23.6 PREPARE SUMMARY\n=======================')
    print(f"Status: {report.get('status')}")
    for sid, row in report.get('sessions', {}).items():
        print(f"  {sid}: processed={row.get('processed',0)} cached={row.get('cached',0)} cache={row.get('cache_mb',0):.1f}MB errors={len(row.get('errors',[]))}")
    return 0 if report.get('status') == 'ok' else 1

if __name__ == '__main__':
    raise SystemExit(main())
