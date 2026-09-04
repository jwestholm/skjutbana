from __future__ import annotations
import argparse
from src.engine.ai.training_v223.proposal import expand_session
from src.engine.ai.training_v223.trainer_v2233 import prepare_dense_sessions
from src.engine.ai.training_v223.heatmap_v2236 import prepare_heatmap_sessions
from src.engine.ai.training_v223.trainer_v2236 import train_direct_heatmap_v2236


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default='latest')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    expand_session(args.session)
    prepare_dense_sessions(session=args.session)
    prepare_heatmap_sessions(session=args.session)
    report = train_direct_heatmap_v2236(quick=args.quick, prepare=False)
    return 0 if report.get('status') == 'ok' else 1

if __name__ == '__main__':
    raise SystemExit(main())
