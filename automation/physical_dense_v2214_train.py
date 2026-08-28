from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.engine.offline.fullframe_benchmark_v2214 import train_physical_dense_v2214
from src.engine.offline.physical_dense_v2214 import DEFAULT_MODEL_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="Train V2.21.4 physical dense ranker on DEVELOPMENT full-frame shots only")
    parser.add_argument("--root", default="content/ai/candidate_shadow_v216")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--config", default="content/ai/physical_dense_v2214.json")
    parser.add_argument("--report", default="content/ai/reports/v2214/physical_dense_training_v2214.json")
    args = parser.parse_args()

    report = train_physical_dense_v2214(
        Path(args.root),
        model_path=Path(args.model),
        config_path=Path(args.config),
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    t = report["training"]
    print("V2.21.4 PHYSICAL DENSE TRAINING")
    print("================================")
    print(f"Development full-frame shots : {report['development_fullframe_shots_used']}")
    print(f"Protected not used           : {report['protected_fullframe_shots_not_used']}")
    print(f"Dense pool DEV oracle20      : {t['development_dense_pool_oracle20']:.4f}")
    print(f"Dense pool DEV oracle42      : {t['development_dense_pool_oracle42']:.4f}")
    print(f"Positive samples             : {t['positive_samples']}")
    print(f"Negative pool samples        : {t['negative_pool_samples']}")
    print(f"Pairwise examples            : {t['pair_count_stage1']} + {t['pair_count_stage2']}")
    print(f"Loss stage1                  : {t['stage1_loss_head_tail'][0]:.4f} -> {t['stage1_loss_head_tail'][1]:.4f}")
    print(f"Loss stage2                  : {t['stage2_loss_head_tail'][0]:.4f} -> {t['stage2_loss_head_tail'][1]:.4f}")
    print(f"Model                        : {report['model_path']}")
    print(f"Report                       : {report_path}")
    print("NEXT: run automation.physical_dense_v2214_benchmark with this frozen model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
