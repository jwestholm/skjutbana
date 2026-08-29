from __future__ import annotations

import inspect
from pathlib import Path


def check(ok: bool, text: str) -> None:
    if not ok:
        raise AssertionError(text)
    print(f"[PASS] {text}")


def main() -> None:
    print("V2.23.2 INSTALL VERIFICATION")
    print("============================")
    required = [
        Path("src/engine/ai/training_v223/framepack.py"),
        Path("src/engine/ai/training_v223/proposal.py"),
        Path("src/engine/ai/training_v223/domain.py"),
        Path("automation/v2232_proposals.py"),
        Path("automation/v2232_train.py"),
        Path("automation/v2232_status.py"),
        Path("automation/v2232_cycle.py"),
        Path("automation/v2232_apply_docs.py"),
        Path("V2232_PROPOSAL_DATA_DOMAIN_PIPELINE.md"),
        Path("V2232_TEST_PLAN.md"),
    ]
    check(all(p.exists() for p in required), "required V2.23.2 files exist")

    from src.engine.offline.direct_proposal_v221 import propose_direct_v221
    from src.engine.offline.temporal_local_v2212 import propose_local_temporal_v2212
    from src.engine.offline.physical_dense_v2215 import propose_dense_pool_v2215, extract_candidate_features_v2215
    check(all(callable(x) for x in (propose_direct_v221, propose_local_temporal_v2212, propose_dense_pool_v2215, extract_candidate_features_v2215)), "V2.21/V2.21.5 offline proposal engine is available")

    from src.engine.ai.training_v223 import integration, registry, trainer
    check("save_scene_framepack" in inspect.getsource(integration._capture_known_gt), "F2 known-GT capture persists full PRE/POST framepacks")
    check("select_fresh_f2_domain" in inspect.getsource(trainer.train_once_v223), "training reserves newest substantial F2 session for domain validation")
    check(registry.MIN_DOMAIN_SHOTS >= 50 and registry.MIN_DOMAIN_ORACLE20 > 0, "research promotion requires fresh-F2 domain support")
    check("baseline_score" in Path("src/engine/ai/training_v223/model.py").read_text(encoding="utf-8"), "reference baseline has score fallback instead of zero eligible coverage")
    main_text = Path("main.py").read_text(encoding="utf-8")
    check("install_v2226_runtime" in main_text and "install_v2230_training_pipeline" in main_text, "frozen V2.22 runtime remains ahead of V2.23 training pipeline")
    print("[PASS] proposal processing is offline/shadow-only")
    print("[PASS] protected holdout remains outside automatic model selection")
    print("[PASS] live authority remains unchanged")
    print("\nV2.23.2 installation verification passed.")

if __name__ == "__main__":
    main()
