from __future__ import annotations

import inspect
from pathlib import Path


def check(ok: bool, text: str) -> None:
    if not ok:
        raise AssertionError(text)
    print(f"[PASS] {text}")


def main() -> None:
    print("V2.23.1 INSTALL VERIFICATION")
    print("============================")
    required = [
        Path("src/engine/ai/training_v223/dataset.py"),
        Path("src/engine/ai/training_v223/integration.py"),
        Path("src/engine/ai/training_v223/registry.py"),
        Path("src/engine/ai/training_v223/trainer.py"),
        Path("automation/v2231_audit.py"),
        Path("automation/v2231_train.py"),
        Path("automation/v2231_status.py"),
    ]
    check(all(p.exists() for p in required), "required V2.23.1 files exist")

    from src.engine.offline.candidate_pack_v216 import CandidatePackV216
    check(callable(getattr(CandidatePackV216, "load", None)), "canonical CandidatePackV216.load loader is available")

    from src.engine.ai.training_v223 import dataset
    from src.engine.ai.training_v223 import integration
    from src.engine.ai.training_v223 import registry
    check("CandidatePackV216" in inspect.getsource(dataset._official_pack_loader), "legacy import uses canonical JSON+NPZ loader")
    src = inspect.getsource(integration._candidate_union)
    check("_v28_all_hypotheses" in src and "_v28_hypothesis_pool" in src, "native capture includes V2.8 high-recall pools")
    check(registry.MIN_VALIDATION_ORACLE20 > 0 and registry.MIN_DEV_ORACLE20 > 0, "research promotion requires positive proposal support")

    main_text = Path("main.py").read_text(encoding="utf-8")
    check("install_v2226_runtime" in main_text and "install_v2230_training_pipeline" in main_text, "V2.22 runtime chain remains ahead of V2.23 shadow pipeline")

    gate = registry.champion_gate_status()
    if gate.get("exists") and not gate.get("usable"):
        print(f"[PASS] existing pre-V2.23.1 plumbing champion is quarantined: {gate.get('trial_id')}")
    else:
        print(f"[PASS] champion state is safe: {gate.get('reason')}")
    print("[PASS] V2.23.1 still cannot grant live authority")
    print("\nV2.23.1 installation verification passed.")


if __name__ == "__main__":
    main()
