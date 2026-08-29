from __future__ import annotations

import inspect
from pathlib import Path


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def main() -> None:
    print("V2.23.0 INSTALL VERIFICATION")
    print("============================")
    required = [
        Path("src/engine/ai/training_v223/schema.py"),
        Path("src/engine/ai/training_v223/dataset.py"),
        Path("src/engine/ai/training_v223/model.py"),
        Path("src/engine/ai/training_v223/trainer.py"),
        Path("src/engine/ai/training_v223/integration.py"),
        Path("automation/v2230_selftest.py"),
    ]
    check(all(p.exists() for p in required), "required V2.23 files exist")
    from src.engine.ai.training_v223 import FEATURE_NAMES, SCHEMA_VERSION
    check(SCHEMA_VERSION == "2.23.0" and len(FEATURE_NAMES) >= 15, "schema/physical feature contract imports")
    from src.engine.ai.training_v223.integration import install_v2230_training_pipeline
    check(callable(install_v2230_training_pipeline), "runtime capture integration imports")
    main_text = Path("main.py").read_text(encoding="utf-8")
    check("install_v2230_training_pipeline" in main_text, "main.py installs V2.23 after runtime patches")
    check("install_v2226_runtime" in main_text, "V2.22.6 runtime chain preserved")
    check("eligible_for_live_authority\": False" in Path("src/engine/ai/training_v223/registry.py").read_text(encoding="utf-8"), "registry hard-codes no live authority")
    check("holdout_evaluated_for_selection\": False" in Path("src/engine/ai/training_v223/trainer.py").read_text(encoding="utf-8"), "automatic trainer does not evaluate protected holdout for selection")
    print("\nV2.23.0 installation verification passed.")


if __name__ == "__main__":
    main()
