from __future__ import annotations

import json
from pathlib import Path
from src.engine.ai.training_v223.dataset import compile_dataset
from src.engine.ai.training_v223.registry import load_champion_entry, load_registry


def main() -> None:
    dataset = compile_dataset(include_legacy=True)
    split = dataset.split()
    champion = load_champion_entry()
    registry = load_registry()
    print("V2.23.0 STATUS")
    print("==============")
    print(f"Dataset: {dataset.summary()}")
    print(f"Split: dev={len(split.development)} validation={len(split.validation)} protected_holdout={len(split.holdout)} provisional={split.provisional}")
    print(f"Registered challengers: {len(registry.get('models', []))}")
    if champion:
        print(f"Research shadow champion: {champion.get('trial_id')} kind={champion.get('kind')}")
        print(f"Metrics: {champion.get('metrics')}")
    else:
        print("Research shadow champion: none")
    latest = Path("content/ai/training_v223/reports/latest.json")
    print(f"Latest train report: {latest if latest.exists() else 'none'}")
    print("Live authority: unchanged / NO")


if __name__ == "__main__":
    main()
