from __future__ import annotations

from pathlib import Path
from src.engine.ai.training_v223.dataset import compile_dataset
from src.engine.ai.training_v223.registry import champion_gate_status, load_champion_entry, load_registry


def main() -> None:
    dataset = compile_dataset(include_legacy=True)
    split = dataset.split()
    champion = load_champion_entry()
    gate = champion_gate_status()
    registry = load_registry()
    print("V2.23.1 STATUS")
    print("==============")
    print(f"Dataset: {dataset.summary()}")
    print(
        f"Split: dev={len(split.development)} oracle20={sum(int(r.oracle20) for r in split.development)} "
        f"validation={len(split.validation)} oracle20={sum(int(r.oracle20) for r in split.validation)} "
        f"protected_holdout={len(split.holdout)} provisional={split.provisional}"
    )
    print(f"Registered challengers: {len(registry.get('models', []))}")
    if champion:
        print(f"Champion file: {champion.get('trial_id')} kind={champion.get('kind')}")
        print(f"Champion usable in shadow: {gate.get('usable')} ({gate.get('reason')})")
        if not gate.get("usable"):
            print(f"Gate failures: {gate.get('gate', {}).get('reasons', [])}")
        print(f"Metrics: {champion.get('metrics')}")
    else:
        print("Research shadow champion: none")
    latest = Path("content/ai/training_v223/reports/latest.json")
    print(f"Latest train report: {latest if latest.exists() else 'none'}")
    print("Live authority: unchanged / NO")


if __name__ == "__main__":
    main()
