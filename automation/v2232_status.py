from __future__ import annotations

from pathlib import Path
from src.engine.ai.training_v223.dataset import compile_dataset
from src.engine.ai.training_v223.domain import select_fresh_f2_domain
from src.engine.ai.training_v223.framepack import discover_framepacks
from src.engine.ai.training_v223.proposal import PROPOSAL_ROOT
from src.engine.ai.training_v223.registry import champion_gate_status, load_champion_entry, load_registry


def main() -> None:
    ds = compile_dataset(include_legacy=True)
    domain = select_fresh_f2_domain(ds.records)
    engineering = type(ds)(records=domain.engineering_records, legacy_report=ds.legacy_report)
    split = engineering.split()
    framepacks = discover_framepacks()
    proposal_files = list(PROPOSAL_ROOT.glob('*/shot_*.json')) if PROPOSAL_ROOT.exists() else []
    champion = load_champion_entry(); gate = champion_gate_status(); reg = load_registry()
    print("V2.23.2 STATUS")
    print("==============")
    print(f"Dataset: {ds.summary()}")
    print(f"Fresh F2 domain: session={domain.session_id} shots={len(domain.records)} oracle20={sum(int(r.oracle20) for r in domain.records)} reason={domain.reason}")
    print(f"Engineering split: dev={len(split.development)} oracle20={sum(int(r.oracle20) for r in split.development)} validation={len(split.validation)} oracle20={sum(int(r.oracle20) for r in split.validation)} protected_holdout={len(split.holdout)}")
    print(f"Framepacks: {len(framepacks)} | proposal sidecars: {len(proposal_files)}")
    print(f"Registered challengers: {len(reg.get('models', []))}")
    if champion:
        print(f"Champion file: {champion.get('trial_id')} kind={champion.get('kind')}")
        print(f"Champion usable in shadow: {gate.get('usable')} ({gate.get('reason')})")
        if not gate.get('usable'):
            print(f"Gate failures: {gate.get('gate',{}).get('reasons',[])}")
        print(f"Validation metrics: {champion.get('metrics')}")
        print(f"Fresh-domain metrics: {champion.get('domain_metrics')}")
    else:
        print("Research shadow champion: none")
    latest = Path('content/ai/training_v223/reports/latest.json')
    print(f"Latest train report: {latest if latest.exists() else 'none'}")
    print("Live authority: unchanged / NO")

if __name__ == '__main__':
    main()
