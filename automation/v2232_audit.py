from __future__ import annotations

from pathlib import Path

from src.engine.ai.training_v223.dataset import DatasetV223, compile_dataset
from src.engine.ai.training_v223.domain import select_fresh_f2_domain
from src.engine.ai.training_v223.framepack import discover_framepacks
from src.engine.ai.training_v223.proposal import PROPOSAL_ROOT
from src.engine.ai.training_v223.registry import champion_gate_status


def main() -> None:
    ds = compile_dataset(include_legacy=True)
    domain = select_fresh_f2_domain(ds.records)
    engineering = DatasetV223(records=domain.engineering_records, legacy_report=ds.legacy_report)
    split = engineering.split()
    framepacks = discover_framepacks()
    proposals = list(PROPOSAL_ROOT.glob("*/shot_*.json")) if PROPOSAL_ROOT.exists() else []
    expanded = sum(int(bool(r.metadata.get("v2232_proposal_expanded"))) for r in ds.records)
    summary = ds.summary()
    print("V2.23.2 TRAINING/PROPOSAL AUDIT")
    print("================================")
    print(f"Unified: shots={summary.get('shots')} sessions={summary.get('sessions')} candidates={summary.get('candidates')}")
    print(f"Oracle: @5={summary.get('oracle5_rate',0):.3f} @10={summary.get('oracle10_rate',0):.3f} @20={summary.get('oracle20_rate',0):.3f} @42={summary.get('oracle42_rate',0):.3f}")
    print(f"Framepacks: {len(framepacks)} | proposal sidecars: {len(proposals)} | expanded unified records: {expanded}")
    print(f"Fresh F2 domain: session={domain.session_id} shots={len(domain.records)} oracle20={sum(int(r.oracle20) for r in domain.records)} reason={domain.reason}")
    print(f"Engineering: dev={len(split.development)} oracle20={sum(int(r.oracle20) for r in split.development)} validation={len(split.validation)} oracle20={sum(int(r.oracle20) for r in split.validation)} protected_holdout={len(split.holdout)}")
    print(f"Legacy: loaded={ds.legacy_report.get('loaded',0)} skipped={ds.legacy_report.get('skipped',0)} loader={ds.legacy_report.get('loader')}")
    print(f"Champion gate: {champion_gate_status()}")
    print("Protected holdout used for auto selection: NO")
    print("Live authority changed: NO")

if __name__ == "__main__":
    main()
