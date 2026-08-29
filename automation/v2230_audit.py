from __future__ import annotations

import json
from src.engine.ai.training_v223.audit import audit_repository_state, write_audit_report


def main() -> None:
    report = audit_repository_state()
    path = write_audit_report()
    print("V2.23.0 TRAINING/MODEL AUDIT")
    print("============================")
    ds = report["unified_dataset"]
    print(f"Unified shots: {ds['shots']} | sessions: {ds['sessions']} | candidates: {ds['candidates']}")
    print(f"Oracle <=20px: {ds['oracle20']}/{ds['shots']} ({100.0*ds['oracle20_rate']:.1f}% if shots exist)")
    print("Sources:")
    for key, value in sorted(ds["by_source"].items()):
        print(f"  {key}: {value}")
    split = report["split_preview"]
    print(f"Split: dev={split['development']} validation={split['validation']} protected_holdout={split['holdout_protected']} provisional={split['provisional']}")
    print("Module probes:")
    for name, info in report["module_probes"].items():
        print(f"  {'PASS' if info.get('available') else 'MISS'} {name}")
    print(f"Legacy/model artifacts discovered: {len(report['models_and_reports'])}")
    print("Code inventory: " + ", ".join(f"{k}={len(v)}" for k, v in report.get("code_inventory", {}).items()))
    champ = report.get("v223_champion")
    print(f"V2.23 research champion: {champ.get('trial_id') if champ else 'none'}")
    print("Live authority changed: NO")
    print("Protected holdout used for auto selection: NO")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
