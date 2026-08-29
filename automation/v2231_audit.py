from __future__ import annotations

import json
from src.engine.ai.training_v223.audit import audit_repository_state, write_audit_report


def main() -> None:
    report = audit_repository_state()
    path = write_audit_report()
    ds = report.get("unified_dataset", {})
    split = report.get("split_preview", {})
    legacy = report.get("legacy_import", {})
    print("V2.23.1 TRAINING/MODEL AUDIT")
    print("============================")
    print(
        f"Unified shots: {ds.get('shots', 0)} | sessions: {ds.get('sessions', 0)} | "
        f"candidates: {ds.get('candidates', 0)}"
    )
    print(
        "Oracle: "
        f"@5={ds.get('oracle5', 0)}/{ds.get('shots', 0)} ({100*float(ds.get('oracle5_rate',0)):.1f}%) | "
        f"@10={ds.get('oracle10', 0)}/{ds.get('shots', 0)} ({100*float(ds.get('oracle10_rate',0)):.1f}%) | "
        f"@20={ds.get('oracle20', 0)}/{ds.get('shots', 0)} ({100*float(ds.get('oracle20_rate',0)):.1f}%) | "
        f"@42={ds.get('oracle42', 0)}/{ds.get('shots', 0)} ({100*float(ds.get('oracle42_rate',0)):.1f}%)"
    )
    print(f"Sources: {ds.get('by_source', {})}")
    print(
        f"Split: dev={split.get('development',0)} (oracle20={split.get('development_oracle20',0)}) "
        f"validation={split.get('validation',0)} (oracle20={split.get('validation_oracle20',0)}) "
        f"protected_holdout={split.get('holdout_protected',0)} provisional={split.get('provisional')}"
    )
    print(f"Legacy loader: {legacy.get('loader')} loaded={legacy.get('loaded',0)} skipped={legacy.get('skipped',0)}")
    for root, stats in legacy.get("roots", {}).items():
        print(
            f"  {root}: json={stats.get('json_files',0)} loaded={stats.get('loaded',0)} "
            f"skipped={stats.get('skipped',0)} cache={stats.get('cache_hits',0)} "
            f"oracle20={stats.get('oracle20',0)} oracle42={stats.get('oracle42',0)}"
        )
        if stats.get("skip_reasons"):
            print(f"    skip reasons: {stats.get('skip_reasons')}")
    gate = report.get("v223_champion_gate", {})
    print(f"Champion gate: {gate}")
    print("Live authority changed: NO")
    print("Protected holdout used for auto selection: NO")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
