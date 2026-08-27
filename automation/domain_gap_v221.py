from __future__ import annotations

import argparse
from pathlib import Path

from src.engine.offline.domain_gap_v221 import DomainGapConfigV221, profile_domain_gap_v221, write_domain_gap_report_v221


def main() -> int:
    p = argparse.ArgumentParser(description="Profile V2.20 synthetic vs projector/camera candidate-domain gap")
    p.add_argument("--synthetic-root", type=Path, default=Path("content/ai/candidate_synthetic_v220"))
    p.add_argument("--physical-root", type=Path, default=Path("content/ai/candidate_shadow_v216"))
    p.add_argument("--v217-model", type=Path, default=Path("content/ai/reports/v217/new_hole_ai_v217.npz"))
    p.add_argument("--synthetic-cache", type=Path, default=Path("content/ai/reports/v218/v220_cache"))
    p.add_argument("--physical-cache", type=Path, default=Path("content/ai/reports/v218/v220_physical_cache"))
    p.add_argument("--out", type=Path, default=Path("content/ai/reports/v221/domain_gap_v221.json"))
    p.add_argument("--max-candidates", type=int, default=24000)
    a = p.parse_args()
    report = profile_domain_gap_v221(
        synthetic_root=a.synthetic_root,
        physical_root=a.physical_root,
        v217_model=a.v217_model,
        synthetic_cache=a.synthetic_cache,
        physical_cache=a.physical_cache,
        config=DomainGapConfigV221(max_candidates_per_domain=max(1000, a.max_candidates)),
    )
    write_domain_gap_report_v221(a.out, report)
    print("V2.21 DOMAIN GAP PROFILE")
    print("========================")
    print(f"Synthetic candidates : {report['sampled_candidates']['synthetic']}")
    print(f"Physical DEV cand.    : {report['sampled_candidates']['physical_development']}")
    print(f"Synthetic <=20 rows  : {report['near_gt20_candidates']['synthetic']}")
    print(f"Physical DEV <=20    : {report['near_gt20_candidates']['physical']}")
    pg = report["physical_groups"]
    print(f"Physical groups       : dev_used={pg['development_used']} confirmation_protected={pg['confirmation_protected']} holdout_protected={pg['holdout_protected']}")
    clf = report["group_domain_classifier"]
    print(f"Group domain AUC     : {clf['auc']:.4f} ({clf['interpretation']})")
    print(f"Shortcut warning     : {report['shortcut_warning']}")
    print("\nTop candidate-feature shifts:")
    for row in report["feature_shift_all_candidates"][:12]:
        print(f"  {row['feature']:<30s} KS={row['ks']:.3f} |SMD|={row['abs_smd']:.3f} W={row['wasserstein']:.4f}")
    if report["feature_shift_near_gt20"]:
        print("\nTop <=20px feature shifts:")
        for row in report["feature_shift_near_gt20"][:8]:
            print(f"  {row['feature']:<30s} KS={row['ks']:.3f} |SMD|={row['abs_smd']:.3f}")
    print("\nNEXT:", report["next_action"])
    print(f"Report: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
