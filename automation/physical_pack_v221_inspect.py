from __future__ import annotations

import argparse
from pathlib import Path

from src.engine.offline.physical_pack_audit_v221 import audit_physical_packs_v221, write_physical_pack_audit_v221


def main() -> int:
    p = argparse.ArgumentParser(description="Audit V2.16 candidate packs for V2.21 full-frame/direct-proposal readiness")
    p.add_argument("--root", type=Path, default=Path("content/ai/candidate_shadow_v216"))
    p.add_argument("--out", type=Path, default=Path("content/ai/reports/v221/physical_pack_audit_v221.json"))
    a = p.parse_args()
    report = audit_physical_packs_v221(a.root)
    write_physical_pack_audit_v221(a.out, report)
    print("V2.21 PHYSICAL PACK AUDIT")
    print("=========================")
    print(f"Root                    : {report['root']}")
    print(f"Packs                   : {report['packs']}")
    print(f"Sessions                : {len(report['sessions'])}")
    print(f"Split provisional       : {report['split_is_provisional']}")
    av = report["availability"]
    print(f"GT patch shots          : {av['gt_patch_shots']}")
    print(f"Recent PRE patch shots  : {av['recent_pre_patch_shots']}")
    print(f"Full recent PRE shots   : {av['full_recent_pre_shots']}")
    print(f"Full POST shots         : {av['full_post_shots']}")
    print(f"Direct-ready full shots : {av['full_frame_direct_ready_shots']}")
    print(f"Oracle <=20 all         : {report['candidate_oracle_all']['20']:.4f}")
    for name in ("development", "confirmation", "holdout"):
        row = report["splits"][name]
        print(f"  {name:12s}: shots={row['shots']} full_ready={row['full_frame_ready']} oracle20={row['oracle']['20']:.4f}")
    print(f"Can direct benchmark now: {report['can_benchmark_direct_proposals_now']}")
    if report.get("next_capture_requirement"):
        print("NEXT:", report["next_capture_requirement"])
    print(f"Report                  : {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
