from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.engine.offline.media_bank_v219 import audit_media, read_media_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit V2.19 media bank for split leakage / missing provenance")
    parser.add_argument("--manifest", default="content/ai/media_bank_v219/media_manifest.jsonl")
    parser.add_argument("--out", default="content/ai/media_bank_v219/media_audit.json")
    parser.add_argument("--near-hamming", type=int, default=3)
    args = parser.parse_args()
    assets = read_media_manifest(Path(args.manifest))
    report = audit_media(assets, near_duplicate_hamming=args.near_hamming)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print("V2.19 MEDIA BANK AUDIT")
    print("=====================")
    print(f"Assets                       : {report['assets']}")
    print(f"Exact cross-split duplicates : {len(report['exact_cross_split_duplicates'])}")
    print(f"Near cross-split duplicates  : {len(report['near_cross_split_duplicates'])}")
    print(f"Unknown license/source meta  : {len(report['unknown_license_paths'])}")
    print(f"Frozen-holdout leakage gate  : {report['ok_for_frozen_holdout']}")
    if report['near_duplicate_check_skipped']:
        print("Near-duplicate pair scan skipped (>20k perceptual hashes); exact duplicate gate still ran.")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
