from __future__ import annotations

from src.engine.ai.training_v223.dataset import (
    LEGACY_ROOTS,
    _v217_split_lookup,
    load_legacy_candidate_record,
)


def main() -> None:
    print("V2.23.1 LEGACY PACK PROBE")
    print("========================")
    failures = 0
    for root in LEGACY_ROOTS:
        paths = sorted(root.glob("**/shot_*.json")) if root.exists() else []
        if not paths:
            print(f"[SKIP] {root}: no packs")
            continue
        base = root.parent if root.name == "sessions" else root
        lookup = _v217_split_lookup(base) if "candidate_shadow_v216" in str(root) else {}
        record, reason = load_legacy_candidate_record(paths[0], root, split_lookup=lookup)
        if record is None:
            failures += 1
            print(f"[FAIL] {root}: {paths[0]} -> {reason}")
        else:
            print(
                f"[PASS] {root}: candidates={len(record.candidates)} "
                f"oracle20={record.oracle20} nearest={record.nearest_distance_px:.1f}px "
                f"split={record.split_hint} loader={record.metadata.get('legacy_loader')} reason={reason}"
            )
    if failures:
        raise SystemExit(f"Legacy probe failed for {failures} root(s)")
    print("\nLegacy canonical-loader probe passed.")


if __name__ == "__main__":
    main()
