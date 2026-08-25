from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.engine.offline.archive import discover_shot_cases
from src.engine.offline.shot_case import write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "V2.12: inspect an existing shooting-image archive, pair before/after "
            "images without modifying the archive, and write a portable JSONL manifest."
        )
    )
    parser.add_argument("root", type=Path, help="Archive root on the shooting PC")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("content/ai/offline/v212/archive_manifest.jsonl"),
        help="Output JSONL manifest (default: content/ai/offline/v212/archive_manifest.jsonl)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("content/ai/offline/v212/archive_summary.json"),
        help="Output summary JSON",
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after N paired shots (smoke test)")
    parser.add_argument(
        "--no-image-stats",
        action="store_true",
        help="Do not open images; faster inventory but no background/shape statistics",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"ERROR: archive root does not exist or is not a directory: {root}")
        return 2

    cases, summary = discover_shot_cases(
        root,
        inspect_images=not args.no_image_stats,
        limit=args.limit,
    )
    manifest_count = write_manifest(args.manifest, cases)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("V2.12 OFFLINE ARCHIVE INSPECTION")
    print("================================")
    print(f"Archive root        : {root}")
    print(f"Image files         : {summary.image_files}")
    print(f"Candidate groups    : {summary.candidate_groups}")
    print(f"Paired shots        : {summary.paired_shots}")
    print(f"With ground truth   : {summary.labelled_shots}")
    print(f"Without ground truth: {summary.unlabelled_shots}")
    print(f"Ambiguous/unpaired  : {summary.ambiguous_groups}")
    print(f"Ignored images      : {summary.ignored_images}")
    print(f"Standalone labelled : {summary.standalone_labelled_images}  (future Hole-AI assets, not replay pairs)")
    print(f"Shape mismatches    : {summary.shape_mismatch_shots}")
    if summary.background_types:
        print("Background estimate :")
        for name, count in sorted(summary.background_types.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {name:16s} {count}")
    print(f"Manifest            : {args.manifest} ({manifest_count} shots)")
    print(f"Summary             : {args.summary}")
    if summary.examples_ambiguous:
        print("\nFirst ambiguous groups:")
        for value in summary.examples_ambiguous[:10]:
            print(f"  {value}")
    if summary.examples_unclassified:
        print("\nFirst unclassified filenames (useful if archive naming differs):")
        for value in summary.examples_unclassified[:10]:
            print(f"  {value}")
    if summary.examples_standalone_labelled:
        print("\nFirst standalone labelled hole patches:")
        for value in summary.examples_standalone_labelled[:10]:
            print(f"  {value}")
    print("\nSource archive was READ ONLY; no source files were changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
