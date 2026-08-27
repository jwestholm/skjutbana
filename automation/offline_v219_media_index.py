from __future__ import annotations

import argparse
from pathlib import Path

from src.engine.offline.media_bank_v219 import (
    DEFAULT_MANIFEST,
    DEFAULT_SUMMARY,
    default_media_roots,
    index_media_roots,
    summarise_media,
    write_media_manifest,
    write_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Index local images/videos for V2.19 offline scenario generation")
    parser.add_argument("roots", nargs="*", help="Media directories/files. Default: content/ai/media_bank + assets")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--split-salt", default="skjutbana-v219-media-split-v1")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    roots = [Path(value) for value in args.roots] if args.roots else default_media_roots(repo_root)
    assets = index_media_roots(roots, repo_root=repo_root, split_salt=args.split_salt)
    write_media_manifest(Path(args.manifest), assets)
    summary = summarise_media(assets)
    write_summary(Path(args.summary), summary)

    print("V2.19 MEDIA BANK INDEX")
    print("=====================")
    print(f"Roots         : {[str(path) for path in roots]}")
    print(f"Assets        : {summary['assets']}")
    print(f"Families      : {summary['families']}")
    print(f"Kinds         : {summary['by_kind']}")
    print(f"Splits        : {summary['by_split']}")
    print(f"Categories    : {summary['by_category']}")
    print(f"Unknown license metadata: {summary['unknown_license']}")
    print(f"Manifest      : {args.manifest}")
    print("\nWhole media sources/families stay in one split; video frames never split independently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
