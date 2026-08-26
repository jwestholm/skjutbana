from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.engine.offline.hole_dataset_v213 import build_dataset_split, discover_hole_assets


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect content/ai/holes for V2.13 Hole-AI training without modifying source data.")
    p.add_argument("--root", type=Path, default=Path("content/ai/holes"))
    p.add_argument("--seed", type=int, default=21301)
    p.add_argument(
        "--holdout-backgrounds",
        default="black,checker,gray,bubbles",
        help="Comma-separated backgrounds excluded from training and used as novel-background holdout.",
    )
    p.add_argument("--json-out", type=Path, default=Path("content/ai/reports/v213/hole_archive_inspect.json"))
    p.add_argument("--no-image-check", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        return 2
    backgrounds = tuple(value.strip() for value in args.holdout_backgrounds.split(",") if value.strip())
    assets, summary = discover_hole_assets(root, inspect_images=not args.no_image_check)
    split = build_dataset_split(assets, holdout_backgrounds=backgrounds, seed=args.seed)

    print("V2.13 HOLE DATASET INSPECTION")
    print("=============================")
    print(f"Root                 : {root}")
    print(f"Synthetic PNG/JSON   : {summary.synthetic_png} / {summary.synthetic_json}")
    print(f"Real PNG/JSON        : {summary.real_png} / {summary.real_json}")
    print(f"Usable synthetic     : {summary.paired_synthetic}")
    print(f"Usable REAL          : {summary.paired_real}")
    print(f"Missing sidecar      : {summary.missing_sidecar}")
    print(f"Invalid JSON         : {summary.invalid_json}")
    print(f"Unreadable images    : {summary.unreadable_images}")
    print("\nSynthetic backgrounds:")
    for name, count in summary.synthetic_backgrounds.most_common():
        print(f"  {name:20s} {count}")
    print("\nSynthetic sessions:")
    for name, count in summary.synthetic_sessions.most_common():
        split_name = split.session_assignment.get(name, "background-only/unknown")
        print(f"  {name:20s} {count:6d}  -> {split_name}")
    print("\nProposed split:")
    for name, count in split.counts().items():
        print(f"  {name:24s} {count}")
    print(f"Novel backgrounds    : {list(split.holdout_backgrounds)}")
    print("\nANTI-CENTRE-BIAS RULE:")
    print("  The original 128x128 image centre is metadata, not a model cue.")
    print("  Training crops will be centred on RANDOM candidate positions and the")
    print("  network must also predict the hole offset from that candidate centre.")
    print("  REAL hole_*.png files are holdout-only and never used for training.")

    payload = {"schema_version": "2.13", "archive": summary.to_dict(), "split": split.to_dict()}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport               : {args.json_out}")
    print("Source images were READ ONLY; no content/ai/holes files were changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
