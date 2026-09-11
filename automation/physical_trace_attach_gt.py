"""Attach an independently verified camera-space label to one trace."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.engine.physical_trace import PhysicalTraceRecorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shot-id", type=int, required=True)
    parser.add_argument("--camera-x", type=float, required=True)
    parser.add_argument("--camera-y", type=float, required=True)
    parser.add_argument("--label-source", default="manual_verified")
    args = parser.parse_args()
    path = PhysicalTraceRecorder(args.root).attach_ground_truth(args.shot_id, (args.camera_x, args.camera_y), label_source=args.label_source)
    print(path)


if __name__ == "__main__":
    main()
