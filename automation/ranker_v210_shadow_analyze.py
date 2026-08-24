from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("content/ai/ranking_v210/shadow_sessions")


def _latest_session(session: str | None) -> Path | None:
    if session:
        path = ROOT / session
        return path if path.is_dir() else None
    if not ROOT.is_dir():
        return None
    folders = [path for path in ROOT.iterdir() if path.is_dir()]
    return max(folders, key=lambda path: path.stat().st_mtime) if folders else None


def _load(path: Path) -> list[dict[str, Any]]:
    rows = []
    for file in sorted(path.glob("shot_*.json")):
        try:
            row = json.loads(file.read_text(encoding="utf-8"))
            if isinstance(row, dict):
                rows.append(row)
        except Exception:
            pass
    return rows


def _metrics(rows: list[dict[str, Any]], block: str, radius: int) -> dict[str, Any]:
    key = f"rank_{radius}"
    ranks = []
    for row in rows:
        value = row.get(block, {}).get(key)
        if value is not None:
            try:
                ranks.append(int(value))
            except Exception:
                pass
    if not ranks:
        return {"covered": 0, "top1_pct": 0.0, "top3_pct": 0.0, "median_rank": None, "mrr": None}
    array = np.asarray(ranks, dtype=np.float64)
    return {
        "covered": len(ranks),
        "top1_pct": round(100.0 * float(np.mean(array <= 1)), 3),
        "top3_pct": round(100.0 * float(np.mean(array <= 3)), 3),
        "median_rank": float(np.median(array)),
        "mrr": round(float(np.mean(1.0 / array)), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze V2.10 V8 shadow on unseen camera shots")
    parser.add_argument("--session", type=str, default=None)
    args = parser.parse_args()
    folder = _latest_session(args.session)
    if folder is None:
        print("No V2.10 shadow session found.")
        raise SystemExit(1)
    rows = _load(folder)
    if not rows:
        print("V2.10 shadow session contains no rows.")
        raise SystemExit(1)

    print("=" * 78)
    print("V2.10 V8 SHADOW HOLDOUT ANALYSIS")
    print("=" * 78)
    print(f"Session: {folder.name}")
    print(f"Shots:   {len(rows)}")
    loaded = sum(bool(row.get("v8_shadow", {}).get("loaded")) for row in rows)
    print(f"Rows with V8 model loaded: {loaded}/{len(rows)}")
    print()
    for radius in (10, 20, 42):
        actual = _metrics(rows, "actual", radius)
        v8 = _metrics(rows, "v8_shadow", radius)
        print(
            f"<= {radius:2d}px  ACTUAL top1={actual['top1_pct']:6.2f}% top3={actual['top3_pct']:6.2f}% "
            f"med={str(actual['median_rank']):>6s} | "
            f"V8 top1={v8['top1_pct']:6.2f}% top3={v8['top3_pct']:6.2f}% "
            f"med={str(v8['median_rank']):>6s}"
        )
    print("=" * 78)
    print("V8 is shadow-only; these numbers do not change the game's selected hit.")


if __name__ == "__main__":
    main()
