from __future__ import annotations

"""Static V2.24.0 status.

The shot-context registry is process-local, so a standalone CLI process cannot
inspect snapshots held by a separately running game.  Runtime shots with game
regions print a [V2.24.0 GAME-CONTEXT] line instead.
"""

from src.engine.input.object_hit_v2223 import SCHEMA_VERSION


def main() -> None:
    print("V2.24.0 STATUS")
    print("===============")
    print(f"Game hit context schema: {SCHEMA_VERSION}")
    print("Game API: optional game.get_hit_regions()")
    print("Coordinates: game-local AABB -> four-corner camera AABB")
    print("No regions / transform unavailable: global detector fallback")
    print("Runtime diagnostics: [V2.24.0 GAME-CONTEXT] in main.py output")
    print("Local physical search: NEXT (V2.24.1)")
    print("Live authority: unchanged / NO new authority")


if __name__ == "__main__":
    main()
