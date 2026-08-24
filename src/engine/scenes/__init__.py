"""Startup hooks for measured AI detector/ranker experiments.

V2.8 remains authoritative. V2.9 captures the full hypothesis dataset. V2.10
loads only a new V8 SHADOW ranker and never changes the game's selected hit.
"""
from __future__ import annotations

try:
    from src.engine.ai.ranker_v6_extension import install_ranker_v6_extension

    install_ranker_v6_extension(source="scenes.__init__")
except Exception as exc:
    print(f"[RANKER-V6] startup error: {exc!r}")

try:
    from src.engine.ai.ranker_v7_extension import install_ranker_v7_extension

    install_ranker_v7_extension(source="scenes.__init__")
except Exception as exc:
    print(f"[RANKER-V7] V2.9 startup error: {exc!r}")

try:
    from src.engine.ai.ranker_v8_extension import install_ranker_v8_extension

    install_ranker_v8_extension(source="scenes.__init__")
except Exception as exc:
    print(f"[RANKER-V8] V2.10 startup error: {exc!r}")
