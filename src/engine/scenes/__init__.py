"""Startup hooks for the measured AI detector/ranker experiments.

The V2.8 integration remains authoritative. V2.9 is installed after it and is
strictly observational/shadow-only.
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
