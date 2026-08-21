from .camera_manager import camera_manager
from .hit_scanner import HitScanner, hit_scanner

# Detector V2 is deliberately installed as a hybrid wrapper around the existing
# HitScanner candidate generator. Any import/initialisation failure leaves the
# legacy detector untouched so experimental code cannot prevent the game from
# starting.
try:
    from .candidate_generator_v2 import install_candidate_generator_v2

    install_candidate_generator_v2(HitScanner)
except Exception as exc:
    print(f"[DETECTOR-V2] unavailable, legacy detector kept: {exc}")

__all__ = ["camera_manager", "hit_scanner"]
