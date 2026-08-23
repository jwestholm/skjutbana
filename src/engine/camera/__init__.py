from .camera_manager import camera_manager
from .hit_scanner import HitScanner, hit_scanner

# Detector V2 is deliberately installed as a hybrid wrapper around the existing
# HitScanner candidate generator. Any import/initialisation failure leaves the
# legacy detector untouched so experimental code cannot prevent the game from
# starting.
try:
    from .candidate_generator_v2 import install_candidate_generator_v2

    # V2.4 is additive: patch the tested V2/V2.3 CandidateGeneratorV2 class
    # before the existing installer creates its engine instance. If V2.4 fails,
    # ordinary V2 can still be installed below.
    try:
        from .detector_v24_extension import apply_detector_v24_extension

        apply_detector_v24_extension()
    except Exception as exc:
        print(f"[DETECTOR-V2.4] unavailable, V2 core kept: {exc}")

    # V2.5 is additive on top of the measured V2.4 detector. It leaves the
    # original V2.4 tile candidates untouched, adds refined centre hypotheses,
    # and records localisation + shadow-accumulator telemetry.
    try:
        from .detector_v25_extension import apply_detector_v25_extension

        apply_detector_v25_extension()
    except Exception as exc:
        print(f"[DETECTOR-V2.5] unavailable, V2.4 kept: {exc}")

    install_candidate_generator_v2(HitScanner)
except Exception as exc:
    print(f"[DETECTOR-V2] unavailable, legacy detector kept: {exc}")

__all__ = ["camera_manager", "hit_scanner"]
