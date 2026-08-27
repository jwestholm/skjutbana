# V2.20.2 — Alpha backgrounds + frayed-paper hole appearance

## Why
Visual QA of V2.20.1 showed that game/sprite PNG assets can contain transparent pixels. When those pixels are effectively shown against black, a new dark bullet hole can become almost invisible if GT lands outside the foreground sprite.

## Transparent media handling
`scenario_generator_v220.py` now loads still images with `cv2.IMREAD_UNCHANGED`.

If an image has a meaningful alpha channel, transparent areas are composited onto a deterministic procedural background. The fallback background contains:

- a diagonal multi-colour gradient,
- a weak broad accent wash,
- low-frequency colour noise,
- a minimum brightness floor so it cannot collapse to black.

The background is deterministically selected from the media identity, so the same source/index produces the same world background and reproducibility is preserved.

Opaque images are unchanged.

## Hole appearance
`hole_patch_bank_v220.py` retains the compact V2.20/V2.20.1 hole model, but the RGB stamp now includes subtle extra physical cues:

- dark core,
- warm/light torn-paper rim,
- weak irregular fraying,
- sparse lighter tonal flecks,
- slight non-neutral core/rim tint.

The change is deliberately subtle: the hole must remain small and camera-like, not become a large high-contrast cartoon marker.

## Selftest
A new test creates a genuinely transparent RGBA asset and verifies that:

- transparent regions do not become black,
- the fallback keeps colour variation,
- the generated new hole still passes V2.20.1 visibility QA.

This remains offline/shadow/training infrastructure only and does not change live hit authority.
