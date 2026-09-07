# V2.24.3 — Why the local ROI moved upward

## Physical evidence from V2.24.2

The first dedicated game-context shooting test showed:

- game regions were present and transformed (`game=7 camera=7`),
- moving-target snapshots and classification worked,
- overlap/no-shoot logic worked,
- EMPTY regions correctly used the old global detector,
- but many region-enabled shots immediately logged
  `V2.24.1 GLOBAL-FALLBACK reason=outside_detector_roi`,
- and successful shots could occur without any V2.24.1 LOCAL-SEARCH line.

That combination means the problem was integration, not merely target semantics.

## Root cause A — content_rect default origin

HitScanner and HitInput explicitly treat `content_rect` as viewport-local. When
an explicit `content_rect` is absent, the older default returned a copy of the
viewport rectangle including viewport.x/y. HitScanner then added viewport.x/y
to content.x/y again before inverse homography. This can displace the ordinary
camera ROI.

V2.24.3 corrects the implicit runtime value to local `(0, 0, w, h)`.

## Root cause B — V2.24.1 hook position

V2.24.1 wrapped `CandidateGeneratorV2._extract_candidates()`. That is not the
first common point for every live proposal branch. Legacy/V1 proposals and the
V2 waiting-post-peak path can exist before or without that call.

V2.24.3 therefore applies the frozen region mask in
`HitScanner._frame_roi_mask()`. All first-pass detector branches consume that
mask.

## Authority invariant

Object context still only says **where to search first**. It never says that a
shot occurred and never moves final XY into a region. A hit still requires the
normal physical PRE->POST evidence and tracking/confirmation chain.

## Rescue invariant

V2.22.5 FULL-RESCUE remains the escape hatch. When rescue is requested,
V2.24.3 returns the original global ROI before high-recall extraction.
