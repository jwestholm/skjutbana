# V2.22.5 append-only documentation patch

The updater in `automation/v2225_apply_docs.py` appends these sections without replacing older history.

## CURRENT_STATE.md

```markdown
---
## V2.22.5 Fast Proposal + Local Confirmation — 2026-08-29

- V2.22.4-r2 proved audio dispatch and async AI shadow are no longer the dominant common-path latency. A physical test showed main acknowledgement ~31 ms and AI shadow ~70 ms, while the first async CV job was ~1.26 s.
- Detector stage profiling isolated V2 `_extract_candidates` as the largest measured stage (~0.73 s in that shot). The full live pipeline also attempted another whole-ROI detector pass to obtain tracking persistence, causing timeout before the second result returned.
- V2.22.5 changes normal live semantics to **GLOBAL PROPOSE -> LOCAL CONFIRM**. One async global proposal seeds ordinary HitScanner tracks; later camera frames validate small PRE->POST patches around existing candidate coordinates instead of searching the full ROI again.
- The live V2 worker now has a sparse FAST extractor using dilation/local maxima + bounded top-K/NMS. The historical high-recall extractor remains available as one explicit FULL rescue and remains the offline/F2 path.
- Re-hit/hole-in-hole remains legal when fresh temporal evidence exists. Local confirmation never drags/interpolates candidate XY.
- Startup/calibration audio is consumed while HitScanner is not ACTIVE and cannot become a delayed gameplay shot.
```

## HIT_DETECTION_PLAN.md

```markdown
---
## V2.22.5 runtime funnel — global proposal is not persistence

Do not use repeated whole-viewport/whole-ROI search merely to increment a candidate track's persistence counter.

Normal gameplay funnel:

1. audio timestamp and game/object snapshot,
2. one async GLOBAL FAST candidate proposal,
3. preserve camera XY/provenance through V1/V2 merge + cleanup,
4. LOCAL PRE->POST confirmation around the proposed coordinates on a later real camera frame,
5. existing HitScanner track association/readiness,
6. immediate camera->game HitEvent,
7. AI/advisory work remains off the critical path.

If local confirmation cannot validate any proposal, permit one FULL high-recall rescue. Treat rescue frequency as an explicit runtime metric: the desired mature system should rarely need it.

The same local physical-change primitive is a future building block for object-first games: object hit regions can ask whether a fresh physical change occurred inside their frozen shot-time area without first requiring a globally authoritative miss coordinate.
```

## AI_CONTEXT.md

```markdown
---
## V2.22.5 proposal / confirmation semantics

- `candidate` means a proposed physical location, not yet proof of a new hit.
- `local_confirm` means a later timestamped frame provides compact PRE->POST change near the same candidate XY.
- Local confirmation does not move XY. Diagnostic best-offset values are not an authority coordinate.
- An unchanged old hole may remain hole-like but should fail current-shot temporal confirmation.
- A re-hit at an old hole may pass because fresh PRE->POST evidence exists.
- Live FAST extraction is not a replacement for the full research/high-recall extractor. FULL rescue and offline/F2 use preserve the broader evidence source.
- AI ranking/fusion must preserve provenance: FAST proposal, full rescue, local confirmation, V1/V2 source and known-hole context remain distinguishable.
```
