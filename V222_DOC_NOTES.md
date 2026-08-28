# V2.22 documentation notes

The V2.22 implementation was designed after reviewing the existing project documentation, in particular the current-state, AI-context and hit-detection planning material.

Some older documentation describes the AI only as a ranker over the current detector's hotspots. That remains true for the legacy `SimpleAIMemory` path, but V2.22 adds a higher-level resolver contract that can also consume **independent external proposal/ranking experts** such as the V2.21.x physical full-frame work.

For V2.22 and later, use these rules when older wording conflicts:

1. `HitScanner` remains the live baseline and coordinate-emission owner.
2. `AIRuntime` evolves into the integration/shot-intelligence layer.
3. `ShotResolverV222` chooses between discrete candidate clusters; it never blends XY coordinates.
4. Independent physical experts may propose candidates not present in the original small detector list, but must publish a short list to the resolver.
5. Heavy vision must run outside the synchronous resolver path.
6. `advisory` is the required first live mode.
7. Resolver `confidence` is uncalibrated until a held-out calibration stage is implemented.
8. Game target/hotspot information is context/evidence, never ground truth.
9. The existing V2.21.5 dense pool is currently treated as a promising proposal source, not a live-authority model.

`V222_SHOT_RESOLVER.md` is the canonical V2.22 design document and `V222_TEST_PLAN.md` is the verification procedure.
