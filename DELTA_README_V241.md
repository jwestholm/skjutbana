# Skjutbana V2.24.1 delta

## Object-aware local physical search

Overlay this ZIP on the **tested V2.24.0 dev checkout**.

V2.24.1 is the first live consumer of V2.24.0 `HitRegion` camera AABBs. It does
not create a new detector and does not give objects hit authority. Instead it
constrains the first V2.22.5 FAST physical proposal pass to expanded/merged
shot-time object regions. Existing V2.22.5 PRE->POST confirmation remains the
physical gate and its one FULL-RESCUE remains completely global.

### Main changes

- new `src/engine/shot_object_local_v241.py`,
- installed after V2.22.6 from `main.py`,
- frozen V2.24.0 camera HitRegions are expanded by 36 camera px by default,
- overlapping windows are transitively merged,
- local windows intersect the detector's existing perspective/valid ROI,
- only the live `shot-cv-v2224` worker is changed; offline/F2/replay extraction
  stays on its previous path,
- `target`, `no_shoot`, `breakable`, etc. are all physical-search context,
- no candidate coordinate is snapped to an object,
- missing/bad context fails open to today's global detector path,
- queued V2.22.5 FULL-RESCUE bypasses the object mask and is global,
- `v241_*` diagnostics expose region count, merged count and searched fraction,
- `GAME_DEVELOPMENT.md` is updated for the new runtime contract,
- living project MDs are updated append-only/idempotently by the docs command.

### Install / verify

From the repository root:

```bash
unzip -o skjutbana_v2.24.1_object_local_physical_search_delta.zip -d .
python3 -m automation.v241_apply_docs
python3 -m automation.v241_selftest
python3 -m automation.v241_verify_install
python3 -m automation.v241_status
python3 main.py
```

`v241_apply_docs` first makes sure the V2.24.0 foundation sections exist, then
adds V2.24.1 to `ARCHITECTURE.md`, `HIT_DETECTION_PLAN.md`, `CURRENT_STATE.md`
and `ROADMAP.md`. It is safe to run more than once. `ROADMAP.md` is created if
it does not yet exist.

### What to look for at runtime

Games without `get_hit_regions()` should behave exactly as before.

For a shot with game context:

```text
[V2.24.0 GAME-CONTEXT] shot=... game=... camera=... transform=...
[V2.24.1 LOCAL-SEARCH] shot=... regions=... merged=... valid=...% margin=36px
```

If local physical proof is insufficient and V2.22.5 requests rescue:

```text
[V2.24.1 GLOBAL-FALLBACK] shot=... reason=v2225_full_rescue
[V2.22.5 FULL-RESCUE] shot=... using high-recall extractor
```

### Regression status in the build environment

- V2.24.1 selftest: PASS
- V2.24.1 install verification: PASS
- V2.24.0 HitRegion/context selftest: PASS
- V2.22.5 FAST/local-confirmation selftest: PASS
- Python compile: PASS
- V2.24.0 + V2.24.1 docs patch repeated: idempotent PASS

### Next planned version

**V2.24.2**: dedicated game-context test scene with moving target/no-shoot
objects, overlap, edge cases, empty context and shots immediately outside a
region. That is where local-search success, global fallback, false object
attraction, XY error, latency and transform correctness get measured physically.
