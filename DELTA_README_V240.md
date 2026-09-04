# Skjutbana V2.24.0 delta

## Game Hit Context foundation

Overlay this ZIP on the current `dev` branch after V2.23.6.

V2.24.0 reuses and extends the existing V2.22.3 object-hit/shot-critical
foundation. It does not add another runtime installer and does not change live
hit authority.

### Main changes

- stable `src.engine.input.hit_regions.HitRegion` game-facing AABB API,
- game-local coordinates are canonical,
- four-corner game/screen -> camera AABB transform,
- camera -> game point helper,
- `GameScene` proxies optional `game.get_hit_regions()`,
- `OverlayScene` proxies the provider through normal scene wrapping,
- V2.22.3 shot-time snapshot now stores game and camera AABBs,
- missing regions or unavailable transforms remain safe/global-fallback cases,
- old V2.22.3 screen-polygon/AI context stays compatible,
- `GAME_DEVELOPMENT.md` documents future game/Object/Breakable usage.

### Install/test

```bash
unzip -o skjutbana_v2.24.0_game_hit_context_foundation_delta.zip -d .
python3 -m automation.v240_selftest
python3 -m automation.v240_verify_install
python3 -m automation.v240_status
python3 -m automation.v240_apply_docs
python3 main.py
```

Existing games require no changes.

### Next planned version

V2.24.1 consumes the frozen camera AABBs for local physical PRE->POST search,
then falls back to the ordinary global detector when no region is supported.
