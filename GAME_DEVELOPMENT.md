# Game development — hit context and future objects

## Status

This document is the game-facing source of truth from V2.24.0 onward for how a
game may expose approximate hit-search regions to the hit engine.

Existing games do **not** need to change. `get_hit_regions()` is optional.

## 1. Existing game contract

A game module still exposes:

```python
def create_game(game_root, viewport):
    return MyGame(game_root, viewport)
```

The returned game object may implement the existing lifecycle methods such as
`update(dt)`, `render(screen)`, `handle_event(event)`, `on_enter()` and
`on_exit()`.

V2.24.0 adds one optional method:

```python
def get_hit_regions(self):
    ...
```

If it is absent or returns an empty sequence, hit detection stays in the normal
global mode.

## 2. HitRegion: simple and fast

New games should import:

```python
from src.engine.input.hit_regions import HitRegion
```

Example:

```python
def get_hit_regions(self):
    return (
        HitRegion(
            object_id="balloon_17",
            x=420,
            y=180,
            width=60,
            height=60,
            role="target",
        ),
    )
```

The region is an approximate AABB only. Do not create meshes, polygons or image
masks for the hit detector.

`get_hit_regions()` must be cheap: O(number of active regions), no file I/O, no
OpenCV, no blocking work and no expensive collision calculations.

## 3. Coordinate contract

`HitRegion.x/y/width/height` are always **viewport-local / game-local pixels**.
They are never camera coordinates and never absolute desktop/screen coordinates.

The engine owns the transform:

```text
game-local AABB
      |
      | + viewport offset
      v
screen/projector coordinates
      |
      | existing calibration / homography / scanport mapping
      v
camera coordinates
```

Because perspective calibration is not a uniform scale, the engine transforms
all four rectangle corners and then creates a camera-space AABB. Games never do
this themselves.

If camera transformation is unavailable, the object-aware path must not guess.
The normal global detector remains the fallback.

### Games that currently render with absolute screen coordinates

Some current helpers (for example parts of `range_projection`) can return
screen-absolute positions. Convert them when exposing hit context:

```python
from src.engine.input.hit_regions import HitRegion

region = HitRegion.from_screen_rect(
    "target_1",
    screen_rect,
    self.viewport,
    role="target",
)
```

Rendering may continue to use absolute coordinates. The hit-context API remains
game-local.

## 4. Approximate search rectangle is not exact game collision

A HitRegion answers:

> Where should the physical hit engine search first?

It does **not** answer:

> Did the player definitely hit this object?

A future round balloon may expose a rectangular AABB to the detector. After the
physical hit engine returns the actual `HitEvent.game_x/game_y`, the game may
still use an exact circle/ellipse collision test.

The same applies to head/body zones, irregular sprites, targets and breakable
objects.

## 5. Shot-time snapshot

The V2.22.3 shot-critical loop already snapshots game context before normal
scene update when a microphone shot is dispatched. V2.24.0 extends that snapshot
with game-local and camera-local AABBs.

This matters for moving objects. The hit engine must use the position displayed
at the shot, not where an object moves a frame later.

Normal scene wrapping is:

```text
OverlayScene -> GameScene -> game instance
```

V2.24.0 proxies `get_hit_regions()` through both wrapper layers so the existing
shot-critical snapshot can see the game provider.

## 6. Roles

`role` is lightweight context. Suggested values include:

- `target`
- `no_shoot`
- `breakable`
- `neutral`
- game-specific strings when useful

Role is not hit authority. A `target` region cannot turn an unsupported camera
change into a hit, and a `no_shoot` region is still just context until a real
physical hit has been resolved.

## 7. No objects is a valid game

All of these are valid:

```python
# No method at all.
class Game:
    pass
```

```python
def get_hit_regions(self):
    return ()
```

```python
def get_hit_regions(self):
    return tuple(obj.hit_region() for obj in self.objects if obj.hittable)
```

The first two use the global detector exactly as before.

## 8. Future Object/Item layer

V2.24.0 deliberately does not introduce a large ECS/object framework. The
planned V2.25 layer can build on this API with concepts such as:

```text
GameObject
HittableObject
BreakableObject
ObjectManager
```

A future object may contain rich gameplay state (sprite, velocity, HP,
animation, virtual distance, physical dimensions), but the hit engine should
still receive only a flattened `HitRegion` AABB.

Example direction:

```python
class BreakableObject(GameObject):
    def get_hit_region(self):
        if self.destroyed:
            return None
        return HitRegion(...)
```

`ObjectManager.get_hit_regions()` can then gather active regions for the game's
`get_hit_regions()` implementation.

## 9. Range projection relationship

The existing range projection engine remains separate. Objects may use it to
calculate on-screen size and position from virtual distance and physical size.

```text
range_projection
      |
      v
GameObject screen/game geometry
      |
      v
HitRegion (game-local AABB)
```

`range_projection` should not know about GameObjects, HitScanner or OpenCV.

## 10. What V2.24.0 does and does not do

V2.24.0 provides:

- stable game-facing `HitRegion` API,
- optional `game.get_hit_regions()`,
- OverlayScene/GameScene proxying,
- shot-time geometry freezing through the existing V2.22.3 runtime,
- game-local -> screen -> camera AABB transformation,
- camera -> game point helper,
- transform provenance and fail-safe behaviour,
- legacy V2.22.3 compatibility.

V2.24.0 does **not** yet make local object-aware physical search authoritative.
That is V2.24.1. Until then, the existing global hit path remains unchanged.

<!-- V2.24.1 OBJECT_LOCAL_PHYSICAL_SEARCH -->
## 11. V2.24.1 — object-aware local physical search

V2.24.1 is the first runtime consumer of the frozen camera-space HitRegions.
The contract from V2.24.0 does not change: games still expose approximate
viewport-local/game-local AABBs through `get_hit_regions()`.

At PANG the V2.24.0 snapshot freezes the object geometry. The new live path is:

```text
PANG
  |
  v
frozen camera HitRegions
  |
  +-- add camera-space safety margin (default 36 px)
  +-- merge overlapping windows
  +-- intersect with the detector's existing valid perspective ROI
  v
V2.22.5 FAST physical proposal extraction inside those windows
  |
  v
V2.22.5 PRE->POST local confirmation
  |
  +-- physical proof -> ordinary resolver / HitEvent
  |
  +-- no proof / zero proposals -> ONE existing V2.22.5 FULL-RESCUE
                                   (GLOBAL, not region-masked)
```

### Authority rule

A region means **"search here first"**, never **"this object was hit"**.

- `target`, `no_shoot`, `breakable` and other roles are searched equally.
- No candidate coordinate is snapped to a region or object centre.
- A shot just outside an object is not moved onto it.
- Physical PRE->POST evidence remains mandatory.
- Exact object collision still happens in the game after a physical
  `HitEvent.game_x/game_y` exists.

### Fallback behaviour

The normal/global V2.22.5 path is used unchanged when:

- the game exposes no HitRegions,
- no shot-time snapshot exists,
- game->camera transformation was unavailable,
- transformed regions do not intersect the detector ROI,
- region data is invalid or exceeds the safety cap,
- V2.22.5 requests its one FULL-RESCUE pass.

The FULL-RESCUE is deliberately not masked by game context. This is the main
false-negative safety valve for V2.24.1.

### Runtime diagnostics

For an object-aware shot, expect a line like:

```text
[V2.24.1 LOCAL-SEARCH] shot=17 regions=4 merged=2 valid=8.7% margin=36px
```

If the existing rescue is needed:

```text
[V2.24.1 GLOBAL-FALLBACK] shot=17 reason=v2225_full_rescue
[V2.22.5 FULL-RESCUE] shot=17 using high-recall extractor
```

`last_window_debug` also receives `v241_*` fields including region count,
merged-window count and the fraction of the original valid detector ROI searched.

### Performance scope

V2.24.1 restricts the **candidate/proposal extraction** stage by replacing the
`valid` mask passed to the already-installed V2.22.5 FAST extractor. Some
upstream evidence maps may still be computed for the normal perspective ROI.
This is intentional for the first object-aware release: it gives us a small,
reversible integration before considering deeper ROI/crop changes.

### Tunables

Runtime settings (defaults):

```text
object_local_search_enabled_v241 = true
object_local_search_margin_px_v241 = 36.0
object_local_search_max_regions_v241 = 256
object_local_search_log_v241 = true
```

Do not tune the margin for benchmark scores before the V2.24.2 physical test
scene exists. The first goal is correctness and false-attraction measurement.

---

## V2.24.2 — Hit Context Test scene

Before V2.25 introduces reusable objects, use `Hit Context Test (V2.24.2)` as
the physical acceptance harness for the HitRegion bridge.

The scene intentionally uses plain local test rectangles rather than a shared
GameObject hierarchy. It implements the same contract future games use:

```python
from src.engine.input.hit_regions import HitRegion

def get_hit_regions(self):
    return tuple(...)
```

Cases visible in the scene:

- stationary `target`,
- stationary `no_shoot`,
- moving `target`,
- overlapping `target` + `no_shoot`,
- target near a viewport edge,
- an outside-region challenge,
- `E` toggles EMPTY regions so hit detection must use the ordinary global path.

For camera/audio hits the scene reads the latest frozen shot context and draws
its game regions in cyan for a few seconds. This makes the PANG-time position
visible even if the moving object has advanced before HitEvent delivery.

The scene does **not** grant object authority. Returned `HitEvent.game_x/game_y`
is classified against frozen/current geometry only for diagnostics. It never
snaps hit XY to a box and never changes detector evidence thresholds.

<!-- V2.24.3 GAME_DEVELOPMENT -->
## V2.24.3 detector integration note

Games still expose the same viewport-local `get_hit_regions()` contract. No game API changed. V2.24.3 only moves consumption of those frozen camera AABBs upward to HitScanner's first-pass ROI so every live proposal branch sees the same region restriction. Object roles remain context only; final game collision uses returned HitEvent XY.

<!-- V2.24.4 GAME_DEV_WORKING_SPACE -->
## V2.24.4 — detector working-space note

Game authors still provide `HitRegion` in **game/viewport-local XY**. Games must
not know about camera crops or detector worker coordinates. The engine owns the
entire transform chain:

`game-local -> screen -> full camera -> detector working space`

V2.24.4 adds the final internal step using the live V2.22.1 analysis geometry.
This does not change the game-facing API and games must never pre-transform or
scale their HitRegions for the camera/detector.

<!-- V2.25.0 GAME_OBJECT_FOUNDATION -->
## V2.25.0 — GameObject contract

Prefer `src.engine.game_objects` and composition. GameObject owns identity,
projected game-local geometry and lifecycle; hit shape, ballistic body, damage,
motion and reactions are independent capabilities. `make_living_object()` and
`make_breakable_object()` are convenience presets, not inheritance requirements.

For every camera shot, ObjectManager should resolve exact collision from
`HitEvent.shot_id` against the frozen PANG snapshot. Never use current moving
geometry when the matching frozen snapshot exists and never alter HitEvent XY.
See `GAME_OBJECT_SYSTEM.md` for the stable API.

<!-- V2.25.1 OBJECT_REGION_PHYSICAL_PROPOSAL -->
## V2.25.1 — detector fairness around GameObjects

`GameObject.get_hit_region()` still means only "search here first". V2.25.1 may
partition those regions so every physical area contributes a bounded proposal set,
but it must not infer which object the player intended to hit. Exact object shape,
z-order, health, penetration and reactions remain downstream of resolved HitEvent XY.

For overlapping objects at the same projected location, detector work may be grouped;
ObjectManager must still resolve all frozen objects at the final XY in gameplay order.

<!-- V2.25.2 REGISTERED_FRESHNESS_AUTHORITY -->
## V2.25.2 — GameObject context and physical authority

A GameObject HitRegion may supply a physical search partition, but a local hit cannot
be emitted merely because a legacy/bank candidate lies inside that region. Normal local
authority must have registered immediate PRE→POST freshness at the same physical XY.
This remains independent of role, health, penetration and reactions. Exact object
collision still begins only after HitEvent XY exists.

<!-- V2.25.3 CROSS_THREAD_NOVELTY_AUTHORITY -->
## V2.25.3 — repeated hotspot handling remains physical

GameObjects still contribute only search regions. V2.25.3 can prefer a newly appearing
camera-space physical change over a hotspot repeatedly confirmed on earlier shots, but
it may not prefer a target over no-shoot or move XY into an object. Re-hit is legal and
can recover via stronger registered evidence.

<!-- V2.25.3-r3 FULL_FILE_DELIVERY -->
## V2.25.3-r3 – full-file delivery

Packaging-only correction after the V2.25.3 runtime work. Future delivery for this development line uses complete replacement files only: no prepare/apply scripts and no menu/settings mutation helpers. `content/menu.json` is shipped as a complete schema version 1 file with the diagnostic games already present. Central configuration files are not replaced unless the version actually requires a source change.

