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
