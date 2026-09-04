from __future__ import annotations

"""V2.24.0 game-hit-context selftest."""

from src.engine.input.object_hit_v2223 import (
    CameraHitRegionV240,
    GameHitRegionV240,
    ObjectHitRegistryV2223,
    game_rect_to_screen_polygon,
    transform_camera_point_to_game,
    transform_game_rect_to_camera_aabb,
)


def _check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"[PASS] {name}")


def main() -> None:
    print("V2.24.0 SELFTEST")
    print("===============")

    region = GameHitRegionV240("balloon", 10, 20, 30, 40, role="target")
    _check("HitRegion is a simple game-local AABB", region.rect == (10, 20, 30, 40))

    mapped = GameHitRegionV240.from_mapping({
        "id": "crate",
        "x": 2,
        "y": 4,
        "w": 20,
        "h": 10,
        "role": "breakable",
    })
    _check("mapping API defaults to game-local coordinates", mapped is not None and mapped.role == "breakable")

    vp = (100.0, 200.0, 800.0, 600.0)
    screen_region = GameHitRegionV240.from_screen_rect(
        "screen_target", (150, 260, 30, 40), vp
    )
    _check(
        "screen-absolute convenience converts back to game-local",
        screen_region.rect == (50.0, 60.0, 30.0, 40.0),
    )

    poly = game_rect_to_screen_polygon(region, vp)
    _check(
        "game-local rectangle adds viewport offset",
        poly == ((110.0, 220.0), (140.0, 220.0), (140.0, 260.0), (110.0, 260.0)),
    )

    def screen_to_camera(x: float, y: float):
        # Deliberately anisotropic transform; proves width/height are NOT scaled
        # naively and all four corners are used.
        return 2.0 * x + 5.0, 3.0 * y - 7.0

    cam = transform_game_rect_to_camera_aabb(region, vp, screen_to_camera)
    _check("four-corner game->camera AABB transform", isinstance(cam, CameraHitRegionV240))
    assert cam is not None
    _check(
        "camera AABB has transformed bounds",
        abs(cam.x - 225.0) < 1e-6
        and abs(cam.y - 653.0) < 1e-6
        and abs(cam.width - 60.0) < 1e-6
        and abs(cam.height - 120.0) < 1e-6,
    )

    def camera_to_screen(x: float, y: float):
        return (x - 5.0) / 2.0, (y + 7.0) / 3.0

    game_point = transform_camera_point_to_game(cam.x, cam.y, vp, camera_to_screen)
    _check(
        "camera->game point transform returns viewport-local coordinates",
        game_point is not None
        and abs(game_point[0] - region.x) < 1e-6
        and abs(game_point[1] - region.y) < 1e-6,
    )

    class NoObjectsGame:
        pass

    class EmptyScene:
        def get_hit_regions(self):
            return ()

    class SimpleScene:
        def get_hit_regions(self):
            return [GameHitRegionV240("t1", 5, 7, 20, 25)]

    registry = ObjectHitRegistryV2223()
    empty = registry.snapshot(1, 123.0, scene=EmptyScene())
    _check("games may expose zero hit objects", empty.game_regions == ())

    simple = registry.snapshot(2, 124.0, scene=SimpleScene())
    _check("shot-time snapshot freezes game regions", len(simple.game_regions) == 1)
    _check("snapshot keeps V2.22 legacy compatibility", len(simple.regions) in (0, 1))

    class WrappedGame:
        def get_hit_regions(self):
            return [GameHitRegionV240("wrapped", 1, 2, 3, 4)]

    class GameHolder:
        def __init__(self):
            self.game = WrappedGame()

    class Wrapper:
        def __init__(self):
            self.inner = GameHolder()

    wrapped = registry.snapshot(3, 125.0, scene=Wrapper())
    _check(
        "defensive wrapper traversal reaches game provider",
        len(wrapped.game_regions) == 1 and wrapped.game_regions[0].object_id == "wrapped",
    )

    context = registry.game_context(2)
    _check(
        "game context schema exposes V2.24 AABBs",
        context is not None and context.get("schema") == "2.22.3" and context.get("game_hit_context_schema") == "2.24.0" and "game_hit_regions" in context,
    )

    _check("V2.24.0 contains no live-hit authority", not hasattr(registry, "push_camera_hit"))

    print("\nAll V2.24.0 selftests passed.")


if __name__ == "__main__":
    main()
