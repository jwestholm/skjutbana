from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

from src.engine.game_objects import (
    BallisticBody,
    DamageModel,
    DurabilityLayer,
    EffectAction,
    GameObject,
    HitShapeSpec,
    ObjectGeometry,
    ObjectManager,
    PenetrationMode,
    ProjectileProfile,
    ReactionRule,
    make_breakable_object,
    make_living_object,
)
from src.engine.shot_context_v250 import (
    ShotEmissionContextV250,
    _install_hit_input_bridge,
    _install_scanner_bridge,
    annotate_hit_event_v250,
)

ROOT = Path(__file__).resolve().parents[1]


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"[PASS] {label}")


def hit(x: float, y: float, shot_id=None):
    return SimpleNamespace(game_x=float(x), game_y=float(y), shot_id=shot_id)


def test_geometry() -> None:
    obj = GameObject(
        "ellipse",
        ObjectGeometry(100, 50, 200, 100),
        hit_shape=HitShapeSpec.ellipse(),
    )
    region = obj.get_hit_region()
    check("exact shape still publishes its AABB", region is not None and (region.x, region.y, region.width, region.height) == (100.0, 50.0, 200.0, 100.0))

    manager = ObjectManager()
    manager.add(obj)
    # Inside AABB but outside ellipse at the extreme corner.
    result = manager.resolve_hit(hit(101, 51))
    check("exact ellipse rejects AABB-only corner", result.unclaimed)
    result2 = manager.resolve_hit(hit(200, 100))
    check("exact ellipse accepts its centre", len(result2.interactions) == 1 and result2.interactions[0].object_id == "ellipse")


def test_identity_and_metadata() -> None:
    manager = ObjectManager()
    obj = GameObject(
        "car_4_window_left",
        ObjectGeometry(10, 20, 50, 40),
        entity_id="car_4",
        part_id="window_left",
        ballistic_body=BallisticBody(material_id="glass", penetration_mode=PenetrationMode.ALWAYS, penetration_resistance=0.2),
    )
    manager.add(obj)
    region = obj.get_hit_region()
    snap = region.metadata["v250_object_snapshot"]
    check("entity id is frozen", snap["entity_id"] == "car_4")
    check("part id is frozen", snap["part_id"] == "window_left")
    check("material id is frozen", snap["ballistic_body"]["material_id"] == "glass")
    check("manager generation is frozen", snap["generation"] == 1)


def test_frozen_snapshot_after_motion() -> None:
    manager = ObjectManager()
    obj = manager.add(make_living_object("moving", ObjectGeometry(100, 100, 80, 80), health=3))
    frozen_region = obj.get_hit_region()
    frozen = SimpleNamespace(shot_id=17, game_regions=(frozen_region,))
    manager._snapshot_provider = lambda sid: frozen if sid == 17 else None
    obj.geometry.x = 600
    obj.geometry.y = 400
    result = manager.resolve_hit(hit(140, 140, 17))
    check("shot_id selects frozen PANG geometry", result.used_frozen_snapshot and len(result.interactions) == 1)
    check("frozen hit applies to moved object instance", result.interactions[0].object_id == "moving")
    check("ObjectManager never changes physical XY", result.game_x == 140.0 and result.game_y == 140.0)


def test_stale_generation_guard() -> None:
    manager = ObjectManager()
    old = manager.add(make_breakable_object("reuse", ObjectGeometry(0, 0, 50, 50), integrity=1))
    region = old.get_hit_region()
    frozen = SimpleNamespace(shot_id=8, game_regions=(region,))
    manager._snapshot_provider = lambda sid: frozen
    manager.remove("reuse")
    new_obj = manager.add(make_breakable_object("reuse", ObjectGeometry(0, 0, 50, 50), integrity=1))
    before = new_obj.damage_model.layers[0].current
    result = manager.resolve_hit(hit(25, 25, 8), ProjectileProfile(profile_id="p", damage=1))
    after = new_obj.damage_model.layers[0].current
    check("reused object id receives a new generation", new_obj.generation == 2)
    check("delayed shot does not damage new generation", before == after and result.unclaimed)
    check("stale-object diagnostic event emitted", any(e.event_type == "shot.stale_object" for e in manager.events.history))


def test_penetration_chain() -> None:
    manager = ObjectManager()
    rear = manager.add(make_living_object(
        "rear", ObjectGeometry(0, 0, 100, 100), health=3,
        penetration_mode=PenetrationMode.NEVER, z_index=2,
    ))
    glass = manager.add(GameObject(
        "glass", ObjectGeometry(0, 0, 100, 100), object_type="glass", z_index=10,
        ballistic_body=BallisticBody(material_id="glass", penetration_mode=PenetrationMode.ALWAYS, penetration_resistance=0.4),
        damage_model=DamageModel([DurabilityLayer("integrity", 1)]),
    ))
    projectile = ProjectileProfile(profile_id="medium", damage=1, penetration_power=2.0, max_object_hits=4)
    result = manager.resolve_hit(hit(50, 50), projectile)
    check("front object is resolved first", [i.object_id for i in result.interactions][:2] == ["glass", "rear"])
    check("penetrable front object allows rear hit", result.interactions[0].penetrated and result.interactions[1].blocked)
    check("rear living object receives damage", rear.damage_model.layers[0].current == 2.0)
    check("front glass receives damage independently", glass.damage_model.layers[0].current == 0.0)


def test_blocking_body() -> None:
    manager = ObjectManager()
    front = manager.add(GameObject(
        "hard", ObjectGeometry(0, 0, 100, 100), z_index=10,
        ballistic_body=BallisticBody(material_id="hard", penetration_mode=PenetrationMode.NEVER, receives_damage=False),
    ))
    manager.add(make_living_object("behind", ObjectGeometry(0, 0, 100, 100), health=3, z_index=1))
    result = manager.resolve_hit(hit(50, 50), ProjectileProfile(profile_id="high", damage=9, penetration_power=99, max_object_hits=5))
    check("NEVER body blocks regardless of configured projectile power", len(result.interactions) == 1 and result.interactions[0].object_id == front.object_id and result.stopped)


def test_damage_reactions_and_effect_requests() -> None:
    manager = ObjectManager()
    box = manager.add(make_breakable_object(
        "box", ObjectGeometry(0, 0, 50, 50), integrity=1,
        break_effect="dust", break_sound="wood_break",
    ))
    manager.resolve_hit(hit(20, 20), ProjectileProfile(profile_id="p", damage=1))
    kinds = [(e.payload.get("kind"), e.payload.get("name")) for e in manager.effect_requests]
    check("break terminal event deactivates object", not box.active and box.state == "broken")
    check("dust effect requested", ("spawn_effect", "dust") in kinds)
    check("sound cue requested", ("play_sound", "wood_break") in kinds)
    terminals = [e for e in manager.events.history if e.event_type == "object.broken"]
    check("terminal event fired once", len(terminals) == 1)


def test_reaction_projectile_conditions() -> None:
    manager = ObjectManager()
    obj = GameObject(
        "fuel", ObjectGeometry(0, 0, 50, 50),
        ballistic_body=BallisticBody(penetration_mode=PenetrationMode.NEVER, receives_damage=False),
        reactions=[ReactionRule.on(
            "shot.hit",
            EffectAction.spawn_effect("explosion"),
            require_projectile_tags={"explosive"},
        )],
    )
    manager.add(obj)
    manager.resolve_hit(hit(20, 20), ProjectileProfile(profile_id="plain", tags=frozenset()))
    check("projectile-tag reaction does not fire for ordinary profile", not manager.effect_requests)
    manager.resolve_hit(hit(20, 20), ProjectileProfile(profile_id="special", tags=frozenset({"explosive"})))
    check("projectile-tag reaction can request special effect", any(e.payload.get("name") == "explosion" for e in manager.effect_requests))


def test_shot_context_bridge() -> None:
    event = SimpleNamespace(source="camera")
    annotate_hit_event_v250(event, ShotEmissionContextV250(42, 123.5, "matched"))
    check("annotation carries scanner shot_id", event.shot_id == 42)
    check("annotation carries peak timestamp", math.isclose(event.shot_peak_ts, 123.5))

    mouse = SimpleNamespace(source="mouse")
    annotate_hit_event_v250(mouse, None)
    check("non-scanner/debug hit explicitly has shot_id None", mouse.shot_id is None)

    delivered = []

    class FakeHitInput:
        def _build_event_from_camera(self, *args, **kwargs):
            return SimpleNamespace(source="camera")
        def _notify(self, event):
            delivered.append((getattr(event, "shot_id", "missing"), getattr(event, "shot_peak_ts", "missing")))
            return event

    class FakeScanner:
        def _emit_track_result(self, track, scanner_event):
            inp = FakeHitInput()
            built = inp._build_event_from_camera()
            inp._notify(built)
            return built

    _install_hit_input_bridge(FakeHitInput)
    _install_scanner_bridge(FakeScanner)
    scanner = FakeScanner()
    returned = scanner._emit_track_result(None, SimpleNamespace(shot_id=77, peak_ts=9.25, state="matched"))
    check("scanner wrapper annotates before simulated subscriber delivery", delivered == [(77, 9.25)] and returned.shot_id == 77)
    debug_event = SimpleNamespace(source="mouse")
    FakeHitInput()._notify(debug_event)
    check("notify bridge normalizes mouse/debug event shot_id", delivered[-1][0] is None)


def test_menu_patch() -> None:
    from automation.v250_apply_menu import patch_menu_text
    entry = json.loads((ROOT / "menu_games_entry_v250.json").read_text(encoding="utf-8"))
    sample = '{\n  "children": [\n    {"id":"games","children":[\n      {"id":"old","title":"Old"}\n    ]}\n  ]\n}\n'
    patched, changed, found = patch_menu_text(sample, entry)
    check("menu games folder found", found)
    check("menu entry inserted", changed and "game_objects_test_v250" in patched)
    patched2, changed2, found2 = patch_menu_text(patched, entry)
    check("menu patch is idempotent", found2 and not changed2 and patched2 == patched)


def test_install_order_source() -> None:
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    i244 = source.rfind("install_v244_runtime(App)")
    i250 = source.rfind("install_v250_runtime(App)")
    check("V2.25 installer comes after V2.24.4", i244 >= 0 and i250 > i244)


def main() -> None:
    print("V2.25.0 SELFTEST")
    print("===============")
    test_geometry()
    test_identity_and_metadata()
    test_frozen_snapshot_after_motion()
    test_stale_generation_guard()
    test_penetration_chain()
    test_blocking_body()
    test_damage_reactions_and_effect_requests()
    test_reaction_projectile_conditions()
    test_shot_context_bridge()
    test_menu_patch()
    test_install_order_source()
    print("\nAll V2.25.0 selftests passed.")


if __name__ == "__main__":
    main()
