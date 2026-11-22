"""Selftest: opt-in touch_radius edge rule on hit shapes (disc touches shape = inside)."""
from src.engine.game_objects.geometry import HitShapeSpec, ObjectGeometry, shape_contains


def main():
    g = ObjectGeometry(x=100, y=100, width=20, height=30)
    for spec in (HitShapeSpec.rect(), HitShapeSpec.ellipse(), HitShapeSpec.polygon(((0, 0), (1, 0), (1, 1), (0, 1))),
                 HitShapeSpec.union(HitShapeSpec.rect(width=0.5), HitShapeSpec.ellipse(x=0.5, width=0.5))):
        plain = spec.snapshot(g)
        assert "touch_radius" not in plain
        assert shape_contains(plain, 110, 115) and not shape_contains(plain, 121.5, 115)
        from dataclasses import replace
        edged = replace(spec, touch_radius=2.0).snapshot(g)
        assert edged["touch_radius"] == 2.0
        assert shape_contains(edged, 110, 115)
        assert shape_contains(edged, 121.5, 115), spec.kind          # 1.5 px outside, radius 2 -> touches
        assert not shape_contains(edged, 122.5, 115), spec.kind      # 2.5 px outside -> no
        assert not shape_contains(edged, 110, 132.5), spec.kind
    print("edge rule selftest OK")


if __name__ == "__main__":
    main()
