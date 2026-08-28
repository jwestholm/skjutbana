# V2.22.2 r2 — real HitScanner installer fix

## Why r2 exists

The first V2.22.1/V2.22.2 live run printed:

```text
[V2.22.1] ROI optimizer unavailable; continuing full-frame: 'HitScanner' object has no attribute 'HitScanner'
[V2.22.2] Fast-path cleanup unavailable; continuing with V2.22.1: 'HitScanner' object has no attribute 'HitScanner'
```

`src.engine.camera.__init__` exports the live singleton as `hit_scanner`.  The
V2.22.1 and V2.22.2 installers used a dotted import whose local binding could
resolve to that package attribute rather than to the `src.engine.camera.hit_scanner`
submodule.  The installer then attempted `hs_module.HitScanner`, which fails when
`hs_module` is already the singleton.

r2 resolves the submodule explicitly with `importlib.import_module()` and adds a
real-module installation smoke test.  No detector thresholds, geometry, resolver
weights, model files, or game-coordinate transforms are changed by this r2.

## Expected startup

Both lines below must appear and there must be no `unavailable` line:

```text
[V2.22.1] Perspective ROI/edge-guard HitScanner patch installed
[V2.22.2] cursor/novelty/ridge/backlog fast-path patch installed
```

Only after these lines are visible is a physical shot run a valid V2.22.1/2.22.2
performance test.
