# V2.24.4 physical test plan

## Stage A — software

Run:

```bash
python3 -m automation.v244_prepare
python3 -m automation.v244_selftest
python3 -m automation.v244_verify_install
python3 -m automation.v244_status
```

All commands should pass before physical testing.

## Stage B — startup

Start:

```bash
python3 main.py
```

Open **Spel -> Hit Context Test (V2.24.4)**.

Startup should include:

```text
[V2.24.3] HitScanner-level object ROI installed ...
[V2.24.4] full-camera -> detector working-space HitRegion mapping installed ...
[V2.24.4 TESTSCENE] entered ...
```

## Stage C — short physical matrix

No new photographs are required if shots are placed approximately as in the
previous V2.24.2/V2.24.3 runs. Suggested sequence:

1. stationary green target,
2. red no-shoot,
3. moving target,
4. overlap area,
5. edge target,
6. yellow area just outside `outside_challenge`,
7. press `E` and fire one arbitrary EMPTY/global shot.

The key acceptance criterion is **not** that every existing detector ranking is
perfect. It is that the region-enabled first pass now truly searches the mapped
object regions instead of silently returning a global mask.

## Expected region-enabled log

Every normal region-enabled shot should contain:

```text
[V2.24.0 GAME-CONTEXT] ... camera=7 transform=homography
[V2.24.4 ROI-MAP] ...
[V2.24.4 LOCAL-ROI] ... region=<NON-ZERO>% ...
```

`ROI-MAP` should show:

- full camera dimensions,
- V2.22.1 crop origin and size,
- detector working image dimensions,
- derived scale,
- one example full-camera region and its mapped work-space rectangle.

For the current V2.22.1 implementation, `work` should normally equal crop size
and scale should normally be approximately `(1.0000,1.0000)`.

## Global rescue

If the local pass cannot establish physical evidence, V2.22.5 may request one
FULL-RESCUE. Then expect:

```text
[V2.24.4 GLOBAL-RESCUE-ROI] shot=N ...
[V2.22.5 FULL-RESCUE] shot=N using high-recall extractor
```

That pass must use the ordinary full calibrated V2.22.1 ROI, not the object mask.

## EMPTY/global

After `E`, `objects=0` is expected and there should be no V2.24.4 ROI-MAP for
that shot. The ordinary global detector remains the compatibility path.

## Physical acceptance gate

Proceed to V2.25.0 only when:

- region-enabled shots have non-zero `region` coverage,
- ROI-MAP camera->work values look plausible,
- stationary target and no-shoot can be classified at their real positions,
- moving-target snapshot still behaves correctly,
- outside-region shots are not magnetised onto a target,
- EMPTY/global still works,
- FULL-RESCUE, when used, is explicitly global.
