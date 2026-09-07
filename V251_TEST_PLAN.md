# V2.25.1 physical test plan

## Software gate

Run:

```bash
python3 -m automation.v251_prepare
python3 -m automation.v251_selftest
python3 -m automation.v251_verify_install
python3 -m automation.v251_status
```

Then start `python3 main.py` and open **Spel -> Game Objects Test (V2.25.1)**.

## Physical series

Use the same five areas as V2.25.0 so the prior photographs remain a useful layout
reference:

1. BREAKABLE / integrity
2. LIVING / health
3. STATIC NO-SHOOT / BLOCKS
4. MOVING LIVING
5. glass/rear penetration stack

The exact old holes do not need to be re-used; shoot normal fresh shots in those
objects. Use projectile 1 unless specifically testing penetration. For the glass/rear
stack, also run projectile 2 or 3 if needed to exercise the gameplay penetration
chain after detector XY is correct.

## Expected diagnostics

For each normal object-context shot, expect several lines such as:

```text
[V2.25.1 REGION-PROPOSAL] shot=1 object=crate ... candidates=... best_sigma=...
[V2.25.1 REGION-PROPOSAL] shot=1 object=living_target ...
[V2.25.1 REGION-POOL] shot=1 regions=5 raw=... balanced=...
[V2.25.1 REGION-CONFIRM] shot=1 tested=... v2225=... balanced=... groups=...
[V2.25.1 OBJECT-HIT] shot=1 frozen=True ...
```

Glass + rear target should normally appear as one physical proposal group because
they occupy the same physical search area.

## Acceptance signals

- `shot_id` is still present and `frozen=True`.
- REGION-POOL is bounded; one area does not consume ~100 candidates alone.
- REGION-CONFIRM normally emits <=8 balanced candidates, not ~75-129.
- physical final XY follows the actual new holes across widely separated objects.
- at least the static breakable/living/no-shoot shots resolve to their actual objects.
- no object attraction: deliberate outside-object hits remain outside.
- FULL rescue, when used, is logged as the existing global rescue path.

Accuracy is judged against physical hole location, not whether gameplay produced a
desirable result.
