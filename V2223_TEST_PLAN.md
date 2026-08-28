# V2.22.3 physical verification

Run from repository root after extracting the DELTA over the already working V2.22.2-r2 tree.

## 1. Static tests

```bash
git status --short
git diff --check

python3 -m automation.v2223_selftest
python3 -m automation.v2223_verify_install
python3 -m automation.v2222_selftest
python3 -m automation.v2221_selftest
python3 -m automation.offline_v222_selftest
python3 -m automation.runtime_v222_selftest
```

Apply the append-only MD updates once:

```bash
python3 -m automation.v2223_apply_docs
```

The command is idempotent; rerunning it does not append duplicate V2.22.3 sections.

## 2. Startup only — no shots yet

```bash
python3 main.py
```

Expected new line:

```text
[V2.22.3] shot-critical runtime + object-hit shadow foundation installed
```

V2.22.1 and V2.22.2 must still report successful installation and there must be no `unavailable` message.

## 3. AI Training cursor + dispatch test: 3 real shots

Use AI Training, AI mode `advisory`, no F2.

Expected cursor behaviour:

- after calibration / while armed: pointer hidden;
- PANG: optional wait/hourglass should appear almost immediately when the **main loop** acknowledges the audio event;
- after hit is resolved and manual GT is requested: normal pointer visible;
- after GT click / re-arm: hidden again.

Keys:

- `F3`: toggle mouse-shot/debug cursor override;
- `F4`: toggle latency wait cursor.

For each physical shot capture these lines:

```text
[V2.22.3 AUDIO-PRIORITY]
[V2.22.3 SHOT]
[PRE-SHOT] recent ring frame ... (V2.22.3)
[V2.22.1 ROI]
[V2.22.2 FAST]
[SHOT #N] HIT/MISS
[V2.22.3 LATENCY]
[V2.22.3 VISIBLE]
[V2.22 RESOLVER]
```

### Critical checks

1. `main_ack` should normally be tens of ms, not 1-2 seconds.
2. V2.22.2 `frames=` should no longer stay `0->0` on every shot. The scanner now owns the ring pickup cursor.
3. `PRE-SHOT` should normally say `recent ring frame`, not static fallback.
4. ROI must remain `mode=homography` and the hit marker must remain geometrically correct.
5. Real PANG -> visible marker should feel materially faster; exact log is the source of truth.

Stop after 3 shots if any of these are wrong.

## 4. If the three shots are good: 10 + 2 shots

- 8 ordinary new holes distributed over the playfield,
- 2 near/re-hit cases close to holes already made in the same session,
- 2 extra ordinary shots if desired.

Return the full terminal block. We will compute median/p95 dispatch, scanner, HitEvent and visible latency and then decide whether V2.22.4 should attack detector slow-path variance or move to the direct object-region fast path.

## 5. Object-hit API test

V2.22.3 does not require AI Training to expose game objects, so `objects=0` is normal there. The automated selftest validates object snapshot/evaluation.

For a game scene that implements `get_hit_regions()`, expect shadow-only logs such as:

```text
[V2.22.3 SHOT] shot=42 dispatch=18.2ms objects=17
[V2.22.3 OBJECT SHADOW] shot=42 regions=17 object=enemy_7 conf=... rank=... local=(...,...)
```

No object shadow result may alter live gameplay authority in V2.22.3.
