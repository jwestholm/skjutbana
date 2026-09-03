# V2.23.4 Test Plan

## 1. Install verification

```bash
python3 -m automation.v2234_selftest
python3 -m automation.v2234_verify_install
python3 -m automation.v2234_status
```

Expected: all PASS, one substantial patch session may not exist until prepare is run, and live authority remains NO.

## 2. Compile existing 100-shot patch bank

No projector is needed.

```bash
python3 -m automation.v2234_prepare --session latest
```

Expected progress:

```text
[V2.23.4 PATCH] 1/100 ...
[V2.23.4 PATCH] 5/100 ...
...
[V2.23.4 PATCH] 100/100 ...
```

This is a one-time disk cache. A repeated command should report most/all shots as cached.

## 3. Bootstrap learnability test

```bash
python3 -m automation.v2234_train --quick --no-prepare
python3 -m automation.v2234_status
```

Send the complete `TRAIN SUMMARY` and status output.

Primary metrics:

- best patch model kind,
- retention20@128,
- retention20@512,
- median positive rank20,
- patch-model conditional Top1@20,
- final-ranker conditional Top1/Top3@20,
- bootstrap learnability gate PASS/FAIL.

Target decision gate:

```text
R512 >= 0.80
R128 >= 0.60
median positive rank <= 100
```

If this fails badly (for example R512 around 0.4 and median rank several hundred), do not add more hand-tuned features. Re-evaluate the patch representation/model capacity.

## 4. Only after bootstrap success: collect a second F2 ×100

The second substantial dense session becomes fresh-domain validation.

After F2:

```bash
python3 -m automation.v2234_cycle --session latest --quick
python3 -m automation.v2234_status
```

The newest session must be excluded from patch/final model selection and evaluated only after selection.

## 5. Rollback

V2.23.4 changes only training/shadow integration. To roll back the experiment, restore the V2.23.3 versions of:

- `src/engine/ai/training_v223/integration.py`
- `src/engine/ai/training_v223/__init__.py`

and remove/ignore the new `patch_v2234`, `patch_model_v2234`, `trainer_v2234`, and `automation/v2234_*` files. Existing framepacks/proposals/rich caches remain valid.
