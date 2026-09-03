# Skjutbana V2.23.5 — Registered Evidence Patch Ranker

Delta on top of V2.23.4. Copy the archive contents over the repository root.

## Why

V2.23.4 proved that raw PRE/POST candidate patches were the wrong first-stage representation on the current bootstrap. Dense proposal oracle@20 stayed ~92%, while the best raw patch reducer retained only ~30% at Top512 and had median positive rank ~949. V2.23.5 therefore reuses the image processing that already succeeds: the registered/photometrically compensated V2.21 physical evidence maps that generate the dense proposal pool.

## New first-stage contract

Evidence channels:

- blackhat_gain
- tophat_gain
- persistent_abs
- gradient_gain
- persistent_dark
- persistent_bright
- fused
- compact_change

Each dense candidate receives a 27x27 local crop pooled to 9x9 per channel. The candidate bank is GT-free. Training-only GT anchors use offsets <=6 px.

Patch labels:

- <=6 px: positive NEW-hole-centred patch
- >6 px and <=42 px: neutral; never used as negative
- >42 px: negative for current NEW-hole learning

Final localisation metrics remain <=20 px and <=42 px.

## Iterative hard-negative mining

Training has three stages:

1. physical/dense/image-hard + random negatives,
2. score every training candidate and mine the model's top false candidates,
3. score again and mine a larger second hard-negative set.

Fresh-domain sessions are never mined or used for model selection.

## Install / test

```bash
cd ~/skjutbana/skjutbana
unzip -o skjutbana_v2.23.5_registered_evidence_hardmine_delta.zip

git diff --check
python3 -m automation.v2235_selftest
python3 -m automation.v2235_verify_install
python3 -m automation.v2235_status
```

Prepare the existing latest 100-shot dense session:

```bash
python3 -m automation.v2235_prepare --session latest
```

Then run the bootstrap learner without preparing again:

```bash
python3 -m automation.v2235_train --quick --no-prepare
python3 -m automation.v2235_status
```

Do not create another F2 session until the bootstrap result has been reviewed.

## Bootstrap decision gate

The one-session bootstrap is engineering-only. V2.23.5 asks for a large improvement over V2.23.4:

- retention20@512 >= 70%
- retention20@128 >= 45%
- median positive rank20 <= 200

A pass is still not live authority. It only justifies collecting a second independent F2 session for fresh-domain validation.

## Rollback

Restore the V2.23.4 delta versions of:

- `src/engine/ai/training_v223/integration.py`
- `src/engine/ai/training_v223/__init__.py`

and remove the V2.23.5-specific modules/automation files. V2.23.5 writes new data only below `content/ai/training_v223/evidence_v2235/`; older V2.23.2–V2.23.4 data is not modified.

Live authority remains unchanged / NO.
