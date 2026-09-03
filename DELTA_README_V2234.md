# V2.23.4 DELTA — Patch NewHole Candidate Model

Apply this ZIP on top of an installation that already contains V2.23.3.

V2.23.4 deliberately changes the first-stage learned reducer. V2.23.3 proved that the dense V2.21.5 teacher pool has high proposal recall, but its tabular reducer discarded too many real positives before the final ranker saw them. V2.23.4 therefore learns directly from candidate-centred PRE/POST image patches.

The live hit path is unchanged. All new models remain offline/shadow-only.

## New pipeline

1. Reuse V2.23.2 full-frame PRE/POST framepacks.
2. Reuse V2.21.5 dense proposal coordinates and V2.23.3 numeric caches.
3. Compile a one-time 5-channel patch bank per candidate:
   - PRE grayscale
   - mean POST grayscale
   - amplified absolute PRE→POST change
   - signed PRE→POST change
   - temporal persistence
4. Train two image-model families with NumPy only:
   - patch MLP baseline
   - tiny learned CNN
5. Rank the full dense pool with the selected patch model.
6. Keep the best 512 candidates and run the existing final listwise ranker with patch score added as evidence.

No PyTorch/TensorFlow dependency is required.

## Important label discipline

GT-centred and slightly jittered patches are allowed as *training-only positive examples*. They are never inserted into the proposal pool, never counted in oracle metrics, and never available at inference time.

The 20–42 px band remains neutral for NEW-hole ranking. Candidates farther than 42 px are valid NOT-NEW negatives for this shot; they are not automatically treated as static non-hole examples.

## Fresh-domain discipline

With only one substantial dense F2 session, V2.23.4 uses a deterministic same-session bootstrap split to answer only whether the image model can learn the task. It cannot create a research champion.

Once a second substantial F2 session exists, the newest session is held out completely. Patch-model and final-ranker trial selection uses engineering validation only. The fresh-domain session is evaluated only after the winning model has already been chosen.

## First commands

```bash
python3 -m automation.v2234_selftest
python3 -m automation.v2234_verify_install
python3 -m automation.v2234_status
python3 -m automation.v2234_prepare --session latest
python3 -m automation.v2234_train --quick --no-prepare
python3 -m automation.v2234_status
```

Do not run a new F2 ×100 before reviewing the bootstrap result. The key V2.23.4 metrics are patch-model `retention20@128`, `retention20@512`, and `median_positive_rank20`.
