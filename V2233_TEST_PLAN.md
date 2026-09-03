# V2.23.3 verification plan

## A — Installation

From repository root:

```bash
python3 -m automation.v2233_selftest
python3 -m automation.v2233_verify_install
python3 -m automation.v2233_status
```

Expected: all selftests and install checks pass, and live authority remains `NO`.

## B — Prepare the existing 100-shot V2.23.2 dense session

No projector is needed:

```bash
python3 -m automation.v2233_prepare --session latest
```

This computes rich PRE/POST features and compact numeric caches. The first pass can take time and use significant memory because it processes full 4K frames. Progress is printed every few shots.

Run it a second time if desired. It should predominantly report cached rich data rather than recomputing image maps.

## C — Bootstrap learning proof on the existing session

With only one substantial dense-expanded session, run:

```bash
python3 -m automation.v2233_train --quick --no-prepare
```

Expected split:

```text
mode=single_session_bootstrap
train≈80
validation≈20
fresh_domain=0
```

This is deliberately **not** a champion/domain claim. It answers only:

> Can the learned physical reducer move the true candidate from rank hundreds/thousands into a small top-K pool on held-out shots from the same capture session?

Primary metrics:

- reducer validation `retention20_at_k` for 32/64/128/256/512/1024;
- `median_positive_rank20`;
- final-ranker conditional Top1@20 and Top3@20.

A useful result should show a major rank reduction. If positive rank remains hundreds after rich features/pairwise learning, change model approach rather than adding more detector filters.

## D — Create the second independent dense F2 session

After the bootstrap proof, run a new F2 x100 session on the projector/camera.

V2.23.3 will attempt the shadow cycle automatically at session end. For clean logs, manual offline execution is recommended:

```bash
python3 -m automation.v2233_cycle --session latest --quick
```

This reuses V2.23.2 proposal caches where available, computes rich evidence, builds numeric caches and trains/evaluates the cascade.

Now the expected split changes to:

```text
mode=fresh_session_domain
older dense session -> engineering train/validation
newest dense session -> untouched fresh domain
```

This is the first meaningful generalisation test.

## E — Decision metrics

Do not judge only Top-1.

First require the reducer to preserve the V2.23.2 95% raw proposal recall reasonably well:

```text
fresh-domain retention20@512
fresh-domain retention20@256
fresh-domain median positive rank
```

Then inspect final ranking:

```text
conditional Top1@20
conditional Top3@20
MRR@20
```

If reducer retention is high but final Top-1 remains poor, the next work is final patch/ranking modelling. If reducer retention itself is poor, improve rich physical/NewHole representation before touching live detection.

## F — Overnight work

Only after the second-session domain test is healthy should longer runs be worthwhile. V2.23.3 caches the expensive full-frame feature extraction so repeated optimizer/model experiments are cheap relative to proposal generation.
