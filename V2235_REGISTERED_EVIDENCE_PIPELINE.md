# V2.23.5 — Registered Evidence Patch Ranker + Iterative Hard-Negative Mining

## Decision leading to this version

V2.23.2 established high proposal recall with the dense V2.21.5 teacher (~95% oracle@20 on a fresh 100-shot projector/camera session). V2.23.3's tabular reducer and V2.23.4's raw PRE/POST patch learner both failed to preserve enough of that recall. V2.23.4's best bootstrap result retained ~30% of oracle-positive shots at Top512 and had median positive rank ~949.

The next experiment therefore changes the representation rather than increasing raw-model size.

## Representation

V2.23.5 recomputes the same direct registered physical evidence used upstream of dense proposals. The eight channels are blackhat/tophat gain, persistent abs/dark/bright, gradient gain, fused, and compact change. Each map is robustly normalised per shot without GT, then candidate-centred 27x27 regions are pooled to 9x9.

The learner receives spatial evidence that is already registration- and scene-compensation-aware. It no longer needs to rediscover those operations from 75 bootstrap training shots.

## Labels

The patch task is intentionally stricter than the final hit tolerance. A candidate <=6 px from GT is positive. Candidates between 6 and 42 px are neutral because the true event may be off-centre in their patch. Only candidates >42 px are current-NEW-hole negatives. GT-centred/jittered anchors are training-only and never enter the proposal pool or evaluation oracle.

## Hard-negative mining

Stage 1 uses strong dense-score, map-percentile, V2.23.3 NEW-hole-heuristic, centre-evidence, and random negatives. After fitting, the model scores every candidate in every training shot. The highest-scoring >42px errors are mined and used for stage 2. The process repeats with a larger mined set for stage 3.

No validation or fresh-domain candidate is mined into training.

## Gates

Single-session bootstrap:

- R512 >= 0.70
- R128 >= 0.45
- median rank <= 200

Fresh-domain research gate, once a second substantial session exists:

- domain R512 >= 0.80
- domain R128 >= 0.55
- domain median rank <= 150
- final conditional Top1@20 >= 0.10

All gates are shadow/research only. Live authority remains false.
