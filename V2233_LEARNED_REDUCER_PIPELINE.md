# V2.23.3 — Learned Candidate Reduction + Rich NEW-hole Ranking

## Why this version exists

V2.23.2 answered the most important proposal question on a completely new F2 projector/camera session:

- current/live oracle <=20 px: **6%**
- local oracle <=20 px: **15%**
- dense V2.21.5 oracle <=20 px: **95%**
- dense/union oracle <=42 px: **96%**
- mean dense pool: about **9,359 candidates/shot**

So the physical information is present. The dominant problem is now ranking/reduction, not discovering whether the hit exists at all.

The old V2.23.2 rankers failed on that fresh domain because the positive candidate was often buried hundreds of ranks down. An additional structural reason was exposed: the only fully dense-expanded 100-shot F2 session was reserved as fresh-domain validation, so the ranker had almost no same-domain dense examples to learn from.

V2.23.3 therefore adds a dedicated dense-candidate learning stage.

## Pipeline

```text
F2 labelled framepack
  PRE + POST1..3 + GT label
          |
          v
V2.21.5 dense teacher (offline)
  ~8k-11k GT-free candidate hypotheses
          |
          v
V2.23.3 rich PRE/POST map features
  computed once per frame, sampled at every candidate
          |
          v
pairwise learned REDUCER
  keep top 512 (metrics also @32/64/128/256/1024)
          |
          v
final listwise ranker on the retained pool
          |
          v
fresh-F2 domain gate when >=2 dense sessions exist
          |
          v
research shadow only; live authority remains NO
```

## Rich full-frame evidence

The reducer does not receive GT coordinates or GT distance as model input. GT is label-only.

For each dense candidate V2.23.3 samples GT-free maps derived from the saved PRE and up to three POST frames:

- small/medium/large absolute PRE->POST change;
- peak local change;
- darkening and brightening separately;
- signed change;
- temporal persistence;
- temporal variance;
- PRE and POST local standard deviation;
- local edge gain;
- small-vs-large compactness;
- centre/ring change contrast;
- dark-change fraction;
- a deterministic `newhole_heuristic` used only as a mining/reference signal.

The computation is map-based: expensive image operations are performed once per full frame and candidate extraction is indexed sampling. Ten thousand candidates therefore do not require ten thousand independent image crops.

Rich features are stored separately as compressed numeric NPZ files beside V2.23.2 proposal sidecars. The large proposal JSON is not duplicated.

## Compact numeric training cache

Repeated overnight experiments should not repeatedly parse large proposal JSON files and reconstruct Python dictionaries.

V2.23.3 compiles each prepared shot into a numeric NPZ cache containing:

- candidate XY;
- compact reducer feature matrix;
- GT distance labels (not model inputs);
- a reproducible baseline score;
- GT XY metadata.

All NPZ loading uses `allow_pickle=False`.

## Pairwise reducer objective

The reducer learns the ordering directly.

For each training shot:

- candidates <=20 px from GT are positives;
- candidates from 20..42 px are **neutral** and are not trained as negatives;
- candidates >42 px are valid NEW-hole negatives;
- hard negatives are sampled from dense physical score and the rich NEW-hole heuristic;
- additional random negatives preserve broad coverage.

The pairwise loss asks the model to score a real positive higher than hard physical negatives. This avoids asking one softmax to reason over roughly ten thousand rows at once.

## The metric that matters first

Reducer success is measured by **positive retention after reduction**, not only Top-1.

Reports include conditional oracle retention at:

- top 32;
- top 64;
- top 128;
- top 256;
- top 512;
- top 1024.

For example, if raw dense oracle20 is 95% and reducer `retention20@512` is 95%, the two-stage system still has about 90% overall proposal availability at <=20 px while reducing ~9,500 rows to 512.

Only then is final Top-1/Top-3 ranking interpreted.

## One-session bootstrap vs real domain validation

At installation time the shooting PC is expected to have only one substantial fully dense-expanded F2 session.

V2.23.3 handles this explicitly:

### One substantial dense F2 session

Use a deterministic 80/20 shot split:

- ~80 shots train;
- ~20 shots same-session validation;
- status is `single_session_bootstrap`;
- the purpose is only to prove whether the reducer can learn the dense ranking problem;
- **no research/domain-generalisation claim is allowed**.

### Two or more substantial dense F2 sessions

The newest session is untouched fresh-domain validation. Older dense sessions become engineering data.

Only this mode may pass the V2.23.3 research cascade gate.

A research cascade currently requires at least:

- fresh-domain reducer retention20@512 >= 0.90; and
- final fresh-domain conditional Top1@20 >= 0.10.

This gate grants research-shadow status only. It never grants game authority.

## F2 integration

F2 still captures the same full framepacks. At F2 completion V2.23.3 now schedules the complete shadow cycle:

1. reuse/create dense proposal sidecars;
2. compute rich PRE/POST evidence;
3. compile numeric reducer cache;
4. train reducer and final ranker;
5. evaluate according to bootstrap/fresh-domain discipline.

The worker is a daemon thread, so for controlled experiments it is still preferable to exit the app and run the offline commands manually. Nothing is lost; all expensive intermediate data are cached.
