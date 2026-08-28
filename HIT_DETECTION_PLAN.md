# HIT DETECTION PLAN — current source of truth

**Updated:** 2026-08-27  
**Current planned version:** V2.21 — Physical Domain Bridge + Direct Proposals  
**Final target:** >=95% correct hit detection on unseen physical mixed-background sessions before AI becomes authoritative in normal games.

---

## 1. Problem definition

The system must determine the **new physical bullet hole created by the current shot** among:

- 0..n old physical holes,
- projector/game graphics,
- moving images/video,
- camera noise and exposure variation,
- paper/fibre texture,
- hard edges/text/UI,
- detector false positives,
- possible re-hits / hole-in-hole cases.

The useful evidence sources are not interchangeable:

- **V1/V2 classical CV candidates** — fast-ish physical proposal source, but currently misses too many real hits.
- **V9 physical ranker/features** — historically useful among real candidates.
- **Static Hole-AI (V2.15)** — answers approximately “does this patch look like a hole?”; old and new holes are both positive.
- **NEW-hole temporal AI (V2.17/V2.18)** — answers approximately “is this the hole created by the current shot?” from PRE/POST evidence.
- **Known-hole state** — soft session context only; it is incomplete and must never be treated as ground truth for every physical hole on the board.
- **Projector/render context** — future residual source: what the projector was expected to show vs what the camera observed.
- **Game context/spatial priors** — future soft priors only.
- **Direct AI proposals** — required to break the current candidate-recall ceiling when V1/V2 never proposes the true hit location.

The final detector should eventually fuse several independent sources rather than delegate authority to one model.

---

## 2. Non-negotiable design rules

1. **Physical evidence dominates.** Game/object/spatial context may bias ranking but must not invent a hit unsupported by image evidence.
2. **Old hole != negative hole.** Static Hole-AI must regard old and new real holes as hole-positive.
3. **NOT-NEW != NON-HOLE.** A candidate far from the current GT can be an old real hole and is only negative for the current-shot novelty task.
4. **Known holes are soft context.** The registry is session-local/incomplete and can contain errors.
5. **Hole-in-hole remains possible.** Never hard-reject proximity to an old hole.
6. **Generated validation/holdout must remain frozen.** Never train on them after inspecting their results.
7. **Physical confirmation/holdout must remain protected.** Existing provisional shot splits may be reported, but the long-term acceptance gate requires independent physical sessions.
8. **Synthetic success is not physical proof.** Simulator scores are development evidence only.
9. **No live authority changes before gates pass.** Current game hit order/coordinates remain unchanged while models are shadow/offline.
10. **Every experiment must preserve provenance.** Seed, source, split, model, feature version, capture session and configuration must be recoverable.

---

## 3. Current architecture

```text
physical shot
   |
   +--> audio peak / shot event
   |
   +--> PRE / recent PRE / POST camera frames
   |
   +--> V1/V2 classical CV candidate generation
   |       |
   |       +--> physical/V9 features
   |       +--> patch evidence
   |
   +--> static Hole-AI evidence (V2.15)
   |
   +--> NEW-hole temporal embedding (V2.17)
   |
   +--> listwise same-shot ranker + offset head (V2.18)
   |
   +--> future direct full-frame proposals (V2.21)
   |
   +--> future multi-source fusion
```

The key distinction is now explicit:

- **proposal recall:** is a candidate near the real hit present at all?
- **ranking:** if it is present, can the system put it first?

A ranker cannot exceed its proposal oracle.

---

## 4. Current measured physical ceiling

Existing physical V2.16 session: 100 labelled shots, one physical/projector session, provisional 60/20/20 shot split.

### Raw+ranked union oracle within 20 px

| Split | Oracle |
|---|---:|
| Development | 40% |
| Confirmation | 35% |
| Holdout | 50% |

Even a perfect ranker cannot exceed these numbers without new proposals.

### Current ranking on the same physical data

| Split | Current Top-1 | Synthetic-trained V2.18 Top-1 | Synthetic-trained V2.18 median GT rank |
|---|---:|---:|---:|
| Development | 6.67% | 0% | 110 |
| Confirmation | 5% | 0% | 125 |
| Holdout | 10% | 0% | 104 |

This is the decisive current constraint.

---

## 5. What V2.20.2 proved

The V2.19 media-world idea was repaired through V2.20/V2.20.1/V2.20.2:

- RGB is canonical; grayscale is derived only for legacy detectors.
- static PRE/POST share one camera state,
- hole rendering uses compact stamps rather than long signed residual artefacts,
- weak/invalid hole renders are caught by QA,
- still-image generation was made much faster,
- transparent sprites are composited over deterministic coloured/textured procedural backgrounds instead of black,
- hole appearance includes a dark core plus subtle torn/frayed light paper cues.

This simulator is useful for **controlled diversity and learning experiments**.

It is not yet physically faithful enough to be the only training domain.

---

## 6. Latest synthetic results

### Generated train session

- seeds: `1..100`
- 100 groups
- 38,400 candidates
- 98/100 groups had a candidate <=20 px in the first generated training set
- V2.18 trained for 32 epochs

Observed V2.18 Top-1 on the generated training session split:

- development: **96.67%**
- confirmation: **100%**
- holdout: **95%**

### Frozen generated validation

- seeds: `9000001..9000100`
- 100/100 saved
- 100/100 had candidate <=20 px and <=42 px
- V2.18 model was frozen; no retraining

Observed V2.18 Top-1:

- development: **98.33%**
- confirmation: **100%**
- holdout: **100%**

Therefore V2.18 is not merely memorising seeds `1..100`; it generalises extremely well **inside the V2.20.2 synthetic domain**.

---

## 7. Decisive synthetic -> physical result

The exact same generated-data-trained V2.18 model was then benchmarked without retraining on the real V2.16 physical candidate session.

Result:

- V2.18 Top-1: **0% development / 0% confirmation / 0% holdout**.
- median GT rank: **110 / 125 / 104** on ranked candidates.
- on raw+ranked union, median GT rank became **235 / 287 / 185.5**.

The offset/refinement head retained a small amount of useful local/geometric signal in some subsets, but ranking transfer failed.

### Current conclusion

```text
synthetic train -> unseen synthetic       PASS, ~98-100% Top-1
synthetic train -> existing physical      FAIL, 0% Top-1
```

This is a large **domain gap**, not a reason to increase epochs.

---

## 8. Active next step: V2.21

V2.21 must address two different bottlenecks in the correct order.

### V2.21-A — physical data audit + domain-gap profiler

1. Inspect the existing physical candidate-pack NPZ schema and determine whether raw/full recent-PRE and POST frames are actually available.
2. If full frames are missing, extend future shadow capture to save lossless full-frame recent-PRE + selected POST frames in a storage-aware mode.
3. Compare synthetic vs physical distributions for:
   - V2.17 embedding/features,
   - current/V9 features,
   - local signed/absolute change,
   - darkening,
   - local contrast,
   - candidate count/rank,
   - nearest-GT distance,
   - old-hole proximity.
4. Produce a machine-readable domain-gap report and identify the largest shifted features.
5. Audit for accidental simulator shortcuts/leakage, especially any feature that indirectly encodes generated GT/hole stamp construction.

### V2.21-B — direct full-frame proposal baseline

Candidate recall is a hard ceiling and must be raised before spending major effort on ranking.

Build a proposal source that scans PRE/POST directly:

```text
recent PRE + POST sequence
        |
        +--> temporal evidence maps
        +--> local residual/darkening/persistence
        +--> optional NEW-hole model heatmap
        |
        +--> local maxima / NMS
        |
        +--> AI_DIRECT candidates
        |
        +--> union with V1/V2 candidates
```

First gates on protected physical data:

- beat current physical union oracle consistently,
- initial target: >=70% <=20 px oracle on confirmation/holdout,
- next target: >=85%,
- long-term target before authority work: >=95% with acceptable false-candidate volume.

No ranking model can solve the project until proposal recall approaches the final target.

### V2.21-C — physical-domain bridge for ranking

Once full physical PRE/POST material is available:

- use **physical development-only** frames as camera-domain bases,
- model real temporal nuisance from physical data,
- insert new-hole events at new locations without touching protected physical confirmation/holdout,
- mix physical-domain worlds with V2.20.2 worlds,
- retrain listwise ranker,
- evaluate frozen on generated validation and physical confirmation/holdout.

The bridge must teach camera-domain appearance without memorising acceptance shots.

---

## 9. Suggested future versioning

### V2.21 — Physical Domain Bridge + Direct Proposals

- domain-gap report,
- full-frame capture audit/support,
- direct-proposal baseline,
- physical-domain generated worlds,
- frozen physical transfer benchmark.

### V2.22 — Champion/challenger offline trainer

Only after V2.21 shows meaningful physical transfer and better proposal oracle:

- continuously generate fresh train seeds,
- mix synthetic and physical-domain training worlds,
- mine hard cases,
- train challengers for a time budget (`--hours N`),
- compare to champion on frozen generated validation + protected physical confirmation,
- keep/reject automatically,
- never train on generated holdout or physical acceptance sessions.

### V2.23+ — learned fusion + projector/game context + live shadow

- direct AI proposals,
- V1/V2/V9,
- static Hole-AI,
- NEW-hole listwise ranker,
- projector expected-vs-observed residual,
- known-hole soft context,
- game-object/spatial soft priors,
- calibrated confidence/abstention,
- live shadow before authority.

---

## 10. Final authority gate

AI/camera selection must not become authoritative in normal games until all of the following are true:

- >=95% final correct hit detection on **unseen independent physical mixed-background sessions**,
- proposal oracle itself comfortably exceeds the final target,
- no catastrophic background/motion class,
- wrong-hit and false-hit rate are acceptable,
- latency is acceptable for gameplay,
- hole-in-hole and old-hole cases are explicitly represented,
- no evaluation/training leakage,
- fallback/debug path remains available,
- mouse/API/camera hits still converge on the same game hit interface.

---

## 11. Immediate decision

**Do not start a 12-hour V2.18 synthetic-only optimization run.**

The next useful code work is V2.21. The latest experiment has already answered the synthetic-only question: the ranker can learn the simulator, but the simulator/model representation does not yet transfer to the camera, while real candidate recall remains too low.


## 12. V2.21 implementation checkpoint — 2026-08-27

Implemented in the V2.21 delta:

- physical pack full-frame audit,
- candidate-level synthetic vs projector/camera domain-gap profiler,
- direct full-frame PRE->POST proposal engine (shadow/offline),
- proposal oracle/rescue benchmark,
- opt-in/storage-aware full-frame automation capture using true recent PRE + two POST frames,
- no live authority/order change.

The existing old candidate session should be audited first. If full frames are absent, do not synthesize a direct-proposal score from candidate patches: collect a small fresh automation/projector-camera capture with V2.21 full-frame shadow storage enabled.

The direct-proposal engine must improve proposal **coverage** before more listwise-ranker optimization. Initial gate remains >=70% union oracle <=20 px on confirmation and holdout, then >=85%, with >=95% required across independent physical sessions before authority work.

---

## V2.21.1 — short full-frame projector/camera gate (2026-08-27)

Observed V2.21 domain-gap profile:

- group domain AUC **1.0000** (severe/trivial separation),
- `known_hole_distance_scaled` is the strongest shortcut (KS **0.988**),
- temporal and V2.17 embedding feature distributions are also strongly shifted.

Decision: **do not run more synthetic-only V2.18 training yet**.

V2.21.1 adds a one-shot short capture controller. First run is 30 white
projector/camera rounds with online learning frozen and full recent PRE + two
full POST frames saved. The first target is to raise CURRENT+AI_DIRECT
candidate oracle recall at <=20 px toward/above **70%** before ranking work
continues. Live authority remains unchanged.

---

## V2.21.2 / V2.21.3 observed physical full-frame results — 2026-08-28

The first honest 30-round full-frame projector/camera capture changed the immediate priority.

### V2.21.2

- CURRENT oracle <=20 px: **26.67%**.
- CURRENT oracle <=42 px: **90.00%**.
- V2.21 global AI_DIRECT oracle <=20/42 px: **3.33% / 3.33%**.
- anchored local temporal oracle <=20 px: **53.33%**.
- CURRENT + local temporal oracle <=20 px: **63.33%**.
- CURRENT + local temporal oracle <=42 px: **93.33%**.
- local temporal proposals rescued **11/30** CURRENT misses at <=20 px.
- registration was essentially zero; there is no useful single global calibration offset.

### V2.21.3

Plateau-aware consensus and a candidate-derived target mask were tested as a hand-written refinement. They were **rejected**:

- final union <=20 px fell to **46.67%** overall,
- confirmation final union <=20 px fell to **16.67%**,
- holdout final union <=20 px was **50.00%**,
- masked global direct proposals rescued **0** CURRENT misses at <=20 px,
- V2.21.2 remains the better handcrafted baseline at **63.33%** <=20 px overall.

Decision: stop adding hand-written temporal heuristics for now. Preserve V2.21.2 as the handcrafted baseline and move to a learned physical-domain dense proposal ranker.

## V2.21.4 — Learned physical-domain dense temporal ranker

V2.21.4 is offline/shadow-only and uses the 30 honest full-frame packs already captured.

Training contract:

- fit only on full-frame **DEVELOPMENT** shots,
- never fit/normalise/mine hard negatives from confirmation or holdout,
- create a broad GT-free temporal proposal pool using deliberately lower map thresholds than V2.21.3,
- learn a pairwise linear ranker from physical PRE->POST evidence,
- freeze the model before evaluating protected splits,
- keep V2.21.2 local temporal union as the baseline/fallback comparator.

The key diagnostic is now two-stage:

1. **dense pool oracle** — if this is low, candidate generation is still the bottleneck;
2. **learned top-K oracle/rank** — if the pool is high but top-K is low, the learned ranker/data volume is the bottleneck.

Initial V2.21.4 gates remain offline only:

- development dense-pool oracle <=20 px should reach at least 90% before ranking is blamed,
- frozen confirmation/holdout final union should target >=70% <=20 px first,
- the split remains provisional until additional independent full-frame projector/camera sessions exist,
- no live authority regardless of one-session benchmark quality.
