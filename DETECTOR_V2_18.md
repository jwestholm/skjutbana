# Detector V2.18 — Candidate-aware NEW-hole Ranking + Offset Refinement

## Why V2.18 exists

V2.17 produced a very diagnostic result on the first real 100-shot candidate capture:

- NEW/NOT-NEW patch classification learned strongly (confirmation AUC **0.935817**),
- but candidate ranking stayed essentially unusable (Top-1 **0%**, Top-3 0% on confirmation/holdout),
- the candidate pool still had only 25–45% raw <=20px oracle recall depending on the provisional split.

This means the before/after representation contains useful information, but the **pointwise classification objective is the wrong objective for the real task**. The live problem is not “classify this patch independently”; it is:

> For this shot, rank the correct/new-hole candidate above all the other candidates generated for the same shot.

V2.18 changes the training target accordingly. It remains offline/shadow only.

## Architecture

V2.18 deliberately reuses the V2.17 temporal image learner rather than throwing it away:

```text
candidate PRE + POST
       |
       v
frozen V2.17 temporal backbone
       |
       +--> 96-d learned embedding
       +--> V2.17 NEW probability
       +--> V2.17 dx/dy
       +--> temporal scalar features
                    |
                    v
         per-shot relative context
                    |
                    v
     V2.18 listwise ranking head
                    |
             score(candidate)
             residual dx/dy
```

The V2.17 backbone is frozen in this version. This is intentional: with one provisional physical session, retraining the full image network and ranking head together would make it too easy to overfit without knowing which part helped.

## Listwise objective

Candidates are trained **as a group belonging to one shot**. Distance to GT gives graded relevance:

- <=8 px: 1.00
- <=12 px: 0.92
- <=20 px: 0.78
- <=32 px: 0.42
- <=42 px: 0.16
- >42 px: 0

This prevents a candidate 21–30 px from the new hole from being treated as a categorical “not-new” patch even though it can contain useful new-hole signal and may be refinable to the correct centre.

The loss combines:

1. listwise softmax cross-entropy over the entire candidate group,
2. hard pairwise pressure against high-scoring candidates >=55 px from GT,
3. candidate->GT offset refinement loss for candidates up to 42 px.

## Offset refinement and effective recall

V2.18 measures two different ceilings:

- **raw oracle**: did CV emit a candidate within the requested radius?
- **refined oracle**: after the learned candidate offset, does any emitted candidate land within the radius?

Refined oracle is diagnostic, not authority. It answers whether AI localisation can rescue near-miss CV candidates before we build full direct AI proposals. To avoid a misleading "teleport" metric, V2.18 only credits refined-oracle rescue to candidates that were already <=42 px from GT, which is the range on which the offset head is trained; total predicted refinement is also magnitude-limited.

## Known holes

The audit confirmed that `HitScanner.known_holes` already exists and future V2.17 captures snapshot it. V2.18 records known-hole distance diagnostically where available, but does **not** feed it as a hard neural feature and never excludes a candidate solely because it overlaps an old hole.

Why: a true re-hit/hole-in-hole is valid, and the registry is session-local/incomplete. NEW-hole before/after evidence is responsible for novelty; the known-hole registry remains soft fusion context for a later version.

## Embedding cache

The expensive part is extracting the V2.17 before/after representation for tens of thousands of real candidates. V2.18 therefore caches frozen candidate embeddings under:

```text
content/ai/reports/v218/embedding_cache/
```

The cache is invalidated automatically if the source candidate pack or V2.17 model changes. The first run can take time; subsequent epochs/benchmarks and the future overnight trainer reuse the small cached embeddings.

## Commands

```bash
python3 -m automation.newhole_v218_selftest
python3 -m automation.newhole_v218_inspect
python3 -m automation.newhole_v218_train
python3 -m automation.newhole_v218_benchmark
python3 -m automation.newhole_v218_verify
```

The first `inspect` or `train` may spend most of its time creating the embedding cache. This is expected and should happen only once per source/model revision.

## What success means

On the existing seed-65432 dataset, V2.18 is still **provisional** because all 100 shots are one physical/projected session. The immediate engineering gate is:

- candidate Top-1/Top-3 or median GT rank improves over V2.17 on confirmation,
- and/or learned offset refinement adds measurable <=20px signal,
- the report also shows whether V2.18 has reached the strongest existing V2.16 confirmation Top-1 (V9/fusion where available); merely beating V2.17's 0% is not presented as the final ranking win,
- no live authority is granted.

If this works, the next version is the first justified **offline champion/challenger / overnight learning loop**. If it does not, we should inspect candidate-group features rather than merely add more V2.17 classification epochs.
