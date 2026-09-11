# Post-fix score comparability audit

This is read-only development analysis of the latest complete physical pool.
No live ranking or candidate generation changed.

## Exact score ordering

The final eligible selector compares onset distance first and `-best_score`
second. In all seven @42-positive failures onset is tied, so the observed
selection is `max(track.best_score)`.

| source | eligible tracks | median | p25 | p75 | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| FAST | 312 | 13.41 | 11.45 | 14.95 | 16.49 | 17.24 | 21.03 |
| V1 | 557 | 5.06 | 3.17 | 8.14 | 10.22 | 11.30 | 34.22 |
| UNKNOWN | 104 | 8.82 | 5.11 | 14.14 | 16.82 | 17.16 | 22.63 |

The PRE fix removed the old FAST ceiling saturation, but source distributions
remain different. FAST has a much higher typical score than V1; this is still
semantic scale evidence, not proof that source normalization alone will solve
selection.

## True versus selected tracks

| shot | true source/score/rank | same-source percentile | selected source/score | ratio | gap |
|---:|---|---:|---|---:|---:|
| 1 | V1 / 6.65 / 53 | 0.71 | FAST / 17.30 | 2.60 | 10.65 |
| 2 | V1 / 6.18 / 59 | 0.50 | FAST / 17.92 | 2.90 | 11.73 |
| 3 | FAST / 15.32 / 9 | 0.76 | FAST / 21.03 | 1.37 | 5.71 |
| 4 | V1 / 18.31 / 1 | 0.99 | V1 / — | — | — |
| 6 | V1 / 6.46 / 55 | 0.60 | FAST / 17.78 | 2.75 | 11.33 |
| 7 | V1 / 11.28 / 33 | 0.91 | FAST / 20.34 | 2.00 | 9.06 |
| 8 | FAST / 9.70 / 23 | 0.43 | FAST / 20.67 | 2.13 | 10.98 |
| 9 | V1 / 34.22 / 1 | 0.99 | V1 / — | — | — |
| 10 | FAST / 13.28 / 24 | 0.32 | FAST / 20.49 | 1.54 | 7.21 |

The true V1 tracks in shots 1, 2, 6 and 7 are not uniformly weak within V1;
shots 1 and 7 are above the 90th percentile. Shot 3, 8 and 10 show that
within-FAST weakness also occurs. The supported conclusion is both cross-source
scale mismatch and a deeper within-source detector-scoring problem.

## Source formulas

The documented V2 formula is:

```text
raw = .16*v2_saliency + .23*center_darkening
    + .15*local_contrast_gain + .11*blackhat_value
    + .06*min(v2_zscore, 25)
score = clip(raw, 3.6, 35.0)
```

Known-hole penalties multiply this score by 0.15, 0.4 or 0.7. Vault/bank
carry can replace it with prior `best_score + repeat_bonus + .35`. FAST uses
the same nominal score field but a different sparse temporal proposal path;
V1 and UNKNOWN records have other provenance and no common calibrated
likelihood interpretation. Equal field names do not establish comparability.

## Read-only rank counterfactuals

True-track ranks under exact eligible-pool replay (shots 1,2,3,4,5,6,7,8,9,10):

```text
raw:               53,59,9,1,85,55,33,23,1,24
source percentile: 29,50,19,1,85,38,8,61,2,71
source rank:       36,63,15,2,85,52,10,40,1,45
common compact:    10,3,18,7,84,21,3,19,4,NA
common dark:       43,19,44,9,69,7,2,37,68,33
```

These are diagnostic rankings, not validated selectors. Source normalization
moves some V1 truths upward but leaves several failures low; a single static
common feature is promising on a few rows but has no physical validation and
does not justify promotion.

## Conclusion

**PROVEN:** PRE correction removed saturation but did not make source scores
semantically comparable. Final ranking uses raw historical best score.

**STRONG HYPOTHESIS:** the correct architecture is source-specific proposal
generation followed by common, source-independent impact evidence.

**RESEARCH_ONLY:** source normalization and common-feature rescoring. No live
change is frozen.

## Common-evidence follow-up

A source-independent registered compactness measurement is available across
all 973 eligible tracks. It improves true-track recovery at @10/@20 but does
not win any of the seven previously failed pairwise events. This supports the
source-specific-proposal/common-impact-confirmation architecture as a
hypothesis, not as a validated selector.
