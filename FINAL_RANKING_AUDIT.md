# Final ranking audit

## Exact ordering

The live eligible-track key is:

```text
(onset_distance_to_audio_peak, -track.best_score)
```

For every comparable pair in the latest physical session, onset distance is
identical or effectively tied. The selected track wins at the second field,
`-best_score`; readiness and local confirmation are eligibility gates, not
ranking terms. There is no source field or confirmation feature in the final
tuple.

| shot | true rank / score | selected score | selected source | decisive field |
|---:|---:|---:|---|---|
| 1 | 53 / 6.647 | 17.298 | FAST | best_score |
| 2 | 59 / 6.183 | 17.918 | FAST | best_score |
| 3 | 9 / 15.315 | 21.027 | FAST | best_score |
| 4 | 1 / 18.314 | — | V1 | selected |
| 5 | 84 / 4.056 | 22.629 | UNKNOWN | best_score; no @42 oracle |
| 6 | 55 / 6.455 | 17.782 | FAST | best_score |
| 7 | 33 / 11.279 | 20.337 | FAST | best_score |
| 8 | 23 / 9.698 | 20.674 | FAST | best_score |
| 9 | 1 / 34.216 | — | V1 | selected |
| 10 | 24 / 13.279 | 20.490 | FAST | best_score |

The seven @42-positive failures are therefore precise ranking failures: the
true track is eligible, but the final selector compares its lower historical
best score against a higher score from an unrelated eligible track. This is
stronger evidence than the shorthand “FAST won.”

## Score provenance

False winners have higher current/best score, but the complete physical export
does not preserve every intermediate component for every historical observation
well enough to attribute the advantage to one term without inference. The
available candidate records show the same V2 saliency/center-change/local
contrast/DoG/z-score formula and source-specific FAST/Vault history. The final
selector does not compensate for those different provenance paths.

## Recurrence check

Selected false-winner coordinates are separated by more than 40 px from one
another in this ten-shot session, so there is no demonstrated recurring
selected nuisance location. Broad clustering of all 973 active tracks becomes
dense at a 15 px radius and is not a meaningful recurrence signal. A prior
event penalty is therefore not justified from this dataset.

## Status

**PROVEN:** final ordering is onset distance, then descending historical
`best_score`; all seven physical losses are decided by the score field.

**STRONG HYPOTHESIS:** source/provenance score semantics make nuisance tracks
look better than physically correct tracks.

**RESEARCH_ONLY:** prior-event recurrence remains a diagnostic idea only. No
recurrence penalty or ranker is frozen or promoted.
