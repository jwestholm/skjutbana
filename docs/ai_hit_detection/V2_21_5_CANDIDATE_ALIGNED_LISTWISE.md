# V2.21.5 candidate-aligned listwise dense ranker

## Result that triggered the change

V2.21.4 separated the problem cleanly:

- raw physical dense proposal pool: excellent recall (~8.7k candidates/shot; 90% <=20 px, 100% <=42 px over the 30 full-frame shots);
- learned Top-512: poor recall (13.3% <=20 px overall; 0% on confirmation).

Therefore V2.21.5 leaves live authority unchanged and focuses only on the compression/ranking step.

## Important semantic change from V2.21.4

V2.21.5 has **no forced positive jitter candidates**. Every candidate used for fitting is first produced by a GT-free dense proposal pass. GT is applied afterward only to assign training targets and offline metrics.

This makes the train task identical in form to inference: rank the proposals the system truly generated.

## Data isolation

- DEVELOPMENT full-frame shots: fitting, standardisation, hard mining and DEVELOPMENT-only cross-fit.
- CONFIRMATION: frozen evaluation only.
- HOLDOUT: frozen evaluation only.
- Existing old 100 packs without full frames remain unavailable to this full-frame learner.

When only one 30-shot full-frame session exists, the legacy split remains provisional. Protected results are useful evidence but not final physical acceptance.

## Metrics to watch

### Proposal ceiling

`dense_pool oracle20` and `dense_pool oracle42`.

If these regress materially from V2.21.4, fix proposal generation before ranking.

### DEVELOPMENT-only cross-fit

`top512_oracle20` is the first ranking gate. This must improve substantially over the V2.21.4 learned ranker before interpreting protected results optimistically.

### Frozen protected ranking

Compare:

- `v2212_union`
- `learned_512` / `learned_1024`
- `v2212_plus_learned_512` / `v2212_plus_learned_1024`

The final objective remains a much smaller candidate set with high physical recall, eventually followed by final multi-source ranking/fusion.
