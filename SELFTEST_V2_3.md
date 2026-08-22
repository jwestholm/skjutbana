# V2.3 isolated self-tests performed before packaging

These are code-level smoke tests, not substitutes for the physical projector/camera benchmark.

## Python source validation

All changed Python files were compiled from source with `compile(..., "exec")` after the final edits.

## Pairwise Ranker V3

A synthetic positive candidate and deliberately strong wrong candidate were repeatedly pair-trained.

Observed in the isolated model test:

```text
before: positive=0.50, negative=0.50
after:  positive≈0.91, negative≈0.38
```

Persistence reload reproduced the score and reset returned the ranker to an empty model.

A second integration smoke test used `AIRuntime.rank_candidates()` with ten deliberately misleading high-score negatives. The GT-like candidate moved from rank 6 to rank 1 after online pairwise training, and the existing `memory.reset()` reset the V3 model through the new callback.

## Candidate-bank reserve

Test setup:

```text
frame 1: GT-like V1 candidate at (100,100)
frame 2: 200 unrelated current candidates
```

V2.3 output:

```text
201 candidates
1 historical carried candidate at the GT location
```

This verifies that a full 200-candidate current frame no longer forces historical bank capacity to zero.

## Deterministic benchmark hook

With `benchmark_control.json` set to seed `12345`, two independent F2-start calls produced identical Python and NumPy random values after the seed hook.

## Strong-signal candidate extraction

Small synthetic temporal blobs with multiple weak signal strengths were inserted into noisy arrays. The V2.3 multi-path candidate extractor placed a candidate directly at the injected location in the smoke cases.

## Early diagnostics registration

Synthetic GT was registered before any detector frame. A diagnostic skeleton existed immediately with:

```text
ground truth
benchmark seed
frames_seen = 0
```

This specifically targets earlier runs where 1000 synthetic shots sometimes yielded only 985-994 diagnostics rows.

## Analyzer

A synthetic V2.3 JSON diagnostic set was passed through `summarise()` and `print_summary()` successfully, including deterministic-seed reporting and the new V3 fields.
