# V2.21.3 selftest

Run from repo root:

```bash
python3 -m automation.offline_v2213_selftest
```

Expected checks:

- plateau-aware anchored consensus recovers a compact multi-source blob in the presence of a stronger elongated plateau,
- candidate-derived target mask includes the candidate/target region and excludes unrelated image corners,
- target-masked direct proposals ignore a stronger nuisance blob outside the target mask,
- proposal generation is deterministic,
- proposal APIs do not accept GT.

Then run the physical benchmark:

```bash
python3 -m automation.direct_proposal_v2213_benchmark \
  --root content/ai/candidate_shadow_v216
```

Output report:

```text
content/ai/reports/v2213/fullframe_benchmark_v2213.json
```

Debug images:

```text
content/ai/reports/v2213/debug/
```
