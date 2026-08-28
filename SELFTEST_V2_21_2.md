# V2.21.2 selftest

Run from repository root:

```bash
python3 -m automation.offline_v2212_selftest
```

Expected final line:

```text
All V2.21.2 selftests passed.
```

Then run the real 30-pack diagnostic:

```bash
python3 -m automation.direct_proposal_v2212_benchmark \
  --root content/ai/candidate_shadow_v216
```

The command writes a JSON report and a small set of miss/debug PNGs under `content/ai/reports/v2212/`.
