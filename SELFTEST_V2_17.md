# V2.17 selftest

Run from repository root:

```bash
python3 -m automation.newhole_v217_selftest
```

Expected end:

```text
All V2.17 selftests passed.
```

The selftest checks four non-negotiable behaviours:

1. a before/after NEW-hole model learns a newly appearing hole while an
   unchanged old hole is correctly negative for **newness**,
2. V2.16 hard-negative candidate identity is interpreted as `NOT NEW`, never as
   a static `non-hole` label,
3. future candidate packs preserve the additional true recent-pre camera patch
   and timestamp while legacy/reference PRE remains intact,
4. saved models preserve `shadow_only` and semantic metadata.

The selftest is plumbing/semantic validation. It is not a claim about real
shooting-range accuracy.
