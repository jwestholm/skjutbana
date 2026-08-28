# V2.21.4 selftest

Run from repository root:

```bash
python3 -m automation.offline_v2214_selftest
```

Expected checks:

- broad GT-free dense pool contains the synthetic temporal mark,
- pairwise ranker trains on DEVELOPMENT-style samples,
- frozen ranker retains a held-out synthetic mark in top-K,
- model save/load is deterministic,
- public inference functions accept no GT argument.

This selftest proves plumbing/semantics only. It does not prove physical accuracy.
