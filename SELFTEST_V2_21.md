# SELFTEST V2.21

Run from repo root:

```bash
python3 -m automation.offline_v221_selftest
```

Expected checks:

1. direct proposal engine finds a newly introduced hole-like PRE->POST event without GT input,
2. direct proposals are deterministic for identical frozen inputs,
3. storage-aware candidate capture saves full recent PRE + bounded POST stack while omitting the large reference PRE when configured,
4. capture provenance includes `v221_full_frame_direct`,
5. a deliberately shifted mock domain is detected by the group-level diagnostic classifier,
6. live hit authority/order remains untouched.

The selftest does not claim physical detector accuracy.
