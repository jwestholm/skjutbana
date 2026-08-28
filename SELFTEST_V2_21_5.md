# V2.21.5 selftest

From repository root:

```bash
python3 -m automation.offline_v2215_selftest
```

It verifies:

1. the broad proposal API has no GT argument;
2. compact temporal evidence survives a strong elongated nuisance pattern;
3. listwise training uses actual candidate rows and records zero forced GT/jitter proposals;
4. DEVELOPMENT-style shot-level cross-fit can retain a learnable candidate;
5. model save/load leaves frozen inference scores unchanged.

A synthetic selftest validates software semantics only. It is not a physical accuracy claim.
