# SELFTEST V2.20

Run:

```bash
python3 -m automation.offline_v220_selftest
```

Checks:
- deterministic scenario spec and RGB output reproduction,
- observed output does not collapse to grayscale,
- new hole passes compact/local visibility QA,
- static image scenarios keep PRE/POST drift very low,
- dynamic procedural scenarios still preserve world semantics.
