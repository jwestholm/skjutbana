# V2.23.3 documentation patch

Append the V2.23.3 state to `CURRENT_STATE.md`, `HIT_DETECTION_PLAN.md` and `AI_CONTEXT.md` using:

```bash
python3 -m automation.v2233_apply_docs
```

Core state to preserve:

- V2.23.2 fresh F2 result: current@20 6%, local@20 15%, dense/union@20 95%, dense/union@42 96%, ~9.36k dense candidates/shot.
- Proposal discovery is therefore no longer the dominant offline blocker.
- Existing V2.23.2 rankers failed on the fresh dense F2 domain (Top1@20 0), so ranking/reduction is the current focus.
- V2.23.3 adds GT-free full-frame PRE/POST rich evidence and pairwise reduction before final ranking.
- With only one substantial dense F2 session, V2.23.3 uses a same-session 80/20 bootstrap proof and refuses domain/generalisation claims.
- A second independent F2 x100 session becomes the first real fresh-domain gate.
- V2.22 live authority remains frozen/unmodified.
