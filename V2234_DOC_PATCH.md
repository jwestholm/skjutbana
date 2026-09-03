# V2.23.4 documentation patch

Run:

```bash
python3 -m automation.v2234_apply_docs
```

The command appends the V2.23.4 strategic state to `CURRENT_STATE.md`, `HIT_DETECTION_PLAN.md`, and `AI_CONTEXT.md` if those files exist and do not already contain the section.

The section records that V2.23.3's tabular reducer failed the intended decision gate (high dense oracle, poor retention), so V2.23.4 changes model class to learned PRE/POST candidate patches rather than adding more scalar heuristics. It also records training-only GT anchors, honest candidate/oracle metrics, validation-only trial selection, fresh-domain discipline, and shadow-only authority.
