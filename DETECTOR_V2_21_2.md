# Detector V2.21.2 — Physical Full-Frame Diagnostic + Local Temporal Refinement

## Why this exists

The first V2.21 full-frame projector/camera run produced 30 honest recent-PRE + POST packs.
The V2.21 global direct proposal prototype then measured:

- current oracle @20: 26.67%,
- current oracle @42: 90.00%,
- global AI_DIRECT oracle @20/@42: 3.33%,
- rescued current misses @20: 0,
- AI_DIRECT emitted the full 180 proposal cap on average.

This changes the diagnosis. On this white projector/camera session the old detector is often **near** the GT but not within 20 px. The first global direct map is meanwhile dominated by nuisance and is not ready to rescue misses.

V2.21.2 therefore does not train anything and does not change live authority. It measures three questions separately:

1. Is there a systematic current-candidate/GT coordinate bias inside the <=42px hits?
2. Is the real hole signal present in any temporal map at GT but buried below the global top-N cutoff?
3. Can evidence-backed local temporal maxima, searched only near existing current candidates, turn <=42px coverage into <=20px coverage?

## New local temporal proposal path

For every current V1/V2 candidate, V2.21.2 searches a 48 px circular neighborhood in:

- persistent darkening,
- black-hat gain,
- compact change,
- persistent absolute change,
- fused V2.21 evidence.

Only actual local maxima with local contrast are emitted. The code deliberately does **not** create a geometric grid around candidates; that would inflate oracle recall without image evidence.

This path cannot rescue a shot with no current candidate nearby. Its purpose is coordinate refinement/expansion. Global AI_DIRECT remains separately measured for true rescue.

## Evaluation discipline

The 130-pack root currently has only two sessions, so the legacy 60/20/20 split remains provisional. Use DEVELOPMENT diagnostics to choose the next implementation. Confirmation/holdout are reported but must not be tuned against.

No live candidate order, hit coordinate or game behavior changes in V2.21.2.
