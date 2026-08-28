# V2.22.2 — Cursor, novelty, ridge and backlog cleanup

## Status

Live/shadow fast-path delta. This does **not** make the V2.21.5 physical dense
ranker authoritative. V2.22 ShotResolver remains advisory-first.

## Why this version exists

The first V2.22.1 physical run proved that the perspective-aware playfield edge
guard removes the previous border candidate rows. The remaining live failure
modes were different:

- a visible mouse cursor is a projected moving object and can generate a strong
  PRE/POST difference;
- static physical holes remain visually hole-like and can receive large
  morphology scores even though they are not the *new* shot;
- a slight board/projector shift can create a long horizontal family of false
  candidates;
- after a slow analysis pass, many 4K frames can accumulate and the legacy
  update loop converts all of them even though detection uses the newest frame.

## Cursor guard

The cursor is not hidden permanently because manual AI training needs the mouse
after the shot to label the true hit.

At audio peak V2.22.2:

1. stores current pygame mouse position and visibility;
2. hides the OS cursor;
3. transforms a small square around that pre-shot screen position through the
   existing inverse homography and removes it from the detector ROI;
4. restores the cursor when no shot event remains open.

Therefore the cursor is absent from post-shot frames, while its disappearance
from the PRE-shot position is also prevented from becoming a candidate.

## Old holes / novelty

V2.22.2 does not blindly delete every point near an old hole. A real projectile
can legitimately re-hit an existing hole.

- registered hole + weak PRE->POST change => stale, reject before tracking;
- registered hole + strong fresh PRE->POST change => keep as fresh re-hit;
- unregistered static hole appearance => heavily demote when PRE-shot evidence
  is informative, but do not hard-delete it solely because it looks static.

This uses the already existing `pre_shot_change` feature rather than adding a
new model dependency.

## Horizontal ridge cleanup

Filtering in raw camera `y` would be wrong under perspective: a physical
horizontal playfield line can be sloped in the camera image.

V2.22.2 therefore takes only the already-produced candidate list and performs
one vectorised camera->screen homography. Candidates are grouped into narrow
screen-y bands. A band is considered a motion ridge only when it contains many
candidates and spans a large fraction of playfield width.

Most ridge candidates are removed. Strong fresh PRE-shot candidates survive,
and as a final safety valve at least the strongest candidate in a detected ridge
is kept so a real shot is never made impossible merely because it landed on a
moving band.

## Backlog thinning

While a shot is open, V2.22.2 bounds `get_new_frames_since_last_pickup()` to
three temporally spread frames, always including the newest. This prevents a
slow pass from creating a feedback loop where the next update spends time
converting dozens of already-obsolete 4K frames.

The camera ring buffer itself is untouched, so timestamped PRE-shot history
remains available to the existing V2.22.1 logic.

## Diagnostics

Two new lines matter during testing:

```text
[V2.22.2 CLEAN] shot=... candidates=A->B stale=S demoted=D ridges=R ridge_removed=X fresh_saved=F cleanup=...ms
[V2.22.2 FAST]  shot=... frames=N->K drop=... detector=...ms cleanup=...ms overhead=...ms update=...ms
```

Interpretation for the next optimization:

- `detector` dominates and is >~500–700 ms: next step should be a true
  coarse-to-fine detector (lower-resolution proposal map + full-resolution local
  refinement), not more Python ranking tweaks;
- `overhead` dominates with a large frame backlog: optimize frame ingestion and
  history representation further;
- detector/update are fast but HIT `e2e` remains high: track-confirm / event
  timing is the next bottleneck.

## Artifact-mask warning

The observed `Artifact mask too aggressive (... active), disabling` warning is
left fail-open in V2.22.2. The mask automatically disabling itself is safer than
letting a questionable calibration hide valid shots. Recalibrating that mask is
a separate experiment after the fast path is stable.
