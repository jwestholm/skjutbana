# AI context — V2.25.3 cross-shot physical novelty

When modifying hit detection after V2.25.3, preserve these invariants:

1. Worker and main HitScanner are different execution contexts. Do not store shot
   authority state only on a worker scanner instance.
2. Cross-thread shot state must be keyed by `shot_id` and validated by peak timestamp
   or equivalent generation identity.
3. Candidate history uses canonical full-camera XY. CandidateGenerator itself runs in
   V2.22.1 crop-local coordinates before the crop is rebased.
4. Cross-shot recurrence is a physical prior only. It must not inspect target role,
   no-shoot role, object type, owner, health, damage, penetration or game score.
5. A previously seen coordinate is not forbidden. Re-hits/hole-in-hole remain legal;
   stronger new PRE->POST evidence can recover recurrence penalty.
6. Never change `camera_x`, `camera_y`, `game_x` or `game_y` to make a candidate fit an
   object. Search/score metadata is allowed; coordinate snapping is not.
7. Keep V2.22.5 FULL rescue global and capable of bypassing every V2.25.x local selector.
8. GameObject collision remains downstream of physical HitEvent XY.

V2.25.3 is not a learned model and does not promote V2.23 shadow AI to live authority.
