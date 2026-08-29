# V2.22.6 — Frame-Unique Tracking

## Old behaviour

`HitScanner._update_tracks()` iterated every candidate and incremented a nearby track immediately. Four candidates around one physical location in the same camera frame could therefore produce four `hits`.

This mixed two different concepts:

- multiple algorithms/features agreeing in one image,
- the physical change persisting into later images.

Only the second is temporal confirmation.

## New behaviour

Each track may receive at most one temporal observation per `frame_ts`.

Additional candidates inside `track_merge_radius_px` during that same frame:

- may raise the track's maximum score,
- reset missed-frame aging,
- are counted as `v2226_same_frame_support`,
- do **not** change `hits`, `last_seen_ts`, or authoritative XY.

A genuinely later frame may increment `hits` once. This makes V2.22.5 Local Confirm the intended cheap persistence step after one global proposal pass.

## Re-hit / hole-in-hole

No new hard known-hole reject is introduced. A new shot close to an existing physical hole can still match the spatial track if a later current-shot frame provides fresh confirmation.
