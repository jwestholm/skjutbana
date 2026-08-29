# V2.22.6 Plan — Frame-Unique Tracking + Audio Near-Miss Telemetry

## Why this version exists

Physical V2.22.5 testing showed two separate problems after the large latency improvements:

1. A single global result could contain several nearby candidates and immediately produce `hits=2..4` on one `HoleTrack`. That means `hits` was not actually measuring persistence over time. It also meant V2.22.5 Local Confirm could be bypassed completely.
2. One physical shot produced no audio event at all. There was no telemetry below `AudioPeakEvent`, so it was impossible to tell whether the absolute threshold, adaptive noise ratio, crest factor, or cooldown rejected the sound.

V2.22.6 fixes measurement semantics before any further ranker tuning.

## Tracking contract

`HoleTrack.hits` is temporal evidence only.

- frame N + 4 nearby proposals => `hits=1`, `same_frame_support=4`
- frame N+k Local Confirm => `hits=2`
- same frame accidentally processed twice => still no additional hit

The strongest candidate in one frame keeps authoritative XY. Duplicate proposals in that frame do not get averaged into an artificial point.

## Audio contract

Audio trigger behaviour is unchanged in this version. Strong rejected transients become observable:

```text
[V2.22.6 AUDIO-RAW] NEAR-MISS ... reject=abs,crest
```

A real accepted trigger is also logged at the audio-thread decision point:

```text
[V2.22.6 AUDIO-RAW] TRIGGER ...
```

This separates microphone/threshold decisions from later main-thread dispatch telemetry.

## Expected next physical test

First test only 3 shots. For each accepted shot confirm:

1. `AUDIO-RAW TRIGGER` appears before `AUDIO-PRIORITY`.
2. first global result logs `V2.22.6 TRACK ... temporal=0` for new tracks and does not immediately fake multiple hits from one frame.
3. `V2.22.5 LOCAL-CONFIRM` appears on a later frame.
4. the next `V2.22.6 TRACK` shows a temporal match and the hit can emit.

If a shot is physically fired but not accepted, capture the `AUDIO-RAW NEAR-MISS` line. Do not lower thresholds until its reject reason is known.
