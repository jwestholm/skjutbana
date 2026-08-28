# V2.22.2 documentation notes

Reviewed alongside the existing hit-detection roadmap and V2.22/V2.22.1 notes.

V2.22.2 does not change the long-term architecture:

```text
audio shot
   -> camera / physical / history / game experts
   -> discrete candidate evidence
   -> ShotResolver
   -> one final camera XY
   -> existing HitInput homography
   -> game XY
```

The version is a live fast-path cleanup motivated by physical testing, not a new
training-model generation. The V2.21.5 dense pool remains an offline/shadow
physical expert candidate for later parallel integration.

Important retained principles:

- game context is a weak prior, never ground truth;
- canonical detection coordinates remain camera XY;
- candidate cleanup can use a cheap vectorised screen transform, but final game
  coordinates are still produced only by the existing HitInput path;
- every new filter is fail-open/configurable and should be validated with real
  physical shots before autonomous F2/night runs.
