# V2.25.3 — Cross-thread registered authority + cross-shot physical novelty

## Why this version exists

The V2.25.2 physical five-shot run proved that the registered detector frame was
computed in the CV worker, but the main HitScanner did not see the same readiness
state. The worker logged `REGISTERED-READY` and the main authority path later logged
`no_registered_frame` for the same shot. The reason was instance-local state on two
different scanner objects.

V2.25.2 also showed that its deliberately permissive registered freshness gate was
not selective enough on the heavily used physical board: nearly every candidate in
every object region could pass. Stable projector/board hotspots therefore remained
competitive across consecutive shots.

## Runtime chain

```
audio PANG
   |
   +--> frozen HitRegions / shot_id
   |
   +--> V2.22.4 CV worker
            |
            +--> V2.25.1 per-region proposals
            +--> V2.25.2 registered PRE->POST evidence
            +--> V2.25.3 shared READY bridge --------------+
            +--> compare with previous confirmed camera XY |
            +--> physical novelty metadata                 |
                                                           v
                                                main HitScanner authority
                                                           |
                                                V2.22.5 temporal confirmation
                                                           |
                                                V2.25.3 novelty selector
                                                           |
                                                     physical HitEvent
                                                           |
                                                     GameObject collision
```

## Shared readiness

Readiness is process-local and protected by a lock. It is keyed by shot id plus peak
timestamp, so a reused shot id after reset cannot silently inherit an old ready flag.
The worker writes the bridge; the main scanner reads the same bridge.

## Cross-shot novelty

Candidate history is stored in canonical full-camera XY, not detector-crop-local XY.
The V2.22.1 crop origin and V2.24.4 scale are applied before history comparison.
Only earlier shot ids/peaks participate; later registered frames from the same shot do
not count as history against themselves.

A candidate close to a previously confirmed hotspot receives a soft recurrence
penalty. A spatially new candidate is favoured. The score is based only on physical
evidence:

- distance to prior confirmed camera hotspots;
- registered PRE->POST signature gain;
- within-region physical excess;
- region candidate sparsity;
- compact registered change;
- V2.22.5 second-frame confirmation.

No target/no-shoot role, health, damage, projectile, player, score or object priority
is used.

## Re-hit / hole-in-hole

History is never a hard exclusion. A recurrent coordinate can recover the penalty if
the new registered PRE->POST signature strengthens materially. This preserves the
architectural rule that old-hole proximity may lower confidence but cannot make a
physical re-hit illegal.

## Confirmation ordering

V2.25.3 calls the original V2.22.5 local physical confirmation before V2.25.1/2
balancing. This is intentional: a spatially novel candidate must not be removed by an
older absolute-score balance before novelty can be evaluated. The output is then
balanced per physical region and globally bounded.

## Rescue

If the shared worker frame never becomes ready, or no novelty-authority track survives,
V2.25.3 requests the existing exactly-once V2.22.5 FULL rescue. Once consumed, the
selector bypasses all V2.25.x local authority and uses the original global physical
selector. No object snapping is introduced.
