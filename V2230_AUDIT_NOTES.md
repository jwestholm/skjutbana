# V2.23.0 audit notes

The V2.23 build was based on the relevant current training/runtime files plus historical model documentation and reports available for the project. The installer also includes `v2230_audit`, which must be run on the shooting PC to inventory artifacts that exist only under `content/ai/` locally.

## Existing pieces retained

- F1/F2 AI Training scene and its automatic GT clicks.
- `AIRuntime` / `SimpleAIMemory` online learner for continuity/comparison.
- V2.16 candidate-pack family as the primary reusable real-candidate bridge.
- V2.17 semantics: static hole appearance and current-new-hole evidence are different labels/tasks.
- V2.19 generated-world/candidate-pack pipeline.
- V2.21.5 dense pool/model artifacts; dense proposal generation is treated as a source, not automatically as champion ranker.
- Entire V2.22.1–V2.22.6 live runtime chain remains authority/foundation.

## Main gap found

There was no single owner for:

1. discovering native + legacy shot groups,
2. enforcing one stable candidate feature contract,
3. session-aware train/validation/protected-holdout discipline,
4. training multiple challengers,
5. validating them on the same candidate groups,
6. keeping a research champion registry,
7. letting F2 and CLI use the same engine.

V2.23.0 supplies that owner.

## Deliberate limitations of first V2.23 release

- The first V2.23 rankers use scalar candidate evidence. Existing pixel Hole-AI/NewHole-AI models are audited but not silently fused until their installed artifacts/coverage are confirmed by the shooting-PC audit.
- V2.23 does not create direct image proposals. If the actual candidate pool misses GT, this release reports the miss rather than manufacturing a positive.
- Automatic holdout evaluation is intentionally absent. A holdout repeatedly consulted by an overnight optimizer is not a holdout.
- Research champion != live champion. Live authority is unchanged.
