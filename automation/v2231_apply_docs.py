from __future__ import annotations

from pathlib import Path

MARKERS = {
    "CURRENT_STATE.md": "### V2.23.1 — high-recall training pool + support-gated champion",
    "HIT_DETECTION_PLAN.md": "### V2.23.1 — proposal recall before ranking",
    "AI_CONTEXT.md": "### V2.23.1 training contract",
}

SECTIONS = {
"CURRENT_STATE.md": r'''

---
### V2.23.1 — high-recall training pool + support-gated champion

V2.23.0 plumbing was verified end-to-end with one 100-shot F2 run, but the native capture used the already-truncated final candidate list: 100 shots / 9,704 candidates / only 10 shots with an actual <=20 px candidate. Its 20-shot validation subset contained zero <=20 candidates, so the automatically promoted linear model was a plumbing artifact, not evidence of model quality.

V2.23.1 fixes the data boundary rather than tuning the ranker. New F2/manual captures union V2.8 `_v28_all_hypotheses`, `_v28_hypothesis_pool`, core/recall pools and the final/live lists after `rank_with_funnel`; retention/deduplication is GT-independent. Proposal oracle @5/@10/@20/@42 is reported separately from ranking quality.

Legacy V2.16/V2.20 packs are now loaded through the canonical `CandidatePackV216.load(path)` JSON+NPZ contract instead of assuming candidates live directly in JSON. Converted records are cached under `content/ai/training_v223/cache/legacy_v2231/` so autonomous rounds do not reread ~GB-scale legacy patch arrays.

Research champion promotion now requires meaningful development + validation positive support and improvement over the current baseline. The old V2.23.0 zero-oracle champion remains on disk for audit but is quarantined and cannot be loaded for shadow scoring. Protected holdout remains excluded from all automatic selection. Live authority remains unchanged/NO.
''',
"HIT_DETECTION_PLAN.md": r'''

---
### V2.23.1 — proposal recall before ranking

Training is a two-stage problem and metrics must never conflate them:

1. **Proposal quality:** did the GT-free candidate sources propose something within 5/10/20/42 px of the true current hit?
2. **Ranking quality:** conditional on a usable candidate being present, can the challenger place it Top-1/Top-3?

A ranker is not allowed to receive credit or blame for a shot whose correct neighbourhood is absent. Conversely, a high-recall pool is not success unless ranking can reduce it to a correct final choice.

Native F2 capture now preserves V2.8 micro-hypotheses/recall pools instead of only the displayed final list. Historical V2.16/V2.20 candidate packs are imported with their canonical loader and split semantics; `candidate_synthetic_v220_validation` is protected holdout. No GT-forced candidate may become a training candidate or improve oracle metrics.

Promotion gate: enough development positives, enough validation shots, enough validation oracle@20 support, and measurable improvement over baseline are required even for a research/shadow champion. Protected holdout is never used to choose a challenger. Live authority remains a later physical-session gate.
''',
"AI_CONTEXT.md": r'''

---
### V2.23.1 training contract

Current model work should prefer reusable candidate groups over more live-runtime heuristics. F2/manual capture is append-only; old V2.16/V2.20 packs are converted once and cached. The V2.8 all-micro-hypothesis / recall pools are legitimate GT-free proposal sources and may be used for training. GT is used only after proposal creation for labels, diagnostics, split metrics and supervised fitting.

Keep three semantics separate: static physical-hole appearance; current NEW-hole evidence; final within-shot candidate ranking. Policy/bookkeeping fields, GT distance, current model score and forced-GT storage helpers remain forbidden as physical model inputs. V2.21.5 dense physical proposal remains an independent high-recall expert/research source to be fused deliberately later rather than silently mixed into labels.

A `research_shadow_champion` is only a benchmark champion. V2.23.1 quarantines pre-gate V2.23.0 champions with zero/insufficient positive validation support. No V2.23 model grants game authority.
''',
}


def main() -> None:
    for name, marker in MARKERS.items():
        path = Path(name)
        if not path.exists():
            print(f"[SKIP] {name} not found")
            continue
        text = path.read_text(encoding="utf-8")
        if marker in text:
            print(f"[OK] {name}: V2.23.1 section already present")
            continue
        with path.open("a", encoding="utf-8") as handle:
            handle.write(SECTIONS[name].replace("\\n", "\n"))
        print(f"[PASS] {name}: V2.23.1 section appended")


if __name__ == "__main__":
    main()
