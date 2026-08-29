from __future__ import annotations

from pathlib import Path

MARKERS = {
    "CURRENT_STATE.md": "### V2.23.2 — proposal learning + fresh-F2 domain validation",
    "HIT_DETECTION_PLAN.md": "### V2.23.2 — recover the hit before ranking it",
    "AI_CONTEXT.md": "### V2.23.2 proposal/domain training contract",
}

SECTIONS = {
"CURRENT_STATE.md": r'''

---
### V2.23.2 — proposal learning + fresh-F2 domain validation

A fresh 100-shot V2.23.1 F2/projector run showed that the live/V2.8 training union was still only about one hundred candidates per shot and recovered the labelled hit within 20 px in only about one shot out of ten. The V2.23.1 research champion also failed to generalise to that fresh projector/camera session. Therefore ranking is no longer allowed to hide proposal failure or domain shift.

V2.23.2 stores one full recent PRE grayscale frame plus up to three unique full POST frames for each labelled F1/F2/manual shot. These framepacks feed the existing offline V2.21 direct, V2.21.2 local and V2.21.5 dense physical proposal engines. Ground truth is applied only after proposal generation for oracle diagnostics and labels. Proposal sidecars are cached and merged GT-independently into the unified V2.23 dataset.

The newest substantial F2/projector session (>=50 shots) is excluded from fitting and used as a fresh-domain research gate. A challenger must have reference-baseline coverage, positive oracle support and measurable improvement on both ordinary validation and that fresh F2 domain before it can become a research/shadow champion. Pre-V2.23.2 champions without domain metrics are quarantined. Protected holdout remains outside automatic selection and live authority remains NO.

The historical `center_bias` F2 sampler was actually uniform. V2.23.2 makes it a soft centre-biased distribution with 25% uniform exploration; separate uniform/edge/corner modes remain for robustness testing.
''',
"HIT_DETECTION_PLAN.md": r'''

---
### V2.23.2 — recover the hit before ranking it

Treat perception as two independent supervised problems. **Proposal learning** asks whether a GT-free camera algorithm proposed a candidate near the new physical hit. **Ranking learning** asks which candidate should win only after such a neighbourhood exists. Report oracle @5/@10/@20/@42 separately from Top-1/Top-3/MRR.

F2/manual labelled shots now persist full PRE/POST frame evidence so offline proposal algorithms can be rerun after live detection failed. Reuse the V2.21 direct temporal maps, V2.21.2 local proposals and V2.21.5 dense physical pool. Dense processing is allowed to be expensive offline; gameplay remains on the frozen V2.22 path. Candidate retention/deduplication must never use GT.

Research promotion requires a reproducible reference baseline and generalisation to the newest substantial F2 projector/camera session that was excluded from fitting. Protected holdout is still untouched by automatic selection. Do not fuse Hole-AI/NewHole-AI or grant live authority until this proposal/domain boundary is trustworthy.
''',
"AI_CONTEXT.md": r'''

---
### V2.23.2 proposal/domain training contract

For labelled F1/F2/manual shots, preserve full recent PRE plus up to three unique POST grayscale frames in a compressed JSON+NPZ framepack. GT coordinates are metadata labels only and are forbidden from direct/local/dense proposal generation. Offline proposal results are cached as sidecars and may contribute physical dense features/provenance to the unified ranking record.

The newest F2/projector session with at least 50 shots is a fresh-domain research gate and must not enter model fitting. Once a newer substantial F2 session exists, an older one may re-enter engineering data according to the normal split policy. This creates a conservative one-session lag for self-learning instead of training and validating on the same projector run.

`baseline_rank` is preferred for reference ranking; where legacy/native data lacks it, captured `baseline_score` may be sorted deterministically. A research champion must beat this reference on both ordinary validation and fresh-F2 domain with sufficient positive proposal support. Protected holdout remains untouched; no V2.23 model has live authority.
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
            print(f"[OK] {name}: V2.23.2 section already present")
            continue
        with path.open("a", encoding="utf-8") as handle:
            handle.write(SECTIONS[name].replace("\\n", "\n"))
        print(f"[PASS] {name}: V2.23.2 section appended")


if __name__ == "__main__":
    main()
