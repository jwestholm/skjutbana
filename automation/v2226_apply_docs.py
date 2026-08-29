from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SECTIONS = {
    "CURRENT_STATE.md": '''\n---\n## V2.22.6 Frame-Unique Tracking + Audio Near-Miss Telemetry — 2026-08-29\n\n- Physical V2.22.5 tests reduced visible shot latency to about 0.58-0.59 s on two registered shots; FAST extraction was about 82-84 ms and the async global CV job about 0.41-0.45 s.\n- V2.22.5 also exposed a tracking semantics bug: several nearby V1/V2/bank candidates from one camera frame could increment `HoleTrack.hits` several times. This allowed same-frame spatial agreement to masquerade as temporal persistence and could bypass Local Confirm.\n- V2.22.6 makes `hits` frame-unique: a track gains at most one temporal hit for a camera timestamp. Additional nearby candidates are retained separately as `same_frame_support`.\n- The intended live path is therefore one GLOBAL FAST proposal observation followed by a later LOCAL-CONFIRM observation. Reprocessing the same physical frame cannot confirm a track.\n- A physical test also produced one shot with no AudioPeakEvent. V2.22.6 does not retune audio thresholds yet; it logs strong rejected transients with absolute/dynamic/crest/cooldown rejection reasons so the next physical series can identify the failing gate without guessing.\n''',
    "HIT_DETECTION_PLAN.md": '''\n---\n## V2.22.6 tracking evidence semantics\n\n`HoleTrack.hits` is strictly temporal evidence. One physical camera frame may increment a given track at most once. Multiple nearby candidates from V1/V2/bank/rescue on the same frame are **same-frame source/spatial support**, not additional temporal hits.\n\nNormal confirmation funnel:\n\n1. one audio event creates one shot context,\n2. GLOBAL FAST proposal on camera frame N creates/seeds tracks with `hits=1`,\n3. same-frame candidate clusters may raise support/provenance but not `hits`,\n4. LOCAL-CONFIRM on a genuinely later camera frame N+k may increment the track to `hits=2`,\n5. existing age/readiness logic can then emit the hit,\n6. full high-recall rescue remains fallback, not ordinary persistence.\n\nAudio trigger recall is now measured explicitly. Strong rejected transients must report the failed gate (`abs`, `noise`, `crest`, `cooldown`) before thresholds are changed. Prefer evidence-based retuning over blindly lowering the trigger.\n''',
    "AI_CONTEXT.md": '''\n---\n## V2.22.6 frame-unique evidence contract\n\n- `track.hits` means observations on different camera timestamps.\n- `same_frame_support` means multiple proposals agree spatially in one image; it is useful ranking/fusion evidence but is not persistence.\n- A candidate cluster from one frame must never satisfy a multi-frame confirmation requirement by itself.\n- V2.22.5 LOCAL-CONFIRM is the preferred cheap second temporal observation after a global proposal.\n- Audio near-miss telemetry is diagnostic only. It does not create soft shot events or change trigger authority in V2.22.6.\n''',
}


def main() -> None:
    for name, section in SECTIONS.items():
        path = ROOT / name
        if not path.exists():
            print(f"[WARN] {name} not found; leaving unchanged")
            continue
        text = path.read_text(encoding="utf-8")
        heading = next((line.strip() for line in section.splitlines() if line.strip().startswith("## V2.22.6")), "## V2.22.6")
        if heading in text:
            print(f"[OK] {name}: V2.22.6 section already present")
            continue
        with path.open("a", encoding="utf-8") as fh:
            if text and not text.endswith("\n"):
                fh.write("\n")
            fh.write(section)
        print(f"[UPDATED] {name}")


if __name__ == "__main__":
    main()
