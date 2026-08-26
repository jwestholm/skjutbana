from __future__ import annotations
import argparse
from pathlib import Path
from src.engine.offline.new_hole_training_v217 import DEFAULT_HARDNEG_MANIFEST, DEFAULT_ROOT, TrainingConfigV217, discover_samples_v217, split_samples_v217

def main() -> int:
    p=argparse.ArgumentParser(description='Inspect V2.17 NEW-hole before/after training data')
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT); p.add_argument('--hard-negatives',type=Path,default=DEFAULT_HARDNEG_MANIFEST)
    args=p.parse_args()
    samples,summary=discover_samples_v217(args.root,hardnegative_manifest=args.hard_negatives,config=TrainingConfigV217())
    split,provisional=split_samples_v217(args.root,samples)
    print('V2.17 NEW-HOLE DATASET INSPECTION'); print('================================')
    print(f"Candidate packs             : {summary['candidate_packs']}")
    print(f"Usable shots                : {summary['usable_shots']}")
    print(f"Sessions                    : {len(summary['sessions'])} {summary['sessions']}")
    print(f"Samples                     : {summary['samples']}")
    print(f"Positive NEW-hole samples   : {summary['positives']}")
    print(f"Negative NOT-NEW samples    : {summary['negatives_newhole']}")
    print(f"Likely old-hole negatives   : {summary['likely_old_hole_negatives']}")
    print(f"V2.16 hardneg manifest used : {summary['hardnegative_manifest_used']}")
    print(f"True recent-pre shots       : {summary.get('recent_pre_shots',0)} (legacy packs fall back safely)")
    print(f"Split provisional           : {provisional}")
    for name,rows in split.items():
        print(f"  {name:12s}: samples={len(rows):4d} pos={sum(s.label==1 for s in rows):3d} neg={sum(s.label==0 for s in rows):4d} old_like={sum(s.label==0 and s.likely_old_hole for s in rows):3d}")
    print('\nSEMANTIC CONTRACT')
    print('  Hole-AI: old hole AND new hole are both hole-like positives.')
    print("  NewHole-AI V2.17: only the current shot's new hole is positive.")
    print('  A V2.17 negative may be an OLD REAL HOLE. It is NOT a static non-hole label.')
    return 0
if __name__=='__main__': raise SystemExit(main())
