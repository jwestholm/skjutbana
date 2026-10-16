"""Read-only GT timing audit. Reuses the original-frame loader and hash contract.

No camera/device, calibration, candidates or labels are opened. Missing recorded
images remain unavailable; raw backend timestamps never become exposure truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from automation import physical_trace_label_crops as crops
from automation.physical_gt_capture import STATES, frame_timing


def audit(capture):
    manifest=crops.load_gt_review(capture)
    root=Path(manifest['capture_path'])
    recorded=json.loads((root/'gt_review/manifest.json').read_text())
    source={s['shot_id']:s for s in recorded['shots']}
    rows=[]
    for shot in manifest['shots']:
        sid=shot['shot_id'];event=source[sid];pre=[];neutral={}
        for f in shot['frames']:
            for entry in f['crops']:
                meta=json.loads((root/entry['metadata_path']).read_text())
                timing=meta.get('timing') or frame_timing(
                    SimpleNamespace(timestamp=entry['timestamp'],acquisition=entry['acquisition']),
                    event['peak_ts'],event.get('audio_acquisition'),entry['role'])
                if entry['role']=='PRE':
                    # Old captures cannot retroactively pass a rule they did not
                    # record, even when a raw timestamp happens to be old enough.
                    proof=shot['pre_causality'][len(pre)]
                    pre.append(dict(role='PRE',status='CLOCK_GUARDED' if proof['pre_cutoff_verified'] else 'UNQUALIFIED',
                        source=entry['path'],image_sha256=entry['source_image_sha256'],
                        delta_ms_to_audio=proof['delta_ms_to_audio'],pre_cutoff_verified=proof['pre_cutoff_verified'],
                        causal_pre_certified=False,reason=proof['reason'],timing=timing))
                elif entry['role']=='GT_SETTLED':
                    neutral[entry['projector_state']]=dict(status='AVAILABLE',source=entry['path'],
                        image_sha256=entry['source_image_sha256'],pixel_sha256=entry['pixel_sha256'],
                        frame_timestamp=entry['timestamp'],timing=timing)
        rows.append(dict(shot_id=sid,capture_status=shot['capture_status'],
            pre=pre or [dict(role='PRE',status='UNAVAILABLE',source=None,image_sha256=None,
                delta_ms_to_audio=None,pre_cutoff_verified=False,causal_pre_certified=False,
                reason=event.get('pre_selection',{}).get('reason') or 'NO_SAVED_PRE',
                timing=event.get('pre_selection',{}).get('timing'))],
            settled={state:neutral.get(state,dict(status='UNAVAILABLE',source=None,image_sha256=None,
                reason='NO_HASH_VALID_SAVED_FRAME')) for state in STATES},missing=shot['missing']))
    crops.assert_inputs(manifest)
    return dict(schema='physical_gt_audit_v1',capture_path=str(root),shots=rows,
        input_hashes=dict(sorted(manifest['input_hashes'].items())),
        conclusion='RECEIVE_TIME_IS_NOT_EXPOSURE; CUTOFF_PASS_IS_NOT_CAUSAL_CERTIFICATION',
        validation='IMPLEMENTED_OFFLINE_VERIFIED; PHYSICAL_ACCEPTANCE_PENDING')


def print_table(result):
    print('SHOT | PRE role/status | delta-to-audio ms | cutoff pass | causal certified | reason')
    for shot in result['shots']:
        for pre in shot['pre']:
            delta=pre['delta_ms_to_audio']
            print(f"{shot['shot_id']} | PRE/{pre['status']} | {delta:.3f}" if delta is not None else
                  f"{shot['shot_id']} | PRE/{pre['status']} | UNAVAILABLE",end='')
            print(f" | {pre['pre_cutoff_verified']} | {pre['causal_pre_certified']} | {pre['reason']}")
        for state,frame in shot['settled'].items():
            print(f"  GT_SETTLED {state}: {frame['status']} sha256={frame['image_sha256'] or 'UNAVAILABLE'}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,help='Optional NEW JSON file outside source capture under /data/skjutbana')
    args=parser.parse_args()
    # Reject forbidden inputs before any manifest/device access.
    capture=crops.checked(args.capture)
    output=crops.checked(args.output) if args.output else None
    if output and (not output.is_relative_to(Path('/data/skjutbana')) or output.is_relative_to(capture) or output.exists()):
        parser.error('output must be a NEW file outside capture under /data/skjutbana')
    result=audit(capture)
    if output:
        # create-once; no baseline or source rewrite
        with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print_table(result)


if __name__=='__main__':main()
