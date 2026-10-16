"""Opt-in, post-decision human review capture; never a physical evidence source.

Owns only the research pause/display schedule. CameraFrame, scanner/audio lifecycle,
PhysicalTraceRecorder and physical_trace_label retain their existing ownership.
"""
from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time

SCHEMA = 'physical_gt_capture_v1'
FRAME_SCHEMA = 'physical_gt_review_frame_v1'
STATES = {'WHITE': (255,255,255), 'BLACK': (0,0,0), 'MID_GRAY': (127,127,127)}
DELAYS = dict(settle_s=.5, background_s=.6, restore_s=1., frame_timeout_s=2.)
# Research reference guard, not a new live detector threshold. Reuse the older
# ring-reference 500ms offset, but NEVER fall back to a newer/nearest frame.
PRE_GUARD_S = .5
PRE_RULE = 'camera_ring_receive_and_publish_before_audio_minus_500ms_v1'
TIMING_SCHEMA = 'physical_gt_timing_v1'


def frame_timing(frame, peak_ts, audio_acquisition, role):
    """Explicit measured clocks and unknowns; never infer exposure from backend PTS."""
    acq = dict(getattr(frame, 'acquisition', {}) or {})
    proof = pre_provenance(frame, peak_ts, audio_acquisition)
    stream, sequence = acq.get('stream_id'), acq.get('sequence')
    exposure = acq.get('exposure_monotonic')
    exposure_available = type(exposure) in (int, float) and math.isfinite(exposure)
    return dict(schema=TIMING_SCHEMA, role=role,
        frame_source_id=f'{stream}:{sequence}' if stream is not None and sequence is not None else None,
        camera_index=acq.get('camera_index'), backend_name=acq.get('backend_name'),
        source_stream_id=stream, source_sequence=sequence,
        sequence_semantics='APPLICATION_SUCCESSFUL_READ_COUNTER_NOT_DEVICE_SEQUENCE',
        device_sequence=None, v4l2_timestamp_flags=None,
        frame_timestamp=getattr(frame,'timestamp',None),
        read_start_monotonic=acq.get('read_start_monotonic'), read_end_monotonic=acq.get('read_complete_monotonic'),
        receive_wall_ts=acq.get('receive_wall_ts'), publication_monotonic=acq.get('published_monotonic'),
        publication_semantics='AFTER_TRANSFORM_BEFORE_RING_APPEND_NOT_EXPOSURE',
        audio_peak_ts=peak_ts, audio_peak_semantics='RECEIPT_ANCHORED_ESTIMATE_NOT_DEVICE_ACQUISITION',
        audio_trigger_receipt_estimate_monotonic=(audio_acquisition or {}).get('trigger_receipt_estimate_monotonic'),
        delta_ms_to_audio=proof['delta_ms_to_audio'], publication_delta_ms_to_audio=proof['publication_delta_ms_to_audio'],
        configured_pre_margin_ms=PRE_GUARD_S*1000, selection_rule=PRE_RULE if role=='PRE' else 'NOT_A_PRE_REFERENCE',
        pre_cutoff_verified=proof['pre_cutoff_verified'] if role=='PRE' else None,
        exposure_timestamp_available=exposure_available, exposure_timestamp=exposure if exposure_available else None,
        exposure_timestamp_verified=False, backend_timestamps_raw=deepcopy(acq.get('backend_timestamps_raw')),
        causal_pre_certified=False, certification_reason=proof['causal_pre_limitation'] if role=='PRE' else 'NOT_A_PRE_REFERENCE',
        unavailable_reason='NO_SAVED_FRAME' if frame is None else None)


def pre_provenance(frame, peak_ts, audio_acquisition):
    """Audit receipt clocks only; device audio latency/exposure are unbounded.

    Passing the guard is NOT proof that the pixels precede physical damage.
    Keep that distinction explicit rather than invent an impact timestamp.
    """
    acq = dict(getattr(frame, 'acquisition', {}) or {})
    audio = dict(audio_acquisition or {})
    stamp = getattr(frame, 'timestamp', None)
    receive = acq.get('receive_wall_ts')
    started, completed, published = (acq.get(k) for k in
        ('read_start_monotonic', 'read_complete_monotonic', 'published_monotonic'))
    audio_mono = audio.get('trigger_receipt_estimate_monotonic')
    finite = lambda v: type(v) in (int, float) and math.isfinite(v)
    complete = all(finite(v) for v in (stamp, receive, started, completed, published, peak_ts, audio_mono))
    consistent = (complete and stamp == receive and started <= completed <= published
        and audio.get('timestamp_semantics') == 'RECEIPT_ANCHORED_ESTIMATE_NOT_DEVICE_ACQUISITION')
    passed = bool(consistent and receive <= peak_ts-PRE_GUARD_S and published <= audio_mono-PRE_GUARD_S)
    return dict(selection_rule=PRE_RULE, guard_ms=PRE_GUARD_S*1000,
        audio_peak_ts=peak_ts, audio_acquisition=audio,
        frame_receive_wall_ts=receive, frame_read_start_monotonic=started,
        frame_read_complete_monotonic=completed, frame_published_monotonic=published,
        delta_ms_to_audio=(stamp-peak_ts)*1000 if finite(stamp) and finite(peak_ts) else None,
        publication_delta_ms_to_audio=(published-audio_mono)*1000 if finite(published) and finite(audio_mono) else None,
        allowed_pre_cutoff_ts=peak_ts-PRE_GUARD_S if finite(peak_ts) else None,
        allowed_pre_cutoff_monotonic=audio_mono-PRE_GUARD_S if finite(audio_mono) else None,
        pre_cutoff_verified=passed, causal_pre_certified=False,
        causal_pre_limitation='AUDIO_UPSTREAM_DELAY_AND_PHYSICAL_IMPACT_TIME_UNVERIFIED',
        physical_impact_timestamp=None,
        reason='RECEIPT_GUARD_PASSED_NOT_PHYSICAL_CERTIFICATION' if passed else
            ('MISSING_OR_INCONSISTENT_CLOCK_PROVENANCE' if not consistent else 'NEWER_THAN_PRE_CUTOFF'))


def select_review_pre(ring, peak_ts, audio_acquisition):
    """Use the existing shot-time selector on qualified native camera frames."""
    from src.engine.shot_critical_v2223 import select_recent_pre_frame_v2223
    eligible = [f for f in ring if pre_provenance(f, peak_ts, audio_acquisition)['pre_cutoff_verified']]
    return select_recent_pre_frame_v2223(eligible, peak_ts,
        target_offset_s=PRE_GUARD_S, latest_safe_offset_s=PRE_GUARD_S)


def checked(path):
    if 's03' in str(path).lower():
        raise ValueError('S03 forbidden')
    p = Path(path).resolve()
    if 's03' in str(p).lower() or not p.is_relative_to(Path('/data/skjutbana')):
        raise ValueError('new development artifacts under /data/skjutbana only')
    return p


def atomic(path, value):
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def prepare(capture, shots=3):
    """Fresh process-only research config, called by physical_collection."""
    capture = checked(capture)
    if type(shots) is not int or not 1 <= shots <= 20:
        raise ValueError('GT review event limit must be 1..20')
    root = capture/'gt_review'; root.mkdir(exist_ok=False)
    config = dict(schema=SCHEMA, capture_path=str(capture), event_limit=shots,
        delays=DELAYS, backgrounds=list(STATES), live_authority=False,
        capture_play_sha256=hashlib.sha256((capture/'play.json').read_bytes()).hexdigest(),
        timing_contract='receive/read timing, NOT certified exposure or projector photometry',
        pre_selection_rule=PRE_RULE, pre_guard_ms=PRE_GUARD_S*1000,
        timing_schema=TIMING_SCHEMA, pre_binding='FIRST_DISPATCHED_EVENT_OBSERVATION_REFERENCE_ONLY',
        role_contract='PRE receipt/publication <= audio estimate minus500ms; NOT certified before physical damage; LIVE_POST before live closure; GT_SETTLED after closure, human inspection ONLY')
    config['source_sha256'] = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
        ('automation/physical_gt_capture.py','src/engine/physical_trace.py','src/engine/shot_async_v2224.py',
         'src/engine/camera/camera_manager.py')}
    atomic(root/'config.json', config)
    atomic(root/'manifest.json', dict(**config, status='PREPARED', shots=[]))
    (root/'REPORT.md').write_text('''# Independent physical GT capture v1
LIVE PHYSICAL EVIDENCE != GT REVIEW CAPTURE. No physical labels exist yet.
Wait for the ordinary result and GT phase; do not shoot during neutral backgrounds
or rearming. Same game instance is paused, then restored. No detector thresholds,
scoring rules or live selection change. Native BGR NPY pixels, not screenshots.

Three-shot observability plan (operator performs, not an automated test):
1. Clean visible surface. 2. Near old damage/tape. 3. Dark/projected detail.
Each audio event consumes one slot, including false triggers. Record independently
whether a projectile was fired; aim/game reaction is not truth. Inspect original
PRE and WHITE/BLACK/MID_GRAY with physical_trace_label --gt-review. UNKNOWN is valid.
Acceptance is operator-identifiable new holes in 3/3 events, NOT detector accuracy.

manifest.json lists missing/interrupted frames. Each NPY has a JSON provenance
sidecar with file/pixel hashes, camera orientation/acquisition, frame time, role,
projector command and presentation time. Device exposure/projector settling are
not hardware-certified. gt_capture_timing.csv records post-decision overhead.
PRE requires receipt AND publication at least500ms before the recorded audio
estimate and complete clock provenance. Missing/too-recent history is UNAVAILABLE.
pre_cutoff_verified is a clock guard, NOT causal_pre_certified: unbounded audio
upstream latency and unknown physical impact time prevent physical certification.
Neutral frames never enter ordinary physical trace.frames or same-shot inputs.
''')
    return root


def closed_event(event, scanner, registry, worker, audio):
    """No array copies/I/O. Both authorities and worker must be terminal."""
    snap = registry.snapshot_for_shot(event.shot_id)
    query = getattr(snap, 'object_query', None)
    return (event.state != 'pending' and query is not None and query.final and query.published
        and not any(e.state == 'pending' for e in scanner.audio_events)
        and not audio.pending_events_snapshot() and not worker.has_pending())


class ReviewCapture:
    """Research-only display/pause adapter. Default runtime never constructs it."""
    def __init__(self, root, *, writer=None):
        self.root = checked(root)
        self.config = json.loads((self.root/'config.json').read_text())
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if (self.config.get('schema') != SCHEMA or self.config.get('live_authority') is not False
                or self.manifest.get('status') != 'PREPARED' or self.manifest.get('shots')):
            raise ValueError('fresh prepared non-authoritative GT capture required')
        capture = checked(self.config['capture_path'])
        if self.root != capture/'gt_review' or hashlib.sha256((capture/'play.json').read_bytes()).hexdigest() != self.config['capture_play_sha256']:
            raise ValueError('GT capture binding mismatch')
        from src.engine.physical_trace import PhysicalTraceRecorder
        self.writer = writer or PhysicalTraceRecorder(self.root, enabled=True,
            writer_workers=1, queue_maxsize=12, max_queued_bytes=256*1024*1024)
        self.active = False
        self.handled = set()
        self.observed = {}
        self.frozen_pre = {}
        self.current = None
        self.phase = None
        self.closed = False

    def freeze_pre(self, scanner, camera):
        """Opt-in event-start hook: retain a native reference, no pixels copied/I/O.

        CameraManager publishes newly allocated read/transform arrays. A readonly
        view pins that storage after ring eviction without changing its flags for
        other consumers. The existing writer copies pixels AFTER live closure.
        A missing reference is frozen as missing; never retry against later pixels.
        """
        if self.active or self.closed:
            return
        new = sorted((e for e in scanner.audio_events if e.shot_id not in self.observed),key=lambda e:e.shot_id)
        if not new or len(self.observed)>=self.config['event_limit']:
            return
        from src.engine.camera.camera_manager import CameraFrame
        ring = camera.get_ring_snapshot()
        for event in new[:self.config['event_limit']-len(self.observed)]:
            started = time.monotonic()
            audio = deepcopy((getattr(event,'audio_trigger',{}) or {}).get('diagnostics',{}).get('acquisition') or {})
            frame = select_review_pre(ring,event.peak_ts,audio)
            if frame is not None:
                pixels=frame.frame_bgr.view();pixels.setflags(write=False)
                frame=CameraFrame(pixels,frame.timestamp,deepcopy(frame.acquisition))
            proof=pre_provenance(frame,event.peak_ts,audio)
            proof.update(status='AVAILABLE' if frame is not None else 'UNAVAILABLE',
                binding_stage='FIRST_DISPATCHED_EVENT_OBSERVATION_REFERENCE_ONLY',
                bound_monotonic=started,bound_wall_ts=time.time(),
                freeze_ms=(time.monotonic()-started)*1000,
                timing=frame_timing(frame,event.peak_ts,audio,'PRE'))
            if frame is None:proof['reason']='NO_QUALIFIED_OLD_RING_FRAME_AT_EVENT_START'
            self.frozen_pre[event.shot_id]=frame
            self.observed[event.shot_id]=dict(shot_id=event.shot_id,event_id=event.shot_id,
                peak_ts=event.peak_ts,audio_acquisition=audio,pre_selection=proof,live_state=event.state)

    def observe(self, app, scanner, camera, registry, worker, audio, controller):
        if self.active or len(self.handled) >= self.config['event_limit']:
            return
        events = sorted(scanner.audio_events, key=lambda e: e.shot_id)
        for event in events:
            if event.shot_id in self.handled:
                continue
            if not closed_event(event, scanner, registry, worker, audio):
                return
            self.handled.add(event.shot_id)
            now = time.time(); mono = time.monotonic()
            row = dict(shot_id=event.shot_id, event_id=event.shot_id, peak_ts=event.peak_ts,
                audio_acquisition=dict((getattr(event,'audio_trigger',{}) or {}).get('diagnostics',{}).get('acquisition') or {}),
                shot_created_at=getattr(event,'created_at',None),
                live_closed_ts=now, live_closed_monotonic=mono, live_state=event.state,
                physical_truth_status='UNKNOWN', status='CAPTURING', frames=[], missing=[],
                gt_phase_start_monotonic=mono, candidate_overlay=False)
            bound=self.observed.get(event.shot_id)
            pre=self.frozen_pre.pop(event.shot_id,None)
            if bound:
                row['audio_acquisition']=deepcopy(bound['audio_acquisition'])
                row['pre_selection']=deepcopy(bound['pre_selection'])
            else:
                row['pre_selection']=pre_provenance(None,event.peak_ts,row['audio_acquisition'])
                row['pre_selection'].update(status='UNAVAILABLE',reason='NO_EVENT_START_BINDING',
                    timing=frame_timing(None,event.peak_ts,row['audio_acquisition'],'PRE'))
            self.manifest['shots'].append(row); self.current = row
            if any(e.shot_id > event.shot_id for e in events):
                row.update(status='UNAVAILABLE', reason='ANOTHER_EVENT_BEFORE_REVIEW')
                row['missing'] = ['PRE','LIVE_POST',*STATES]
                self.checkpoint(); self.current = None
                print(f'[GT-CAPTURE WARNING] shot={event.shot_id} later event before review; no fabricated binding', flush=True)
                return
            # PRE was pinned at event start; normal POST is still taken here.
            ring = camera.get_ring_snapshot()
            posts = [f for f in ring if event.peak_ts <= f.timestamp <= now]
            self.active = True
            self.app, self.scanner, self.camera = app, scanner, camera
            self.audio, self.controller = audio, controller
            self.scene = app.scene
            self.scanner_was_enabled = scanner.enabled
            scanner.disable(); audio.stop()
            if pre is None:
                print(f'[GT-CAPTURE WARNING] shot={event.shot_id} PRE UNAVAILABLE: no clock-qualified ring frame at least500ms before audio',flush=True)
            for role, frames in [('PRE', [] if pre is None else [pre]), ('LIVE_POST', posts[:2])]:
                if not frames: row['missing'].append(role)
                for i, frame in enumerate(frames): self.save_frame(frame, role, f'{role.lower()}_{i}')
            self.phase = 'SETTLE'; self.since = time.monotonic()
            self.checkpoint()
            print(f'[GT-CAPTURE] shot={event.shot_id} live closed; research pause. DO NOT SHOOT until rearmed.', flush=True)
            return

    def save_frame(self, frame, role, name, background=None):
        sid = self.current['shot_id']
        rel = f'shots/shot_{sid:08d}/{name}.npy'
        acq = dict(getattr(frame, 'acquisition', {}) or {})
        meta = dict(schema=FRAME_SCHEMA, shot_id=sid, event_id=sid,
            capture_path=self.config['capture_path'], role=role, live_authority=False,
            source_filename=rel, frame_timestamp=frame.timestamp, acquisition=acq,
            orientation={k:acq.get(k) for k in ('rotation','mirror_horizontal','mirror_vertical','raw_shape','frame_shape')},
            projector_state=background or 'GAME_UNMODIFIED',
            projector_rgb=list(STATES[background]) if background else None,
            settle_delay_s=self.config['delays']['background_s'] if background else None,
            presentation_monotonic=self.since if background else None,
            live_closed_monotonic=self.current['live_closed_monotonic'],
            exposure_age_certified=False, candidate_overlay=False)
        meta['timing']=frame_timing(frame,self.current['peak_ts'],self.current['audio_acquisition'],role)
        if role == 'PRE':
            proof = pre_provenance(frame, self.current['peak_ts'], self.current['audio_acquisition'])
            if not proof['pre_cutoff_verified']:
                raise ValueError('GT PRE refused: '+proof['reason'])
            meta.update(proof)
        accepted = self.writer.enqueue_review_frame(self.root/rel, frame.frame_bgr, meta)
        self.current['frames'].append(dict(path=rel, role=role, projector_state=meta['projector_state'],
            status='QUEUED' if accepted else 'DROPPED'))
        if not accepted:
            self.current['missing'].append(background or name)
            print(f'[GT-CAPTURE WARNING] shot={sid} writer refused {name}; frame explicitly missing', flush=True)

    def present(self, phase, pygame):
        self.phase = phase
        if phase in STATES:
            self.app.screen.fill(STATES[phase])
        else:
            self.scene.render(self.app.screen)
        pygame.display.flip()
        self.since = time.monotonic()  # CPU return from flip, not measured photon onset.
        if phase in STATES:
            print(f'[GT-CAPTURE] shot={self.current["shot_id"]} {phase} inspection capture; DO NOT SHOOT', flush=True)

    def tick(self, pygame):
        """No scanner.update, audio dispatch, game simulation or input during pause."""
        self.app.clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.app.quit(); return
        self.camera.update()
        now = time.monotonic(); delay = self.config['delays']
        if self.phase == 'SETTLE':
            if now-self.since >= delay['settle_s']: self.present('WHITE', pygame)
        elif self.phase in STATES:
            boundary = self.since+delay['background_s']
            # Require acquisition after display settling, never reuse latest_frame.
            fresh = [f for f in self.camera.get_ring_snapshot()
                if isinstance((getattr(f,'acquisition',{}) or {}).get('read_start_monotonic'), (int,float))
                and f.acquisition['read_start_monotonic'] >= boundary]
            if now >= boundary and fresh:
                self.save_frame(fresh[0], 'GT_SETTLED', self.phase.lower(), self.phase)
                self.advance(pygame)
            elif now >= boundary+delay['frame_timeout_s']:
                self.current['missing'].append(self.phase)
                print(f'[GT-CAPTURE WARNING] shot={self.current["shot_id"]} {self.phase} no qualified new camera frame', flush=True)
                self.advance(pygame)
        elif self.phase == 'RESTORE' and now-self.since >= delay['restore_s']:
            # Wait for the single camera reader to have crossed the restore
            # barrier before clearing. An in-flight neutral read cannot then
            # publish behind the purge and contaminate the next armed window.
            recovered = any((getattr(f,'acquisition',{}) or {}).get('read_start_monotonic',0)
                            >= self.since+delay['restore_s'] for f in self.camera.get_ring_snapshot())
            if not recovered:
                if now-self.since >= delay['restore_s']+delay['frame_timeout_s']:
                    self.current['missing'].append('RESTORED_CAMERA_FRAME')
                    print('[GT-CAPTURE ERROR] camera did not recover after normal scene restore; capture stopped',flush=True)
                    self.app.quit()
                return
            # Camera reader stays alive; remove EVERY review frame before rearming.
            # Existing ring/pickup ownership under the camera lock, research only.
            with self.camera._lock:
                self.camera._ring.clear()
                self.camera._last_pickup_count = self.camera._read_count
                self.camera.latest_frame = None
            if not self.writer.flush(timeout=5.):  # never carry full-frame writes into next live shot
                self.current['missing'].append('WRITER_DRAIN')
                print('[GT-CAPTURE ERROR] writer did not drain; capture stopped before rearming',flush=True)
                self.app.quit(); return
            self.current['gt_phase_wall_s'] = time.monotonic()-self.current['gt_phase_start_monotonic']
            self.current['status'] = 'COMPLETE' if not self.current['missing'] else 'PARTIAL'
            self.checkpoint()
            if self.scanner_was_enabled: self.scanner.enable()
            if self.audio.start() is False:
                self.current['missing'].append('AUDIO_RESTART')
                print('[GT-CAPTURE ERROR] audio failed to restart; stop shooting and inspect runtime.log',flush=True)
                self.app.quit(); return
            self.controller.last_seen_peak_ts = float(getattr(self.audio,'last_peak_ts',0.) or 0.)
            print(f'[GT-CAPTURE] shot={self.current["shot_id"]} review saved; normal scene restored. Wait for scanner READY.', flush=True)
            self.active = False; self.current = None

    def advance(self, pygame):
        states = list(STATES); i = states.index(self.phase)
        self.checkpoint()
        self.present(states[i+1] if i+1 < len(states) else 'RESTORE', pygame)

    def checkpoint(self):
        for row in self.manifest['shots']:
            for entry in row['frames']:
                sidecar = (self.root/entry['path']).with_suffix('.json')
                if sidecar.exists():
                    entry['status'] = 'SAVED'
                    entry['metadata'] = str(sidecar.relative_to(self.root))
                    entry['metadata_sha256'] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
                    meta=json.loads(sidecar.read_text())
                    entry['timing']=meta.get('timing')
                    entry['image_sha256']=meta.get('image_sha256')
                elif entry['status'] != 'DROPPED' and not self.writer.stats()['flush_pending']:
                    entry['status'] = 'MISSING'
                if entry['status'] in ('MISSING','DROPPED'):
                    missing=entry['projector_state'] if entry['role']=='GT_SETTLED' else entry['role']
                    if missing not in row['missing']:row['missing'].append(missing)
            if row['status'] == 'COMPLETE' and any(f['status'] != 'SAVED' for f in row['frames']):
                row['status'] = 'PARTIAL'
        self.manifest['writer'] = self.writer.stats()
        self.manifest['status'] = 'CLOSED' if self.closed else 'CAPTURING'
        atomic(self.root/'manifest.json', self.manifest)
        path = self.root/'gt_capture_timing.csv'; tmp = path.with_suffix('.csv.tmp')
        with tmp.open('w') as f:
            writer = csv.DictWriter(f,fieldnames=['shot_id','status','live_closed_ts','gt_phase_wall_s','role','projector_state','frame_status','copy_ms','writer_ms'])
            writer.writeheader()
            for row in self.manifest['shots']:
                for entry in row['frames'] or [dict(role='UNAVAILABLE',projector_state=None,status='MISSING')]:
                    meta = json.loads((self.root/entry['metadata']).read_text()) if entry.get('metadata') else {}
                    writer.writerow({**{k:row.get(k) for k in ('shot_id','status','live_closed_ts','gt_phase_wall_s')},
                        'role':entry['role'],'projector_state':entry['projector_state'],'frame_status':entry['status'],
                        'copy_ms':meta.get('copy_ms'),'writer_ms':meta.get('writer_ms')})
        os.replace(tmp,path)
        stats = self.writer.stats()
        if stats['writer_errors'] or stats['queue_drops'] or (self.closed and stats['flush_pending']):
            print(f'[GT-CAPTURE WARNING] write integrity: errors={stats["writer_errors"]} drops={stats["queue_drops"]} pending={stats["flush_pending"]}',flush=True)

    def close(self):
        if self.closed: return
        for sid, event in self.observed.items():
            if sid not in self.handled:
                self.manifest['shots'].append(dict(**event,status='UNAVAILABLE',
                    reason='LIVE_NOT_CLOSED_BEFORE_EXIT',physical_truth_status='UNKNOWN',frames=[],
                    missing=['PRE','LIVE_POST',*STATES]))
        if self.current and self.active:
            self.current['status'] = 'INTERRUPTED'
            self.current['gt_phase_wall_s'] = time.monotonic()-self.current['gt_phase_start_monotonic']
            saved = {e['projector_state'] for e in self.current['frames']}
            self.current['missing'] = sorted(set(self.current['missing']) | (set(STATES)-saved))
        self.writer.shutdown(timeout=5.)
        self.frozen_pre.clear()
        self.closed = True
        self.checkpoint()


def from_environment():
    root = os.environ.get('SKJUTBANA_GT_CAPTURE_ROOT')
    return ReviewCapture(root) if root else None
