"""Opt-in engine presentation history; no hit decisions, camera reader or game code.

Screen pixels are copied while the engine still owns the presented surface.
Only bounded references are bound at audio dispatch. Hashing, deduplication and
artifact I/O use a background worker and the existing PhysicalTraceRecorder.
CPU presentation times do not certify projector photons or camera exposure.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np

from automation.physical_gt_capture import atomic, checked

SCHEMA='physical_render_history_v1'
FRAME_SCHEMA='physical_render_content_v1'
MAX_PIXELS=1920*1080
MAX_BYTES=64*1024*1024
MAX_FRAMES=120
MAX_EVENTS=20
POST_CONTEXT_S=.35


def copy_render_pixels(screen):
    """One immutable BGR copy; native Pygame/CV conversion avoids strided Python work."""
    import pygame
    import cv2
    w,h=screen.get_size()
    rgba=np.frombuffer(pygame.image.tobytes(screen,'RGBA'),dtype=np.uint8).reshape(h,w,4)
    pixels=cv2.cvtColor(rgba,cv2.COLOR_RGBA2BGR)
    pixels.setflags(write=False)
    return pixels


def renderer_status(scene):
    """Do not mistake an external game's launcher window for projected content."""
    chain=[];seen=set();instances=[]
    while scene is not None and id(scene) not in seen:
        seen.add(id(scene));instances.append(id(scene));chain.append(type(scene).__module__+'.'+type(scene).__name__)
        if getattr(scene,'shot_input_mode',None)=='SHOT_ONLY' or getattr(scene,'process',None) is not None:
            return dict(status='UNAVAILABLE',reason='SHOT_ONLY_OR_EXTERNAL_RENDERER',owners=chain,instances=instances)
        scene=getattr(scene,'inner',None) or getattr(scene,'game',None)
    return dict(status='AVAILABLE',reason='ENGINE_SURFACE_ONLY_NOT_PHOTON_CERTIFIED',owners=chain,instances=instances)


def prepare(capture):
    capture=checked(capture);root=capture/'render_history';root.mkdir(exist_ok=False)
    config=dict(schema=SCHEMA,capture_path=str(capture),max_pixels=MAX_PIXELS,
        max_bytes=MAX_BYTES,max_frames=MAX_FRAMES,max_events=MAX_EVENTS,post_context_s=POST_CONTEXT_S,
        capture_play_sha256=hashlib.sha256((capture/'play.json').read_bytes()).hexdigest(),
        live_authority=False,projectile_truth=False,
        timing='CPU_FLIP_CALL_RETURN_NOT_PHOTON_EXPOSURE; audio is receipt-anchored estimate')
    config['source_sha256']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
        (__file__,'src/engine/physical_trace.py','src/engine/shot_async_v2224.py','automation/physical_collection.py')}
    atomic(root/'config.json',config)
    atomic(root/'manifest.json',dict(**config,status='PREPARED',shots=[]))
    return root


def frame_binding(meta,audio):
    mono=audio.get('trigger_receipt_estimate_monotonic')
    good=type(mono) in (float,int) and math.isfinite(mono)
    delta=(meta['flip_end_monotonic']-mono)*1000 if good else None
    return dict(**meta,delta_ms_to_audio_estimate=delta,
        relative_to_audio_estimate=('BEFORE' if delta<=0 else 'AFTER') if good else 'UNAVAILABLE',
        exposure_at_impact_certified=False)


class RenderHistoryCapture:
    """Small bounded research recorder around existing render/freeze ownership."""
    def __init__(self,root,*,writer=None):
        self.root=checked(root);self.config=json.loads((self.root/'config.json').read_text())
        c=self.config
        if c.get('schema')!=SCHEMA or c.get('live_authority') is not False:raise ValueError('render config required')
        capture=checked(c['capture_path'])
        if self.root!=capture/'render_history' or hashlib.sha256((capture/'play.json').read_bytes()).hexdigest()!=c['capture_play_sha256']:
            raise ValueError('render capture binding mismatch')
        if json.loads((self.root/'manifest.json').read_text()).get('status')!='PREPARED':raise ValueError('fresh render capture required')
        from src.engine.physical_trace import PhysicalTraceRecorder
        self.writer=writer or PhysicalTraceRecorder(self.root,enabled=True,writer_workers=1,
            queue_maxsize=4,max_queued_bytes=MAX_PIXELS*3*2)
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='render-history')
        self.ring=deque();self.bytes=0;self.seq=0;self.pending=[];self.futures=[];self.seen=set()
        self.manifest=dict(**c,status='RUNNING',shots=[]);self.closed=False;self.failed=False
        self.contents={};self.owner=threading.get_ident();self.scene_key=None;self.warnings=set()
        self.future_rows={}
        atomic(self.root/'manifest.json',self.manifest)

    def _main(self):
        if threading.get_ident()!=self.owner:raise RuntimeError('Pygame/history requires main thread')

    def _warn(self,reason):
        if reason not in self.warnings:
            print('[RENDER-CAPTURE WARNING] '+reason,flush=True);self.warnings.add(reason)

    def call(self,method,*args):
        """Optional diagnostic failures cannot alter detector/game dispatch."""
        try:getattr(self,method)(*args)
        except Exception as exc:
            self.failed=True;self._warn('DIAGNOSTIC_DISABLED '+repr(exc))

    def presented(self,screen,scene,flip_start,flip_end,wall):
        self._main()
        if self.closed or self.failed:return
        info=renderer_status(scene);self.seq+=1
        key=(id(scene),tuple(info['instances']))
        if key!=self.scene_key:
            # Never bridge unrelated presentation owners or lose the boundary silently.
            self.ring.clear();self.bytes=0;self.scene_key=key
            for job in self.pending:job['row']['missing'].append('PRESENTATION_OWNER_CHANGED')
        if info['status']!='AVAILABLE':
            for job in self.pending:job['row']['missing'].append(info['reason'])
            self.ring.clear();self.bytes=0;self._warn(info['reason']);return
        w,h=screen.get_size()
        if w*h>MAX_PIXELS:
            for job in self.pending:job['row']['missing'].append('RENDER_SIZE_EXCEEDS_BOUNDED_CAPTURE')
            self.ring.clear();self.bytes=0;self._warn('RENDER_SIZE_EXCEEDS_BOUNDED_CAPTURE');return
        started=time.perf_counter()
        # Owned copy BEFORE a later render mutates the screen; hashes/writes deferred.
        pixels=copy_render_pixels(screen)
        meta=dict(render_id=self.seq,flip_start_monotonic=flip_start,flip_end_monotonic=flip_end,
            flip_return_wall_ts=wall,copy_ms=(time.perf_counter()-started)*1000,
            render_size=[w,h],renderer=info,scene_generation=id(scene),
            coordinate_space='PYGAME_CLIENT_PIXELS',photon_timing_verified=False)
        entry=(meta,pixels);self.ring.append(entry);self.bytes+=pixels.nbytes
        while len(self.ring)>MAX_FRAMES or self.bytes>MAX_BYTES:
            self.bytes-=self.ring.popleft()[1].nbytes
        for job in self.pending:
            if flip_end<=job['end_monotonic'] and job['scene_key']==key:
                if sum(p.nbytes for _,p in job['frames'])+pixels.nbytes<=MAX_BYTES and len(job['frames'])<MAX_FRAMES:
                    job['frames'].append(entry)
                else:job['row']['missing'].append('EVENT_IMAGE_MEMORY_BOUND')
        self.poll(flip_end)

    def observe(self,scanner,registry,mapper,scene):
        self._main()
        if self.closed or self.failed:return
        self.poll(time.monotonic())
        for event in list(scanner.audio_events):
            sid=getattr(event,'shot_id',None)
            if type(sid) is not int or sid<=0 or sid in self.seen:continue
            if len(self.seen)>=MAX_EVENTS:
                self._warn('EVENT_LIMIT_REACHED');break
            self.seen.add(sid)
            now=time.monotonic();status=renderer_status(scene)
            audio=deepcopy((getattr(event,'audio_trigger',{}) or {}).get('diagnostics',{}).get('acquisition') or {})
            context=deepcopy(registry.game_context(sid))
            h=getattr(mapper,'homography',None)
            calibration=dict(camera_to_screen=h.tolist() if isinstance(h,np.ndarray) else None,
                calibration=deepcopy(getattr(mapper,'calibration_data',{})),
                active_method=(context or {}).get('transform_method'),
                matrix_semantics='SAVED_HOMOGRAPHY_NOT_ASSUMED_ACTIVE_FOR_SCANPORT',
                binding='AT_AUDIO_DISPATCH_SAME_MAIN_THREAD_NOT_EXPOSURE',physical_transfer_available=False)
            row=dict(schema=SCHEMA,shot_id=sid,peak_ts=event.peak_ts,audio_acquisition=audio,
                bound_monotonic=now,renderer=status,shot_game_context=context,calibration=calibration,
                post_context_end_monotonic=now+POST_CONTEXT_S,
                window_semantics='BOUNDED_CPU_HISTORY_PLUS_POST_DISPATCH_CONTEXT_NOT_LIVE_EVIDENCE',
                camera_trace_ref=f'object_crops/shots/shot_{sid:08d}/trace.json',
                live_authority=False,projectile_truth=False,selected_render_at_physical_impact=None,
                physical_projection_identity_certified=False,missing=[],frames=[],status='PENDING')
            if status['status']!='AVAILABLE':row['missing'].append(status['reason'])
            if context is None:row['missing'].append('FROZEN_SHOT_GEOMETRY_UNAVAILABLE')
            frames=list(self.ring) if status['status']=='AVAILABLE' and self.scene_key and self.scene_key[0]==id(scene) else []
            if not frames:row['missing'].append('PRESENTED_HISTORY_UNAVAILABLE')
            if len(self.pending)+len(self.futures)>=2:
                row.update(status='UNAVAILABLE');row['missing'].append('BACKGROUND_BACKLOG_BOUND')
                # Metadata-only jobs remain bounded by MAX_EVENTS. Preserve the
                # rejected event off-thread too, even if finalization is interrupted.
                self._submit(dict(row=row,frames=[]))
                self._warn('BACKGROUND_BACKLOG_BOUND');continue
            self.pending.append(dict(row=row,frames=frames,scene_key=self.scene_key,end_monotonic=now+POST_CONTEXT_S))

    def poll(self,now):
        self._main()
        for f in list(self.futures):
            if not f.done():continue
            self.futures.remove(f);pending_row=self.future_rows.pop(f)
            try:row=f.result()
            except Exception as exc:
                self.failed=True;self._warn('ARTIFACT_WRITE_FAILED '+repr(exc))
                row=pending_row;row.update(status='UNAVAILABLE',error=repr(exc));row['missing'].append('ARTIFACT_WRITE_FAILED')
            if row.get('error'):
                self.failed=True;self._warn('ARTIFACT_WRITE_FAILED '+row['error'])
            self.manifest['shots'].append(row)
        for job in list(self.pending):
            if now>=job['end_monotonic']:
                self.pending.remove(job);self._submit(job)

    def _submit(self,job):
        f=self.pool.submit(self._persist_guarded,job);self.futures.append(f);self.future_rows[f]=job['row']

    def _persist_guarded(self,job):
        try:return self._persist(job)
        except Exception as exc:
            row=job['row'];row.update(status='UNAVAILABLE',error=repr(exc))
            row['missing']=sorted(set(row['missing']+['ARTIFACT_WRITE_FAILED']))
            # If even this fails, poll/close still preserve the shot in the manifest.
            atomic(self.root/f'shot_{row["shot_id"]:08d}.json',row)
            return row

    def _persist(self,job):
        row=job['row'];frames=job['frames'];start=time.perf_counter()
        for meta,pixels in frames:
            sha=hashlib.sha256(pixels.tobytes()).hexdigest()
            if sha not in self.contents:
                path=self.root/'frames'/f'{sha}.npy'
                md=dict(schema=FRAME_SCHEMA,shot_id=None,role='ENGINE_RENDER_CONTENT',
                    live_authority=False,projectile_truth=False,coordinate_space='PYGAME_CLIENT_PIXELS')
                if not self.writer.enqueue_render_frame(path,pixels,md) or not self.writer.flush(timeout=30):
                    raise OSError('bounded render image writer failed')
                info=json.loads(path.with_suffix('.json').read_text())
                if info['pixel_sha256']!=sha:raise ValueError('render pixels changed after binding')
                self.contents[sha]=dict(path=str(path.relative_to(self.root)),image_sha256=info['image_sha256'],
                    metadata_sha256=hashlib.sha256(path.with_suffix('.json').read_bytes()).hexdigest(),
                    pixel_sha256=sha,writer_ms=info['writer_ms'])
            row['frames'].append(dict(frame_binding(meta,row['audio_acquisition']),
                **self.contents[sha]))
        row['presentations_before_audio_estimate']=sum(f['relative_to_audio_estimate']=='BEFORE' for f in row['frames'])
        row['presentations_after_audio_estimate']=sum(f['relative_to_audio_estimate']=='AFTER' for f in row['frames'])
        row['audio_alignment_available']=bool(row['frames']) and all(f['relative_to_audio_estimate']!='UNAVAILABLE' for f in row['frames'])
        row['post_window_elapsed']=not row.pop('interrupted',False)
        row['post_presentations_retained']=sum(f['flip_end_monotonic']>row['bound_monotonic'] for f in row['frames'])
        if not row['post_window_elapsed']:row['missing'].append('INTERRUPTED_POST_PRESENTATION_WINDOW')
        row['missing']=sorted(set(row['missing']))
        row['status']='AVAILABLE_CPU_HISTORY' if frames and not row['missing'] else 'INCOMPLETE' if frames else 'UNAVAILABLE'
        row['background_ms']=(time.perf_counter()-start)*1000
        atomic(self.root/f'shot_{row["shot_id"]:08d}.json',row)
        return row

    def close(self):
        self._main()
        if self.closed:return
        for job in self.pending:
            job['row']['interrupted']=True
            self._submit(job)
        self.pending.clear();self.pool.shutdown(wait=True);self.poll(float('inf'))
        metrics=self.writer.shutdown(timeout=30)
        self.closed=True
        self.manifest.update(status='INCOMPLETE' if self.failed or any(r['status']!='AVAILABLE_CPU_HISTORY' for r in self.manifest['shots']) else
            'CLOSED' if self.manifest['shots'] else 'NO_CAPTURED_EVENTS',
            writer=metrics,unique_images=len(self.contents),warnings=sorted(self.warnings))
        atomic(self.root/'manifest.json',self.manifest)
        self.ring.clear();self.bytes=0


def from_environment():
    root=os.environ.get('SKJUTBANA_RENDER_HISTORY_ROOT')
    return RenderHistoryCapture(root) if root else None


def audit(capture):
    """Read-only audit: saved pixels/sidecars only, never today's display/calibration."""
    capture=checked(capture);root=checked(capture/'render_history')
    if root!=capture/'render_history' or (root/'manifest.json').is_symlink():
        raise ValueError('render root/manifest symlink refused')
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('schema')!=SCHEMA or Path(manifest['capture_path'])!=capture:
        raise ValueError('render manifest binding mismatch')
    if hashlib.sha256((capture/'play.json').read_bytes()).hexdigest()!=manifest['capture_play_sha256']:
        raise ValueError('capture play identity changed')
    rows=[];validated={};by_id={r['shot_id']:r for r in manifest.get('shots',[])}
    # IDs are bounded by the recorder; direct filenames also recover completed
    # sidecars after interruption before final manifest publication.
    import re
    for path in sorted(root.iterdir()):
        if not re.fullmatch(r'shot_\d{8}\.json',path.name):continue
        sid=int(path.stem[5:])
        if path.is_symlink():raise ValueError('shot symlink refused')
        row=json.loads(path.read_text())
        if row['shot_id']!=sid or row['schema']!=SCHEMA:raise ValueError('shot identity mismatch')
        by_id[sid]=row
    for sid,row in sorted(by_id.items()):
        if row.get('live_authority') is not False or row.get('projectile_truth') is not False:
            raise ValueError('research-only boundary missing')
        for f in row['frames']:
            raw=root/f['path'];image=checked(raw)
            if not image.is_relative_to(root) or raw.is_symlink():raise ValueError('image escapes render root')
            signature=(f['image_sha256'],f['pixel_sha256'],f['metadata_sha256'],tuple(f['render_size']))
            if image in validated:
                if validated[image]!=signature:raise ValueError('inconsistent reused content identity')
                continue
            sidecar=image.with_suffix('.json')
            if sidecar.is_symlink():raise ValueError('metadata symlink refused')
            a=np.load(image,allow_pickle=False)
            md=json.loads(sidecar.read_text())
            if (hashlib.sha256(image.read_bytes()).hexdigest()!=f['image_sha256'] or
                hashlib.sha256(a.tobytes()).hexdigest()!=f['pixel_sha256'] or
                hashlib.sha256(sidecar.read_bytes()).hexdigest()!=f['metadata_sha256'] or
                md.get('schema')!=FRAME_SCHEMA or md.get('coordinate_space')!='PYGAME_CLIENT_PIXELS' or
                md.get('role')!='ENGINE_RENDER_CONTENT' or md.get('live_authority') is not False or
                md.get('projectile_truth') is not False or md.get('shot_id') is not None or
                a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3 or
                list(a.shape)!=md.get('shape') or list(a.shape[:2][::-1])!=f['render_size']):
                raise ValueError('render source hash/schema mismatch')
            validated[image]=signature
        rows.append(dict(shot_id=sid,status=row['status'],presentations=len(row['frames']),
            unique_images=len({f['pixel_sha256'] for f in row['frames']}),
            presentations_before_audio_estimate=row.get('presentations_before_audio_estimate'),
            audio_alignment_available=row.get('audio_alignment_available',False),
            missing=row['missing'],physical_projection_identity_certified=False))
    return dict(schema=SCHEMA,capture=str(capture),manifest_status=manifest['status'],shots=rows,
        live_authority=False,projectile_truth=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',required=True,type=Path)
    a=p.parse_args();print(json.dumps(audit(a.capture),indent=2))


if __name__=='__main__':main()
