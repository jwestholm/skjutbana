"""Opt-in Pygame adapter for physical_optical_lab, using existing command/event bus.

Runs only in a dedicated SKJUTBANA_OPTICAL_LAB=1 process. No audio dispatch,
scanner, game update, HitEvent, calibration save, or GT capture runs in this loop.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
import math
import threading
import time

from automation.physical_optical_lab import (OpticalRun, configuration, plans, render_probe,
    snapshot_mapping, qualified_frame)
from src.engine.events.event_bus import event_bus


class OpticalController:
    def __init__(self, app, camera, *, mapping=None):
        self.app=app;self.camera=camera;self.mapping_override=mapping;self.owner=threading.get_ident()
        # This process never runs a shot/GT consumer. Keep the SAME camera owner
        # and reader, but do not retain its normal 300 native 4K frames (~7.5GB).
        # Default camera/gameplay retention is untouched in ordinary processes.
        if hasattr(camera, '_ring'):
            with camera._lock:
                camera._ring=deque(camera._ring,maxlen=8)
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='optical-measurement')
        self.run=None;self.active=False;self.pending=[];self.final_future=None
        self.need_present=True;self.transition=None;self.render_id=0;self.request=None
        self.stop_status=None;self.reason=None;self.cursor=None;self.diagnostics={}

    def _main(self):
        if threading.get_ident()!=self.owner:raise RuntimeError('optical Pygame work must stay on main thread')

    def command(self, command):
        self._main()
        if command.command=='opticalLabCancel':
            if not isinstance(command.args,dict) or not self.request or command.args.get('run_id')!=self.request['run_id']:
                raise ValueError('cancel run identity mismatch')
            self.stop_status='INTERRUPTED';self.reason='OPERATOR_CANCEL';command.reply_success(dict(status='CANCELLING'));return
        if command.command!='opticalLabStart':raise ValueError('only optical lab commands allowed in research mode')
        if self.active:raise ValueError('optical experiment already active')
        args=command.args
        if not isinstance(args,dict) or not isinstance(args.get('run_id'),str) or not args['run_id'] or len(args['run_id'])>100:
            raise ValueError('explicit bounded run id required')
        issued=args.get('issued_wall_ts')
        if type(issued) not in (int,float) or not math.isfinite(issued):raise ValueError('command issue timestamp required')
        raw=args['config'];config=configuration(raw['mode'],**{k:v for k,v in raw.items() if k not in ('mode','schema')})
        if config['synthetic']:raise ValueError('synthetic mode belongs to offline CLI, not hardware process')
        if self.mapping_override is None:
            from src.engine.input.hit_input import hit_input
            from src.engine.settings import load_content_rect,load_viewport_rect
            mapping=snapshot_mapping(hit_input,load_viewport_rect(),load_content_rect(),self.app.screen.get_size())
        else:mapping=self.mapping_override
        steps=plans(mapping) if config['mode']=='probes' else []
        self.run=OpticalRun(args['output_dir'],config,mapping)
        self.run.manifest['automation_request_id']=args['run_id'];self.run.checkpoint()
        self.request=dict(args);self.active=True;self.pending=[];self.final_future=None;self.stop_status=None;self.reason=None
        self.probes=steps;self.probe_index=0;self.phase=0;self.need_present=True;self.transition=None;self.cursor=None
        self.diagnostics=dict(processor_backpressure_frames=0,invalid_provenance_frames=0,qualified_frames=0,
            requested_native_rate=True,exposure_timing_certified=False,started_monotonic=time.monotonic())
        self.diagnostics['research_camera_ring_capacity']=getattr(getattr(self.camera,'_ring',None),'maxlen',None)
        command.reply_success(dict(status='ACCEPTED',run_id=args['run_id']))
        self.emit('started',status='ACCEPTED')

    def emit(self,event,**data):
        event_bus.emit('opticalLab.'+event,dict(run_id=self.request['run_id'],output_dir=str(self.run.root),**data),source='OpticalLab')

    def present(self, pygame):
        probe=self.probes[self.probe_index] if self.active and self.probes and self.phase==1 and not self.stop_status else None
        applied=time.monotonic();render_probe(self.app.screen,probe);pygame.display.flip();presented=time.monotonic();self.render_id+=1
        if not self.active or self.stop_status:return
        state='MID_GRAY' if probe is None else probe['probe_id']
        self.transition=dict(transition_id=self.render_id,render_frame_id=self.render_id,state=state,
            role=self.role,probe_id=self.probes[self.probe_index]['probe_id'] if self.probes else None,
            command_issued_wall_ts=self.request['issued_wall_ts'],main_applied_monotonic=applied,
            presented_monotonic=presented,presented_wall_ts=time.time(),
            presentation_semantics='CPU_FLIP_RETURN_NOT_PHOTON_ONSET',
            render_id_semantics='OPTICAL_TRANSITION_COUNTER_NOT_PROJECTOR_FRAME_NUMBER',photon_timestamp=None)
        with (self.run.root/'transitions.jsonl').open('a') as f:f.write(json.dumps(self.transition)+'\n')
        self.emit('presented',**self.transition)

    @property
    def role(self):
        return ('OPTICAL_BASELINE','OPTICAL_PROBE','OPTICAL_RECOVERY')[self.phase] if self.probes else 'OPTICAL_PASSIVE'

    def tick(self, pygame, automation_event):
        self._main();self.app.clock.tick(60)
        self.app._post_automation_events()
        for event in pygame.event.get():
            if event.type==pygame.QUIT or (event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE):
                self.app.quit();return
            if event.type==automation_event:self.app._handle_automation_event(event)
        if self.need_present:
            self.present(pygame);self.need_present=False
        if not self.active:return
        self.camera.update()
        if self.final_future is not None:
            if self.final_future.done():
                try:
                    m=self.final_future.result();self.emit('completed',status=m['status'],reason=m['reason'],synthetic=False)
                except Exception as exc:self.emit('failed',reason=str(exc))
                self.active=False;self.need_present=True
            return
        while self.pending and self.pending[0].done():
            f=self.pending.pop(0)
            try:f.result()
            except Exception as exc:self.stop_status='FAILED';self.reason=str(exc)
            if self.probes and not self.stop_status:
                self.phase+=1
                if self.phase==3:self.phase=0;self.probe_index+=1
                if self.probe_index==len(self.probes):self.stop_status='COMPLETED'
                else:self.need_present=True
        if self.stop_status:
            if not self.pending:
                render_probe(self.app.screen);pygame.display.flip()
                self.diagnostics['capture_wall_s']=time.monotonic()-self.diagnostics['started_monotonic']
                self.final_future=self.pool.submit(self.run.finish,self.stop_status,self.reason,self.diagnostics.copy())
            return
        if self.need_present:return
        if self.probes and self.pending:return
        now=time.monotonic();c=self.run.config;start=self.transition['presented_monotonic']+c['settle_s']
        end=start+c['duration_s'] if not self.probes else start+c['frame_timeout_s']
        for frame in self.camera.get_ring_snapshot():
            acq=frame.acquisition or {};identity=(acq.get('stream_id'),acq.get('sequence'))
            if self.cursor is not None and identity[0]==self.cursor[0] and type(identity[1]) is int and identity[1]<=self.cursor[1]:continue
            if not identity[0] or type(identity[1]) is not int:
                self.diagnostics['invalid_provenance_frames']+=1;continue
            self.cursor=identity
            if not qualified_frame(frame,self.transition,c['settle_s']):continue
            if acq['read_start_monotonic']>end:continue
            self.diagnostics['qualified_frames']+=1
            if len(self.pending)>=2:
                self.diagnostics['processor_backpressure_frames']+=1;continue
            plan=self.probes[self.probe_index] if self.probes else None
            self.pending.append(self.pool.submit(self.run.add,frame,self.role,self.transition,plan))
            if self.probes:break
        if now>=end and not (self.probes and self.pending):
            self.stop_status='UNAVAILABLE' if self.probes else 'COMPLETED'
            self.reason='NO_FRAME_AT_PRESENTATION_BARRIER' if self.probes else None

    def close(self):
        self._main()
        try:
            if self.active:
                for f in self.pending:
                    try:f.result()
                    except Exception:pass
                if self.final_future is not None:self.final_future.result()
                else:self.run.finish('INTERRUPTED','APP_EXIT',self.diagnostics.copy())
        finally:
            self.pool.shutdown(wait=True);self.active=False
