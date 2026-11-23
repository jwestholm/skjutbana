"""Local OBJECT measurement companion to precision_flash; all decisions use frozen shapes.

No gameplay damage or guessed depth here. Main thread owns the state machine;
workers get immutable masks/calibration/frame snapshots and return evidence only.
Camera XY remains raw/distorted. A missed local observation is UNKNOWN, not a miss.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass,replace
import hashlib,json,math,threading
import numpy as np
import cv2


@dataclass(frozen=True)
class GlowConfig:
    color: tuple = (230,220,150)
    radius_px: float = 12.
    pastel_mix: float = .30
    ring_width_px: int = 1
    reference_s: float = .40
    measurement_s: float = .18
    min_interval_s: float = 2.0       # minimum dark/rest interval, including after an interrupted glow
    settle_s: float = .35
    initial_wait_s: float = .25
    capture_wait_s: float = .25
    reference_frames: int = 5
    measurement_frames: int = 3
    receive_latency_s: float = .066
    exposure_s: float = .0167
    projector_min_s: float = .02
    projector_max_s: float = .06
    query_timeout_s: float = 8.
    max_mask_fraction: float = .65

    @classmethod
    def from_scene(cls,scene):
        data=getattr(scene,'precision_glow',None) or {};values={}
        for key,value in data.items():
            if key not in cls.__dataclass_fields__:continue
            if key=='color':
                if len(value)!=3:raise ValueError('glow color requires RGB')
                values[key]=tuple(max(0,min(255,int(v))) for v in value);continue
            n=float(value)
            if not math.isfinite(n) or n<0:raise ValueError('invalid glow '+key)
            values[key]=int(n) if key.endswith('_frames') else n
        values['min_interval_s']=max(2.,values.get('min_interval_s',2.))
        values['reference_frames']=max(3,values.get('reference_frames',5))
        values['measurement_frames']=max(2,values.get('measurement_frames',3))
        values['reference_s']=max(.10,min(1.5,values.get('reference_s',.40)))
        values['measurement_s']=max(.016,min(.50,values.get('measurement_s',.18)))
        values['query_timeout_s']=max(1.,min(8.,values.get('query_timeout_s',8.)))
        if values.get('projector_min_s',.02)>values.get('projector_max_s',.06):raise ValueError('projector delay bounds reversed')
        values['max_mask_fraction']=min(.65,max(.05,values.get('max_mask_fraction',.65)))
        values['radius_px']=max(2.,min(80.,values.get('radius_px',12.)))
        values['pastel_mix'] = min(.55, values.get('pastel_mix', .30))
        values['ring_width_px'] = max(1, min(3, int(values.get('ring_width_px', 1))))
        return cls(**values)


def _shape(region):
    return (region.metadata.get('v250_object_snapshot') or {}).get('shape')


def raster_shape(canvas,shape):
    """Rasterize exact explicit geometry, never substitute an object's AABB."""
    kind=shape.get('kind')
    if kind=='union':
        for part in shape.get('parts',()):raster_shape(canvas,part)
    elif kind=='polygon':
        pts=np.asarray(shape['points'],float)
        if len(pts)<3 or not np.isfinite(pts).all():raise ValueError('invalid polygon')
        cv2.fillPoly(canvas,[np.rint(pts).astype(np.int32)],1)
    elif kind in ('rect','circle','ellipse'):
        x,y,w,h=(float(shape[k]) for k in ('x','y','width','height'))
        if not all(math.isfinite(v) for v in (x,y,w,h)) or w<=0 or h<=0:raise ValueError('invalid shape')
        if kind=='rect':cv2.rectangle(canvas,(round(x),round(y)),(round(x+w),round(y+h)),1,-1)
        else:cv2.ellipse(canvas,(round(x+w/2),round(y+h/2)),(max(1,round(w/2)),max(1,round(h/2))),0,0,360,1,-1)
    else:raise ValueError('unsupported exact shape '+str(kind))


def measurement_circles(regions, radius):
    """One frozen measurement circle per entity, including its visible parts."""
    from src.engine.game_objects.geometry import shape_bounds
    groups = {}
    for region in regions:
        if not region.enabled:
            continue
        shape = _shape(region)
        if not shape:
            raise ValueError('exact shape missing')
        key = (region.metadata.get('v250_object_snapshot') or {}).get('entity_id', region.object_id)
        groups.setdefault(key, []).append(shape_bounds(shape))
    circles = []
    for bounds in groups.values():
        x = min(b[0] for b in bounds)
        y = min(b[1] for b in bounds)
        right = max(b[0] + b[2] for b in bounds)
        bottom = max(b[1] + b[3] for b in bounds)
        circles.append(((round((x + right) / 2), round((y + bottom) / 2)),
                        math.ceil(math.hypot(right - x, bottom - y) / 2 + radius)))
    return circles


def object_highlight_mask(regions, size):
    """Exact visible object silhouettes; measurement circles are kept separate."""
    mask = np.zeros((int(size[1]), int(size[0])), np.uint8)
    for region in regions:
        if region.enabled:
            shape = _shape(region)
            if not shape:
                raise ValueError('exact shape missing')
            raster_shape(mask, shape)
    return mask


def ring_mask(mask, circles, ring_width=1):
    ring = np.zeros(mask.shape, np.uint8)
    for center, radius in circles:
        cv2.circle(ring, center, max(1, radius - 1), 1, max(1, int(ring_width)))
    return ring


def presentation_mask(mask, circles, ring_width=1):
    return ((mask > 0) | (ring_mask(mask, circles, ring_width) > 0)).astype(np.uint8)


def glow_mask(regions, size, radius):
    """Frozen parts plus measurement circles; circles never enlarge collision.

    Retain the lane's identical mask after damage. Only the active, shot-frozen
    object shapes can receive a hit; illuminated neighbours can establish a miss.
    """
    regions = tuple(regions)
    mask = np.zeros((int(size[1]), int(size[0])), np.uint8)
    for region in regions:
        if region.enabled:
            shape = _shape(region)
            if not shape:
                raise ValueError('exact shape missing')
            raster_shape(mask, shape)
    for center, circle_radius in measurement_circles(regions, radius):
        cv2.circle(mask, center, circle_radius, 1, -1)
    if not mask.any():
        raise ValueError('no shootable shapes')
    return mask




def pastel_highlight(rgb, mask, circles, *, mix=.30, ring_color=(230, 220, 150), ring_width=1):
    """Subtly lighten exact object pixels, preserving hue, with thin yellow rings."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (*mask.shape, 3):
        raise ValueError('highlight requires viewport RGB pixels and its frozen mask')
    if not np.isfinite(mix) or not 0 <= mix <= .55:
        raise ValueError('invalid pastel mix')
    pixels = rgb.astype(np.float32)
    light = np.rint(pixels + (255.0 - pixels) * float(mix)).clip(0, 255).astype(np.uint8)
    output = rgb.copy()
    output[mask > 0] = light[mask > 0]
    ring = ring_mask(mask, circles, ring_width)
    output[ring > 0] = ring_color
    overlay = ((mask > 0) | (ring > 0)).astype(np.uint8)
    return output, overlay


def freeze_projection(viewport):
    """Freeze the same canonical mapping selected by the existing hit registry."""
    from src.engine.input.hit_regions import object_hit_registry_v2223
    from src.engine.input.hit_input import hit_input
    from src.engine import lens_model
    from src.engine.settings import load_scanport_rect
    method,_,_=object_hit_registry_v2223._camera_transformers()
    lens=deepcopy(getattr(hit_input,'lens',None)) if method.startswith('homography') else None
    if method.startswith('homography') and hit_input.homography is not None:
        H=np.array(lens['H'] if lens else hit_input.homography,dtype=float).copy()
    elif method=='scanport':
        s=load_scanport_rect()
        if s is None:raise ValueError('no scanport')
        vx,vy,vw,vh=viewport
        H=np.array([[vw/s.w,0,vx-s.x*vw/s.w],[0,vh/s.h,vy-s.y*vh/s.h],[0,0,1]],float)
    else:raise ValueError('no calibrated mapping')
    if not np.isfinite(H).all() or np.linalg.matrix_rank(H)<3:raise ValueError('invalid calibrated transform')
    fingerprint=hashlib.sha256(json.dumps(dict(method=method,H=H.tolist(),lens=None if lens is None else
        {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in lens.items()},viewport=list(viewport)),sort_keys=True).encode()).hexdigest()
    def points(p):
        p=np.asarray(p,dtype=float).reshape(-1,2)
        if lens:p=lens_model.undistort_points(p,lens)
        return cv2.perspectiveTransform(p.reshape(-1,1,2),H).reshape(-1,2)
    points.state=dict(method=method,H=H.tolist(),lens=None if lens is None else {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in lens.items()},viewport=list(viewport))
    return points,fingerprint


def camera_mask(mask,frame_shape,viewport,map_points):
    """Lens-aware camera-pixel remap, tiled to bound transient allocation."""
    h,w=frame_shape[:2];out=np.zeros((h,w),np.uint8);vx,vy,_,_=viewport
    for start in range(0,h,128):
        yy,xx=np.mgrid[start:min(h,start+128),:w]
        p=map_points(np.column_stack((xx.ravel(),yy.ravel())))
        mx=(p[:,0]-vx).reshape(xx.shape).astype(np.float32);my=(p[:,1]-vy).reshape(xx.shape).astype(np.float32)
        out[start:start+len(xx)]=cv2.remap(mask,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
    return out


def distance_mm(shape,xy,mm_per_px):
    """Distance from observed centre to geometric boundary; never expands collision."""
    x,y=xy;sx,sy=mm_per_px;kind=shape.get('kind')
    if kind=='union':return min(distance_mm(p,xy,mm_per_px) for p in shape['parts'])
    if kind=='polygon':points=np.asarray(shape['points'],float)
    elif kind=='rect':
        a,b,w,h=(shape[k] for k in ('x','y','width','height'));points=np.array([(a,b),(a+w,b),(a+w,b+h),(a,b+h)])
    else:
        a,b,w,h=(shape[k] for k in ('x','y','width','height'));theta=np.linspace(0,2*np.pi,257)[:-1]
        points=np.column_stack((a+w/2+w/2*np.cos(theta),b+h/2+h/2*np.sin(theta)))
    points=(points*np.array([sx,sy])).astype(np.float32)
    return max(0.,-float(cv2.pointPolygonTest(points,(x*sx,y*sy),True)))


class ObjectGlowQuery:
    """Small registry-compatible result sink; no dense/ordinary evidence can finish it."""
    evidence_method='object_glow';object_hit_requested=True
    def __init__(self,snapshot,clock):
        self.snapshot=snapshot;self.clock=clock;self.shot_id=snapshot.shot_id;self.peak_ts=snapshot.peak_ts
        self.game_regions=snapshot.game_regions;self.regions=snapshot.camera_regions
        self.final=False;self.published=False;self.result=None;self.hypotheses=[]
        self.search_pixels=0;self.first_post_ts=None
    def finish(self,status='UNCERTAIN',reason='cancelled'):
        if self.final:return self.result
        return self.complete(dict(state='UNKNOWN',object_id=None,hypothesis=None,reason=reason),[],reason)
    def complete(self,decision,hypotheses,reason,**extra):
        if self.final:return self.result
        self.final=True;now=self.clock();self.hypotheses=hypotheses
        status='PHYSICAL_HIT' if hypotheses else 'UNCERTAIN'
        self.result=dict(mode='OBJECT',evidence_method='object_glow',result_semantics='OBJECT_HIT',
            shot_id=self.shot_id,peak_ts=self.peak_ts,decision_ts=now,final=True,status=status,
            reason=reason,object_decision=decision,hypotheses=hypotheses,
            timing=dict(audio_to_decision_ms=(now-self.peak_ts)*1000.),**extra)
        return self.result
    def compact_record(self,**kwargs):
        return {**(self.result or dict(shot_id=self.shot_id,evidence_method='object_glow')),'gameplay':kwargs.get('gameplay'),'current':kwargs.get('current')}


class ObjectGlow:
    """One intro + one measurement per quiet burst, using PrecisionFlash's raw ring."""
    def __init__(self,parent):
        self.parent=parent;self.clock=parent.clock;self.monotonic=parent.monotonic;self.log=parent.log
        self.key=None;self.scene=None;self.enabled=False;self.state='IDLE';self.reference=None
        self.mask=None;self.camera_valid=None;self.projection=None;self.viewport=None;self.cfg=GlowConfig()
        self.queries=[];self.peaks=[];self.seen=[];self.on=None;self.off=None;self.requested_on=None
        self.last_off=-1e9;self.last_off_mono=(self.monotonic() if parent.flash_windows else -1e9);self.context_since=-1e9;self.windows=[];self.is_reference=False;self._surface=None
        self.generation=0;self.worker_result=None;self.lock=threading.Lock();self.results=[]
        self._awaiting_disabled_off=False;self.map_id=None;self.ready=False;self.last_peak=-1e9;self.mm_per_px=(1.,1.)
        self.circles=[];self.highlight_mask=None;self.highlight_rgb_sha256=None;self._palette_rgb=None
    def _notify(self,status):
        cb=getattr(self.scene,'on_precision_glow_state',None)
        if callable(cb):
            try:cb(dict(state=status,ready=self.ready,key=self.key))
            except Exception as exc:self.log('[OBJECT-GLOW] state callback failed: '+str(exc))
    def disable(self,reason='disabled'):
        self.generation+=1;self.enabled=False;self.ready=False
        if self.on is not None and self.off is None:
            self.off=self.clock();self.last_off=self.off;self.last_off_mono=self.monotonic();self.windows[-1]=(self.on,self.off)
            self._awaiting_disabled_off=True
        for q in self.queries:q.finish(reason=reason)
        self.queries=[];self.reference=None;self.state='IDLE';self._notify(reason);self.key=None
    def sync(self,scene):
        self.scene=scene
        provider=getattr(scene,'get_precision_glow_context',None)
        context=provider() if callable(provider) else None
        if not context:
            if self.key is not None:self.disable('no_glow_surface');self.key=None
            return
        cfg=GlowConfig.from_scene(scene);viewport=tuple(context['viewport'])
        projection,map_id=freeze_projection(viewport)
        key=(str(context['key']),map_id,repr(cfg))
        if key!=self.key:
            self.disable('surface_changed');self.scene=scene;self.key=key;self.enabled=True;self.cfg=cfg
            self.viewport=viewport;self.projection=projection;self.map_id=map_id
            self._presentation_mask=None
            self.frozen_regions=deepcopy(context['regions'])
            self.initial_regions={r.object_id:deepcopy(_shape(r)) for r in self.frozen_regions}
            self.mask=glow_mask(self.frozen_regions,viewport[2:],cfg.radius_px)
            self.highlight_mask=object_highlight_mask(self.frozen_regions,viewport[2:])
            self.circles=measurement_circles(self.frozen_regions,0)
            self._palette_rgb=None;self.highlight_rgb_sha256=None
            if float(self.mask.mean())>cfg.max_mask_fraction:raise ValueError('local mask too large')
            self.context_since=self.monotonic()
            self.camera_valid=None;self._surface=None;self.peaks=[];self.seen=[];self.state='REFERENCE_WAIT'
            self.mm_per_px=tuple(context.get('mm_per_px',(1.,1.)))
            self._notify('show_targets')
        else:
            self.enabled=True
            active=context.get('active_regions',self.frozen_regions)
            signature=tuple(sorted((r.object_id,json.dumps(_shape(r),sort_keys=True)) for r in active))
            if signature!=getattr(self,'active_signature',None) and self.state in ('IDLE','REFERENCE_WAIT'):
                if any(r.object_id not in self.initial_regions or _shape(r)!=self.initial_regions[r.object_id] for r in active):
                    raise ValueError('moved/new shapes require a new glow context key')
                new=glow_mask(active,viewport[2:],cfg.radius_px) if active else np.zeros_like(self.mask)
                self.mask &= new  # no newly lit pixel can borrow an unlit reference
                active_highlight=object_highlight_mask(active,viewport[2:]) if active else np.zeros_like(self.highlight_mask)
                self.highlight_mask &= active_highlight
                self.circles=measurement_circles(active,0)
                self._presentation_mask=presentation_mask(self.highlight_mask,self.circles,cfg.ring_width_px)
                self._surface=None
                self.frozen_regions=deepcopy(active);self.active_signature=signature
                self.camera_valid=None;self._surface=None
    def on_peak(self,ts):
        if not self.enabled or any(abs(ts-p)<1e-6 for p in self.seen):return
        self.seen=(self.seen+[float(ts)])[-64:];self.last_peak=max(self.last_peak,float(ts));self.peaks=(self.peaks+[float(ts)])[-128:]
        if self.is_reference and self.state in ('SHOWING','CAPTURING','MEASURING'):
            self.ready=False
    def attach(self,query):
        if query in self.queries:return
        query.clock=self.clock
        self.queries.append(query)
        self.on_peak(query.peak_ts)
        if not self.enabled:query.finish(reason='object_glow_not_enabled')
        elif self.reference is None:query.finish(reason='no_causal_glow_reference')
        elif query.peak_ts<=self.reference.get('latest_exposure_end',-1e18):query.finish(reason='peak_precedes_glow_reference')
        elif not getattr(query.snapshot,'transform_available',False) or not callable(getattr(query.snapshot,'object_camera_to_game',None)):
            query.finish(reason='no_frozen_object_projection')
        else:
            h,w=self.camera_valid.shape if self.camera_valid is not None else (300,500)
            probes=np.array([[w*.25,h*.25],[w*.50,h*.50],[w*.75,h*.75]])
            own=self.projection(probes)-np.asarray(self.viewport[:2])
            frozen=np.array([query.snapshot.object_camera_to_game(*p) for p in probes],dtype=float)
            if frozen.shape!=own.shape or not np.isfinite(frozen).all() or float(np.max(np.abs(frozen-own)))>.05:
                query.finish(reason='reference_projection_changed')
    def excludes_frame(self,ts):
        return any(ts>=a and (b is None or ts<=b+self.parent.cfg.exclude_tail_s) for a,b in self.windows[-8:])
    def _start(self,reference):
        self.is_reference=reference;self.state='SHOWING';self.on=None;self.off=None;self.requested_on=self.clock();self.requested_mono=self.monotonic()
        self.generation+=1;self._notify('show_targets' if reference else 'measuring')
    def tick(self):
        if not self.enabled:return
        now=self.clock()
        self.queries=[q for q in self.queries if not q.published and (not q.final or now-q.peak_ts<10.)]
        for q in self.queries:
            if not q.final and now-q.peak_ts>self.cfg.query_timeout_s:q.finish(reason='object_glow_timeout')
        mono=self.monotonic();spaced=mono-self.last_off_mono>=self.cfg.min_interval_s
        if self.state=='REFERENCE_WAIT':
            if spaced and now-self.last_peak>=self.cfg.settle_s and mono-self.context_since>=self.cfg.initial_wait_s:self._start(True)
        elif self.state=='IDLE' and self.reference is not None:
            if any(not q.final for q in self.queries) and spaced and now-self.last_peak>=self.cfg.settle_s:self._start(False)
        elif self.state=='SHOWING':
            duration=self.cfg.reference_s if self.is_reference else self.cfg.measurement_s
            if self.on is not None and mono-self.on_mono>=duration:self.state='REMOVING'
            elif self.on is None and mono-self.requested_mono>1.:self._failed('highlight_not_presented')
        elif self.state=='CAPTURING' and mono-self.last_off_mono>=self.cfg.capture_wait_s:self._start_measurement()
        elif self.state=='MEASURING':
            with self.lock:result,self.worker_result=self.worker_result,None
            if result is not None:
                generation,result=result
                if generation==self.generation:self._finish(result)
    def render(self,screen):
        if not self.enabled or self.state!='SHOWING':return
        import pygame
        if not screen.get_rect().contains(pygame.Rect(self.viewport)):
            self._failed('highlight_viewport_outside_screen')
            return
        if self._surface is None:
            # Freeze the displayed palette once for this lane. Reference and
            # measurement use identical pixels even after a part breaks.
            if self._palette_rgb is None:
                original=pygame.surfarray.array3d(screen.subsurface(pygame.Rect(self.viewport))).transpose(1,0,2)
                self._palette_rgb,self._presentation_mask=pastel_highlight(original,self.highlight_mask,self.circles,
                    mix=self.cfg.pastel_mix,ring_color=self.cfg.color,ring_width=self.cfg.ring_width_px)
            # Damage may shrink the alpha mask, but remaining pixels must keep
            # the reference palette; a new tint/ring cannot masquerade as a hole.
            rgb=self._palette_rgb
            self.highlight_rgb_sha256=hashlib.sha256(rgb[self._presentation_mask>0].tobytes()).hexdigest()
            self._surface=pygame.Surface(self.mask.shape[::-1],pygame.SRCALPHA)
            pygame.surfarray.blit_array(self._surface,rgb.transpose(1,0,2))
            alpha=pygame.surfarray.pixels_alpha(self._surface)
            alpha[:]=self._presentation_mask.T*255;del alpha
        screen.blit(self._surface,self.viewport[:2])
    def presented(self,now):
        if self._awaiting_disabled_off:
            self._awaiting_disabled_off=False;self.off=float(now);self.last_off=self.off;self.last_off_mono=self.monotonic()
            if self.windows:self.windows[-1]=(self.windows[-1][0],self.off)
        if not self.enabled:return
        if self.state=='SHOWING' and self.on is None:
            self.on=float(now);self.on_mono=self.monotonic();self.parent.flash_count+=1;self.windows=(self.windows+[(self.on,None)])[-16:]
            # Preserve the initial reference in shot-window source recordings too.
            try:
                from src.engine.source_recording import get_source_recorder
                recorder=get_source_recorder()
                if recorder is not None:
                    recorder.note_trigger(self.on,source='object_glow_reference' if self.is_reference else 'object_glow_measurement')
                    recorder._event('object_glow_presented',presented_ts=self.on,presented_monotonic=self.on_mono,reference=self.is_reference,
                        key=list(self.key),config=self.cfg.__dict__,flash_config=self.parent.cfg.__dict__,viewport=list(self.viewport),
                        highlight_rgb_sha256=self.highlight_rgb_sha256,
                        projection=getattr(self.projection,'state',None),
                        regions=[dict(object_id=r.object_id,shape=_shape(r),entity_id=(r.metadata.get('v250_object_snapshot') or {}).get('entity_id',r.object_id)) for r in getattr(self,'frozen_regions',())],
                        mask_sha256=hashlib.sha256(self.mask.tobytes()).hexdigest())
            except Exception as exc:self.log('[OBJECT-GLOW] source provenance unavailable: '+str(exc))
        elif self.state=='REMOVING':
            self.off=float(now);self.last_off=self.off;self.last_off_mono=self.monotonic();self.windows[-1]=(self.on,self.off);self.state='CAPTURING'
    def _failed(self,reason):
        self.ready=False;self.reference=None;self.state='FAILED'
        for q in self.queries:q.finish(reason=reason)
        self.log('[OBJECT-GLOW] '+reason);self._notify(reason)
    def _start_measurement(self):
        ring=self.parent.ring_source() if self.parent.ring_source else []
        frames=[f for f in ring if self.on-.6<=f.timestamp<=min(self.clock(),self.off+.20)]
        generation=self.generation;reference=self.reference;on,off=self.on,self.off
        cfg=self.cfg;mask=self.mask.copy();projection=self.projection;viewport=self.viewport
        camera_valid=self.camera_valid;is_reference=self.is_reference;map_id=self.map_id
        cutoff=min((p for p in self.peaks if p>=on),default=None)
        display_s=self.last_off_mono-self.on_mono
        clock_changed=abs((off-on)-display_s)>.05
        self.state='MEASURING'
        def work():
            try:
                duration=cfg.reference_s if is_reference else cfg.measurement_s
                if clock_changed:raise ValueError('display_wall_clock_changed')
                if display_s>duration+.15:raise ValueError('highlight_display_overrun')
                result=measure_glow(frames,on,off,reference,mask,viewport,projection,cfg,self.parent.cfg,
                    camera_valid=camera_valid,cutoff=cutoff,is_reference=is_reference)
            except Exception as exc:result=dict(valid=False,reason='measurement_error:'+type(exc).__name__+':'+str(exc))
            result.update(reference_flash=is_reference,on=on,off=off,map_id=map_id,cutoff_peak=cutoff,actual_display_s=display_s,mode='object_glow')
            with self.lock:
                if generation==self.generation:self.worker_result=(generation,result)
        threading.Thread(target=work,name='precision-object-glow',daemon=True).start()
    def _finish(self,result):
        finished_here=[]
        image=result.pop('image',None);valid=result.pop('camera_valid',None)
        if result.get('valid'):
            self.reference=image;self.camera_valid=valid;self.ready=True
        if self.is_reference:
            if any(p>=self.on for p in self.peaks):
                self.reference=None;self.ready=False;result.update(valid=False,reason='shot_during_reference')
            self.state='IDLE' if self.ready else 'FAILED'
        else:
            current=[q for q in self.queries if not q.final and q.peak_ts<self.on]
            finished_here=current
            owner=min(current,key=lambda q:q.peak_ts,default=None)
            groups=[]
            for q in sorted(current,key=lambda q:q.peak_ts):
                if not groups or q.peak_ts-groups[-1][0].peak_ts>.15:groups.append([q])
                else:groups[-1].append(q)
            holes=result.get('holes',[])
            for q in current:
                follower=owner is not q and len(groups)==1
                if follower:
                    q.finish(reason='paired_audio_follower')
                    q.result.update(gameplay_owner=False,paired_follower_of=owner.shot_id)
                elif result.get('valid') and len(groups)==1:
                    decision,hypotheses=decide_glow(q,holes,self.projection,self.viewport,self.mm_per_px,result)
                    q.complete(decision,hypotheses,decision['reason'],glow=result)
                else:q.finish(reason=result.get('reason') if not result.get('valid') else 'multi_shot_assignment_unknown')
            self.state='IDLE' if self.reference is not None else 'FAILED'
        if result.get('reason') in ('camera_geometry_changed','new_mask_requires_reference','bipolar_change_camera_or_scene_motion'):
            self.reference=None;self.ready=False;self.state='FAILED'
        self.peaks=[p for p in self.peaks if p>=self.on]
        # Never retain 4K pixels in diagnostic history.
        self.results=(self.results+[result])[-32:]
        self.parent.results=(self.parent.results+[result])[-32:]
        self.log('[OBJECT-GLOW] '+json.dumps(result,default=str))
        try:
            from src.engine.source_recording import get_source_recorder
            recorder=get_source_recorder()
            if recorder is not None:recorder._event('object_glow_result',result=result,decisions=[{k:q.result.get(k) for k in ('shot_id','peak_ts','decision_ts','status','reason','object_decision','gameplay_owner','paired_follower_of')} for q in finished_here if q.final and q.result is not None])
        except Exception as exc:self.log('[OBJECT-GLOW] source result unavailable: '+str(exc))
        self._notify('ready' if self.ready else result.get('reason','unknown'))


def measure_glow(frames,on,off,reference,mask,viewport,projection,cfg,flash_cfg,*,camera_valid=None,cutoff=None,is_reference=False):
    """Strict causal plateau, median; no future frames or GT. Returns camera-space evidence."""
    from src.engine.precision_flash import _gray,find_new_holes
    if not frames:return dict(valid=False,reason='no_frames')
    shape=frames[0].frame_bgr.shape[:2]
    valid=camera_mask(mask,shape,viewport,projection) if camera_valid is None else camera_valid
    ys,xs=np.nonzero(valid)
    if not len(xs):return dict(valid=False,reason='no_calibrated_mask')
    rect=(int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1))
    if reference is not None:
        rx,ry,rw,rh=reference['rect']
        if rect[0]<rx or rect[1]<ry or rect[0]+rect[2]>rx+rw or rect[1]+rect[3]>ry+rh:
            return dict(valid=False,reason='new_mask_requires_reference')
        rect=tuple(reference['rect'])
    x,y,w,h=rect
    if x<0 or y<0 or x+w>shape[1] or y+h>shape[0]:return dict(valid=False,reason='camera_geometry_changed')
    scope=valid[y:y+h,x:x+w]>0
    before=[f for f in frames if f.timestamp<on]
    # Receive timestamps are not exposure timestamps. Safe window includes
    # bounded projector response, exposure duration and measured receive latency.
    candidates=[f for f in frames if f.timestamp-cfg.receive_latency_s-cfg.exposure_s>=on+cfg.projector_max_s
        and f.timestamp-cfg.receive_latency_s<=off+cfg.projector_min_s
        and (cutoff is None or f.timestamp<=cutoff)]
    needed=cfg.reference_frames if is_reference else cfg.measurement_frames
    if len(candidates)<needed:return dict(valid=False,reason='insufficient_causal_plateau',frames=len(candidates),required=needed)
    if not before:return dict(valid=False,reason='no_before_frame')
    grays=[_gray(f.frame_bgr,rect) for f in candidates[-12:]];candidates=candidates[-12:]
    # Each occupied sensor band must be lit; empty bands carry no information.
    bands=[]
    for g in grays:
        bands.append([float(g[a*h//4:(a+1)*h//4][scope[a*h//4:(a+1)*h//4]].mean())
            if scope[a*h//4:(a+1)*h//4].any() else np.nan for a in range(4)])
    bands=np.asarray(bands);occupied=np.isfinite(bands).any(axis=0);b=bands[:,occupied]
    keep=np.all(b>=np.max(b,axis=0)*flash_cfg.plateau_fraction,axis=1)
    if int(keep.sum())<needed:return dict(valid=False,reason='rolling_or_partial_plateau',frames=int(keep.sum()))
    image=np.median(np.stack([g for g,k in zip(grays,keep) if k]),axis=0).astype(np.float32)
    delta=float(np.median(np.abs(image-_gray(before[-1].frame_bgr,rect))[scope]))
    if delta<flash_cfg.min_flash_delta:return dict(valid=False,reason='highlight_not_observed',delta=delta)
    holes=[];stats={}
    if not is_reference and reference is not None:
        if tuple(reference['rect'])!=rect:return dict(valid=False,reason='mask_or_camera_changed')
        holes,stats=find_new_holes(reference['gray'],image,flash_cfg,max_holes=32,valid_mask=scope)
        for hole in holes:hole.update(camera_x=hole['x']+x,camera_y=hole['y']+y)
        if any(h.get('bipolar_change') for h in holes):
            return dict(valid=False,reason='bipolar_change_camera_or_scene_motion',candidate_holes=holes)
    evidence=dict(valid=True,reason='measured',holes=holes,rect=rect,plateau_frames=int(keep.sum()),
        plateau_ts=[float(f.timestamp) for f,k in zip(candidates,keep) if k],delta=delta,
        mask_sha256=hashlib.sha256(valid.tobytes()).hexdigest(),**stats)
    evidence.update(image=dict(gray=image,rect=rect,latest_exposure_end=max(evidence['plateau_ts'])-cfg.receive_latency_s),camera_valid=valid)
    evidence['timing_assumptions']=dict(receive_latency_s=cfg.receive_latency_s,exposure_s=cfg.exposure_s,projector_delay_s=[cfg.projector_min_s,cfg.projector_max_s])
    return evidence


def decide_glow(query,holes,projection,viewport,mm_per_px,evidence):
    from src.engine.game_objects.geometry import shape_contains
    if len(holes)!=1:
        return dict(state='UNKNOWN',object_id=None,hypothesis=None,
            reason='no_new_hole_observed' if not holes else 'multiple_new_holes'),[]
    hole=holes[0];camera_xy=[hole['camera_x'],hole['camera_y']]
    screen=projection([camera_xy])[0];xy=[float(screen[0]-viewport[0]),float(screen[1]-viewport[1])]
    hit=[r for r in query.game_regions if _shape(r) and shape_contains(_shape(r),*xy)]
    distances=[(distance_mm(_shape(r),xy,mm_per_px),r.object_id) for r in query.game_regions if _shape(r)]
    distance,nearest=min(distances,default=(None,None))
    hypothesis=dict(rank=1,camera_xy=camera_xy,game_xy=xy,confirmed=True,confirmation_source='paired_glow_medians',confidence=None,
        evidence_score=hole['mass'],producer='object_glow',frame_ts=max(evidence['plateau_ts']))
    oid=hit[0].object_id if len(hit)==1 else None
    decision=dict(state='OBJECT_HIT' if len(hit)==1 else ('NO_OBJECT_HIT' if not hit else 'UNKNOWN'),
        object_id=oid,hypothesis=hypothesis if len(hit)<=1 else None,supported_object_ids=[r.object_id for r in hit],
        distance_mm=distance,nearest_object_id=nearest,scope='lane_measurement_circles_and_active_frozen_shapes',
        reason='unique_glow_hole_in_frozen_shape' if len(hit)==1 else ('measured_near_miss' if not hit else 'overlapping_shapes'),
        no_physical_impact_established=False,global_rescue_used=False)
    return decision,[hypothesis]
