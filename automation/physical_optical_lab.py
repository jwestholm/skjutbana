"""Bounded optical research measurements. Never projectile truth or live evidence.

CLI uses the existing TCP command queue/EventListener. Synthetic mode opens no
hardware. Geometry replay uses only saved matrices and hash-validated raw images.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from automation.physical_gt_capture import atomic, checked

SCHEMA = 'physical_optical_lab_v1'
FRAME_SCHEMA = 'physical_optical_frame_v1'
KINDS = ('bright_square','dark_square','vertical_edge','horizontal_edge')
POINTS = ((.5,.5),(.25,.25),(.75,.25),(.25,.75),(.75,.75))
DISCLAIMER = 'OPTICAL/HIL != PROJECTILE VALIDATION; PASSIVE_VARIABILITY != HIT EVIDENCE; PROJECTOR_RESPONSE != PHYSICAL IMPACT; SYNTHETIC != PHYSICAL'


def output_path(path):
    path=checked(path)
    if not path.is_relative_to(Path('/data/skjutbana/evaluation_runs')):
        raise ValueError('optical outputs must be under /data/skjutbana/evaluation_runs')
    return path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def configuration(mode, **kwargs):
    c=dict(schema=SCHEMA,mode=mode,duration_s=30.,settle_s=.6,frame_timeout_s=5.,
           tile_px=32,sample_every_s=10.,save_all_frames=False,synthetic=False,seed=1729)
    unknown=set(kwargs)-set(c)
    if unknown:raise ValueError('unknown optical configuration: '+str(sorted(unknown)))
    c.update(kwargs)
    if mode not in ('passive','probes'):raise ValueError('unknown experiment')
    for key,lo,hi in (('duration_s',.1,300),('settle_s',.05,5),('frame_timeout_s',.1,30),('sample_every_s',1,300)):
        if type(c[key]) not in (int,float) or not math.isfinite(c[key]) or not lo<=c[key]<=hi:
            raise ValueError('invalid '+key)
    if type(c['tile_px']) is not int or not 8<=c['tile_px']<=128:raise ValueError('invalid tile size')
    if type(c['seed']) is not int or not 0<=c['seed']<2**32:raise ValueError('invalid seed')
    if any(type(c[k]) is not bool for k in ('synthetic','save_all_frames')):raise ValueError('boolean flags required')
    return c


def snapshot_mapping(mapper, viewport, content, screen_size):
    """Read existing calibration only. No identity/scanport-as-calibration fallback."""
    h=mapper.homography
    valid=bool(mapper._prefers_homography() and h is not None and np.isfinite(h).all()
               and abs(float(np.linalg.det(h)))>1e-12)
    return dict(status='AVAILABLE' if valid else 'UNAVAILABLE',
        reason='EXISTING_HOMOGRAPHY_NOT_PHYSICALLY_REVALIDATED' if valid else 'NO_VALID_SAVED_HOMOGRAPHY',
        camera_to_screen=h.tolist() if valid else None,
        screen_to_camera=np.linalg.inv(h.astype(np.float64)).tolist() if valid else None,
        calibration=deepcopy(mapper.calibration_data),viewport=list(viewport),content=list(content),screen_size=list(screen_size),
        game_space='VIEWPORT_LOCAL_PIXELS',window_position_semantics='DESKTOP_OFFSET_NOT_CAMERA_COORDINATE',
        physical_calibration_validated=False)


def transform(matrix, point):
    # Same perspective primitive as HitInput; supplied saved matrix ONLY.
    if matrix is None:return None
    h=np.asarray(matrix,np.float64)
    if h.shape!=(3,3) or not np.isfinite(h).all() or abs(float(h[2]@[*point,1]))<1e-12:return None
    p=cv2.perspectiveTransform(np.array([[point]],np.float64),h)[0,0]
    return p.tolist() if np.isfinite(p).all() else None


def plans(mapping):
    vx,vy,w,h=mapping['viewport'];sw,sh=mapping['screen_size']
    if w<=0 or h<=0 or vx<0 or vy<0 or vx+w>sw or vy+h>sh:
        raise ValueError('viewport outside render surface; no guessed scaling')
    side=max(4,int(round(min(w,h)*.08)))
    result=[]
    for kind in KINDS:
        for i,(u,v) in enumerate(POINTS):
            x,y=round(vx+u*w-side/2),round(vy+v*h-side/2)
            # Pixel-center convention: mean of the exact rendered integer pixels.
            center=[x+(side-1)/2,y+(side-1)/2]
            result.append(dict(probe_id=f'{kind}_{i}',kind=kind,game_requested=[u*w,v*h],
                normalized_requested=[u,v],screen_rect=[x,y,side,side],screen_center=center,
                game_rendered_center=[center[0]-vx,center[1]-vy],
                expected_camera_xy=transform(mapping['screen_to_camera'],center)))
    return result


def render_probe(screen, probe=None):
    """Only Pygame main thread calls this. No scene/game or camera coordinates."""
    import pygame
    screen.fill((127,127,127))
    if probe is None:return
    rect=pygame.Rect(probe['screen_rect']);kind=probe['kind']
    if kind in ('bright_square','dark_square'):
        value=255 if kind=='bright_square' else 0
        pygame.draw.rect(screen,(value,)*3,rect)
    else:
        pygame.draw.rect(screen,(0,0,0),rect)
        half=rect.copy()
        if kind=='vertical_edge':half.width=rect.width//2
        else:half.height=rect.height//2
        pygame.draw.rect(screen,(255,255,255),half)


def gray(pixels):
    if pixels.dtype!=np.uint8 or pixels.ndim!=3 or pixels.shape[2]!=3:raise ValueError('native BGR uint8 required')
    return cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY)


class TileStats:
    """Streaming tile means: exact moments/range, 1DN histogram median/MAD.

    MAD is of quantized tile means, NOT native-pixel noise. Histogram and previous
    grayscale frame are bounded independently of run length. No frame reservoir.
    """
    def __init__(self, tile):
        self.tile=tile;self.n=0;self.previous=None

    def add(self, pixels):
        im=gray(pixels);h,w=im.shape
        shape=(math.ceil(h/self.tile),math.ceil(w/self.tile))
        if np.prod(shape)>40000:raise ValueError('tile grid exceeds bounded budget')
        values=cv2.resize(im.astype(np.float32),(shape[1],shape[0]),interpolation=cv2.INTER_AREA)
        if not self.n:
            self.shape=im.shape;self.mean=np.zeros(shape,np.float64);self.m2=self.mean.copy()
            self.minimum=values.copy();self.maximum=values.copy();self.hist=np.zeros((values.size,256),np.uint32)
            self.first_brightness=float(im.mean());self.first_tiles=values.copy()
        elif im.shape!=self.shape:raise ValueError('camera dimensions changed during survey')
        self.n+=1;delta=values-self.mean;self.mean+=delta/self.n;self.m2+=delta*(values-self.mean)
        np.minimum(self.minimum,values,out=self.minimum);np.maximum(self.maximum,values,out=self.maximum)
        bins=np.clip(np.rint(values),0,255).astype(np.uint8).ravel()
        self.hist[np.arange(values.size),bins]+=1
        brightness=float(im.mean())
        result=dict(mean_brightness=brightness,brightness_drift=brightness-self.first_brightness,
                    frame_abs_difference=None if self.previous is None else float(cv2.absdiff(im,self.previous).mean()))
        self.previous=im
        # Existing global translation primitive, on tiles; never correct pixels.
        if float(self.first_tiles.std())<1e-3:
            result['motion']=dict(status='UNAVAILABLE',reason='UNIFORM_REFERENCE')
        else:
            from src.engine.offline.accuracy_verifier import align_to_reference
            _,m=align_to_reference(self.first_tiles,values)
            result['motion']=dict(status='ESTIMATE' if m['applied'] else 'UNQUALIFIED',**m,
                units='TILE_GRID_PIXELS',native_scale_xy=[w/shape[1],h/shape[0]])
        return result

    def maps(self):
        if not self.n:return None
        cumulative=self.hist.cumsum(axis=1);median=(cumulative>=(self.n+1)//2).argmax(axis=1)
        distances=np.abs(np.arange(256)[None,:]-median[:,None])
        order=np.argsort(distances,axis=1,kind='stable')
        ordered_hist=np.take_along_axis(self.hist,order,axis=1).cumsum(axis=1)
        index=(ordered_hist>=(self.n+1)//2).argmax(axis=1)
        mad=np.take_along_axis(distances,order,axis=1)[np.arange(len(index)),index]
        return dict(mean=self.mean,stddev=np.sqrt(self.m2/self.n),minimum=self.minimum,maximum=self.maximum,
                    median_1dn=median.reshape(self.mean.shape),mad_1dn=mad.reshape(self.mean.shape))

    def memory_bytes(self):
        if not self.n:return 0
        return sum(a.nbytes for a in (self.previous,self.mean,self.m2,self.minimum,self.maximum,self.hist,self.first_tiles))


def centroid(weights):
    total=float(weights.sum(dtype=np.float64))
    if total<=0:return None
    xs=weights.sum(axis=0,dtype=np.float64);ys=weights.sum(axis=1,dtype=np.float64)
    x=float(xs@np.arange(len(xs))/total);y=float(ys@np.arange(len(ys))/total)
    variance=float(xs@((np.arange(len(xs))-x)**2)/total+ys@((np.arange(len(ys))-y)**2)/total)
    bounds=[int(np.searchsorted(np.cumsum(s),total*q)) for s,q in ((xs,.1),(ys,.1),(xs,.9),(ys,.9))]
    return dict(camera_xy=[x,y],response_spread_px=math.sqrt(max(0,variance)),mass=total,central_80_percent_bbox=bounds)


def response_measure(baseline, probe, recovery, mapping, plan):
    a,b,c=(gray(p).astype(np.float32) for p in (baseline,probe,recovery))
    if a.shape!=b.shape or a.shape!=c.shape:raise ValueError('probe camera dimensions differ')
    # Whole camera image, no expected-location ROI or candidate suggestions.
    residual=np.abs(b-a);recovery_change=np.abs(c-a)
    measured=centroid(np.maximum(residual-recovery_change,0))
    before=centroid(residual);after=centroid(np.abs(b-c))
    result=dict(category='PROJECTOR_RESPONSE',status='AVAILABLE' if measured else 'UNAVAILABLE',
        reason='WEIGHTED_OPTICAL_RESPONSE_NOT_CAUSAL_CERTIFICATION' if measured else 'NO_RESPONSE_ABOVE_RECOVERY_CHANGE',
        measured=measured,recovery_abs_mean=float(recovery_change.mean()),response_abs_mean=float(residual.mean()),
        centroid_reference_disagreement_px=None if not before or not after else float(np.linalg.norm(np.subtract(before['camera_xy'],after['camera_xy']))),
        geometry=dict(status='UNAVAILABLE',reason='MISSING_CALIBRATION_OR_RESPONSE'))
    if mapping['status']=='AVAILABLE' and measured:
        screen=transform(mapping['camera_to_screen'],measured['camera_xy'])
        expected=transform(mapping['screen_to_camera'],plan['screen_center'])
        if screen is None or expected is None:return result
        roundtrip=transform(mapping['camera_to_screen'],expected)
        if roundtrip is None:return result
        vx,vy,_,_=mapping['viewport']
        result['geometry']=dict(status='AVAILABLE',expected_camera_xy=expected,observed_game_xy=[screen[0]-vx,screen[1]-vy],
            software_roundtrip_screen_px=float(np.linalg.norm(np.subtract(roundtrip,plan['screen_center']))),
            observed_roundtrip_game_px=float(np.linalg.norm(np.subtract(screen,plan['screen_center']))),
            interpretation='OPTICAL_RESPONSE_LOCATION_WITH_SPREAD_AND_BASELINE_UNCERTAINTY_NOT_HOLE_XY')
    return result


def frame_record(frame, role, transition, run_id, frame_index):
    acq=deepcopy(getattr(frame,'acquisition',{}) or {})
    return dict(schema=FRAME_SCHEMA,run_id=run_id,frame_index=frame_index,shot_id=None,role=role,
        category='PASSIVE_VARIABILITY' if role=='OPTICAL_PASSIVE' else 'PROJECTOR_RESPONSE',
        live_authority=False,projectile_truth=False,frame_timestamp=frame.timestamp,acquisition=acq,
        frame_source_id=f"{acq.get('stream_id')}:{acq.get('sequence')}",
        read_start_monotonic=acq.get('read_start_monotonic'),read_end_monotonic=acq.get('read_complete_monotonic'),
        receive_wall_ts=acq.get('receive_wall_ts'),publication_monotonic=acq.get('published_monotonic'),
        backend_timestamps_raw=acq.get('backend_timestamps_raw'),transition=deepcopy(transition),
        exposure_timestamp_verified=False,photon_latency_available=False)


def qualified_frame(frame, transition, settle_s):
    acq=getattr(frame,'acquisition',{}) or {}
    times=[acq.get(k) for k in ('read_start_monotonic','read_complete_monotonic','published_monotonic')]
    return (all(type(v) in (float,int) and math.isfinite(v) for v in times)
        and times[0]<=times[1]<=times[2] and times[0]>=transition['presented_monotonic']+settle_s)


class OpticalRun:
    """One append-only research run, processed by one non-Pygame worker."""
    def __init__(self, root, config, mapping):
        from src.engine.physical_trace import PhysicalTraceRecorder
        self.root=output_path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.config=configuration(config['mode'],**{k:v for k,v in config.items() if k not in ('mode','schema')})
        self.mapping=deepcopy(mapping);self.run_id=self.root.name
        self.writer=PhysicalTraceRecorder(self.root,enabled=True,writer_workers=1,queue_maxsize=8,max_queued_bytes=128*1024*1024)
        self.stats=TileStats(config['tile_px']);self.frames=0;self.samples=[];self.probes=[];self.triplet={};self.t0=None
        self.last_sample=None;self.first_source=None;self.last_sequence=None;self.gaps=0;self.processing_ms=[];self.closed=False
        self.manifest=dict(schema=SCHEMA,run_id=self.run_id,status='RUNNING',config=self.config,mapping=self.mapping,
            live_authority=False,projectile_truth=False,disclaimer=DISCLAIMER,samples=self.samples,probes=self.probes,
            created_at=datetime.now(timezone.utc).isoformat(),synthetic=config['synthetic'])
        self.manifest['probe_plan']=plans(self.mapping) if config['mode']=='probes' else []
        sources=['automation/physical_optical_lab.py','automation/physical_optical_runtime.py',
            'src/engine/physical_trace.py','src/engine/app.py','src/engine/shot_async_v2224.py',
            'src/engine/camera/camera_manager.py','src/engine/input/hit_input.py']
        self.manifest['source_sha256']={p:digest(Path(__file__).resolve().parents[1]/p) for p in sources}
        atomic(self.root/'config.json',self.config);self.checkpoint()

    def checkpoint(self):
        self.manifest.update(frames_processed=self.frames,application_sequence_gaps=self.gaps)
        atomic(self.root/'manifest.json',self.manifest)

    def add(self, frame, role, transition, plan=None):
        start=time.perf_counter()
        if not qualified_frame(frame,transition,self.config['settle_s']):raise ValueError('frame predates optical presentation/settle receipt barrier')
        record=frame_record(frame,role,transition,self.run_id,self.frames)
        record['pixel_sha256']=hashlib.sha256(memoryview(np.ascontiguousarray(frame.frame_bgr))).hexdigest()
        acq=record['acquisition'];stream=acq.get('stream_id');sequence=acq.get('sequence')
        if not stream or type(sequence) is not int:raise ValueError('missing camera source identity')
        if self.first_source is None:self.first_source=stream
        if stream!=self.first_source:raise ValueError('camera stream changed during optical run')
        if self.last_sequence is not None:
            if sequence<=self.last_sequence:raise ValueError('duplicate/reordered camera frame')
            self.gaps+=max(0,sequence-self.last_sequence-1)
        self.last_sequence=sequence
        self.frames+=1
        if self.t0 is None:self.t0=frame.timestamp
        save=role!='OPTICAL_PASSIVE' or self.config['save_all_frames'] or self.last_sample is None or frame.timestamp-self.last_sample>=self.config['sample_every_s']
        if role=='OPTICAL_PASSIVE':record['statistics']=self.stats.add(frame.frame_bgr)
        else:
            record['probe_id']=plan['probe_id'];self.triplet[role]=frame.frame_bgr
            if role=='OPTICAL_RECOVERY':
                result=response_measure(self.triplet['OPTICAL_BASELINE'],self.triplet['OPTICAL_PROBE'],self.triplet['OPTICAL_RECOVERY'],self.mapping,plan)
                self.probes.append(dict(plan=plan,**result));self.triplet.clear()
        if save:
            rel=f"{'passive' if role=='OPTICAL_PASSIVE' else 'projector_probes'}/frame_{self.frames:06d}.npy"
            accepted=self.writer.enqueue_optical_frame(self.root/rel,frame.frame_bgr,record)
            self.samples.append(dict(path=rel,role=role,probe_id=record.get('probe_id'),frame_source_id=record['frame_source_id'],status='QUEUED' if accepted else 'DROPPED'))
            self.last_sample=frame.timestamp
        record['processing_ms']=(time.perf_counter()-start)*1000;self.processing_ms.append(record['processing_ms'])
        with (self.root/'frames.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        self.checkpoint()
        return record

    def finish(self,status='COMPLETED',reason=None,diagnostics=None):
        if self.closed:return self.manifest
        self.closed=True;started=time.perf_counter();self.writer.shutdown(timeout=30)
        for row in self.samples:
            p=self.root/row['path'];side=p.with_suffix('.json')
            if p.exists() and side.exists():
                m=json.loads(side.read_text());row.update(status='SAVED',image_sha256=m['image_sha256'],metadata_sha256=digest(side),writer_ms=m['writer_ms'],copy_ms=m['copy_ms'])
            elif row['status']=='QUEUED':row['status']='UNAVAILABLE'
        if not self.frames and status=='COMPLETED':status='UNAVAILABLE';reason='NO_QUALIFIED_CAMERA_FRAMES'
        ws=self.writer.stats()
        if (ws['writer_errors'] or ws['queue_drops'] or ws['flush_pending']
                or any(row['status']!='SAVED' for row in self.samples)):
            status='INCOMPLETE';reason='FRAME_WRITER_INTEGRITY'
        elif status=='COMPLETED' and (diagnostics or {}).get('processor_backpressure_frames',0):
            status='INCOMPLETE';reason='PROCESSOR_BACKPRESSURE_NATIVE_RATE_NOT_SUSTAINED'
        maps=self.stats.maps()
        if maps is not None:
            p=self.root/'passive';p.mkdir(exist_ok=True)
            np.savez_compressed(p/'tile_statistics.npz',**maps)
            self.manifest['passive']=dict(tile_grid=list(maps['mean'].shape),native_shape=list(self.stats.shape),
                tile_mean_stddev_median=float(np.median(maps['stddev'])),tile_mad_1dn_median=float(np.median(maps['mad_1dn'])),
                summary_memory_bytes=self.stats.memory_bytes(),mad_semantics='EXACT_HISTOGRAM_MAD_OF_TILE_MEANS_QUANTIZED_TO_1_DN',
                map_sha256=digest(p/'tile_statistics.npz'),range_semantics='TEMPORAL_MIN_MAX_OF_AREA_AVERAGED_TILES_NOT_NATIVE_PIXEL_EXTREMES')
            flat=maps['stddev'].ravel();order=np.argsort(flat,kind='stable');cols=maps['mean'].shape[1]
            def tiles(indices):
                return [dict(tile_row=int(i//cols),tile_col=int(i%cols),stddev=float(flat[i])) for i in indices]
            self.manifest['passive'].update(lowest_variability_tiles=tiles(order[:10]),highest_variability_tiles=tiles(order[-10:][::-1]),
                classification='RELATIVE_VARIABILITY_ORDERING_NOT_HIT_EVIDENCE')
        self.manifest.update(status=status,reason=reason,diagnostics=diagnostics or {},writer=self.writer.stats(),
            processing_ms=dict(median=float(np.median(self.processing_ms)) if self.processing_ms else None,p95=float(np.percentile(self.processing_ms,95)) if self.processing_ms else None),
            finalize_ms=(time.perf_counter()-started)*1000,device_dropped_frames=None,
            sequence_gap_semantics='APPLICATION_SEQUENCE_GAPS_INCLUDE_INTENTIONAL_SETTLE_WINDOWS_AND_SKIPPED_READS_NOT_DEVICE_DROPS')
        self.checkpoint();atomic(self.root/'summary.json',self.manifest)
        with (self.root/'timing.csv').open('w') as f:
            fields=['frame_index','role','frame_timestamp','read_start_monotonic','read_end_monotonic','receive_wall_ts','publication_monotonic','processing_ms']
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            if (self.root/'frames.jsonl').exists():
                for line in (self.root/'frames.jsonl').read_text().splitlines():
                    rec=json.loads(line);w.writerow({k:rec.get(k) for k in fields})
        if maps is not None:
            rows=[json.loads(line) for line in (self.root/'frames.jsonl').read_text().splitlines()]
            brightness=np.array([r['statistics']['mean_brightness'] for r in rows])
            intervals=np.diff([r['frame_timestamp'] for r in rows])
            lag1=None
            if len(brightness)>3 and brightness[:-1].std()>1e-9 and brightness[1:].std()>1e-9:
                lag1=float(np.corrcoef(brightness[:-1],brightness[1:])[0,1])
            self.manifest['passive'].update(brightness_range=float(np.ptp(brightness)),brightness_drift=float(brightness[-1]-brightness[0]),
                brightness_lag1_correlation=lag1,interval_median_s=float(np.median(intervals)) if len(intervals) else None,
                interval_max_s=float(np.max(intervals)) if len(intervals) else None)
            self.checkpoint();atomic(self.root/'summary.json',self.manifest)
        (self.root/'REPORT.md').write_text(f"# Optical HIL measurement\n\n{DISCLAIMER}\n\nStatus: {status}; synthetic: {self.config['synthetic']}.\nFrames: {self.frames}; probes: {len(self.probes)}.\n\nSee summary.json, frames.jsonl, timing.csv, native NPY/JSON samples and tile_statistics.npz.\nNo physical hit labels or detector decisions exist here. Backend timestamps and flip-return are NOT proven exposure/photon times.\n")
        return self.manifest


def synthetic_run(output, config):
    from types import SimpleNamespace as NS
    mapping=dict(status='AVAILABLE',reason='SYNTHETIC_KNOWN_TRANSFORM',camera_to_screen=[[1,0,-8],[0,1,-6],[0,0,1]],
        screen_to_camera=[[1,0,8],[0,1,6],[0,0,1]],viewport=[8,8,128,64],content=[0,0,128,64],screen_size=[144,80],physical_calibration_validated=False)
    run=OpticalRun(output,config,mapping);rng=np.random.default_rng(config['seed']);seq=0;t=1000.
    def frame(image,transition):
        nonlocal seq,t
        seq+=1;t=max(t+1/30,transition['presented_monotonic']+config['settle_s']+.001)
        return NS(frame_bgr=image,timestamp=t,acquisition=dict(stream_id='SYNTHETIC',sequence=seq,
            read_start_monotonic=t,read_complete_monotonic=t+.001,published_monotonic=t+.002,receive_wall_ts=t,
            backend_name='SYNTHETIC',backend_timestamps_raw=None,exposure_monotonic=None))
    def transition(i):
        return dict(transition_id=i,command_issued_wall_ts=1000.,main_applied_monotonic=t,presented_monotonic=t,
            presentation_semantics='SYNTHETIC_NOT_PHOTON_TIME',render_frame_id=i)
    try:
        if config['mode']=='passive':
            tr=transition(0)
            for i in range(max(2,round(config['duration_s']*30))):
                image=np.clip(127+rng.normal(0,2,(96,160,1))+i*.01,0,255).astype(np.uint8).repeat(3,2)
                run.add(frame(image,tr),'OPTICAL_PASSIVE',tr)
        else:
            import pygame
            screen=pygame.Surface(mapping['screen_size'])
            for p in plans(mapping):
                for role in ('OPTICAL_BASELINE','OPTICAL_PROBE','OPTICAL_RECOVERY'):
                    render_probe(screen,p if role=='OPTICAL_PROBE' else None)
                    rgb=pygame.surfarray.array3d(screen).transpose(1,0,2)
                    image=np.full((96,160,3),127,np.uint8);image[6:86,8:152]=rgb[:,:,::-1]
                    tr=transition(seq);run.add(frame(image,tr),role,tr,p)
        return run.finish()
    except BaseException:
        run.finish('INTERRUPTED');raise


def geometry_audit(source, output):
    source=checked(source);output=output_path(output)
    if output.is_relative_to(source):raise ValueError('analysis must be outside source run')
    m=json.loads((source/'manifest.json').read_text())
    if m.get('schema')!=SCHEMA:raise ValueError('not an optical capture')
    hashes={str(source/'manifest.json'):digest(source/'manifest.json')};samples={}
    for s in m['samples']:
        if s.get('status')!='SAVED':continue
        p=checked(source/s['path'])
        if not p.is_relative_to(source):raise ValueError('image path escapes source')
        if digest(p)!=s['image_sha256'] or digest(p.with_suffix('.json'))!=s['metadata_sha256']:raise ValueError('source hash mismatch')
        meta=json.loads(p.with_suffix('.json').read_text())
        if (meta.get('schema')!=FRAME_SCHEMA or meta.get('role')!=s['role']
                or meta.get('projectile_truth') is not False or meta.get('live_authority') is not False
                or meta.get('shot_id') is not None or meta.get('probe_id')!=s.get('probe_id')):
            raise ValueError('optical source role/identity mismatch')
        hashes[str(p)]=s['image_sha256'];hashes[str(p.with_suffix('.json'))]=s['metadata_sha256']
        if s.get('probe_id'):samples.setdefault(s['probe_id'],{})[s['role']]=p
    results=[]
    saved_plans=m.get('probe_plan',[p['plan'] for p in m.get('probes',[])])
    for plan in saved_plans:
        group=samples.get(plan['probe_id'],{});roles=('OPTICAL_BASELINE','OPTICAL_PROBE','OPTICAL_RECOVERY')
        if all(k in group for k in roles):
            result=response_measure(*(np.load(group[k],allow_pickle=False) for k in roles),m['mapping'],plan)
        else:result=dict(status='UNAVAILABLE',reason='MISSING_SAVED_TRIPLET',geometry=dict(status='UNAVAILABLE'))
        results.append(dict(plan=plan,**result))
    output.mkdir(parents=True,exist_ok=False)
    result=dict(schema=SCHEMA,mode='geometry',synthetic=m['config']['synthetic'],source=str(source),input_hashes=hashes,
        mapping=m['mapping'],probes=results,plan_status='AVAILABLE' if saved_plans else 'UNAVAILABLE',disclaimer=DISCLAIMER)
    atomic(output/'summary.json',result)
    (output/'REPORT.md').write_text('# Saved optical geometry audit\n\n'+DISCLAIMER+'\n\nSee summary.json. Missing triplets/calibration stay UNAVAILABLE; no current camera/calibration used.\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    for mode in ('passive','probes'):
        p=sub.add_parser(mode);p.add_argument('--output-dir',type=Path,required=True)
        p.add_argument('--synthetic','--dry-run',action='store_true',dest='synthetic')
        p.add_argument('--duration',type=float,default=30);p.add_argument('--settle',type=float,default=.6)
        p.add_argument('--save-all-frames',action='store_true');p.add_argument('--seed',type=int,default=1729)
    p=sub.add_parser('geometry');p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='geometry':
        result=geometry_audit(args.input,args.output_dir)
    else:
        c=configuration(args.mode,duration_s=args.duration,settle_s=args.settle,save_all_frames=args.save_all_frames,synthetic=args.synthetic,seed=args.seed)
        if c['synthetic']:result=synthetic_run(args.output_dir,c)
        else:
            from src.engine.communication.tcp_network_handler import EventListener,send_command
            run_id=uuid.uuid4().hex
            with EventListener() as listener:
                send_command('opticalLabStart',dict(run_id=run_id,output_dir=str(output_path(args.output_dir)),config=c,issued_wall_ts=time.time()))
                try:
                    deadline=time.monotonic()+max(c['duration_s'],60*(c['settle_s']+c['frame_timeout_s']))+120
                    while True:
                        remaining=deadline-time.monotonic()
                        if remaining<=0:raise TimeoutError('optical completion deadline exceeded')
                        e=listener.wait_for_event(timeout=remaining)
                        if e.get('data',{}).get('run_id')!=run_id:continue
                        if e['event']=='opticalLab.completed':result=e['data'];break
                        if e['event']=='opticalLab.failed':raise RuntimeError(str(e['data']))
                        if e['event']=='opticalLab.presented':print('[OPTICAL]',e['data']['state'])
                except BaseException:
                    send_command('opticalLabCancel',dict(run_id=run_id));raise
    print(json.dumps(dict(status=result.get('status','ANALYZED'),output=str(args.output_dir),synthetic=result.get('synthetic',False),disclaimer=DISCLAIMER)))


if __name__=='__main__':main()
