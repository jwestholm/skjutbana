"""No-shoot geometry and A/V receipt diagnostics using existing engine owners.

Never starts HitScanner, alters settings/calibration, binds physical shots, or
claims device acquisition timestamps. SPACE begins after operator positioning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import pygame

from config import SCREEN_WIDTH, SCREEN_HEIGHT
from src.engine.camera.aruco_calibrator import (ArucoCalibrator, geometry_context,
    compare_geometry_setup, calibration_fit_quality)
from src.engine.camera.camera_manager import CameraManager
from src.engine.audio.audio_peak_detector import AudioPeakDetector
from src.engine.settings import load_viewport_rect


def write_new(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def next_creation_position(requested, desired_client, observed_client):
    """Compensate measured window-manager placement, never camera residuals.

    SDL's X11 creation request and the eventual decorated client origin can
    differ. One bounded recreation uses this run's readback, not a saved -37.
    The resulting client position still MUST match calibration before capture.
    """
    return [int(r+d-o) for r,d,o in zip(requested,desired_client,observed_client)]


def create_geometry_window(size, desired_client):
    attempts=[]
    requested=list(map(int,desired_client)) if desired_client is not None else None
    for attempt in range(2):
        if requested is not None:
            os.environ['SDL_VIDEO_WINDOW_POS']=','.join(map(str,requested))
        screen=pygame.display.set_mode(size)
        screen.fill((25,25,25));pygame.display.flip()
        # Allow the WM to decorate/reparent and SDL to consume ConfigureNotify.
        # No camera or microphone owner is running during this bounded step.
        previous=None;stable_since=time.monotonic();deadline=stable_since+1.5
        while time.monotonic()<deadline:
            events=pygame.event.get()
            if any(e.type==pygame.QUIT for e in events):raise RuntimeError('operator aborted')
            current=geometry_context().get('window') or {}
            actual=current.get('sdl_desktop_position')
            if actual!=previous:previous=actual;stable_since=time.monotonic()
            if actual is not None and time.monotonic()-stable_since>=.25:break
            pygame.time.wait(10)
        measured=geometry_context(native_window=True).get('window') or {}
        actual=measured.get('sdl_desktop_position')
        attempts.append(dict(creation_request=requested,observed_window=measured))
        if desired_client is None or actual==list(desired_client) or attempt==1 or actual is None:
            break
        requested=next_creation_position(requested,desired_client,actual)
        # Preserve the operator's safe env-before-set_mode approach. Do not use
        # Window.position assignment or move a live display-owned wrapper.
        pygame.display.quit();pygame.display.init()
        pygame.display.set_caption('Skjutbana — no-shoot foundation check')
    return screen,dict(desired_client_position=desired_client,attempts=attempts,
        verified=desired_client is not None and actual==list(desired_client),
        correction_space='WINDOW_CREATION_ONLY_NOT_CAMERA_OR_HOMOGRAPHY')


def geometry_report(context, ending, frames, tolerance=3., placement=None):
    setup=context.get('calibration',{}).get('setup_at_calibration',{})
    comparison=compare_geometry_setup(setup,context)
    stable=compare_geometry_setup(context,ending)
    quality=calibration_fit_quality(context.get('calibration',{}),tolerance)
    frame_size=context.get('calibration',{}).get('camera_frame_size')
    shape_mismatch=any(f.get('acquisition',{}).get('frame_shape',[])[:2]!=list(reversed(frame_size or []))
                       for f in frames if f.get('acquisition',{}).get('frame_shape'))
    if comparison['status']!='MATCH' or stable['status']!='MATCH' or shape_mismatch:
        status='SETUP_MISMATCH'
    elif quality['status']!='PASS':status='CALIBRATION_FIT_POOR'
    elif len(frames)<3 or any(f['status']!='PASS' for f in frames):status='PHYSICAL_GEOMETRY_FAIL'
    else:status='PASS'
    return dict(schema='foundation-geometry-1',status=status,context=context,ending_context=ending,
        setup_comparison=comparison,setup_stability=stable,calibration_fit=quality,
        unchanged_setup=stable['status']=='MATCH',camera_shape_mismatch=shape_mismatch,
        measurement_valid_for_calibrated_setup=comparison['status']=='MATCH' and stable['status']=='MATCH' and not shape_mismatch,
        frames=frames,placement=placement,measured_tolerance_camera_px=tolerance,
        physical_hit_accuracy_claim=False)


def analyse_sync(audio_rows, visual_rows, commands):
    """Receipt-relative response to paired flash/tone commands, not ADC/exposure.

    Known output windows isolate the controlled responses. Missing or ambiguous
    signals fail characterization instead of becoming zero-latency observations.
    """
    pairs=[]
    for command in commands:
        at=command['flip_complete_monotonic']
        before_a=[r['rms'] for r in audio_rows if at-1.5<r['read_complete_monotonic']<at-.1]
        before_v=[r['mean'] for r in visual_rows if at-1.5<r['read_complete_monotonic']<at-.1]
        a=[r for r in audio_rows if at<=r['read_complete_monotonic']<=at+3]
        v=[r for r in visual_rows if at<=r['read_complete_monotonic']<=at+3]
        if not before_a or not before_v or not a or not v:
            pairs.append(dict(command=command,status='MISSING_BASELINE_OR_RESPONSE'));continue
        # Relative half-amplitude crossings describe a commanded step, not a
        # gunshot/audio trigger threshold. Retain all curves for independent review.
        ab=float(np.median(before_a));vb=float(np.median(before_v))
        ar=max(r['rms'] for r in a)-ab;vr=max(r['mean'] for r in v)-vb
        an=float(np.std(before_a));vn=float(np.std(before_v))
        if ar<=max(1e-4,6*an) or vr<=max(3.,6*vn):
            pairs.append(dict(command=command,status='RESPONSE_NOT_DISTINCT',audio_amplitude=ar,visual_amplitude=vr));continue
        aa=next(r for r in a if r['rms']>=ab+ar*.5)
        vv=next(r for r in v if r['mean']>=vb+vr*.5)
        # Reject gaps: a single transition estimate cannot bridge lost records.
        ordered_a=[r for r in audio_rows if at-.1<r['read_complete_monotonic']<=aa['read_complete_monotonic']]
        contiguous=all(y['sample_start_index']==x['sample_end_index_exclusive'] for x,y in zip(ordered_a,ordered_a[1:]))
        pairs.append(dict(command=command,status='OBSERVED' if contiguous else 'PCM_GAP',
            audio_receipt_monotonic=aa['read_complete_monotonic'],visual_receipt_monotonic=vv['read_complete_monotonic'],
            audio_minus_visual_ms=1000*(aa['read_complete_monotonic']-vv['read_complete_monotonic']),
            command_to_audio_ms=1000*(aa['read_complete_monotonic']-at),
            command_to_visual_ms=1000*(vv['read_complete_monotonic']-at),
            audio_sample_start_index=aa['sample_start_index'],visual_sequence=vv['sequence']))
    values=[r['audio_minus_visual_ms'] for r in pairs if r['status']=='OBSERVED']
    return dict(schema='foundation-sync-1',status='CHARACTERIZED_RECEIPT_ONLY' if len(values)>=3 and len(values)==len(commands) else 'INCOMPLETE',
        pairs=pairs,median_audio_minus_visual_ms=float(np.median(values)) if values else None,
        p05_ms=float(np.percentile(values,5)) if values else None,p95_ms=float(np.percentile(values,95)) if values else None,
        range_ms=float(np.ptp(values)) if values else None,
        calibrated_device_offset=False,pre_selection_demonstrated=False,
        limitations='Includes output scheduling, projector/speaker latency, propagation and input buffering. Receipt is not exposure/ADC time; no fixed correction is installed.')


def run_live(mode, output, *, tolerance=3., window_position=None, repetitions=4):
    if mode not in ('geometry', 'sync'):
        raise ValueError('unknown no-shoot diagnostic')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('positive camera-pixel tolerance required')
    if mode == 'sync' and not 3 <= repetitions <= 12:
        raise ValueError('sync requires 3..12 paired observations')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)

    if window_position is None:
        from src.engine.input.hit_input import hit_input
        setup=hit_input.calibration_data.get('setup_at_calibration',{})
        window_position=(setup.get('window') or {}).get('sdl_desktop_position')

    # SDL/Pygame-supported window placement. This must be supplied before
    # display.set_mode(); do not wrap the pygame.display-owned SDL window with
    # the experimental pygame._sdl2 Window API.
    if window_position:
        x, y = map(int, window_position)
        os.environ['SDL_VIDEO_WINDOW_POS'] = f'{x},{y}'

    pygame.init()
    camera=CameraManager();audio=AudioPeakDetector()
    try:
        pygame.display.set_caption('Skjutbana — no-shoot foundation check')
        screen,placement=create_geometry_window((SCREEN_WIDTH,SCREEN_HEIGHT),window_position)
        initial=geometry_context(native_window=True)
        preflight=geometry_report(initial,initial,[],tolerance,placement)
        if preflight['status']=='SETUP_MISMATCH':
            write_new(output/'report.json',preflight)
            print(json.dumps(dict(status=preflight['status'],setup_comparison=preflight['setup_comparison']),indent=2))
            return preflight
        if not camera.start():
            raise RuntimeError(camera.last_error)
        cal=ArucoCalibrator(load_viewport_rect())
        if mode=='geometry' and not cal.available:
            raise RuntimeError('Existing ArUco support unavailable')
        if mode=='sync':
            if not audio.start():raise RuntimeError(audio.last_error or 'audio unavailable')
            if pygame.mixer.get_init() is None:raise RuntimeError('output mixer unavailable')
            rate,fmt,channels=pygame.mixer.get_init()
            if fmt!=-16:raise RuntimeError('diagnostic cue requires signed 16-bit mixer')
            t=np.arange(round(rate*.15))/rate
            wave=(np.sin(2*np.pi*1100*t)*np.sin(np.pi*np.arange(len(t))/len(t))**2*6500).astype(np.int16)
            pcm=wave if channels==1 else np.repeat(wave[:,None],channels,axis=1)
            cue=pygame.sndarray.make_sound(np.ascontiguousarray(pcm))
        font=pygame.font.Font(None,26);clock=pygame.time.Clock()
        begun=None;context=None;last_seq=0;last_audio=0;validations=[];visual=[];sounds=[];commands=[]
        capture_end=None;next_flash=None;flash_until=0.;running=True
        print('NO SHOOTING. Position this client on the projector exactly as the game. SPACE starts; ESC aborts.')
        with (output/'pcm.s16le').open('xb') as pcm_file:
            while running:
                for event in pygame.event.get():
                    if event.type==pygame.QUIT or (event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE):
                        raise RuntimeError('operator aborted; no pass report')
                    if event.type==pygame.KEYDOWN and event.key==pygame.K_SPACE and begun is None:
                        begun=time.monotonic();context=geometry_context(native_window=True)
                        if compare_geometry_setup(initial,context)['status']!='MATCH':
                            report=geometry_report(initial,context,[],tolerance,placement)
                            write_new(output/'report.json',report)
                            print('SETUP_MISMATCH: window/setup changed before capture')
                            return report
                        write_new(output/'geometry_context.json',context)
                        if mode=='geometry':capture_end=begun+7.
                        else:next_flash=begun+2.;capture_end=begun+2.+repetitions*5.
                now=time.monotonic()
                if begun is None:
                    screen.fill((25,25,25));screen.blit(font.render('NO SHOOTING — move window, then SPACE',True,(255,255,255)),(25,60))
                elif mode=='geometry':cal.render_markers(screen)
                else:
                    fire=next_flash is not None and now>=next_flash and len(commands)<repetitions
                    if fire:flash_until=now+.65
                    screen.fill((240,240,240) if now<flash_until else (30,30,30))
                flip_start=time.monotonic();pygame.display.flip();flip_end=time.monotonic()
                if begun is not None and mode=='sync' and fire:
                    playback=audio.play_feedback(cue,owner='no_shoot_sync',shot_id=len(commands)+1,cue='original_1100hz_pulse')
                    commands.append(dict(flip_start_monotonic=flip_start,flip_complete_monotonic=flip_end,
                        audio_play_call_complete_monotonic=time.monotonic(),playback=playback))
                    next_flash+=5.
                camera.update()
                for cf in camera.get_ring_snapshot():
                    meta=cf.acquisition
                    if not meta or meta['sequence']<=last_seq:continue
                    last_seq=meta['sequence']
                    if begun is None:continue
                    if mode=='geometry':
                        if now-begun<2. or len(validations)>=5 or meta['read_complete_monotonic']<begun+2.:continue
                        if validations and meta['read_complete_monotonic']-validations[-1]['acquisition']['read_complete_monotonic']<.4:continue
                        check=cal.validate_existing(cf.frame_bgr,tolerance_camera_px=tolerance)
                        check['acquisition']=dict(meta);validations.append(check)
                        np.save(output/f'geometry_frame_{len(validations):02d}.npy',cf.frame_bgr,allow_pickle=False)
                    else:
                        gray=cv2.cvtColor(cf.frame_bgr,cv2.COLOR_BGR2GRAY)
                        # Same physical viewport through existing inverse, not full
                        # room brightness. A stale geometry makes this inconclusive.
                        pts=[cal.viewport_rect.topleft,cal.viewport_rect.topright,cal.viewport_rect.bottomright,cal.viewport_rect.bottomleft]
                        from src.engine.input.hit_input import hit_input
                        cam=[hit_input._canonical_screen_to_camera(*p) for p in pts]
                        if any(p is None for p in cam):raise RuntimeError('mapping unavailable')
                        mask=np.zeros(gray.shape,np.uint8);cv2.fillConvexPoly(mask,np.int32(cam),1)
                        if not np.any(mask):raise RuntimeError('predicted viewport is outside camera frame; verify geometry first')
                        visual.append(dict(meta,mean=float(np.mean(gray[mask>0]))))
                        np.save(output/f'sync_frame_{meta["sequence"]:06d}.npy',cv2.resize(gray,(960,540)),allow_pickle=False)
                if mode=='sync':
                    for meta,pcm in audio.acquisition_blocks_since(last_audio):
                        last_audio=meta['sequence']
                        if begun is None:continue
                        samples=np.frombuffer(pcm,np.int16).astype(np.float32)/32768
                        sounds.append(dict(meta,rms=float(np.sqrt(np.mean(samples*samples))),peak=float(np.max(np.abs(samples))),
                            file_offset_bytes=pcm_file.tell()))
                        pcm_file.write(pcm)
                if begun is not None and now>=capture_end:running=False
                clock.tick(30)
        ending=geometry_context(native_window=True);unchanged=context['identity_sha256']==ending['identity_sha256']
        if mode=='geometry':
            report=geometry_report(context,ending,validations,tolerance,placement)
        else:
            report=analyse_sync(sounds,visual,commands);report.update(context=context,unchanged_setup=unchanged)
            if not unchanged:report['status']='INCOMPLETE_SETUP_CHANGED'
            write_new(output/'audio_blocks.json',sounds);write_new(output/'visual_frames.json',visual);write_new(output/'commands.json',commands)
        write_new(output/'report.json',report)
        write_new(output/'hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
        print(json.dumps({k:v for k,v in report.items() if k not in ('context','ending_context','frames','pairs')},indent=2))
        return report
    finally:
        camera.stop();audio.stop();pygame.quit()


def foundation_gate(geometry, timing, pre_selection, current_context=None):
    """Explicit prerequisite report; never turns receipt-only sync into PRE proof."""
    reasons=[]
    if geometry.get('schema')!='foundation-geometry-1' or geometry.get('status')!='PASS':reasons.append('geometry not physically verified')
    if not geometry.get('unchanged_setup'):reasons.append('geometry setup changed during validation')
    context=geometry.get('context',{})
    if current_context is not None and context.get('identity_sha256')!=current_context.get('identity_sha256'):
        reasons.append('current setup differs from measured geometry')
    if timing.get('status')!='CHARACTERIZED_RECEIPT_ONLY' or not timing.get('unchanged_setup'):reasons.append('A/V receipt offset/jitter not characterized')
    if timing.get('context',{}).get('identity_sha256')!=context.get('identity_sha256'):reasons.append('timing and geometry setups differ')
    if pre_selection.get('status')!='DEMONSTRATED_CAUSAL_PRE_SELECTION' or not pre_selection.get('no_manual_probe_input'):
        reasons.append('causal event-specific PRE selection not demonstrated')
    # No production PRE policy was changed in this development step. Evidence
    # about an offline alternative cannot authorize the still-old live policy.
    reasons.append('live OBJECT PRE policy still uses receipt-time audio boundary; not validated')
    return dict(status='BLOCKED',geometry=geometry.get('status'),timing=timing.get('status'),reasons=reasons,
        ready_to_shoot=False,no_physical_capture_started=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    for name in ('geometry','sync'):
        p=sub.add_parser(name);p.add_argument('--output',type=Path,required=True)
        p.add_argument('--window-position',type=int,nargs=2)
        if name=='geometry':p.add_argument('--tolerance-camera-px',type=float,default=3.)
        else:p.add_argument('--repetitions',type=int,default=4)
    gate=sub.add_parser('preflight');gate.add_argument('--geometry',type=Path,required=True)
    gate.add_argument('--timing',type=Path,required=True);gate.add_argument('--pre-selection',type=Path,required=True)
    a=parser.parse_args()
    if a.mode=='preflight':
        r=foundation_gate(*(json.loads(p.read_text()) for p in (a.geometry,a.timing,a.pre_selection)))
        print(json.dumps(r,indent=2));return 2
    report=run_live(a.mode,a.output,tolerance=getattr(a,'tolerance_camera_px',3.),
        window_position=a.window_position,repetitions=getattr(a,'repetitions',4))
    return 0 if report['status'] in ('PASS','CHARACTERIZED_RECEIPT_ONLY') else 2


if __name__=='__main__':raise SystemExit(main())
