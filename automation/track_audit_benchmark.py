"""Repeat saved detector inputs with tracing toggled; assert coordinate equality."""
import argparse,json,statistics,time
from pathlib import Path
from src.engine.camera.hit_scanner import HitScanner,AudioShotEvent
from src.engine.shot_track_v2226 import update_tracks_frame_unique_v2226 as update
from src.engine.track_audit import capture
from src.engine.offline.track_replay import reconstruct_legacy

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=10)
 p.add_argument('--tag-producers',action='store_true',help='Apply current async producer metadata to verified legacy event batches')
 a=p.parse_args()
 inputs=[json.loads(f.read_text()) for f in sorted((a.root/'shots').glob('*/trace.json'))];rows=[]
 for d in inputs:
  dec=d['decision_input'];reconstructed=reconstruct_legacy(d);batches=reconstructed['associations'];durations={False:[],True:[]};sizes=[];pretty_sizes=[];encode_times=[];xy=[]
  for repeat in range(a.repeats):
   for enabled in ((False,True) if repeat%2==0 else (True,False)):
    s=HitScanner();s.physical_trace_capture_enabled=enabled;s.last_trace_pipeline_shot_id=d['shot_id'];e=AudioShotEvent(d['shot_id'],d['peak_ts'],d['peak_ts']);s.audio_events.append(e);s._next_track_id=reconstructed['reconstruction']['id_offset']+1
    start=time.perf_counter()
    for batch in batches:
     candidates=[dict(r['candidate']) for r in batch['records']]
     if a.tag_producers:
      for c in candidates:c['v2224_producer_shot_id']=d['shot_id']
     update(s,candidates,batch['frame_ts'])
    chosen=s._best_track_for_event(e)
    snap=capture(s,e,chosen) if enabled else None
    durations[enabled].append((time.perf_counter()-start)*1000);xy.append((chosen.camera_x,chosen.camera_y))
    if snap:
     sizes.append(len(json.dumps(snap,separators=(',',':'))))
     t0=time.perf_counter();pretty_sizes.append(len(json.dumps(snap,indent=2,sort_keys=True)));encode_times.append((time.perf_counter()-t0)*1000)
  assert len(set(xy))==1,'tracing changed selected XY'
  assert xy[0]==(dec['deterministic_selection']['camera_x'],dec['deterministic_selection']['camera_y']),'replay changed live XY'
  rows.append(dict(shot=d['shot_id'],off_ms=statistics.median(durations[False]),on_ms=statistics.median(durations[True]),bytes=statistics.mean(sizes),pretty_bytes=statistics.mean(pretty_sizes),json_encode_ms=statistics.mean(encode_times),tracks=len(s._active_tracks),records=sum(len(b['records']) for b in snap['associations']) if snap else sum(len(b['records']) for b in batches)))
 report=dict(repeats=a.repeats,tag_producers=a.tag_producers,rows=rows,mean_off_ms=statistics.mean(r['off_ms'] for r in rows),mean_on_ms=statistics.mean(r['on_ms'] for r in rows),mean_json_bytes=statistics.mean(r['bytes'] for r in rows),mean_pretty_json_bytes=statistics.mean(r['pretty_bytes'] for r in rows),mean_encode_ms=statistics.mean(r['json_encode_ms'] for r in rows),projected_pretty_bytes={str(n):int(n*statistics.mean(r['pretty_bytes'] for r in rows)) for n in (10,20,100)},projected_bytes={str(n):int(n*statistics.mean(r['bytes'] for r in rows)) for n in (10,20,100)})
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.open('x').write(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
