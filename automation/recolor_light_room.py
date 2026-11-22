"""Package a sprite bundle with the R3 palette or preserved R4 materials; geometry stays fixed.
python3 -m automation.recolor_light_room BASELINE_BUNDLE NEW_BUNDLE
Preserve BASELINE_BUNDLE (e.g. git archive of the previous revision); no overwrites.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
from PIL import Image
from automation.light_room_palette import colorize,stats,EXPORT_FLOOR,protect_render,hit_render


def recolor(source,output,*,realistic=False):
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=False)
    original=(source/'manifest.json').read_bytes();m=copy.deepcopy(json.loads(original))
    if m.get('schema')!='light-room-parts-2':raise ValueError('Use the frozen v2 baseline, not an already recolored bundle')
    for name,digest in m['files_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:raise ValueError('Baseline asset hash mismatch: '+name)
    # Preserve geometry, part ownership, frames and license alongside the new palette.
    for p in source.iterdir():
        if p.is_file():shutil.copyfile(p,output/p.name)
    report=dict(kind='DISPLAY_PROXY_ONLY_NOT_PHYSICAL_VALIDATION',render_preserving=realistic,floor=EXPORT_FLOOR,
                baseline_manifest_sha256=hashlib.sha256(original).hexdigest(),objects={})
    for kind,entity in m['objects'].items():
        rows={}
        for part,spec in entity['parts'].items():
            baseline=Image.open(source/spec['sprite']).convert('RGBA')
            normal=protect_render(baseline) if realistic else colorize(baseline,kind,part=part);normal.save(output/spec['sprite'])
            hit_name=Path(spec['sprite']).stem+'_hit.png';hit=(normal.copy() if m.get('scene')=='living_room' else hit_render(baseline,kind)) if realistic else colorize(baseline,kind,part=part,hit=True);hit.save(output/hit_name)
            spec['hit_sprite']=hit_name
            for name in spec['frames']:
                (protect_render(Image.open(source/name)) if realistic else colorize(Image.open(source/name),kind,part=part)).save(output/name)
            rows[part]=dict(before=stats(baseline),normal=stats(normal),hit=stats(hit))
        report['objects'][kind]=rows
    room_before=Image.open(source/'room.png').convert('RGBA');room=protect_render(room_before) if realistic else colorize(room_before)
    room.convert('RGB').save(output/'room.png');report['room']=dict(before=stats(room_before),after=stats(room))
    m['schema']='light-room-parts-3'
    m['palette']=dict(model='37+16R^2.2+115G^2.2+35B^2.2; RGB normalized 0..1',floor=EXPORT_FLOOR,
                      render_preserving=realistic,baseline_manifest_sha256=report['baseline_manifest_sha256'],hit_state='missing_parts_and_debris' if m.get('scene')=='living_room' else 'contrasting_remaining_parts')
    m['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('*.png'))}
    m['palette_recipes_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path('automation/recolor_light_room.py'),Path('automation/light_room_palette.py')]}
    (output/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (output/'palette_report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output/'SOURCES.md').open('a') as f:
        f.write('\nPalette packaging: automation.recolor_light_room; render_preserving='+str(realistic)+'.\nBaseline alpha/masks/geometry are preserved. Generated sprites retain CC0-1.0.\nSee palette_report.json; DN values are predictions, not physical camera validation.\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('output');p.add_argument('--realistic',action='store_true');a=p.parse_args()
    r=recolor(a.source,a.output,realistic=a.realistic);print(json.dumps(r['room'],indent=2))
