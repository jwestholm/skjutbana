"""Attach natural Cycles room plates to the unchanged R6 target sprite bundle.

python3 -m automation.build_pang_rooms BASELINE RENDER_ROOT NEW_BUNDLE
"""
import argparse,hashlib,json,shutil
import numpy as np
from PIL import Image
from automation.light_room_palette import ensure_floor,stats
from pathlib import Path


def build(baseline,renders,out):
    baseline,renders,out=map(Path,(baseline,renders,out))
    if out.exists():raise FileExistsError(out)
    shutil.copytree(baseline,out)
    m=json.loads((baseline/'manifest.json').read_text());m['baseline_manifest_sha256']=hashlib.sha256((baseline/'manifest.json').read_bytes()).hexdigest()
    # Colored enamel/glaze display variants preserve rendered shading, alpha and
    # exact silhouettes. Only targets are processed; room plates remain raw.
    glazes={'vase':(65,225,155),'jar':(240,220,75),'bowl':(70,220,240),'bottle':(50,225,190),'plant':(235,220,145),'lamp':(245,225,90),'mug':(125,225,240),'clock':(240,225,160),'books':(245,225,160)}
    for kind,base in glazes.items():
        for part_id,part in m['objects'][kind]['parts'].items():
            tint=(95,230,105) if kind=='plant' and part_id=='foliage' else base
            for name in dict.fromkeys([part['sprite'],part['hit_sprite']]+part['frames']):
                image=np.asarray(Image.open(out/name).convert('RGBA')).copy()
                rgb=image[:,:,:3].astype(float)*np.array(tint)[None,None,:]/230.
                image[:,:,:3]=ensure_floor(rgb,floor=128.)
                Image.fromarray(image).save(out/name)
    m['target_glazes']=dict(colors=glazes,scope='target material/color variants only; PBR shading and alpha retained')
    rooms=[]
    for template,label in [('living','Vardagsrum'),('kitchen','Kök'),('study','Arbetsrum')]:
        for furnishing in (0,1):
            name=template+'_'+str(furnishing);raw=renders/name
            image='room_'+name+'.png';shutil.copyfile(raw/'room.png',out/image)
            anchors=json.loads((raw/'layout.json').read_text())['anchors']
            if len(anchors)<12 or any(a.get('unit_height',0)<=0 for a in anchors):raise ValueError('Incomplete room layout '+name)
            rooms.append(dict(id=name,template=template,label=label,furnishing=furnishing,image=image,anchors=anchors,
                raw_sha256={n:hashlib.sha256((raw/n).read_bytes()).hexdigest() for n in ('room.png','layout.json')}))
    shutil.copyfile(out/rooms[0]['image'],out/'room.png')
    previous_report=baseline/'palette_report.json'
    (out/'palette_report.json').write_text(json.dumps(dict(schema='pang-local-palette-7',
        baseline_report_sha256=hashlib.sha256(previous_report.read_bytes()).hexdigest() if previous_report.exists() else None,
        room_scope='natural Cycles plates, no global image lift or floor',
        target_scope='paint/enamel variants of the inherited R6 target PNGs; alpha unchanged',
        local_background=dict(floor_dn=124,margin_px=12,feather_px=32,shape='visible silhouette'),
        room_stats={r['id']:stats(Image.open(out/r['image'])) for r in rooms}),indent=2)+'\n')
    m['model_display_exposure_scope']='inherited R6 target sprites only; none on new room plates'
    m['objects_per_room']=10
    m['rooms']=rooms;m['anchors']=rooms[0]['anchors'];m['room_plate']='natural_original_PBR_no_global_floor_or_lift'
    m['palette']['scope']='target_sprites_only; background floor124 only within active target silhouette+12px, feather32px'
    m['room_recipes_sha256']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        'automation/assets/pang_room_variants.py','automation/assets/render_light_room_parts.py',
        'automation/assets/pang_polyhaven.py','automation/build_pang_rooms.py')}
    m['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.png'))}
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    with (out/'SOURCES.md').open('a') as f:f.write('\nR7: six natural PBR room plates, three templates/two furnishings each.\nNo global camera-floor postprocessing or display gain on room plates.\nOriginal R6 target sprites and UV silhouettes retained; local safety is applied\nonly to active-target surroundings by content/games/light_room_safety.py.\nNew room recipe hashes are separate from inherited target recipe hashes.\n')
    return m

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline');p.add_argument('renders');p.add_argument('output');a=p.parse_args()
    m=build(a.baseline,a.renders,a.output);print('Packaged',len(m['rooms']),'rooms')
