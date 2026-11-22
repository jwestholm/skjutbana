"""Package Blender visible-part masks and break stages; refuses existing output bundles."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
from automation.build_light_room_assets import brighten


def polygons(mask):
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    clean=np.zeros(mask.shape,np.uint8)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]>=4:clean[labels==i]=255
    contours,_=cv2.findContours(clean,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not 1<=len(contours)<=8:raise ValueError('Part needs 1..8 connected silhouettes')
    rebuilt=np.zeros_like(clean);cv2.drawContours(rebuilt,contours,-1,255,-1)
    if not np.array_equal(rebuilt,clean):raise ValueError('Part silhouette contains an occlusion hole; fix the model')
    h,w=clean.shape
    return clean,[[[float(x)/w,float(y)/h] for x,y in c[:,0,:]] for c in contours]


def build(raw,output,*,realistic=False):
    from automation.light_room_palette import protect_render
    prepare = protect_render if realistic else brighten
    raw,output=Path(raw),Path(output);output.mkdir(parents=True,exist_ok=False)
    info=json.loads((raw/'parts.json').read_text())
    m=dict(scene=info.get('scene','atelier'),schema='light-room-parts-2',blender_version=info['blender_version'],
           alpha_threshold=128,objects={},anchors=json.loads((raw/'layout.json').read_text())['anchors'],
           source='automation/assets/render_light_room_parts.py',license='CC0-1.0',
           material_pipeline='render_preserving' if realistic else 'legacy_brighten',
           note='Functional parts and sectors are authored manually; Cell Fracture is an optional offline replacement.')
    m['raw_inputs_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(raw.iterdir()) if p.is_file() and p.suffix in ('.png','.json')}
    if (raw/'source_provenance.json').exists():m['raw_provenance']=json.loads((raw/'source_provenance.json').read_text())
    for name,spec in info['objects'].items():
        original=prepare(Image.open(raw/f'{name}.png'));full=np.array(original)
        labels=np.asarray(Image.open(raw/f'{name}_ids.png').convert('RGBA'))
        ownership=labels[:,:,:3].argmax(axis=2)
        relabelled=0
        union=((full[:,:,3]>=128)&(labels[:,:,3]>=128)).astype(np.uint8)*255
        # A two-sample flat-ID render can leave a 1-pixel island of another label.
        # Repair only enclosed <=4px label specks; never fill real transparent holes.
        # Pixels retain their original shaded RGB and receive exactly one part owner.
        for idx in range(len(spec['parts'])):
            inverse=np.asarray(~((ownership==idx)&(union>0)),dtype=np.uint8)
            cv2.floodFill(inverse,None,(0,0),2)
            count,cc,stats,_=cv2.connectedComponentsWithStats((inverse==1).astype(np.uint8))
            for k in range(1,count):
                island=cc==k
                if stats[k,cv2.CC_STAT_AREA]<=4 and np.all(union[island]>0):
                    relabelled+=int(island.sum());ownership[island]=idx
        x,y,w,h=cv2.boundingRect(union);bbox=(x,y,x+w,y+h)
        entry=dict(id_antialias_relabelled_pixels=relabelled,label={'vase':'Blomvas','jar':'Kruka','bowl':'Skål','bottle':'Flaska','lamp':'Lampa','tv':'TV','books':'Böcker','plant':'Krukväxt','clock':'Klocka','mug':'Mugg'}[name],
                   source_bbox=list(bbox),source_size=list(original.size),
                   pivot=[(info['pivot'][0]-x)/w,(info['pivot'][1]-y)/h],parts={})
        for idx,part in enumerate(spec['parts']):
            mask=((ownership==idx)&(union>0))[y:y+h,x:x+w]
            mask,poly=polygons(mask)
            pixels=full[y:y+h,x:x+w].copy();pixels[:,:,3]=mask
            base=f'{name}_{part["id"]}'
            Image.fromarray(pixels).save(output/(base+'.png'));Image.fromarray(mask).save(output/(base+'_mask.png'))
            paths=[]
            for j,f in enumerate(part['frames']):
                image=prepare(Image.open(raw/f))
                if part['motion']=='shatter':
                    dust=Image.new('RGBA',image.size);draw=ImageDraw.Draw(dust);t=(j+1)/len(part['frames'])
                    for k in range(10):
                        cx=x+w*.5+math.sin(k*2.4)*w*.3*t;cy=y+h*.65-18*t+math.cos(k)*15*t;r=5+8*t
                        draw.ellipse((cx-r,cy-r,cx+r,cy+r),fill=(245,239,200,round(55*(1-t))))
                    image=Image.alpha_composite(image,dust.filter(ImageFilter.GaussianBlur(5)))
                image.save(output/f);paths.append(f)
            entry['parts'][part['id']]=dict(sprite=base+'.png',mask=base+'_mask.png',polygons=poly,
                frames=paths,motion=part['motion'],detach=part['detach'],points=part['points'],cue=part['cue'])
        m['objects'][name]=entry
    room=protect_render(Image.open(raw/'room.png')) if realistic else brighten(Image.open(raw/'room.png'),floor=160)
    room.convert('RGB').save(output/'room.png')
    m['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('*.png'))}
    m['recipes_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('automation/build_light_room_parts.py'),Path(m['source'])]}
    (output/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (output/'SOURCES.md').write_text('''# Asset provenance\n\nAll shipped meshes, room geometry, part masks and animations are original procedural\nproject assets generated by automation/assets/render_light_room_parts.py.\nNo Kenney/Poly Haven/Quaternius model or copyrighted sample is included.\nNo external accounts, downloads or live Blender scene were used.\nGenerated assets only: CC0-1.0; see LICENSE.txt. Existing project source code is not relicensed.\n\nFree external alternatives (not included), checked 2026-11-22:\n- Kenney Furniture Kit: https://kenney.nl/assets/furniture-kit — CC0.\n- Poly Haven Jug 01, Kuutti Siitonen: https://polyhaven.com/a/jug_01 — CC0;\n  https://polyhaven.com/license .\n- Quaternius Ultimate Furniture: https://quaternius.com/packs/ultimatefurniture.html — CC0.\n- Cell Fracture: https://extensions.blender.org/add-ons/cell-fracture/ — GPL-3.0-or-later\n  for the tool, not a model licence. Not installed or copied into the game.\n\nCurrent shards use manually authored functional parts and mesh sectors.\nThe manifest lists hashes for rendered files and exact source recipes.\n''')
    (output/'LICENSE.txt').write_text('Original generated room geometry/render outputs, part sprites, masks and animation frames: CC0 1.0 Universal.\nhttps://creativecommons.org/publicdomain/zero/1.0/\nThis dedication covers these newly generated assets, not existing project source code or Blender/third-party software.\n')
    return m

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('raw');p.add_argument('output');p.add_argument('--realistic',action='store_true');a=p.parse_args()
    m=build(a.raw,a.output,realistic=a.realistic);print('Packaged',len(m['objects']),'types',sum(len(s['parts']) for s in m['objects'].values()),'parts',len(m['files_sha256']),'images')
