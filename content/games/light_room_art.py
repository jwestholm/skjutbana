"""Game-local adapter for headless Blender room/parts. No live 3D renderer."""
from pathlib import Path
import hashlib
import json
import pygame
from src.engine.game_objects import ObjectGeometry

ASSET_ROOT=Path(__file__).resolve().parent/'light_room_parts_assets'


class LightRoomArt:
    def __init__(self,size):
        data=(ASSET_ROOT/'manifest.json').read_bytes()
        self.manifest=json.loads(data);self.manifest_sha256=hashlib.sha256(data).hexdigest()
        self.objects=self.manifest['objects'];self.size=size
        self.background=pygame.transform.smoothscale(pygame.image.load(str(ASSET_ROOT/'room.png')),size)
        self._raw={};self._scaled={};self._raw_hit={};self._scaled_hit={};self._raw_frames={};self._frames={}
        for name,spec in self.objects.items():
            for part,p in spec['parts'].items():
                self._raw[name,part]=pygame.image.load(str(ASSET_ROOT/p['sprite']))
                self._raw_hit[name,part]=pygame.image.load(str(ASSET_ROOT/p.get('hit_sprite',p['sprite'])))
                self._raw_frames[name,part]=[pygame.image.load(str(ASSET_ROOT/f)) for f in p['frames']]

    def geometry(self,name,anchor,height):
        spec=self.objects[name];x0,y0,x1,y1=spec['source_bbox']
        scale=height/(y1-y0);w=round((x1-x0)*scale);h=round(height)
        px,py=spec['pivot'];ax,ay=anchor['anchor']
        return ObjectGeometry(round(ax*self.size[0]-px*w),round(ay*self.size[1]-py*h),w,h)

    def prepare_round(self,items):
        self._scaled.clear();self._scaled_hit.clear();self._frames.clear()
        for item in items:
            size=(round(item['geometry'].width),round(item['geometry'].height))
            for part in self.objects[item['kind']]['parts']:
                for raw,cache in ((self._raw,self._scaled),(self._raw_hit,self._scaled_hit)):
                    scaled=pygame.transform.smoothscale(raw[item['kind'],part],size)
                    # Keep solid alpha opaque despite SDL's integer resampling.
                    alpha=pygame.surfarray.pixels_alpha(scaled);alpha[alpha>=252]=255;del alpha
                    cache[item['kind'],part,size]=scaled
                self.frames(item['kind'],part,size)  # no image I/O/scaling in a shot reaction

    def frames(self,name,part,size):
        key=(name,part,tuple(size))
        if key not in self._frames:
            spec=self.objects[name];x0,y0,x1,y1=spec['source_bbox']
            sx,sy=size[0]/(x1-x0),size[1]/(y1-y0)
            fs=(round(spec['source_size'][0]*sx),round(spec['source_size'][1]*sy))
            if (name,part) not in self._raw_frames:
                self._raw_frames[name,part]=[pygame.image.load(str(ASSET_ROOT/p)) for p in spec['parts'][part]['frames']]
            self._frames[key]=([pygame.transform.smoothscale(f,fs) for f in self._raw_frames[name,part]],
                               (-round(x0*sx),-round(y0*sy)))
        return self._frames[key]

    def render(self,surface,items,broken_ids=(),effect_ids=(),frame=None,effect_sides=None):
        surface.blit(self.background,(0,0));effect_sides=effect_sides or {}
        ordered=sorted(items,key=lambda i:i['geometry'].bottom)
        def debris(item,part,oid,animated):
            g=item['geometry'];size=(round(g.width),round(g.height))
            frames,offset=self.frames(item['kind'],part,size)
            idx=min(max(int(frame or 0),0),len(frames)-1) if animated else len(frames)-1
            image=frames[idx];x=round(g.x)+offset[0];side=effect_sides.get(oid,1)
            if side<0:
                x=round(2*g.center[0]-x-image.get_width());image=pygame.transform.flip(image,True,False)
            if self.objects[item['kind']]['parts'][part]['motion']=='fall':
                # Extra lateral impulse from the observed hit side; no change to evidence/XY.
                x+=round(g.width*.45*(idx+1)/len(frames))*side
            surface.blit(image,(x,round(g.y)+offset[1]))
        # Resting fragments never hide a still-hittable part. This is an explicit
        # presentation rule, not z-order guessed from a physical coordinate.
        for item in ordered:
            for part,oid in item['parts'].items():
                if oid in broken_ids and oid not in effect_ids:debris(item,part,oid,False)
        for item in ordered:
            g=item['geometry'];size=(round(g.width),round(g.height))
            partial=any(oid in broken_ids for oid in item['parts'].values())
            cache=self._scaled_hit if partial else self._scaled
            for part,oid in item['parts'].items():
                if oid not in broken_ids:
                    surface.blit(cache[item['kind'],part,size],(round(g.x),round(g.y)))
        # Moving fragments may cross the scene only while shot queries are closed.
        for item in ordered:
            for part,oid in item['parts'].items():
                if oid in broken_ids and oid in effect_ids:debris(item,part,oid,True)
