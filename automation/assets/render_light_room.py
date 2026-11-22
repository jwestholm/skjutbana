"""Reproducible headless Blender source. Never attach to an interactive Blender.
blender --background --factory-startup --python automation/assets/render_light_room.py -- --out /tmp/light_room_raw
Then: python3 -m automation.build_light_room_assets RAW_DIR ASSET_DIR
"""
import argparse
import math
from pathlib import Path
import sys
import bpy
from mathutils import Vector

p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--room-only',action='store_true')
a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
scene=bpy.context.scene
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=12
scene.cycles.use_denoising=True
scene.render.threads_mode='FIXED';scene.render.threads=2
scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA'
scene.render.film_transparent=True
scene.view_settings.view_transform='Standard';scene.view_settings.look='None'
scene.world.color=(.8,.8,.8)

def material(name,color):
    m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True
    bs=m.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=(*color,1)
    bs.inputs['Roughness'].default_value=.7
    bs.inputs['Emission Color'].default_value=(*color,1);bs.inputs['Emission Strength'].default_value=.35
    return m

def camera(loc,target,scale):
    bpy.ops.object.camera_add(location=loc);c=bpy.context.object;c.rotation_euler=(Vector(target)-c.location).to_track_quat('-Z','Y').to_euler();c.data.type='ORTHO';c.data.ortho_scale=scale;scene.camera=c
    return c
cam=camera((3,-9,4.5),(0,0,1.1),3.9)
for loc,power,size in [((-3,-4,7),650,5),((4,-1,5),400,4)]:
    bpy.ops.object.light_add(type='AREA',location=loc);ob=bpy.context.object;ob.data.energy=power;ob.data.shape='DISK';ob.data.size=size;ob.rotation_euler=(Vector((0,0,1))-ob.location).to_track_quat('-Z','Y').to_euler()

profiles={
 'vase':[(0,.43),(.08,.58),(.3,.78),(.7,.85),(1.15,.74),(1.5,.48),(1.72,.35),(1.9,.36),(2.05,.46)],
 'jar':[(0,.51),(.12,.73),(.4,.86),(.9,.91),(1.35,.86),(1.68,.65),(1.78,.50),(1.9,.52)],
 'bowl':[(.12,.38),(.24,.52),(.45,.71),(.75,.89),(1.05,1.02),(1.25,1.07),(1.33,1.06)],
}
colors={'vase':(.40,.87,.79),'jar':(.83,.87,.40),'bowl':(.42,.82,.90)}
scene.render.resolution_x=320;scene.render.resolution_y=384
for name,profile in ([] if a.room_only else profiles.items()):
    # A closed, smooth solid of revolution, no fine handles/rim lettering.
    prof=[(profile[0][0],0)]+profile+[(profile[-1][0],0)]
    count=64;verts=[(r*math.cos(2*math.pi*j/count),r*math.sin(2*math.pi*j/count),z) for z,r in prof for j in range(count)]
    faces=[];face_groups=[]
    for i in range(len(prof)-1):
        for j in range(count):
            faces.append((i*count+j,i*count+(j+1)%count,(i+1)*count+(j+1)%count,(i+1)*count+j));face_groups.append((i//3,j//8))
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(verts,[],faces);mesh.update()
    ob=bpy.data.objects.new(name,mesh);scene.collection.objects.link(ob);mat=material(name,colors[name]);mesh.materials.append(mat)
    for poly in mesh.polygons:poly.use_smooth=True
    scene.render.filepath=str(out/f'{name}.png');bpy.ops.render.render(write_still=True)
    ob.hide_render=True
    groups={g:[] for g in face_groups}
    for f,g in zip(faces,face_groups):groups[g].append(f)
    shards=[]
    for k,(group,fs) in enumerate(groups.items()):
        used=sorted({i for f in fs for i in f});center=sum((Vector(verts[i]) for i in used),Vector())/len(used)
        indices={v:i for i,v in enumerate(used)}
        me=bpy.data.meshes.new('shard');me.from_pydata([Vector(verts[i])-center for i in used],[],[tuple(indices[i] for i in f) for f in fs]);me.materials.append(mat)
        sh=bpy.data.objects.new('shard',me);scene.collection.objects.link(sh);sh.location=center
        for poly in me.polygons:poly.use_smooth=True
        shards.append((sh,center,k))
    for frame in range(10):
        t=(frame+1)/10
        for sh,c,k in shards:
            v=Vector((c.x,c.y,0));v.normalize()
            sh.location=c+v*(.65*t)+Vector((0,0,.6*t-1.6*t*t))
            sh.rotation_euler=(t*.8*math.sin(k*2.1),t*1.1*math.cos(k*1.3),t*.5)
        scene.render.filepath=str(out/f'{name}_break_{frame:02d}.png');bpy.ops.render.render(write_still=True)
    for sh,_,_ in shards:bpy.data.objects.remove(sh,do_unlink=True)
    bpy.data.objects.remove(ob,do_unlink=True)

# Empty, bright architectural room. No sharp detail in the target search area.
scene.render.resolution_x=760;scene.render.resolution_y=580;scene.render.film_transparent=False
bpy.data.objects.remove(cam,do_unlink=True);cam=camera((0,-11,4.4),(0,1,2.6),9)
for light in [o for o in scene.objects if o.type == 'LIGHT']:
    light.data.energy *= .12
cream=material('ivory plaster',(.50,.62,.40));floor=material('pale floor',(.31,.46,.40))
for name,loc,scale,mat in [('back',(0,4,4),(8,.15,5),cream),('floor',(0,0,-1.25),(8,9,.12),floor),('left',(-4.3,0,4),(.1,8,6),cream),('right',(4.3,0,4),(.1,8,6),cream)]:
    bpy.ops.mesh.primitive_cube_add(size=2,location=loc);ob=bpy.context.object;ob.name=name;ob.scale=scale;ob.data.materials.append(mat)
scene.render.filepath=str(out/'room.png');bpy.ops.render.render(write_still=True)
(out/'blender_version.txt').write_text(bpy.app.version_string+'\n')
print('LIGHT_ROOM_RENDER_COMPLETE',out)
