"""V3.9.2 real GPU shadow maps for the existing ModernGL trench renderer.

One sun map, one moon map, and a downward local projection for the closest
nearby lantern.  These are RENDER-ONLY effects: gameplay LOS, frozen hit rays,
AI perception and geometry data remain untouched.  No new Python dependency.
"""
from __future__ import annotations
import math
import numpy as np

SUN_SHADOW_RES = 1536
MOON_SHADOW_RES = 1024
LAMP_SHADOW_RES = 768
SHADOW_RADIUS_METRES = 42.0
LAMP_SHADOW_MAX_DISTANCE = 18.0


# The shadow-only vertex shader draws existing VBOs without uploading geometry.
# Trench and actor buffers share the same interleaved stride of 36 bytes.
SHADOW_VERT = '''#version 330
in vec3 in_pos;
in vec2 in_uv;
in float in_layer;
uniform mat4 light_mvp;
uniform mat4 model;
uniform int is_actor;
uniform float time_seconds;
void main() {
    vec3 p = in_pos;
    // Cast shadows from the same animated net/fabric shape as the visible VBO.
    if(in_layer > 25.5 && in_layer < 26.5) {
        float free_edge=clamp(in_uv.y,0.0,1.0);
        float breeze=sin(time_seconds*.83 + in_pos.x*.74 + in_pos.z*.63);
        float ripples=sin(time_seconds*1.43 + in_pos.x*2.32 - in_pos.y*1.24);
        p.x+=free_edge*(.024*breeze+.011*ripples);
        p.z+=free_edge*(.035*breeze-.012*ripples);
    }
    vec4 world = is_actor == 1 ? model * vec4(p, 1.0) : vec4(p, 1.0);
    gl_Position = light_mvp * world;
}
'''
SHADOW_FRAG = '''#version 330
void main() { }
'''

# Injected into the shared material shader BEFORE 'void main()'. Actor and
# world shading receive identical light/shadow positions, without changing
# the game's camera, physical hit position or actor animation matrices.
SHADOW_GLSL = '''
uniform int shadow_mode;
uniform sampler2D shadow_sun;
uniform sampler2D shadow_moon;
uniform sampler2D shadow_lamp;
uniform mat4 shadow_sun_mvp;
uniform mat4 shadow_moon_mvp;
uniform mat4 shadow_lamp_mvp;
uniform int shadow_lamp_index;
uniform float shadow_lamp_weight;

float shadow_pcf(sampler2D depth_map, mat4 light_matrix, vec3 world_pos, float bias) {
    vec4 clip = light_matrix * vec4(world_pos,1.0);
    if (abs(clip.w)<0.00001) return 1.0;
    vec3 projected=clip.xyz/clip.w*0.5+0.5;
    // No false dark bands outside the finite shadow map.
    if(projected.z<=0.0 || projected.z>=1.0 ||
       projected.x<=0.003 || projected.x>=0.997 ||
       projected.y<=0.003 || projected.y>=0.997) return 1.0;
    vec2 texel=1.0/vec2(textureSize(depth_map,0));
    float lit=0.0;
    for(int x=-1;x<=1;x++) for(int y=-1;y<=1;y++) {
        float closest=texture(depth_map,projected.xy+vec2(x,y)*texel).r;
        lit += projected.z-bias <= closest ? 1.0 : 0.0;
    }
    return lit/9.0;
}
'''


def orthographic(left,right,bottom,top,near,far):
    if right<=left or top<=bottom or far<=near:
        raise ValueError('invalid orthographic shadow frustum')
    m=np.eye(4,dtype='f4')
    m[0,0]=2.0/(right-left);m[1,1]=2.0/(top-bottom);m[2,2]=-2.0/(far-near)
    m[0,3]=-(right+left)/(right-left)
    m[1,3]=-(top+bottom)/(top-bottom)
    m[2,3]=-(far+near)/(far-near)
    return m


def look_at(eye,target,up=(0.0,1.0,0.0)):
    eye=np.asarray(eye,dtype='f4'); target=np.asarray(target,dtype='f4')
    f=target-eye;length=float(np.linalg.norm(f))
    if length<1e-5:raise ValueError('zero-length light direction')
    f/=length
    u=np.asarray(up,dtype='f4')
    s=np.cross(f,u)
    if float(np.linalg.norm(s))<1e-5:
        u=np.array((0.,0.,1.),dtype='f4')
        s=np.cross(f,u)
    s/=float(np.linalg.norm(s))
    u=np.cross(s,f)
    out=np.eye(4,dtype='f4')
    out[0,:3]=s;out[1,:3]=u;out[2,:3]=-f
    out[:3,3]=-out[:3,:3]@eye
    return out


def perspective(fov,aspect,near,far):
    f=1.0/math.tan(math.radians(fov)/2.0)
    return np.array(((f/aspect,0,0,0),(0,f,0,0),
                     (0,0,-(far+near)/(far-near),-2*far*near/(far-near)),
                     (0,0,-1,0)),dtype='f4')


def directional_matrix(camera_pos,light_direction,span=SHADOW_RADIUS_METRES,
                       resolution=SUN_SHADOW_RES):
    """Snap in LIGHT texels, not half-metre world-axis jumps."""
    direction=np.asarray(light_direction,dtype='f4')
    direction/=max(1e-7,float(np.linalg.norm(direction)))
    centre=np.asarray((camera_pos[0],1.4,camera_pos[2]),dtype='f4')
    matrix=orthographic(-span,span,-span,span,.1,122.) @ look_at(centre+direction*53.,centre)
    origin=matrix[:2,3]*float(resolution)*.5
    matrix[:2,3]+=(np.round(origin)-origin)*(2./resolution)
    return matrix


def lamp_matrix(position):
    """Finite local *downward* shadow cone; does not claim 360 degree shadows."""
    p=np.asarray(position,dtype='f4')
    return perspective(145.,1.,.08,7.5) @ look_at(p,p+(0.,-1.,0.),(0.,0.,1.))


def closest_shadow_lamp(lights,camera_pos,max_distance=LAMP_SHADOW_MAX_DISTANCE):
    if not lights:return -1
    distances=[sum((float(a)-float(b))**2 for a,b in zip(p,camera_pos))
               for p in lights]
    idx=min(range(len(lights)),key=distances.__getitem__)
    return idx if distances[idx]<=max_distance**2 else -1


def lamp_shadow_weight(lights,camera_pos,index):
    """Continuous handover of the one-map budget; no hard 7.5 m activation.

    Prepare outside the old visible transition. At an equal-distance switch
    both candidates have zero weight, so changing map ownership cannot flash.
    """
    if index<0:return 0.
    distances=[math.dist(p,camera_pos) for p in lights]
    def ease(t):
        t=max(0.,min(1.,t));return t*t*(3.-2.*t)
    strength=ease((LAMP_SHADOW_MAX_DISTANCE-distances[index])/6.)
    others=[d for i,d in enumerate(distances) if i!=index]
    if others:strength*=ease((min(others)-distances[index])/3.)
    return strength


class ShadowRenderer:
    """One depth-only program + three reusable depth FBOs, no mesh copies.

    All resident chunks participate, matching the visible world pass.  A finite local light cone from
    ONE nearby lantern avoids six cubemap renders per lamp on GTX 1070.
    """
    def __init__(self,ctx,sun_direction,moon_direction):
        import moderngl
        self.gl=moderngl
        self.ctx=ctx
        self.program=ctx.program(vertex_shader=SHADOW_VERT,fragment_shader=SHADOW_FRAG)
        self.light_dirs=(tuple(sun_direction),tuple(moon_direction))
        self.sizes=(SUN_SHADOW_RES,MOON_SHADOW_RES,LAMP_SHADOW_RES)
        self.maps=[];self.framebuffers=[]
        self.world_vaos={};self.actor_vaos={}
        self.enabled=True
        self.lamp_index=-1
        self.matrices=tuple(np.eye(4,dtype='f4') for _ in range(3))
        try:
            for size in self.sizes:
                depth=ctx.depth_texture((size,size))
                # shadow_pcf samples raw depth with sampler2D and compares in GLSL.
                # Hardware comparison mode requires sampler2DShadow instead.
                depth.compare_func=''
                depth.repeat_x=False;depth.repeat_y=False
                depth.filter=(moderngl.NEAREST,moderngl.NEAREST)
                fbo=ctx.framebuffer(depth_attachment=depth)
                self.maps.append(depth);self.framebuffers.append(fbo)
            self.program['is_actor'].value=0
            self.program['model'].write(np.eye(4,dtype='f4').T.tobytes())
        except Exception:
            self.release()
            raise

    def setup_material(self,program):
        for i,uniform in enumerate(('shadow_sun','shadow_moon','shadow_lamp'),start=2):
            program[uniform].value=i
        program['shadow_mode'].value=0
        program['shadow_lamp_index'].value=-1

    def _world_vao(self,index,vbo):
        old=self.world_vaos.get(index)
        if old and old[0] is vbo:return old[1]
        if old:old[1].release()
        vao=self.ctx.vertex_array(self.program,
            [(vbo,'3f 12x 2f 1f','in_pos','in_uv','in_layer')])
        self.world_vaos[index]=(vbo,vao)
        return vao

    def _actor_vao(self,key,vbo):
        old=self.actor_vaos.get(key)
        if old and old[0] is vbo:return old[1]
        if old:old[1].release()
        vao=self.ctx.vertex_array(self.program,
            [(vbo,'3f 12x 2f 1f','in_pos','in_uv','in_layer')])
        self.actor_vaos[key]=(vbo,vao)
        return vao

    def draw_maps(self,chunks,enemy_renderer,instances,camera_pos,lights,
                  frame_size,route_chunk,time_seconds):
        if not self.enabled:return
        self.lamp_index=closest_shadow_lamp(lights,camera_pos)
        self.lamp_weight=lamp_shadow_weight(lights,camera_pos,self.lamp_index)
        mats=[directional_matrix(camera_pos,self.light_dirs[0]),
              directional_matrix(camera_pos,self.light_dirs[1],resolution=MOON_SHADOW_RES),
              lamp_matrix(lights[self.lamp_index]) if self.lamp_index>=0
              else np.eye(4,dtype='f4')]
        self.matrices=tuple(mats)
        # Garbage-collect dead VBO references so bounded chunk streaming keeps
        # bounded GL state. Actor meshes only rebuild on a new renderer.
        for idx in list(self.world_vaos):
            if idx not in chunks or self.world_vaos[idx][0] is not chunks[idx][0]:
                self.world_vaos.pop(idx)[1].release()
        # Main rendering draws all resident chunks. Route IDs are not spatial
        # bounds on winding terrain; never exclude an already visible caster.
        keys=sorted(chunks)
        prg=self.program
        prg['time_seconds'].value=float(time_seconds)
        try:
            self.ctx.disable(self.gl.BLEND)
            self.ctx.enable(self.gl.DEPTH_TEST)
            for pass_idx,fbo in enumerate(self.framebuffers):
                if pass_idx==2 and self.lamp_index<0:continue
                fbo.use()
                size=self.sizes[pass_idx]
                self.ctx.viewport=(0,0,size,size)
                fbo.clear(depth=1.0)
                prg['light_mvp'].write(self.matrices[pass_idx].T.tobytes())
                prg['is_actor'].value=0
                for index in keys:
                    vbo,vao,n=chunks[index]
                    self._world_vao(index,vbo).render(self.gl.TRIANGLES)
                if getattr(enemy_renderer,'is_skeletal',False):
                    # Identical bone palette, identical bound GPU mesh in
                    # camera and shadow passes: no detached shadow limbs.
                    enemy_renderer.draw_skeletal_shadows(self.matrices[pass_idx],instances)
                else:
                    prg['is_actor'].value=1
                    for key,model in instances:
                        vbo,_=enemy_renderer.meshes[key]
                        self._actor_vao(key,vbo)
                        prg['model'].write(model.T.tobytes())
                        self.actor_vaos[key][1].render(self.gl.TRIANGLES)
        finally:
            # Keep all VFX and the original framebuffer in their usual state.
            self.ctx.screen.use()
            self.ctx.viewport=(0,0,int(frame_size[0]),int(frame_size[1]))
            self.ctx.enable(self.gl.BLEND)

    def bind_material(self,program):
        program['shadow_mode'].value=int(self.enabled)
        program['shadow_lamp_index'].value=self.lamp_index if self.enabled else -1
        try:program['shadow_lamp_weight'].value=getattr(self,'lamp_weight',0.) if self.enabled else 0.
        except KeyError:pass  # Actor materials without local lantern lighting.
        for idx,name in enumerate(('shadow_sun_mvp','shadow_moon_mvp','shadow_lamp_mvp')):
            program[name].write(self.matrices[idx].T.tobytes())
            self.maps[idx].use(location=idx+2)

    def release(self):
        for vbo,vao in self.world_vaos.values():vao.release()
        for vbo,vao in self.actor_vaos.values():vao.release()
        self.world_vaos.clear();self.actor_vaos.clear()
        for fbo in self.framebuffers:fbo.release()
        for m in self.maps:m.release()
        self.framebuffers.clear();self.maps.clear()
        self.program.release()
