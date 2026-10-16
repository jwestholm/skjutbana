"""Experimental GPU single-influence skinning of the existing Skyttegrav mesh.

Three resident bind meshes (equipment variants), sixteen matrices per actor,
one GPU draw per soldier per pass. No CPU skinning, no per-frame VBO creation.
This does NOT load external GLB assets; the same bone palette can be retained
when an artist-rigged, multi-weight glTF importer is added later.
"""
from __future__ import annotations
import math
import numpy as np
from enemy_system import EnemyAgent,EnemySpec,enemy_parts,actor_facing
from skeletal_pose import BONES,bone_matrices

SKIN_VERT = '''#version 330
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
in float in_layer;
in float in_bone;
in float in_prop;
uniform int weapon_mode;
uniform int flash_on;
uniform mat4 bones[16];
uniform mat4 model;
uniform mat4 mvp;
uniform vec3 camera_position;
out vec3 v_normal;
out vec2 v_uv;
flat out int v_layer;
out float v_distance;
out vec3 v_world;
void main() {
    int bone=clamp(int(in_bone+.5),0,15);
    vec4 p=bones[bone]*vec4(in_pos,1.0);
    vec4 w=model*p;
    int prop=int(in_prop+.5);
    if ((prop==1 && weapon_mode!=1) || (prop==2 && weapon_mode!=2) ||
        (prop==3 && flash_on==0)) {
        gl_Position=vec4(2.0,2.0,2.0,1.0);
        v_world=vec3(0.0);
        v_normal=vec3(0.0,1.0,0.0);
        v_uv=in_uv;
        v_layer=int(in_layer+.5);
        v_distance=1000.0;
        return;
    }
    gl_Position=mvp*w;
    v_world=w.xyz;
    v_normal=normalize(mat3(model)*mat3(bones[bone])*in_normal);
    v_uv=in_uv;
    v_layer=int(in_layer+.5);
    v_distance=length(w.xyz-camera_position);
}
'''
SKIN_SHADOW_VERT = '''#version 330
in vec3 in_pos;
in float in_bone;
in float in_prop;
uniform int weapon_mode;
uniform mat4 bones[16];
uniform mat4 model;
uniform mat4 light_mvp;
void main() {
    int prop=int(in_prop+.5);
    if (prop==3 || (prop==1 && weapon_mode!=1) ||
                   (prop==2 && weapon_mode!=2)) {
        gl_Position=vec4(2.0,2.0,2.0,1.0);
        return;
    }
    gl_Position=light_mvp*model*bones[clamp(int(in_bone+.5),0,15)]*vec4(in_pos,1.0);
}
'''
SKIN_SHADOW_FRAG = '''#version 330
void main() {}
'''


def bone_for_part(index,count):
    """The bind-mesh authoring order of enemy_parts(): deterministic anatomy.

    The fixed groups are validated for each equipment variant. Never use
    renderer-selected bone ownership for physical hit tests.
    """
    if not 0<=index<count:raise ValueError('invalid part index')
    if index<12:
        leg=index//6;within=index%6
        base=9+leg*3
        return base + (0 if within in (0,1,3) else 1 if within in (2,4) else 2)
    if index==12:return 0
    weapon_start=count-8
    head_start=weapon_start-11
    arms_start=head_start-10
    if index<arms_start:return 1
    if index<head_start:
        i=index-arms_start
        side=i//5
        local=i%5
        return 3+3*side+(0 if local in (0,1) else 1 if local in (2,3) else 2)
    if index<weapon_start:return 2
    return 15


def to_bind_mesh(parts,shape_data,part_model):
    """Bake static primitive vertices into 1 local-space, bone-tagged array.

    Shapes remain GPU resident. Actor position, skeleton and lighting are
    uniforms, NOT changed vertex buffers. Indexed skeleton palette length=16.
    """
    arrays=[]
    count=len(parts)
    for idx,part in enumerate(parts):
        # Live rifle stock/handguard use the existing dark equipment surface,
        # not the terrain bark layer. Identity, size and weapon bone stay fixed.
        visual_material=10 if idx>=count-8 and part[1]==13 else part[1]
        src=shape_data[(part[0],visual_material)]
        m=part_model((0.,0.,0.),0.,part).astype('f4')
        vertices=src.copy()
        vertices[:,:3]=src[:,:3]@m[:3,:3].T+m[:3,3]
        normal_matrix=np.linalg.inv(m[:3,:3]).T
        normal=src[:,3:6]@normal_matrix.T
        normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-7)
        vertices[:,3:6]=normal
        bone=np.full((len(src),1),float(bone_for_part(idx,count)),dtype='f4')
        prop=np.full((len(src),1),1.0 if idx>=count-8 else 0.0,dtype='f4')
        arrays.append(np.concatenate((vertices,bone,prop),axis=1))
    # One static shovel and a small brief muzzle flash, both attached to the
    # WEAPON bone. All geometry exists permanently on GPU; shader visibility
    # flags decide which prop is visible without buffer reallocations.
    optional=(
        (('cylinder',13,(-.02,.88,.38),(-.02,1.62,.38),.028),2.0),
        (('cube',10,(-.02,1.69,.38),(.135,.135,.029)),2.0),
        (('oval',11,(.004,1.06,1.35),(.055,.055,.12)),3.0),
    )
    for part,tag in optional:
        src=shape_data[part[:2]]
        m=part_model((0.,0.,0.),0.,part).astype('f4')
        vertices=src.copy()
        vertices[:,:3]=src[:,:3]@m[:3,:3].T+m[:3,3]
        normal=src[:,3:6]@np.linalg.inv(m[:3,:3]).T
        normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-7)
        vertices[:,3:6]=normal
        arrays.append(np.concatenate((vertices,
                        np.full((len(src),1),15.,dtype='f4'),
                        np.full((len(src),1),tag,dtype='f4')),axis=1))
    result=np.concatenate(arrays,axis=0)
    if not np.isfinite(result).all():raise ValueError('invalid rig geometry')
    return np.ascontiguousarray(result,dtype='f4')


class SkeletalEnemyRenderer:
    """Skinning is performed by vertex shaders, both camera and shadows."""
    is_skeletal=True
    def __init__(self,ctx,mesh_factory,part_model,actor_fragment,sun,moon):
        self.ctx=ctx
        self.program=ctx.program(vertex_shader=SKIN_VERT,fragment_shader=actor_fragment)
        self.shadow_program=ctx.program(vertex_shader=SKIN_SHADOW_VERT,fragment_shader=SKIN_SHADOW_FRAG)
        self.models={}
        self.part_model=part_model
        self.program['materials'].value=0
        self.program['material_normals'].value=1
        self.program['sun_direction'].value=tuple(sun)
        self.program['moon_direction'].value=tuple(moon)
        try:self.lamp_uniform=self.program['lamp_positions']
        except KeyError:self.lamp_uniform=self.program['lamp_positions[0]']
        self.bones_uniform=self._bones_uniform(self.program)
        self.program['weapon_mode'].value=1
        self.program['flash_on'].value=0
        self.shadow_program['weapon_mode'].value=1
        self.shadow_bones_uniform=self._bones_uniform(self.shadow_program)
        # Generate shape prototypes ONCE, not once per actor and not per frame.
        shapes={}
        try:
            for variant in range(3):
                actor=EnemyAgent(EnemySpec(variant,32.,-.18,'unaware'))
                parts=enemy_parts(actor)
                if len(parts) not in (58,59):raise ValueError('unexpected bind order')
                for part in parts:
                    key=part[:2]
                    if key not in shapes:shapes[key]=mesh_factory(*key)
                # Extra prop meshes use existing reusable template materials.
                for key in (('cylinder',13),('cube',10),('oval',11)):
                    if key not in shapes:shapes[key]=mesh_factory(*key)
                data=to_bind_mesh(parts,shapes,part_model)
                buf=ctx.buffer(data.tobytes())
                vao=ctx.vertex_array(self.program,[(buf,'3f 3f 2f 1f 1f 1f',
                           'in_pos','in_normal','in_uv','in_layer','in_bone','in_prop')])
                sh_vao=ctx.vertex_array(self.shadow_program,
                    [(buf,'3f 24x 1f 1f','in_pos','in_bone','in_prop')])
                self.models[variant]=(buf,vao,sh_vao,len(data))
        except Exception:
            self.release()
            raise
        self.gpu_bytes=sum(b.size for b,_,_,_ in self.models.values())

    @staticmethod
    def _bones_uniform(program):
        try:return program['bones']
        except KeyError:return program['bones[0]']

    @staticmethod
    def _model_matrix(pos,heading):
        c,s=math.cos(heading),math.sin(heading)
        result=np.eye(4,dtype='f4')
        result[:3,:3]=((c,0.,s),(0.,1.,0.),(-s,0.,c))
        result[:3,3]=pos
        return result

    def collect_instances(self,agents,position_at,heading_at):
        out=[]
        for agent in agents:
            pos=position_at(agent.spec)
            angle=actor_facing(agent,heading_at(agent.route_m))
            model=self._model_matrix(pos,angle)
            pose=getattr(agent,'pose_preview',None)
            controller=getattr(agent,'pose_controller',None)
            bones=(controller.sample(agent) if controller is not None
                   else bone_matrices(agent,'auto' if pose is None else pose))
            variant=agent.spec.visual_variant
            out.append((agent.spec.enemy_id%3 if variant is None else variant,model,bones,agent))
        return out

    def draw(self,agents,position_at,heading_at,projection_view,eye,
             fog_strength,normal_enabled,lights,instances=None):
        import moderngl
        if instances is None:instances=self.collect_instances(agents,position_at,heading_at)
        p=self.program
        p['mvp'].write(np.asarray(projection_view,dtype='f4').T.tobytes())
        p['camera_position'].value=eye
        p['fog_strength'].value=fog_strength
        p['use_pbr_normals'].value=int(normal_enabled)
        p['lamp_count'].value=len(lights)
        if lights:self.lamp_uniform.write(np.asarray(lights+[(0.,0.,0.)]*(8-len(lights)),dtype='f4').tobytes())
        for variant,model,bones,actor in instances:
            p['model'].write(model.T.tobytes())
            current_pose=getattr(actor,'pose_preview',None)
            p['weapon_mode'].value=2 if current_pose=='spade_attack' else 1
            p['flash_on'].value=int(
                (actor.state=='firing' and actor.state_time<.115 and
                 current_pose not in ('spade_attack','dead')) or
                (current_pose=='fire' and actor.total_time%1.05<.095))
            self.bones_uniform.write(bones.transpose((0,2,1)).copy().tobytes())
            self.models[variant][1].render(moderngl.TRIANGLES)

    def draw_skeletal_shadows(self,light_mvp,instances):
        import moderngl
        p=self.shadow_program
        p['light_mvp'].write(np.asarray(light_mvp,dtype='f4').T.tobytes())
        for variant,model,bones,actor in instances:
            p['model'].write(model.T.tobytes())
            p['weapon_mode'].value=2 if getattr(actor,'pose_preview',None)=='spade_attack' else 1
            self.shadow_bones_uniform.write(bones.transpose((0,2,1)).copy().tobytes())
            self.models[variant][2].render(moderngl.TRIANGLES)

    def release(self):
        for buf,vao,shadow_vao,_ in self.models.values():
            shadow_vao.release();vao.release();buf.release()
        self.models.clear()
        self.program.release()
        self.shadow_program.release()
