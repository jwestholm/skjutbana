"""V3.8.1 battlefield visuals + instantaneous actor reaction signals.

Deterministic per seed and simulation time; independent of OpenGL and sound.
Impacts do not damage anyone or alter terrain, but may interrupt actor movement.
World coordinates are metres (X, Y up, Z). Events are local and bounded.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
import ctypes
import ctypes.util

MAX_PARTICLES = 420
MAX_PENDING = 18
MAX_AIRCRAFT = 2
# The main game provides world_on_ground(s, lateral) to resolve positions.
# No persistent crater or change to ground mesh is ever produced.


def blast_reaction_level(distance_metres: float, strength: float) -> int:
    """Pure gameplay cue from a REAL WORLD blast distance, not screen pixels.

    0=ignore, 1=look/brace, 2=kneel, 3=drop prone. Not a blast physics
    model and not a damage model. Weak strafing impacts never demand prone.
    """
    if distance_metres < 0 or not math.isfinite(distance_metres):
        raise ValueError('distance must be a non-negative finite number')
    if not math.isfinite(strength):
        raise ValueError('strength must be finite')
    if distance_metres > 110.0 or strength <= 0.0:
        return 0
    if distance_metres <= 32.0 and strength >= .52:
        return 3
    if distance_metres <= 90.0:
        return 2
    return 1


@dataclass(slots=True)
class Particle:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    age: float
    life: float
    size: float
    kind: int  # 0 grit/chip, 1 smoke, 2 dusty plume, 3 brief warm flash
    color: tuple[float, float, float]
    base_y: float

    @property
    def alive(self):
        return self.age < self.life


@dataclass(slots=True)
class Impact:
    at: float
    position: tuple[float, float, float]
    strength: float
    origin: str


@dataclass(slots=True)
class Aircraft:
    x: float
    y: float
    z: float
    dx: float
    dz: float
    speed: float
    lifetime: float
    age: float = 0.0

    def advance(self, dt):
        self.age += dt
        self.x += self.dx * self.speed * dt
        self.z += self.dz * self.speed * dt


class BattlefieldEventDirector:
    """A few convincing **visual-only** impacts, not a war simulation.

    On the normal schedule, a single event is considered every ~38-85 s.
    `trigger()` permits instant local tests without waiting for randomness.
    The RNG is private: chunk generation and enemy encounters are untouched.
    """
    EVENT_TYPES = ('shell', 'barrage', 'flyby', 'near_shell')

    def __init__(self, *, seed=1944):
        self.seed = int(seed)
        self.reset()

    def reset(self):
        self.rng = random.Random(self.seed * 1907 + 318)
        self.elapsed = 0.0
        self.next_event = 21.0 + self.rng.uniform(0.0, 11.0)
        self.particles: list[Particle] = []
        self.pending: list[Impact] = []
        self.aircraft: list[Aircraft] = []
        self.events_started = 0
        self.impacts_spawned = 0
        self.last_kind = ''
        self.enabled = True
        # Only impact events detonating THIS update; never older pending shots.
        # This is an impulse signal, not a persistent world object.
        self.last_detonations: tuple[Impact, ...] = ()

    @property
    def active_count(self):
        return len(self.particles)

    def _impact_point(self, route_m, ground_at, *, spread=1.0):
        # Ahead in the visible field, well outside walkable banks and dugouts.
        # These are particles, not permanent holes or physical obstacles.
        s = max(0.0, route_m + self.rng.uniform(25.0, 76.0) * spread)
        side = -1.0 if self.rng.random() < .5 else 1.0
        lateral = side * self.rng.uniform(18.0, 52.0) * spread
        return tuple(float(x) for x in ground_at(s, lateral))

    def trigger(self, kind, route_m, ground_at):
        """Create a single shot, short salvo or distant flyover.

        Effects do not affect the route, ground state, AI or enemy hitboxes.
        This call is safe on the main thread and never uploads GPU resources.
        """
        if kind not in self.EVENT_TYPES:
            raise ValueError('Unknown battlefield VFX event')
        self.events_started += 1
        self.last_kind = kind
        if kind == 'shell':
            pos = self._impact_point(route_m, ground_at)
            self._queue_impact(self.elapsed + .25, pos, .75, 'shell')
        elif kind == 'near_shell':
            # Developer-only F3 proximity test, outside trench. Real 3D
            # positions, but never modify geometry or damage the soldier.
            s = max(0.0, route_m + self.rng.uniform(7.0, 13.0))
            lateral = (-1.0 if self.rng.random() < .5 else 1.0) * self.rng.uniform(10.0, 16.0)
            pos = tuple(float(x) for x in ground_at(s, lateral))
            self._queue_impact(self.elapsed + .25, pos, .80, 'near_shell')
        elif kind == 'barrage':
            # 2-4 staggered impacts, each with slightly different strength.
            count = self.rng.randrange(2, 5)
            anchor_s=max(0.0,route_m+self.rng.uniform(30.0, 68.0))
            side=-1.0 if self.rng.random()<.5 else 1.0
            anchor_off=side*self.rng.uniform(22.0, 46.0)
            for i in range(count):
                ss=max(0.0,anchor_s+self.rng.uniform(-9.0,9.0))
                off=anchor_off+self.rng.uniform(-9.0,9.0)
                # Each shell uses its OWN ground contact, not the first
                # shell's height projected sideways through nearby hills.
                pos=tuple(float(x) for x in ground_at(ss,off))
                self._queue_impact(self.elapsed + .45 + i * self.rng.uniform(.48, .94),
                                   pos, self.rng.uniform(.55, .95), 'artillery')
        else:  # Flyover: a small distant aircraft silhouette.
            s = max(0.0, route_m + self.rng.uniform(65.0, 105.0))
            left = ground_at(s, -77.0)
            right = ground_at(s, 77.0)
            dx, dz = right[0] - left[0], right[2] - left[2]
            length = max(1e-6, math.hypot(dx, dz))
            dx /= length
            dz /= length
            if self.rng.random() < .5:
                dx, dz = -dx, -dz
                start = right
            else:
                start = left
            if len(self.aircraft) >= MAX_AIRCRAFT:
                self.aircraft.pop(0)
            plane_height = max(left[1], right[1]) + self.rng.uniform(34.0, 51.0)
            self.aircraft.append(Aircraft(float(start[0]), plane_height,
                                          float(start[2]), dx, dz,
                                          self.rng.uniform(18.0, 24.0),
                                          lifetime=8.0))
            # Occasionally a distant bomb; otherwise brief tiny strafing puffs.
            if self.rng.random() < .42:
                bomb = ground_at(s + self.rng.uniform(8.0, 20.0),
                                 (-1.0 if self.rng.random() < .5 else 1.0) *
                                 self.rng.uniform(35.0, 65.0))
                self._queue_impact(self.elapsed + self.rng.uniform(3.1, 4.6),
                                   tuple(float(v) for v in bomb), 1.0, 'distant_bomb')
            else:
                side = -1.0 if self.rng.random() < .5 else 1.0
                for k in range(self.rng.randint(3, 5)):
                    pos = ground_at(s + 4.0*k, side*(29.0 + self.rng.uniform(-5.0, 5.0)))
                    self._queue_impact(self.elapsed + 2.4 + .20*k,
                                       tuple(float(v) for v in pos), .28, 'strafe')

    def bullet_splash(self, position, *, material='earth'):
        """Small, immediate near-miss chips; no terrain changes or player damage.

        Returns a visual cue from a nearby surface, not a physically simulated
        projectile ray. The gameplay rules still require actual enemy LOS.
        """
        if material not in ('earth','wood','stone'):
            raise ValueError('unknown splash material')
        x,y,z=(float(v) for v in position)
        palette={'earth':(.36,.31,.25),'wood':(.33,.24,.15),
                 'stone':(.41,.41,.39)}[material]
        new=[]
        for i in range(11):
            angle=self.rng.uniform(0.,math.tau)
            speed=self.rng.uniform(.5,2.3)
            new.append(Particle(x,y,z,math.cos(angle)*speed,
                                self.rng.uniform(.3,2.0),math.sin(angle)*speed,
                                0.,self.rng.uniform(.35,.8),
                                self.rng.uniform(.014,.052),0,palette,y-.7))
        new.append(Particle(x,y+.1,z,0.,.27,0.,0.,.38,.24,2,
                            tuple(v*.8 for v in palette),y-.7))
        capacity=max(0,MAX_PARTICLES-len(new))
        self.particles=self.particles[-capacity:] if capacity else []
        self.particles.extend(new[:MAX_PARTICLES-len(self.particles)])
        return len(new)

    def _queue_impact(self, at, pos, strength, origin):
        # Future pending effects are bounded, including during forced tests.
        if len(self.pending) >= MAX_PENDING:
            return
        self.pending.append(Impact(float(at), tuple(pos), float(strength), origin))
        self.pending.sort(key=lambda p:p.at)

    def _explode(self, impact):
        """Transient dirt, smoke and small flash; NO crater mesh."""
        x, y, z = impact.position
        strength = impact.strength
        self.impacts_spawned += 1
        # Debris fragments are numerous but mostly fine; no monster explosions.
        debris = round(20 + 25 * strength)
        smoke = round(12 + 18 * strength)
        dust = round(10 + 12 * strength)
        if impact.origin == 'strafe':
            smoke = max(3, smoke//3)
            dust = max(3, dust//2)
        fresh=[]
        for k in range(debris):
            ang=self.rng.uniform(0.0, math.tau)
            v=self.rng.uniform(.8, 3.5)*(.50+.6*strength)
            vy=self.rng.uniform(1.5, 5.8)*(.50+.55*strength)
            color=(.25+.07*self.rng.random(),.225+.052*self.rng.random(),
                   .19+.045*self.rng.random())
            fresh.append(Particle(x,y+.04,z,math.cos(ang)*v,vy,math.sin(ang)*v,
                                  0.0,self.rng.uniform(.6,1.55),
                                  self.rng.uniform(.025,.075),0,color,y))
        for k in range(dust):
            ang=self.rng.random()*math.tau
            fresh.append(Particle(x,y+.08,z,
                                  math.cos(ang)*self.rng.uniform(.35,1.9),
                                  self.rng.uniform(.4,1.9),
                                  math.sin(ang)*self.rng.uniform(.35,1.9),
                                  0.0,self.rng.uniform(1.2,2.5),
                                  self.rng.uniform(.28,.72)*strength,2,
                                  (.39,.35,.29),y))
        for k in range(smoke):
            ang=self.rng.random()*math.tau
            fresh.append(Particle(x,y+self.rng.uniform(.20,.9),z,
                                  math.cos(ang)*self.rng.uniform(.2,1.2),
                                  self.rng.uniform(.75,1.75),
                                  math.sin(ang)*self.rng.uniform(.2,1.2),
                                  0.0,self.rng.uniform(2.7,5.2),
                                  self.rng.uniform(.48,1.13)*(.6+.45*strength),1,
                                  (.24,.245,.24),y))
        # A small amber blink, NOT a fireball. Present very briefly.
        if impact.origin != 'strafe':
            fresh.append(Particle(x,y+.35,z,0,0,0,0.,.09,
                                  .22+.28*strength,3,(1.0,.65,.30),y))
        budget=max(0,MAX_PARTICLES-len(self.particles))
        # Prioritize fresh impact if other smoke is still hanging in the air.
        if len(fresh)>budget:
            discard=min(len(self.particles),len(fresh)-budget)
            self.particles=self.particles[discard:]
        self.particles.extend(fresh[:MAX_PARTICLES-len(self.particles)])

    def update(self, dt, route_m, ground_at, *, allow_events=True):
        """Advance effects in game-time. Paused game never calls this method."""
        dt=max(0.0,min(.20,float(dt)))
        # Clear even when paused; a blast must never replay next frame.
        self.last_detonations = ()
        if dt <= 0.0:
            return ()
        self.elapsed += dt
        if self.enabled and allow_events and self.elapsed >= self.next_event:
            pick=self.rng.random()
            kind='shell' if pick<.47 else 'barrage' if pick<.82 else 'flyby'
            self.trigger(kind, route_m, ground_at)
            self.next_event=self.elapsed+self.rng.uniform(39.0, 82.0)
        # When the camera is temporarily inside a covered dugout/tunnel, do not
        # start a new random event; allow already emitted smoke to dissipate.
        elapsed=self.elapsed
        detonations=[]
        while self.pending and self.pending[0].at <= elapsed:
            impact=self.pending.pop(0)
            self._explode(impact)
            detonations.append(impact)
        self.last_detonations=tuple(detonations)
        surviving=[]
        for p in self.particles:
            p.age += dt
            if not p.alive:
                continue
            p.x+=p.vx*dt
            p.y+=p.vy*dt
            p.z+=p.vz*dt
            if p.kind==0:
                p.vy-=9.2*dt
                if p.y <= p.base_y:
                    continue  # gravel vanishes on ground contact, no deformation
            elif p.kind==1:
                p.vx*=max(0.,1.-.14*dt)
                p.vz*=max(0.,1.-.14*dt)
            elif p.kind==2:
                p.vy*=max(0.,1.-.6*dt)
            surviving.append(p)
        self.particles=surviving
        for plane in self.aircraft:
            plane.advance(dt)
        self.aircraft=[p for p in self.aircraft if p.age<p.lifetime]
        return self.last_detonations


# A reusable GPU particle buffer + two compact OpenGL 3.3 shaders.
# Importing this module for offline tests does not require ModernGL.
PARTICLE_VERTEX = '''#version 330
in vec3 in_pos;
in vec2 in_corner;
in float in_size;
in vec4 in_rgba;
in float in_kind;
uniform mat4 mvp;
uniform vec3 camera_right;
uniform vec3 camera_up;
out vec2 uv;
out vec4 tint;
flat out int kind;
void main(){
    vec3 p=in_pos + in_size*(in_corner.x*camera_right+in_corner.y*camera_up);
    gl_Position=mvp*vec4(p,1.0);
    uv=in_corner;
    tint=in_rgba;
    kind=int(in_kind+0.5);
}
'''
PARTICLE_FRAGMENT = '''#version 330
in vec2 uv;
in vec4 tint;
flat in int kind;
out vec4 out_color;
void main(){
    float r=length(uv);
    float a=tint.a;
    if(kind==1 || kind==2){
        // Soft smoke/dust is translucent, with diffuse edges.
        a*=1.0-smoothstep(.26,.99,r);
    }else if(kind==0){
        // Angular grit rather than perfect circular sparks.
        float edge=.80+.095*sin(atan(uv.y,uv.x)*5.0);
        a*=1.0-smoothstep(edge-.11,edge,r);
    }else{
        a*=1.0-smoothstep(.18,.95,r);
    }
    if(a<.003)discard;
    out_color=vec4(tint.rgb,a);
}
'''
PLANE_VERTEX = '''#version 330
in vec3 in_local;
uniform mat4 mvp;
uniform vec3 plane_position;
uniform vec2 plane_forward;
void main(){
    vec3 f=vec3(plane_forward.x,0.0,plane_forward.y);
    vec3 s=vec3(-f.z,0.0,f.x);
    vec3 pt=plane_position + f*in_local.x + vec3(0.,in_local.y,0.) + s*in_local.z;
    gl_Position=mvp*vec4(pt,1.0);
}
'''
PLANE_FRAGMENT = '''#version 330
out vec4 out_color;
void main(){
    out_color=vec4(.18,.20,.21,1.0);
}
'''


class BattlefieldEffectRenderer:
    """Renderer owns fixed GPU buffers; no particle-specific allocations."""

    def __init__(self, ctx):
        import numpy as np
        self.np=np
        self.ctx=ctx
        self.program=ctx.program(vertex_shader=PARTICLE_VERTEX,
                                 fragment_shader=PARTICLE_FRAGMENT)
        self.plane_program=ctx.program(vertex_shader=PLANE_VERTEX,
                                       fragment_shader=PLANE_FRAGMENT)
        # 3D quads, six vertices x 11 float32 per particle; one reusable VBO.
        self.vbo=ctx.buffer(reserve=MAX_PARTICLES*6*11*4,dynamic=True)
        self.vao=ctx.vertex_array(self.program,
                                  [(self.vbo,'3f 2f 1f 4f 1f',
                                    'in_pos','in_corner','in_size','in_rgba','in_kind')])
        # Model: elongated fuselage, left/right wings, rear tailplanes.
        triangles=[
            (2.4,.0,0.,-.8,.03,-.18,-.8,.03,.18),
            (-.8,.03,-.18,-2.3,.03,-.12,-.8,.03,.18),
            (-2.3,.03,-.12,-2.3,.03,.12,-.8,.03,.18),
            (.45,.0,-.15,-.18,.02,-2.75,-.65,.03,-.15),
            (.45,.0,.15,-.65,.03,.15,-.18,.02,2.75),
            (-1.75,.0,-.12,-2.0,.0,-1.1,-2.15,.0,-.10),
            (-1.75,.0,.12,-2.15,.0,.10,-2.0,.0,1.1),
            (-1.30,.0,0.,-1.93,.90,0.,-2.13,.0,0.),
        ]
        self.plane_vertex_count=len(triangles)*3
        self.plane_vbo=ctx.buffer(np.asarray(triangles,dtype='f4').reshape(-1).tobytes())
        self.plane_vao=ctx.vertex_array(self.plane_program,
                                        [(self.plane_vbo,'3f','in_local')])
        self.particles_drawn=0
        # Ubuntu/ModernGL 5.x: Context.depth_mask is not present. Loading
        # libGL via ctypes.util.find_library() in EACH draw call can spawn
        # expensive ldconfig/library probes 100+ times per second, precisely
        # while blast smoke is active (and SoldierBrain ducks). Resolve this
        # fallback ONCE during renderer construction and reuse the GL funcs.
        self._has_depth_property=(hasattr(type(ctx),'depth_mask') or
                                  hasattr(ctx,'depth_mask'))
        self._gl_library=None
        self._gl_get_boolean=None
        self._gl_depth_mask=None
        if not self._has_depth_property:
            gl=ctypes.CDLL(ctypes.util.find_library('GL') or 'libGL.so.1')
            get_boolean=gl.glGetBooleanv
            get_boolean.argtypes=(ctypes.c_uint,ctypes.POINTER(ctypes.c_ubyte))
            get_boolean.restype=None
            depth_mask=gl.glDepthMask
            depth_mask.argtypes=(ctypes.c_ubyte,)
            depth_mask.restype=None
            self._gl_library=gl
            self._gl_get_boolean=get_boolean
            self._gl_depth_mask=depth_mask

    @staticmethod
    def _appearance(p):
        fraction=max(0.0,min(1.0,p.age/p.life))
        if p.kind==0:
            alpha=.91*(1.0-fraction*.30)
            radius=p.size
        elif p.kind==1:
            alpha=.25*min(1.0,(p.age+.05)*2.9)*(1.0-fraction)**1.25
            radius=p.size*(.35+1.55*fraction)
        elif p.kind==2:
            alpha=.32*(1.0-fraction)**1.3
            radius=p.size*(.4+1.15*fraction)
        else:
            alpha=.54*(1.0-fraction)
            radius=p.size
        return alpha,radius

    def draw(self, director, viewproj, camera_eye, camera_right, camera_up):
        import moderngl
        if director.aircraft:
            self.plane_program['mvp'].write(viewproj.T.astype('f4').tobytes())
            for plane in director.aircraft:
                self.plane_program['plane_position'].value=(plane.x,plane.y,plane.z)
                self.plane_program['plane_forward'].value=(plane.dx,plane.dz)
                self.plane_vao.render(mode=moderngl.TRIANGLES)
        if not director.particles:
            self.particles_drawn=0
            return
        np=self.np
        eye=np.asarray(camera_eye,dtype='f4')
        # Blend distant smoke before nearer particles; renderer remains bounded.
        ordered=sorted(director.particles,
                       key=lambda p:-(p.x-eye[0])**2-(p.y-eye[1])**2-(p.z-eye[2])**2)
        # Opacity and quad radius are computed on the CPU (<=420 particles).
        vertices=[]
        for p in ordered:
            alpha,radius=self._appearance(p)
            if alpha<.007:continue
            color=(p.color[0],p.color[1],p.color[2],alpha)
            data=(p.x,p.y,p.z)
            for a,b in ((-1.,-1.),(1.,-1.),(1.,1.),(-1.,-1.),(1.,1.),(-1.,1.)):
                vertices.extend((*data,a,b,radius,*color,float(p.kind)))
        count=len(vertices)//(6*11)
        self.particles_drawn=count
        if not count:return
        array=np.asarray(vertices,dtype='f4')
        self.vbo.write(array.tobytes())
        self.program['mvp'].write(viewproj.T.astype('f4').tobytes())
        self.program['camera_right'].value=tuple(float(v) for v in camera_right)
        self.program['camera_up'].value=tuple(float(v) for v in camera_up)
        # Keep depth testing but do not let translucent smoke occlude solid
        # geometry or other effect geometry through depth writes.
        # ModernGL Context.depth_mask does not exist on all released versions
        # (notably the runtime used on Ubuntu/GTX 1070). The OpenGL function
        # glDepthMask is stable in the OpenGL 3.3 context. Preserve its actual
        # previous state, including on render failure.
        if self._has_depth_property:
            previous=self.ctx.depth_mask
            self.ctx.depth_mask=False
            try:
                self.vao.render(mode=moderngl.TRIANGLES,vertices=count*6)
            finally:
                self.ctx.depth_mask=previous
        else:
            previous=ctypes.c_ubyte(1)
            self._gl_get_boolean(0x0B72,ctypes.byref(previous))  # GL_DEPTH_WRITEMASK
            self._gl_depth_mask(0)
            try:
                self.vao.render(mode=moderngl.TRIANGLES,vertices=count*6)
            finally:
                self._gl_depth_mask(previous.value)

    def release(self):
        self.vao.release()
        self.vbo.release()
        self.program.release()
        self.plane_vao.release()
        self.plane_vbo.release()
        self.plane_program.release()
