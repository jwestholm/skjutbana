"""Standalone actual-GPU 3D trench proof of concept (Pygame + ModernGL).
No interaction with Skjutbanan's hit detection, settings, menu or rendering.
Run: python3 trench_3d_test.py
"""
from __future__ import annotations

import math
from collections import OrderedDict, deque
import queue
import threading
import os
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
from multiprocessing import get_context
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from enemy_system import (EnemyDirector, EnemyAgent, EnemySpec, actor_facing,
                          enemy_parts, ray_from_screen, encounter_partner_spec, off_trunk)
from arcade_mode import ArcadeRun
from combat_visibility import (player_body,can_see_body,enemy_faces_player,
                              enemy_shown_to_player,mesh_clear_body,
                              enemy_shown_to_player_mesh)
from combat_tactics import difficulty_at, MATERIALS, tactical_cover_lane
from death_screen import DeathScreen
from shadow_lighting import ShadowRenderer, SHADOW_GLSL
from rig_renderer import SkeletalEnemyRenderer
from gltf_actor import GLBSkinnedRenderer, TacticalEnemyRenderer
from skeletal_pose import POSE_LAB, PoseTransition
from shot_geometry import WorldShotMesh, trace_world, cover_name
from battlefield_events import BattlefieldEventDirector, BattlefieldEffectRenderer, blast_reaction_level

try:
    import moderngl
    import numpy as np
    import pygame
    from PIL import Image
except ImportError as exc:
    raise SystemExit(f"Missing dependency: {exc}. Install: python3 -m pip install pygame moderngl numpy pillow") from exc

ROOT = Path(__file__).resolve().parent
# Skjutbana defaults to 800x600 (4:3); the calibrated viewport can differ.
# This is a standalone *window default*, not the physical camera calibration.
W, H = 800, 600
CAMERA_VERTICAL_FOV = 80.0  # Camera geometry remains independent of physical shooting distance.
TEXTURE_SIZE = 1024  # Keep GTX 1070 streaming budget; extra detail via world-space blends.
SUN_DIRECTION = (0.35, 0.085, 0.93)  # Stable world-space celestial direction.
MOON_DIRECTION = (-0.38, 0.40, -0.69)
FOG_DENSITY = 1.0  # Atmospheric haze; close-range aiming remains unobstructed.

# Initial lift from desktop/window-manager placement; F9/F10 move in 12px steps.
WINDOW_RAISE_PIXELS = 28
OBSTACLE_SPACING = 13.0  # sparse, legible traversals without cluttering blind corners
DEFAULT_SEED = 1944
SEED = DEFAULT_SEED
WORLD_BUILD_LOCK = threading.RLock()
SEGMENT = 1.0
CHUNK_LEN = 80.0
JUNCTION_FIRST = 64
JUNCTION_SPACING = 72  # For now all branching remains local; no global graph/A* yet.
BRANCH_LENGTH = 12.0  # base length; individual dead ends vary
CORNER_DEPTH = 1.50   # physical depth into branch reached by curved corner traversal
CORNER_MAIN_OFFSET = 1.15  # metres before junction to start continuous bend
BRANCH_SAFE_OBSERVE_BACK = 2.47  # V3.10.1: compromise: see into opening while stopping behind its edge
COVER_EYE_CLEARANCE = 0.49  # avoid scraping eye against soil bank
BEND_FIRST = 11
BEND_SPACING = 34
# Stream world geometry in the background; only the main thread touches OpenGL.
PREFETCH_AHEAD = 5
RETAIN_BEHIND = 2
UPLOADS_PER_FRAME = 1
# Debug scrub: debounce autorepeated key presses, hold current scene until
# centre, next, and previous chunk exist. Never move into unbuilt geometry.
JUMP_KEY_SETTLE_SECONDS = 0.23
JUMP_MESH_RETRIES = 2
CHUNK_FAILURE_BACKOFF_SECONDS = 12.0  # transient errors may recover
MESH_CACHE_CHUNKS = 6  # bounded CPU-only cache (~40 MB typical)
WALL_HALF_WIDTH = 1.12
FAR = 290.0

MOVES = [
    ('STÅR STILLA', 0.0, 1.55, 0.0),
    ('KNÄSTÅENDE', 0.0, 1.00, 0.0),
    ('LIGGANDE', 0.0, 0.43, 0.0),
    ('KRYPER', 0.30, 0.46, 0.48),
    ('GÅR PÅ HUK', 0.68, 1.04, 0.48),
    ('GÅR', 1.15, 1.55, 0.66),
    ('SNABBGÅR', 1.65, 1.55, 0.74),
    ('JOGGAR', 2.30, 1.53, 0.88),
    ('SPRINGER', 3.35, 1.50, 1.08),
]

VERT = '''#version 330
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
in float in_layer;
uniform mat4 mvp;
uniform float time_seconds;
out vec3 v_normal;
out vec2 v_uv;
flat out int v_layer;
out float v_distance;
out vec3 v_world;
uniform vec3 camera_position;
void main(){
    vec3 mesh_pos=in_pos;
    // Only layer 26 (hanging fabric/net) sways. Sandbags, enemies and soil
    // stay perfectly still. Attachment at uv.y=0 is fully anchored.
    if (in_layer>25.5 && in_layer<26.5) {
        float free_edge=clamp(in_uv.y,0.0,1.0);
        float breeze=sin(time_seconds*.83 + in_pos.x*.74 + in_pos.z*.63);
        float ripples=sin(time_seconds*1.43 + in_pos.x*2.32 - in_pos.y*1.24);
        mesh_pos.x+=free_edge*(.024*breeze + .011*ripples);
        mesh_pos.z+=free_edge*(.035*breeze - .012*ripples);
    }
    gl_Position=mvp*vec4(mesh_pos,1.0);
    v_world=mesh_pos;
    v_normal=in_normal;
    v_uv=in_uv;
    v_layer=int(in_layer+0.5);
    v_distance=length(mesh_pos-camera_position);
}
'''
FRAG = '''#version 330
in vec3 v_normal;
in vec2 v_uv;
flat in int v_layer;
in float v_distance;
in vec3 v_world;
uniform sampler2DArray materials;
uniform sampler2DArray material_normals;
uniform int use_pbr_normals;
uniform int lamp_count;
uniform vec3 lamp_positions[8];
uniform float fog_strength;
uniform vec3 camera_position;
uniform vec3 sun_direction;
uniform vec3 moon_direction;
out vec4 f_color;

// 20 = excavated earth, 21 = trench floor, 22 = world-grid terrain.
// They deliberately do NOT index the 20-layer texture array.
float terrain_pattern(vec3 p) {
    return 0.31*sin(p.x*1.75+p.z*0.87+0.63*sin(p.y*3.0))
         + 0.28*cos(p.z*2.31-p.y*1.82+p.x*0.49)
         + 0.23*sin(p.x*4.12+p.y*3.17-p.z*2.92)
         + 0.11*cos(p.z*7.08+p.y*5.91);
}
bool linear_surface() {
    return v_layer==20 || v_layer==21 || v_layer==8 ||
           v_layer==13 || v_layer==14 || v_layer==15 || v_layer==16 || v_layer==18;
}
vec3 sample_color(vec2 uv,int layer) {
    // This photographed board sheet has horizontal grain; the other sheets
    // and all metric plank UVs use V along the timber.
    if(layer==15)uv=uv.yx;
    vec3 c=texture(materials,vec3(uv,float(layer))).rgb;
    // Albedo is sRGB; normal data stays linear. Preserve unrelated actor/VFX grading.
    return linear_surface() ? pow(c,vec3(2.2)) : c;
}
vec3 detail_normal(vec3 base,vec2 uv,int layer,float strength) {
    if(layer==15)uv=uv.yx;
    vec3 dp1=dFdx(v_world),dp2=dFdy(v_world);
    vec2 a=dFdx(uv),b=dFdy(uv);
    float determinant=a.x*b.y-a.y*b.x;
    if(abs(determinant)<1e-9 || use_pbr_normals==0)return base;
    vec3 t=(dp1*b.y-dp2*a.y)/determinant;
    vec3 bt=(-dp1*b.x+dp2*a.x)/determinant;
    t-=base*dot(base,t);bt-=base*dot(base,bt);
    if(min(dot(t,t),dot(bt,bt))<1e-10)return base;
    // Both axes come from the UV derivatives, not screen winding. This also
    // handles mirrored banks and the underside of the tunnel correctly.
    vec3 nm=texture(material_normals,vec3(uv,float(layer))).xyz*2.0-1.0;
    return normalize(normalize(t)*nm.x*strength+normalize(bt)*nm.y*strength+base*max(.1,nm.z));
}
void soil_texture(vec3 base,int layer,out vec3 color,out vec3 normal) {
    // Shared metre-scale projection across walls, floors, roof and chunk seams.
    // Avoid stretched vertical UVs and keep photographed detail in all orientations.
    vec3 w=pow(abs(base),vec3(4.));w/=max(.0001,w.x+w.y+w.z);
    vec2 ux=v_world.zy*.80,uy=v_world.xz*.80,uz=v_world.xy*.80;
    color=sample_color(ux,layer)*w.x+sample_color(uy,layer)*w.y+sample_color(uz,layer)*w.z;
    normal=normalize(detail_normal(base,ux,layer,.48)*w.x+
                     detail_normal(base,uy,layer,.48)*w.y+
                     detail_normal(base,uz,layer,.48)*w.z);
}
// V3.9.2 real depth shadow maps; shared by world and actors.
''' + SHADOW_GLSL + '''
void main(){
    vec3 n=normalize(v_normal);
    vec3 col;
    vec2 uv=v_uv;
    if (v_layer == 20 || v_layer == 21) {
        vec3 soil,soil_n,inclusions,inclusions_n;
        soil_texture(n,0,soil,soil_n);
        soil_texture(n,v_layer==21 ? 4 : 1,inclusions,inclusions_n);
        float deposits=smoothstep(-.35,.45,terrain_pattern(v_world*.37));
        float blend=(v_layer==21 ? .32 : .22)*deposits;
        col=mix(soil,inclusions,blend);
        n=normalize(mix(soil_n,inclusions_n,blend));
    } else if (v_layer==26) {
        // Dedicated hanging cloth material; 26 is a shader-only material ID.
        // Canvas and dirty bag-fabric blend without new texture dependencies.
        float grime=smoothstep(-.3,.52,terrain_pattern(v_world*1.9));
        col=mix(sample_color(uv*1.2,9),sample_color(uv*.92,19),.40+.34*grime);
    } else if(v_layer==3 || v_layer==22) {
        // World-spaced scorched earth islands between the original humus.
        // These clusters align between streamed chunks (no repeating UVs).
        float scattered=terrain_pattern(v_world*vec3(.23,1.0,.23));
        float scorched=smoothstep(.31,.66,scattered)*.42;
        col=mix(sample_color(uv,3),sample_color(uv*.88,12),scorched);
    } else {
        col=sample_color(uv,v_layer);
    }
    // V3.10: micro-variation is continuous in WORLD metres across chunks,
    // rather than a flat brightness grade or repeated per-triangle bands.
    // This is a fallback enhancement; genuine CC0 Poly Haven albedos/normals
    // remain preferred when installed. No extra sampler or VRAM needed.
    if (v_layer==20 || v_layer==21 || v_layer==22 || v_layer==3) {
        float grit=terrain_pattern(v_world*vec3(3.6,4.1,3.6)+vec3(4.,2.,7.));
        float deposits=smoothstep(.13,.54,terrain_pattern(v_world*.61));
        col *= .96 + .025*grit + .025*deposits;
        if (v_layer==20) {
            // Light clay seams follow the excavated surface; irregular strata
            // break up the otherwise evenly-grey wall shading.
            float strata=sin(v_world.y*9.4 + sin(v_world.x*1.73+v_world.z*.87)*1.9);
            col *= 1.0 - .025*smoothstep(.58,.91,strata)*deposits;
            // Rain streaks and wet seams follow gravity in world space. Keep
            // ceiling/floor response separate; no change to geometry or IDs.
            float vertical=1.0-smoothstep(.35,.82,abs(n.y));
            float rivulet=sin(v_world.x*12.3+v_world.z*9.7
                              +.32*sin(v_world.y*2.8+v_world.z));
            float wet=smoothstep(.48,.91,rivulet)*deposits*vertical;
            float seep=smoothstep(.06,.58,terrain_pattern(
                       v_world*vec3(.8,.22,.8)+vec3(7.,1.,3.)))*vertical;
            col*=mix(vec3(1.0),vec3(.66,.70,.64),clamp(wet*.25+seep*.22,0.0,.4));
        }
    } else if (v_layer==8 || (v_layer>=13 && v_layer<=18)) {
        // Variable timber dampness / soot on the shoring, never uniform beige.
        float staining=terrain_pattern(v_world*vec3(1.3,2.5,1.1)+vec3(2.,0.,8.));
        col*=.84+.19*smoothstep(-.45,.44,staining);
    } else if (v_layer==6) {
        // Patchy leaf clusters: depth-readable foliage without extra textures
        // or alpha billboards. Matches deciduous and layered conifer crowns.
        float canopy=terrain_pattern(v_world*vec3(1.8,1.3,1.8));
        float speck=terrain_pattern(v_world*vec3(4.7,3.6,4.7)+vec3(5.,1.,3.));
        col*=mix(vec3(.69,.75,.65),vec3(1.04,1.02,.91),
                 smoothstep(-.42,.43,canopy+.24*speck));
    }
    // The soil branch already blended matching color/normal channels.
    if (v_layer!=20 && v_layer!=21) {
        int layer=v_layer==22 ? 3 : (v_layer == 26 ? 19 : v_layer);
        n=detail_normal(n,uv,layer,.48);
    }
    vec3 sun=normalize(sun_direction);
    vec3 moon=normalize(moon_direction);
    float diffuse=max(dot(n,sun),0.0);
    float moonlight=max(dot(n,moon),0.0);
    // Dusk: muted ambient earth, directional sunlight and cool moon fill.
    // Only the directional components are shadowed: ambient never disappears.
    float sun_vis=(shadow_mode!=0) ? shadow_pcf(shadow_sun,shadow_sun_mvp,
                                v_world,max(.0006,.0020*(1.0-diffuse))) : 1.0;
    float moon_vis=(shadow_mode!=0) ? shadow_pcf(shadow_moon,shadow_moon_mvp,
                                 v_world,max(.0006,.0018*(1.0-moonlight))) : 1.0;
    float ambient=linear_surface() ? .28 : .37;
    float brightness=ambient + .52*diffuse*sun_vis + .10*moonlight*moon_vis;
    col*=brightness;
    for (int i=0; i<8; i++) {
        if(i>=lamp_count) break;
        vec3 delta=lamp_positions[i]-v_world;
        float r2=dot(delta,delta);
        float nearLamp=pow(max(0.0,1.0-r2/42.25),2.0);
        float facing=max(dot(n,normalize(delta)),0.0);
        float lamp_vis=(shadow_mode!=0 && i==shadow_lamp_index) ?
                       mix(1.0,shadow_pcf(shadow_lamp,shadow_lamp_mvp,v_world,.0025),
                           shadow_lamp_weight) : 1.0;
        vec3 lamp_color=vec3(.31,.19,.075);
        if(linear_surface())lamp_color=pow(lamp_color,vec3(2.2));
        col+=lamp_color*nearLamp*(.23+.77*facing)*lamp_vis;
    }
    if (v_layer==12) {
        vec3 towardEye=normalize(camera_position-v_world);
        vec3 halfLight=normalize(sun+towardEye);
        float spec=pow(max(dot(n,halfLight),0.0),26.0);
        col+=vec3(.085,.077,.058)*spec*sun_vis;
    }
    // V3.9.5: small directional glints from damp patches on the trench floor.
    // The wetness mask is WORLD-space and coherent between streamed chunks.
    // Sun and moon geometry remain authoritative; this is not screen bloom.
    if (v_layer==21) {
        float damp=smoothstep(.02,.59,-terrain_pattern(v_world*.78));
        vec3 v=normalize(camera_position-v_world);
        vec3 half_sun=normalize(sun+v);
        float spec=pow(max(dot(n,half_sun),0.0),36.0);
        float wet_response=damp*.095*diffuse*sun_vis*spec;
        col+=vec3(.65,.63,.55)*wet_response;
    }
    if(linear_surface())col=pow(max(col,vec3(0.0)),vec3(1.0/2.2));
    col=pow(max(col,vec3(0.0)),vec3(.96));
    if (v_layer==11) col=vec3(1.0,.54,.16)*1.8;
    if (v_layer==12) col=mix(col,vec3(.10,.07,.045),.4);
    // Volumetric *approximation*: visible distant mist banks and drifting
    // battlefield smoke patches. Near field (< 9m) remains clear for aiming.
    float outside=smoothstep(1.0,3.1,v_world.y);
    vec2 smokeCell=floor(v_world.xz/19.0);
    float cluster=fract(sin(dot(smokeCell,vec2(12.9898,78.233)))*43758.5453);
    float breeze=.5+.5*sin(v_world.x*.067+v_world.z*.043
                          +sin(v_world.x*.027)*2.1);
    float smoke=pow(max(0.0,breeze),2.0)*smoothstep(.43,.84,cluster);
    float distant=max(0.0,v_distance-8.0);
    float density=(.007 + .015*outside + .013*smoke)*fog_strength;
    float fog=1.0-exp(-distant*density);
    float mist_band=.14*outside*smoothstep(24.,65.,v_distance);
    // Neutral charcoal-brown mist keeps far trench geometry readable without
    // bleaching the wall textures to light-grey as in the earlier preview.
    col=mix(col,vec3(.295,.302,.297),clamp(fog+mist_band,0.,.83));
    // Gentle filmic highlight shoulder + warmer, deeper mid-tones. All shared
    // with enemy shader to avoid disjoint compositing in direct comparison.
    col=max(col,vec3(0.0));
    col=pow(col,vec3(1.075))/(vec3(1.0)+.095*col);
    f_color=vec4(clamp(col,0.0,1.0),1.0);
}
'''
ACTOR_VERT = VERT.replace(
    'uniform vec3 camera_position;',
    'uniform vec3 camera_position;\nuniform mat4 model;'
).replace(
    'gl_Position=mvp*vec4(mesh_pos,1.0);',
    'vec4 world_pos=model*vec4(mesh_pos,1.0);\n    gl_Position=mvp*world_pos;'
).replace(
    'v_world=mesh_pos;',
    'v_world=world_pos.xyz;'
).replace(
    'v_normal=in_normal;',
    'v_normal=normalize(transpose(inverse(mat3(model)))*in_normal);'
).replace(
    'v_distance=length(mesh_pos-camera_position);',
    'v_distance=length(world_pos.xyz-camera_position);'
)

# Enemy-only shading. The trench keeps its calibrated dusk appearance while
# cloth silhouettes receive a soft low-angle fill (not an emissive glow).
# No extra textures, no OpenGL feature beyond GLSL 330.
ACTOR_FRAG = FRAG.replace(
    '    col*=brightness;',
    """    col*=brightness;
    if(v_layer==23 || v_layer==24) {
        vec3 towardCamera=normalize(camera_position-v_world);
        float grazing=pow(1.0-max(dot(n,towardCamera),0.0),2.0);
        col+=vec3(.070,.068,.049)*grazing;
    }"""
)

SKYVERT = '''#version 330
in vec2 xy;
out vec2 uv;
void main(){uv=xy*.5+.5;gl_Position=vec4(xy,0.9999,1.0);}
'''
SKYFRAG = '''#version 330
in vec2 uv;
out vec4 f_color;
uniform vec3 sky_right;
uniform vec3 sky_up;
uniform vec3 sky_forward;
uniform vec3 sun_direction;
uniform vec3 moon_direction;
uniform float tan_half_fov;
uniform float view_aspect;
void main(){
    // A true infinitely distant sky: camera TRANSLATION must not affect it.
    // Derive one world-space viewing ray per pixel using camera orientation.
    vec2 ndc = uv*2.0-1.0;
    vec3 ray = normalize(sky_forward
      + sky_right*(ndc.x*tan_half_fov*view_aspect)
      + sky_up*(ndc.y*tan_half_fov));
    vec3 sun=normalize(sun_direction);
    float altitude=ray.y;
    vec3 horizon=vec3(.39,.345,.33);
    vec3 high=vec3(.105,.15,.205);
    float sky_height=smoothstep(-.03,.70,altitude);
    vec3 col=mix(horizon,high,sky_height);
    float sun_angle=acos(clamp(dot(ray,sun),-1.0,1.0));
    float disc=1.0-smoothstep(.017,.024,sun_angle);
    float glow=exp(-sun_angle*sun_angle/0.018);
    col+=vec3(.27,.13,.06)*glow;
    col=mix(col,vec3(.96,.69,.44),disc*.90);
    vec3 moon=normalize(moon_direction);
    float moonAngle=acos(clamp(dot(ray,moon),-1.0,1.0));
    float moonDisc=1.0-smoothstep(.007,.011,moonAngle);
    col=mix(col,vec3(.68,.74,.82),moonDisc*.70);
    // Atmospheric distant clouds/haze stay fixed in WORLD viewing direction.
    float haze=pow(1.0-smoothstep(-.015,.22,altitude),2.0);
    col=mix(col,vec3(.30,.30,.30),.19*haze);
    // Dark earth backstop is only a last-resort mask for tiny geometry gaps.
    // Its boundary also follows world-space horizon, not screen UVs.
    vec3 soil=vec3(.095,.073,.055);
    col=mix(col,soil,1.0-smoothstep(-.085,-.012,altitude));
    f_color=vec4(col,1.0);
}
'''

def smooth(a, b, t):
    return a + (b-a)*t

# Seeded route in actual X/Z world space. Route distance is arc length, not world Z.
def noise(i, salt=0):
    v=math.sin((i+SEED*.31+salt*17.71)*127.1)*43758.5453
    return v-math.floor(v)


def eased(t):
    t=max(0.,min(1.,t))
    return t*t*(3-2*t)


def turn_for_meter(s):
    """Repeated angular Z/W traverses, separated by calmer stretches.

    Every bend is eased over several metres; exact integral is independent of
    frame rate and chunks. Paired turns prevent endless turning in one direction.
    """
    if SEED != DEFAULT_SEED:
        # A separate integer stream: adding dressing must never reroute a world.
        # Bounded absolute headings preserve forward progress and avoid loops.
        def unit(index):
            v=(SEED ^ 0xA341316C ^ (index*0x9E3779B9)) & 0xffffffff
            v=((v ^ (v>>16))*0x7FEB352D) & 0xffffffff
            v=((v ^ (v>>15))*0x846CA68B) & 0xffffffff
            return (v ^ (v>>16))/4294967296.
        spacing=30.+8.*unit(-1)
        first=9.+6.*unit(-2)
        block=int((s-first)//spacing)
        if block<0:return 0.
        previous=0. if block==0 else (unit(block-1)*2.-1.)*.85
        target=(unit(block)*2.-1.)*.85
        duration=12.+6.*unit(block+1000003)
        start=first+spacing*block
        return (target-previous)*(eased((s-start)/duration)-eased((s-start-1)/duration))
    # Compatibility path: default 1944 retains every original coordinate.
    block=int((s-11)//34)
    if block<0:return 0.0
    start=11+34*block
    pattern=(.87,-1.40,.87,-.82,1.34,-.82)
    magnitude=pattern[block%len(pattern)]*(.82+.27*noise(block,2))
    duration=9.0+noise(block,4)*7.0
    local=magnitude*(eased((s-start)/duration)-eased((s-start-1)/duration))
    # Large-scale directional changes: longer east/west excursions instead
    # of a world that always advances along Z. Compensating turns preserve
    # broad route continuity without generating tight closed loops.
    sweep=0.0
    for begin,angle,span in ((155.,1.30,21.),(274.,-2.35,25.),
                             (412.,1.05,24.)):
        relative=(s%560.0)-begin
        if -1.0<relative<span+1.0:
            sweep+=angle*(eased((relative+1.0)/span)-eased(relative/span))
    # Additional long-wave bends beyond the already validated first encounters.
    # Geometry, obstacles, enemy poses and shots still share this ONE route.
    distant=eased((s-430.0)/90.0)
    meander=distant*(.010*math.sin(.023*s+1.7)+.006*math.sin(.043*s-.9))
    return local+sweep+meander


# Cache integer-metre centreline positions. Each step rotates smoothly.
_ROUTE=[(0.,0.,0.)]  # x,z,heading

def set_world_seed(seed):
    """Switch the existing generator and its integrated centreline atomically."""
    global SEED
    with WORLD_BUILD_LOCK:
        seed=int(seed)
        if seed!=SEED:
            _ROUTE[:]=[(0.,0.,0.)]
            SEED=seed


def route(s):
    s=max(0.,float(s));i=int(s)
    while len(_ROUTE)<=i+1:
        j=len(_ROUTE)
        x,z,h=_ROUTE[-1]
        h+=turn_for_meter(j-.5)
        _ROUTE.append((x+math.sin(h),z+math.cos(h),h))
    x,z,h=_ROUTE[i]
    x2,z2,h2=_ROUTE[i+1]
    f=s-i
    return (x+(x2-x)*f,z+(z2-z)*f,h+(h2-h)*f)


def wall_height(s, side):
    """Terrain surface relative to trench floor, deterministic in route metres.

    Large, slow changes make noticeably shallow and deep trenches, with smaller
    irregularities on top. Left and right share the large-scale profile so the
    trench doesn't look like a sawtooth or open on only one side.
    """
    shared = (0.79 * math.sin(.032 * s + .55)
              + 0.43 * math.sin(.071 * s - .95)
              + 0.18 * math.sin(.145 * s + .8))
    bank = 0.16 * math.sin(.064 * s + side * 1.2)
    return max(1.0, min(3.0, 2.0 + shared + bank))


def sides(s,offset):
    x,z,h=route(s)
    return (x+math.cos(h)*offset,z-math.sin(h)*offset)


def world(s,offset,y):
    x,z=sides(s,offset)
    return (x,y,z)


def junction_marker(index):
    return JUNCTION_FIRST+JUNCTION_SPACING*index


def nearest_junction(s):
    i=round((s-JUNCTION_FIRST)/JUNCTION_SPACING)
    if i<0:return None
    return (junction_marker(i), i)


def junction(s):
    candidate=nearest_junction(s)
    if candidate is None:return None
    marker,i=candidate
    if abs(s-marker)>=2.0:return None
    return (marker,1 if noise(i,23)>.5 else -1)


def junction_side(index):
    # When both sides exist, the SoldierBrain selects one reproducibly.
    return 1 if noise(index,23)>.5 else -1


def junction_kind(index):
    """Local junction topology: one open continuation, T, Y or X.

    'I' sections occur between junctions. This does not imply that side
    branches yet connect into a globally navigable graph.
    """
    return ('T','Y','T','X','T','Y','X','T')[index % 8]


def junction_sides(index):
    return (-1,1) if junction_kind(index)=='X' else (junction_side(index),)


def branch_type(index):
    return 'dugout' if noise(index,1103)<.78 else 'open'


def branch_half_width(index, along):
    # Portal width is shared with the MAIN route excavation mask. Do not alter
    # the mouth: taper variation only after entering the branch, continuously.
    width=2.0 + (1.12-2.0)*eased(along/3.4)
    if branch_type(index)=='dugout':
        room_at=7.4 + 1.2*noise(index,1281)
        width+=.28*math.exp(-((along-room_at)/2.15)**2)
    # Separate excavated chambers and cramped choke points for each seed.
    blend=eased((along-2.0)/3.2)
    period=3.8 + 1.8*noise(index,12013)
    swing=(.075+.115*noise(index,12014))*math.sin(along*math.tau/period+index*1.37)
    slow=.075*math.sin(along*.41+index*.91)
    return max(1.06,width+.035*math.sin(along*.8+index)*eased(along/2.0)
               +(swing+slow)*blend)


def branch_length(index):
    return 9 + int(noise(index,92)*8)  # 9–16m exploratory side tunnels


def branch_center_shift(index, along):
    # Small offset makes side tunnels bent rather than ruler-straight.
    progression=eased((along-3.0)/9.0)
    skew=(.32 if noise(index,1173)>.5 else -.32)*max(0.0,along-3.0) if junction_kind(index)=='Y' else 0.0
    # Narrow S-shaped lateral bends vary by branch, but fade to zero at mouth.
    wiggle=(.09+.15*noise(index,12131))*math.sin(along*.53+index*.71)
    return (noise(index,93)-.5)*2.0*progression + skew + wiggle*progression


def cover_distance(s, side):
    """Eye position is inset from bank even in the narrowest trench.

    This is a cautious geometric clearance rule, not object collision.
    """
    return max(0., min(.58, wall_width(s, side)-COVER_EYE_CLEARANCE))


def bend_marker(index):
    return BEND_FIRST + BEND_SPACING * index


def bend_direction(index):
    """Turn signed from actual generated main route, not assumed from seed."""
    start=bend_marker(index)
    before=route(max(0.,start-1.0))[2]
    after=route(start+15.0)[2]
    return math.atan2(math.sin(after-before), math.cos(after-before))


def near_junction(s, radius=12.):
    candidate=nearest_junction(s)
    return candidate is not None and abs(s-candidate[0])<radius


class MeshBuilder:
    def __init__(self):self.vertices=[]
    def smooth_quad(self,a,b,c,d,layer,uv,center,radii):
        # Radial normals for curved foliage; original quad retains flat faces
        # needed for trench planks, broken rocks and hard corners.
        cx,cy,cz=center
        rx,ry,rz=radii
        for p,q,r,st in ((a,b,c,(uv[0],uv[1],uv[2])),(a,c,d,(uv[0],uv[2],uv[3]))):
            for pos,tc in zip((p,q,r),st):
                x,y,z=pos
                normal=np.array(((x-cx)/(rx*rx),(y-cy)/(ry*ry),(z-cz)/(rz*rz)),dtype=float)
                norm=np.linalg.norm(normal)
                if norm<1e-5:normal=np.array((0.,1.,0.))
                else:normal/=norm
                self.vertices.extend((*pos,*normal,*tc,float(layer)))
    def quad(self,a,b,c,d,layer,uv,normals=None):
        for ids in ((0,1,2),(0,2,3)):
            points=(a,b,c,d)
            normal=vec_normal(*(points[i] for i in ids)) if normals is None else None
            for i in ids:
                n=normal if normals is None else normals[i]
                self.vertices.extend((*points[i],*n,*uv[i],float(layer)))
    def as_array(self):return np.asarray(self.vertices,dtype='f4').reshape(-1,9)


def vec_normal(a,b,c):
    v=np.cross(np.array(b)-a,np.array(c)-a)
    d=np.linalg.norm(v)
    return v/d if d>.00001 else np.array([0.,1.,0.])


def surface_normal(pu0,pu1,pv0,pv1,toward):
    """Differential shading normal; never modifies the rendered/shot positions."""
    a=tuple(pu1[i]-pu0[i] for i in range(3))
    b=tuple(pv1[i]-pv0[i] for i in range(3))
    n=(a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    length=math.sqrt(sum(v*v for v in n))
    if length<1e-10:return tuple(toward)
    sign=1. if sum(n[i]*toward[i] for i in range(3))>=0 else -1.
    return tuple(v*sign/length for v in n)


def earth_surface_normal(s,y_or_lateral,side=0):
    """Global derivative samples keep shared vertices identical across chunks."""
    e=.04
    if side==0:
        def p(t,l):return world(t,l,floor_y(t,l))
        l=y_or_lateral
        return surface_normal(p(s-e,l),p(s+e,l),p(s,l-e),p(s,l+e),(0.,1.,0.))
    y=y_or_lateral
    def p(t,h):return world(t,side*wall_offset(t,side,h,wall_height(t,side)),h)
    center=world(s,0.,y);edge=p(s,y)
    toward=tuple(center[i]-edge[i] for i in range(3))
    return surface_normal(p(s-e,y),p(s+e,y),p(s,y-e),p(s,y+e),toward)


def landscape_type(s):
    # Long regions with soft transitions. The trench walls reflect local soil.
    return int(max(0.,s)//155.) % 5


def wall_width(s, side):
    """Trench half-width: coherent narrowing/widening, asymmetric banks.

    The variation is slow enough to be walkable and deterministic across chunks.
    Width from ~1.55m to ~2.95m overall, with short irregularities.
    """
    valley = (.25 * math.sin(s*.026 + .6)
              + .15 * math.sin(s*.071 - .9)
              + .09 * math.sin(s*.13 + side*1.8))
    bias = .14 * math.sin(s*.035 + side*.8)
    width = WALL_HALF_WIDTH + valley + bias + .035 * math.sin(s*.83+side)
    return max(.77, min(1.58, width))


def wall_offset(s, side, y, height):
    """Actual soil bank surface, with uneven overhangs, undercuts and bedding.

    All vertices at the same world (s, y) use identical coordinates, so
    neighbouring chunks and material bands stitch without T-shaped seams.
    """
    u=max(0.,min(1.,y/max(.1,height)))
    main=wall_width(s,side)
    # The excavation gets wider near the rim, but not uniformly.
    shape=(.14 * u*u - .065 * math.sin(u*math.pi))
    ripple=(.085*math.sin(s*.75 + u*11.1 + side)
            +.042*math.sin(s*2.06 - u*23.0 + side*2))
    # Gradually decrease texture displacement at the floor and lip: adjacent
    # floor and terrain meshes retain an overlapping watertight transition.
    ripple *= math.sin(math.pi*u)**.7
    # Broad excavated scallops, only OUTWARD from the previous bank. Keep
    # foot, lip and covered passages exact: no narrower walkable corridor.
    exposed=1.-eased(tunnel_open(s)*4.)
    scallop=(.10+.12*(.5+.5*math.sin(s*.48+side*1.7)))
    scallop*=math.sin(math.pi*u)**2 * eased((y-.35)/.65)*exposed
    return max(.68,main+shape+ripple)+scallop


def surface_material(s, side, height_fraction):
    # Rocks at the lower cut, deeper clay and loam above, depending on region.
    biome=landscape_type(s)
    if biome==2: return (1, 0, 1, 2)  # rocky rise
    if biome==3: return (0, 1, 2, 3)  # cultivated open field
    if biome==4: return (1, 0, 2, 3)  # shattered / damp terrain
    if biome==1: return (0, 2, 3, 3)  # woodland humus
    return (1, 0, 2, 3)

def tunnel_open(s):
    """Roof envelope 0..1; deterministic world metres, no objects preloaded.

    Each cut-through has an open portal, an underground middle and an exit.
    Kept away from regular junction markers by selecting a broad period.
    """
    u=(max(0.0,float(s))-280.0) % 504.0
    return eased((u-21.0)/5.0)*(1.0-eased((u-45.0)/5.0))


def tunnel_drop(s):
    """World-space ramp under an intact raised earth roof (metres)."""
    u=(max(0.0,float(s))-280.0) % 504.0
    return -0.72*eased((u-11.0)/12.0)*(1.0-eased((u-45.0)/12.0))


def floor_y(s, lateral=0.0):
    """Shared excavated floor height for GPU vertices and first-person eye.

    Fine tread deformation fades towards the bank. No displacement seams at
    neighbouring chunks or special doorway teleports.
    """
    margin=min(wall_width(s,-1),wall_width(s,1))
    edge_factor=1.0-eased((abs(lateral)/max(.3,margin))*.95)
    ruts=(.046*math.sin(s*.38+lateral*2.0)
          +.024*math.cos(s*.91-lateral*3.6)
          +.025*math.sin(s*.16+lateral*5.1))
    return -.048 + tunnel_drop(s) + ruts*edge_factor


def crater_height(s, offset):
    """Local blast craters with a depressed bowl and raised ejected rim.

    Crater sites are computed only from nearby indices, independent of chunk
    loading or camera position. The trench itself stays unaffected.
    """
    if abs(offset)<5.0:return 0.0
    result=0.0
    first=max(0,int(s//24)-2)
    for k in range(first,first+5):
        center_s=11.0+24*k+7.0*(noise(k,360)-.5)
        radius=2.4+3.4*noise(k,361)
        side=-1 if noise(k,362)<.50 else 1
        center_lateral=side*(7.+26.*noise(k,363))
        dist=math.hypot(s-center_s,offset-center_lateral)
        if dist>radius*1.65:continue
        bowl=(1.0-eased(dist/radius))**1.3
        lip=max(0.0,1.0-abs(dist/radius-1.12)/.35)
        result+=-(.35+.62*noise(k,364))*bowl +(.12+.17*noise(k,365))*eased(lip)
    return result


def hill_y(s,side,offset):
    # Same deterministic terrain for meshes and all exterior scenery.
    d=max(0.,abs(offset)-wall_width(s,side))
    base=wall_height(s,side)-.055
    roll=1.0-math.exp(-d/8.0)
    # Mid- and long-distance slopes are substantial enough to silhouette
    # behind the trees, instead of a flat world terminating after 60m.
    hills=(.34*math.sin(s*.027+side*.8+offset*.045)
           +.60*math.sin(offset*.026+s*.018)
           +.24*math.sin(s*.083+offset*.064))
    far_rise=eased((d-12.0)/55.0)*(.35+.82*math.sin(s*.012+offset*.013)**2)
    crater=crater_height(s,offset)*eased((d-1.0)/3.0)
    return base+roll*hills+far_rise+crater


def _cutout_polygon_intersects_ground(patch, footprint):
    """2D SAT overlap for convex X/Z ground patches and cutout footprints.

    Vertex/centroid-only sampling fails when a long ground strip *crosses*
    a narrow buried tunnel without containing a sampled vertex. This test
    examines edges of BOTH polygons, including near-parallel intersections.
    """
    for polygon in (patch, footprint):
        for idx, (ax, az) in enumerate(polygon):
            bx, bz = polygon[(idx + 1) % len(polygon)]
            nx, nz = az - bz, bx - ax
            if nx * nx + nz * nz < 1e-12:
                continue
            pvals = [x * nx + z * nz for x, z in patch]
            fvals = [x * nx + z * nz for x, z in footprint]
            if max(pvals) < min(fvals) - 1e-7 or max(fvals) < min(pvals) - 1e-7:
                return False
    return True


def _branch_footprint(i, side, cx, cz, right, forward, near, along, margin=.06):
    """One swept quad of an excavated side passage, in world X/Z."""
    def edge(d, flank):
        w=branch_half_width(i,d)+margin
        shift=branch_center_shift(i,d)
        return (cx + side*right[0]*(near+d) + forward[0]*(shift+flank*w),
                cz + side*right[1]*(near+d) + forward[1]*(shift+flank*w))
    a,b=along
    return (edge(a,-1),edge(a,1),edge(b,1),edge(b,-1))


def _main_route_footprint(a,b,margin=.06):
    """A navigable cross-section, projected in world X/Z, not route offsets."""
    def point(s,side):
        y=wall_height(s,side)
        edge=max(.35,wall_offset(s,side,y,y)+margin)
        x,z=sides(s,side*edge)
        return (x,z)
    return (point(a,-1),point(a,1),point(b,1),point(b,-1))


def _main_tunnel_footprint(a,b):
    """Roof cuts slightly OUTSIDE the lip, unlike open bank cutouts."""
    # The tunnel earth shell joins the outdoor grid through a finite bank
    # skirt. Clear the same narrow lateral footprint; do not leave roofless
    # cells or remove terrain under an unroofed portal.
    return _main_route_footprint(a,b,.85)


def world_excavation_masks(start):
    """Bounded X/Z spatial hash of actual trench/dugout excavation areas.

    WHY: The old outdoor terrain was generated as up to 145m-long *normal*
    offsets from a bending centreline. Those world-space polygons can fold
    BACK OVER THE TRENCH even hundreds of metres away from their source!
    Protect the whole 3D trench with spatial cutouts, not just its roofs.

    Build each chunk's nearby footprints from the deterministic seed. Spatial
    cells keep triangle queries economical despite the longer routes.
    """
    grid=16.0
    masks={}
    def add(poly):
        xs=[v[0] for v in poly];zs=[v[1] for v in poly]
        x0,x1,z0,z1=min(xs),max(xs),min(zs),max(zs)
        item=(x0,x1,z0,z1,poly)
        for ix in range(math.floor(x0/grid),math.floor(x1/grid)+1):
            for iz in range(math.floor(z0/grid),math.floor(z1/grid)+1):
                masks.setdefault((ix,iz),[]).append(item)
    # Wide world terrain triangles can project across much earlier/later s.
    # The stream remains bounded; extra length affects only cheap 2D masks.
    reach=CHUNK_LEN*4
    first=max(0,math.floor((start-reach-JUNCTION_FIRST)/JUNCTION_SPACING))
    last=max(first,math.ceil((start+CHUNK_LEN+reach-JUNCTION_FIRST)/JUNCTION_SPACING))
    for i in range(first,last+1):
        marker=junction_marker(i)
        cx,cz,h=route(marker)
        right=(math.cos(h),-math.sin(h))
        forward=(math.sin(h),math.cos(h))
        for side in junction_sides(i):
            near=max(wall_width(marker,side),wall_width(marker-2,side),
                     wall_width(marker+2,side))-.18
            length=branch_length(i)
            for k in range(math.ceil(length/.75)):
                a=k*.75;b=min(length,(k+1)*.75)
                # Covered shelters have an outer EARTH shell 1.5m past the
                # interior wall. Cut the world terrain against the whole
                # shell, not just the interior void (V3.1 roof overlap).
                shell_margin=1.58 if branch_type(i)=='dugout' else .06
                add(_branch_footprint(i,side,cx,cz,right,forward,near,
                                      (a,b),margin=shell_margin))
    first_k=max(0,int(start-reach))
    last_k=int(start+CHUNK_LEN+reach)
    for k in range(first_k,last_k):
        a,b=float(k),float(k+1)
        # The cutout MUST match real roof geometry. Previously a roof was
        # claimed whenever either interior probe was nonzero, even for the
        # unroofed metre at each portal. This left unsupported black gaps.
        is_roofed=min(tunnel_open(a),tunnel_open(b))>=.01
        add(_main_tunnel_footprint(a,b) if is_roofed else
            _main_route_footprint(a,b))
    return masks


def ground_patch_intersects_excavation(patch,masks):
    """Exact 2D edge overlap, indexed by coarse world-space hash cells."""
    xs=[p[0] for p in patch];zs=[p[1] for p in patch]
    lo_x,hi_x,lo_z,hi_z=min(xs),max(xs),min(zs),max(zs)
    grid=16.0
    seen=set()
    for ix in range(math.floor(lo_x/grid),math.floor(hi_x/grid)+1):
        for iz in range(math.floor(lo_z/grid),math.floor(hi_z/grid)+1):
            for item in masks.get((ix,iz),()):
                if id(item) in seen:continue
                seen.add(id(item))
                x0,x1,z0,z1,footprint=item
                if x1<lo_x-1e-7 or x0>hi_x+1e-7 or z1<lo_z-1e-7 or z0>hi_z+1e-7:
                    continue
                if _cutout_polygon_intersects_ground(patch,footprint):
                    return True
    return False


def soil_over_excavation(x,z,masks):
    """Compatibility diagnostic: whether a point lies inside an X/Z cutout."""
    p=((x-.0001,z-.0001),(x+.0001,z-.0001),
       (x+.0001,z+.0001),(x-.0001,z+.0001))
    return ground_patch_intersects_excavation(p,masks)



# ---------------------------------------------------------------------------
# Exterior terrain V3.2: a SINGLE world-space heightfield, not curved ribbons.
# ---------------------------------------------------------------------------
# Historically the outer ground was extruded up to 145m along each side of
# the path. Swept ribbons folded back over themselves in X/Z when the trench
# turned: the same position could have TEN different ground surfaces.
# Terrain is now tessellated on one absolute X/Z lattice and emitted once by
# the nearest main-route chunk. Existing excavation masks remove cells from
# the tunnel/trench/dugout footprint, never their solid earth roof shells.
OUTDOOR_TILE_METRES = 12.0
OUTDOOR_CELL_METRES = 3.0
OUTDOOR_RADIUS_METRES = 105.0
OUTDOOR_MASK_SUBDIVISIONS = 5   # 3m / 32 ~9cm at excavation edges


def _terrain_route_segments(start):
    """Local route-window for deterministic nearest-point ownership/height.

    Far-enough overlap means neighbouring chunks calculate the same owner
    even if the route zigzags. These points are pure CPU data, not world props.
    """
    lower=max(0,int(start-CHUNK_LEN*5))
    upper=int(start+CHUNK_LEN*(5+1))+1
    metres=np.arange(lower,upper+1,dtype=float)
    coords=np.asarray([route(float(t))[:2] for t in metres],dtype=float)
    delta=coords[1:]-coords[:-1]
    lens=np.maximum(1e-7,np.sum(delta*delta,axis=1))
    return metres[:-1],coords[:-1],delta,lens


def _nearest_terrain_route(x,z,segments):
    """Return the nearest route metre, signed lateral and planar distance.

    The nearest-point projection is measured against actual X/Z segments,
    never guessed from the route's cumulative distance as a world Z value.
    """
    metres,points,deltas,lengths=segments
    q=np.array((x,z),dtype=float)
    rel=q-points
    projection=np.clip(np.sum(rel*deltas,axis=1)/lengths,0.0,1.0)
    nearest=points+projection[:,None]*deltas
    delta=nearest-q
    index=int(np.argmin(np.einsum('ij,ij->i',delta,delta)))
    point=nearest[index]
    nearest_s=float(metres[index]+projection[index])
    normal=np.array((deltas[index,1],-deltas[index,0]))
    norm=np.linalg.norm(normal)
    if norm>1e-8:normal/=norm
    lateral=float(np.dot(q-point,normal))
    return nearest_s,lateral,float(np.linalg.norm(q-point))


def _exterior_y_world(x,z,segments,point_cache):
    """One well-defined terrain height at each fixed world X/Z coordinate.

    Near the excavation rim, reuse the existing physical bank profile; hills
    and craters continue beyond it. A single function shared by every tile
    guarantees no two outdoor heights at the same world-space position.
    """
    key=(round(x,5),round(z,5))
    stored=point_cache.get(key)
    if stored is not None:return stored
    s,offset,_=_nearest_terrain_route(x,z,segments)
    side=-1 if offset<0 else 1
    y=hill_y(s,side,offset)
    point_cache[key]=y
    return y


def build_exterior_terrain(mesh,start,masks,segments=None):
    """Stream a seamless non-overlapping world-grid landscape for one chunk.

    Tiles have absolute X/Z IDs. Ownership is by closest main-route metre,
    so a terrain square appears in exactly one route chunk rather than being
    duplicated by neighbouring, twisting sweeps. A little subcell recursion
    clips excavations without leaving metre-wide terrain holes beside banks.
    No terrain outside the active/prepared world chunks remains in GPU RAM.
    """
    tile=OUTDOOR_TILE_METRES
    step=OUTDOOR_CELL_METRES
    if segments is None:
        segments=_terrain_route_segments(start)
    section=[route(s) for s in (start,*[start+i*8. for i in range(1,10)],
                                start+CHUNK_LEN)]
    xs=[r[0] for r in section];zs=[r[1] for r in section]
    x0=math.floor((min(xs)-OUTDOOR_RADIUS_METRES)/tile)
    x1=math.ceil((max(xs)+OUTDOOR_RADIUS_METRES)/tile)
    z0=math.floor((min(zs)-OUTDOOR_RADIUS_METRES)/tile)
    z1=math.ceil((max(zs)+OUTDOOR_RADIUS_METRES)/tile)
    height_cache={}
    def emit_cell(ax,az,bx,bz,depth=0):
        polygon=((ax,az),(bx,az),(bx,bz),(ax,bz))
        if ground_patch_intersects_excavation(polygon,masks):
            if depth<OUTDOOR_MASK_SUBDIVISIONS:
                cx=(ax+bx)*.5;cz=(az+bz)*.5
                emit_cell(ax,az,cx,cz,depth+1)
                emit_cell(cx,az,bx,cz,depth+1)
                emit_cell(ax,cz,cx,bz,depth+1)
                emit_cell(cx,cz,bx,bz,depth+1)
            return
        ya=_exterior_y_world(ax,az,segments,height_cache)
        yb=_exterior_y_world(bx,az,segments,height_cache)
        yc=_exterior_y_world(bx,bz,segments,height_cache)
        yd=_exterior_y_world(ax,bz,segments,height_cache)
        # UVs depend ONLY on world X/Z; no seams when a chunk is reloaded.
        mesh.quad((ax,ya,az),(bx,yb,az),(bx,yc,bz),(ax,yd,bz),22,
                  ((ax*.18,az*.18),(bx*.18,az*.18),
                   (bx*.18,bz*.18),(ax*.18,bz*.18)))
    per_side=int(round(tile/step))
    for ix in range(x0,x1):
        ax=ix*tile
        for iz in range(z0,z1):
            az=iz*tile
            center=(ax+tile*.5,az+tile*.5)
            owner_s,_,distance=_nearest_terrain_route(*center,segments)
            if not (start<=owner_s<start+CHUNK_LEN) or distance>OUTDOOR_RADIUS_METRES:
                continue
            for j in range(per_side):
                x= ax+j*step
                for k in range(per_side):
                    z= az+k*step
                    emit_cell(x,z,x+step,z+step)

# --- V1.5 deterministic outdoor scenery, no separate game engine ---

def world_on_ground(s, offset, lift=0.0):
    side = -1 if offset < 0 else 1
    return world(s, offset, hill_y(s, side, offset) + lift)


def exterior_prop_ground(s,offset,masks=None,segments=None,cache=None):
    """Match decoration to ACTUAL generated world-grid floor, or reject it.

    A tree/stone created over a dugout, tunnel portal or a nearby winding
    trench used to stand on the *imagined* local hill surface while the mesh
    was actually CUT OUT. Its visible result was a floating rock/earth chunk.
    """
    x,z=sides(s,offset)
    if masks is not None:
        # A base must stand on REAL triangles, not inside (or just beside)
        # an excavated polygon whose boundary cell is fully removed.
        pad=.18
        if ground_patch_intersects_excavation(
                ((x-pad,z-pad),(x+pad,z-pad),(x+pad,z+pad),(x-pad,z+pad)),
                masks):
            return None
    if segments is None:
        return world_on_ground(s,offset)
    y=_exterior_y_world(x,z,segments,{} if cache is None else cache)
    return (x,y,z)


def bar(mesh, p, q, radius, material, facets=10, repeat=.28):
    """Textured irregular cylindrical timber. Caps + tapered rings avoid tubes.

    No GL state or random calls; deterministic from endpoints and seed.
    Costs are bounded to at most two axial spans per prop for chunk streaming.
    """
    a=np.asarray(p,dtype=float);b=np.asarray(q,dtype=float)
    axis=b-a; length=float(np.linalg.norm(axis))
    if length<.0001:return
    direction=axis/length
    helper=np.array((0.,1.,0.)) if abs(direction[1])<.87 else np.array((1.,0.,0.))
    u=np.cross(direction,helper); u/=max(1e-9,float(np.linalg.norm(u)))
    v=np.cross(direction,u)
    facets=max(6,int(facets))
    spans=2 if radius>=.050 and length>.85 else 1
    seed_phase=a[0]*1.77+a[1]*2.49+a[2]*.81+b[2]*.31
    rings=[]
    for k in range(spans+1):
        t=k/spans
        spine=a+axis*t
        # Minute lateral bend creates real wood silhouettes without camera snap.
        if 0<k<spans:spine+=radius*.12*math.sin(seed_phase*3.0)*u
        ring=[]
        for j in range(facets):
            angle=2*math.pi*j/facets
            irregular=1.0+.065*math.sin(seed_phase*2+j*2.57+k*.63)
            taper=1.0-.07*t
            point=spine+radius*irregular*taper*(math.cos(angle)*u+math.sin(angle)*v)
            ring.append(tuple(point))
        rings.append(ring)
    for k in range(spans):
        for j in range(facets):
            nj=(j+1)%facets
            mesh.quad(rings[k][j],rings[k][nj],rings[k+1][nj],rings[k+1][j],material,
                      ((j/facets,k*length/(spans*repeat)),
                       ((j+1)/facets,k*length/(spans*repeat)),
                       ((j+1)/facets,(k+1)*length/(spans*repeat)),
                       (j/facets,(k+1)*length/(spans*repeat))))
    # Chopped ends: no hollow cylinders visible near camera.
    for end_idx,center in ((0,a),(-1,b)):
        for j in range(facets):
            nj=(j+1)%facets
            # A narrow strip across the cap; face has wood grain rather than sky.
            mesh.quad(tuple(center),rings[end_idx][j],rings[end_idx][nj],tuple(center),material,
                      ((.5,.5),(j/facets,0),((j+1)/facets,0),(.5,.5)))


def blob(mesh, center, radii, layer, segments=12, rows=5, phase=0.0):
    """Convex low-poly irregular ellipsoid. No translucent billboard artifacts."""
    cx,cy,cz=center; rx,ry,rz=radii
    def pt(theta,phi):
        ripple=1+.12*math.sin(phi*5+phase)*math.sin(theta*3+phase)
        return (cx+rx*ripple*math.sin(theta)*math.cos(phi),
                cy+ry*ripple*math.cos(theta),
                cz+rz*ripple*math.sin(theta)*math.sin(phi))
    for iy in range(rows):
        th0=math.pi*iy/rows;th1=math.pi*(iy+1)/rows
        for j in range(segments):
            f0=2*math.pi*j/segments;f1=2*math.pi*(j+1)/segments
            uvs=((j/segments,iy/rows),(j/segments,(iy+1)/rows),
                 ((j+1)/segments,(iy+1)/rows),((j+1)/segments,iy/rows))
            if layer==6 and hasattr(mesh,'smooth_quad'):
                mesh.smooth_quad(pt(th0,f0),pt(th1,f0),pt(th1,f1),pt(th0,f1),layer,uvs,center,radii)
            else:
                mesh.quad(pt(th0,f0),pt(th1,f0),pt(th1,f1),pt(th0,f1),layer,uvs)


def rock_fragment(mesh, center, size, seed_index, material=17, aspect=(1.0,1.0,1.0)):
    """Jagged grounded stone / shrapnel chunk, more readable than smooth blobs.

    Stones are deterministic, slightly flattened on the bottom, and use a
    chipped two-ring profile so small fragments read like broken grey mud rock
    instead of soft mossy boulders.
    """
    rng=random.Random(SEED*1531 + int(seed_index)*9781)
    cx,cy,cz=center
    rx=size*(0.72+.46*rng.random())*aspect[0]
    rz=size*(0.70+.50*rng.random())*aspect[2]
    h=size*(0.40+.34*rng.random())*aspect[1]
    sides=6+rng.randrange(4)
    phase=rng.random()*math.tau
    base=[]; mid=[]; top=[]
    for j in range(sides):
        ang=phase + math.tau*j/sides + rng.uniform(-0.13,0.13)
        radial=.76+.36*rng.random()
        bx=math.cos(ang)*rx*radial
        bz=math.sin(ang)*rz*radial
        base.append((cx+bx*1.05, cy+rng.uniform(0.0,h*0.06), cz+bz*1.05))
        mid.append((cx+bx*(.66+.18*rng.random()),
                    cy+h*(.25+.18*rng.random()),
                    cz+bz*(.66+.18*rng.random())))
        top.append((cx+bx*(.20+.22*rng.random()),
                    cy+h*(.72+.20*rng.random()),
                    cz+bz*(.20+.22*rng.random())))
    chip=rng.randrange(sides)
    px,py,pz=mid[chip]
    mid[chip]=(px*0.985+cx*0.015, py+h*0.04, pz*0.985+cz*0.015)
    tx,ty,tz=top[chip]
    top[chip]=(tx*0.94+cx*0.06, ty-h*0.12, tz*0.94+cz*0.06)
    base_center=(cx,cy+h*0.015,cz)
    top_center=(sum(p[0] for p in top)/sides,
                sum(p[1] for p in top)/sides + h*0.02,
                sum(p[2] for p in top)/sides)
    for j in range(sides):
        nj=(j+1)%sides
        u0,u1=j/sides,(j+1)/sides
        mesh.quad(base[nj],base[j],mid[j],mid[nj],material,
                  ((u0,0.0),(u1,0.0),(u1,0.52),(u0,0.52)))
        mesh.quad(mid[nj],mid[j],top[j],top[nj],material,
                  ((u0,0.48),(u1,0.48),(u1,1.0),(u0,1.0)))
        mesh.quad(base_center,base[j],base[nj],base_center,material,
                  ((.5,.5),(u0,0.0),(u1,0.0),(.5,.5)))
        mesh.quad(top_center,top[nj],top[j],top_center,material,
                  ((.5,.5),(u0,1.0),(u1,1.0),(.5,.5)))


def grass_tuft(mesh, s, lateral, height, seed_index, size=1.0):
    """A few tapered *3D* leaves at the trench lip (no transparency/alpha).

    Strands use crossed vertical leaf triangles so silhouettes look organic.
    Deterministic global spacing avoids chunk seam pops and reloading changes.
    These are decorative, cannot obstruct the soldier or overwrite collision.
    """
    rng=random.Random(SEED*811+seed_index*1777)
    cx,cy,cz=world(s,lateral,height)
    count=rng.randint(8,15)
    for i in range(count):
        ang=2*math.pi*(i+.26*rng.random())/count
        blade_height=(.075+.13*rng.random())*size
        w=(.004+.009*rng.random())*size
        lean=(.022+.07*rng.random())*size
        ax,az=math.cos(ang),math.sin(ang)
        bx,bz=-az,ax
        base=(cx+ax*.028,cy,cz+az*.028)
        p0=(base[0]+bx*w,base[1],base[2]+bz*w)
        p1=(base[0]-bx*w,base[1],base[2]-bz*w)
        tip=(base[0]+ax*lean,cy+blade_height,base[2]+az*lean)
        # One triangle. Culling is disabled; the second degenerate triangle
        # prevents double-sided foliage from becoming opaque rectangles.
        mesh.quad(p0,p1,tip,tip,6,((0.,0.),(.8,0.),(.43,1.),(.43,1.)))


def rim_vegetation_chunk(mesh, start, masks=None, segments=None, ground_cache=None):
    """Mixed sparse clumps, rough stones and leaf litter on either bank rim.

    Only a short world-distance interval is generated for each 80m chunk.
    Fixed global indices make all objects stable on chunk eviction/rebuild.
    """
    first=max(0,math.floor((start-1.3)/2.6)-1)
    last=math.ceil((start+CHUNK_LEN-1.3)/2.6)+1
    for i in range(max(0,first),last+1):
        s=1.3+2.6*i + .32*(noise(i,704)-.5)
        if not start<=s<start+CHUNK_LEN or near_junction(s,4.5):continue
        theme=scene_at(s)
        for side in (-1,1):
            # Some bare stretches preserve the war-torn appearance.
            density=.78 if theme=='forest' else .62 if theme=='marsh' else .25 if theme=='rocky' else .43
            if noise(i,720+side)>density:continue
            off=side*(wall_width(s,side)+.17+.40*noise(i,730+side))
            anchor=exterior_prop_ground(s,off,masks,segments,ground_cache)
            if anchor is None:continue
            y=anchor[1]
            scale=.65+.80*noise(i,740+side)
            grass_tuft(mesh,s,off,y-.015,i*5+side,scale)
            if noise(i,750+side)>.68:
                p=world(s+.19,off+side*.10,y+.015)
                rock_fragment(mesh,(p[0],p[1]-.02,p[2]),.13,i*31+side,7)


def conifer_canopy(mesh, base, height, width, rng):
    """Sparse layered evergreen boughs, not a solid green lollipop.

    Small silhouette-only whorls reuse the existing leaf material and depth /
    shadow pipeline. Each tree is baked into its streamed chunk once.
    """
    cx,cy,cz=base
    levels=5+rng.randrange(3)
    rotation=rng.random()*math.tau
    for level in range(levels):
        fraction=(level+.4)/(levels+.8)
        yy=cy+height*(.22+.74*fraction)
        radius=width*(1.0-fraction)**.77*(.85+.24*rng.random())
        spokes=7 if radius>.38 else 5
        tip_height=(.45+.22*rng.random())*height/levels
        for spoke in range(spokes):
            a=rotation+math.tau*spoke/spokes
            b=a+math.tau/spokes
            # A drooping outer rim and raised inner spine form a real 3D bough.
            center=(cx, yy+tip_height, cz)
            rim_a=(cx+radius*math.cos(a), yy-.23*tip_height, cz+radius*math.sin(a))
            rim_b=(cx+radius*math.cos(b), yy-.23*tip_height, cz+radius*math.sin(b))
            notch=(cx+radius*.63*math.cos((a+b)*.5),
                   yy-.40*tip_height,cz+radius*.63*math.sin((a+b)*.5))
            mesh.quad(center,rim_a,notch,center,6,((.5,1),(0,0),(.5,0),(.5,1)))
            mesh.quad(center,notch,rim_b,center,6,((.5,1),(.5,0),(1,0),(.5,1)))
    # Thin branch ends protruding from a few of the lower whorls.
    for index in range(3):
        angle=rotation+index*2.13
        y=cy+height*(.24+.12*index)
        p=(cx,y,cz)
        q=(cx+width*(.54-.07*index)*math.cos(angle),y-.13,
           cz+width*(.54-.07*index)*math.sin(angle))
        bar(mesh,p,q,.020,13,6,.40)


def broadleaf_canopy(mesh, top, scale, rng):
    """Broken non-spherical deciduous crown with gaps and exposed limbs."""
    count=4+rng.randrange(3)
    for j in range(count):
        angle=math.tau*j/count+rng.uniform(-.28,.28)
        radius=rng.uniform(.34,1.0)*scale
        dx,dz=math.cos(angle)*radius,math.sin(angle)*radius
        yy=top[1]-rng.uniform(.20,.80)*scale
        center=(top[0]+dx,yy,top[2]+dz)
        blob(mesh,center,(rng.uniform(.48,.85)*scale,
                          rng.uniform(.54,1.05)*scale,
                          rng.uniform(.45,.82)*scale),6,
             segments=9,rows=4,phase=rng.uniform(0,math.tau))


def scenery_chunk(mesh, start, masks=None, segments=None, ground_cache=None):
    """Trees, snapped trunks, shrubs, stones and broken wire outside the trench.

    Positions depend only on global world distance + SEED, NOT on chunk loads.
    Places deliberately avoid openings around T junctions.
    """
    rng = random.Random(SEED*1123 + int(start/CHUNK_LEN)*191)
    # Layout is an outdoor strip, excluding the path and its mouth openings.
    for s_index in range(int(start)+2,int(start+CHUNK_LEN)-2,4):
        s=float(s_index)+rng.uniform(-.5,.5)
        if near_junction(s,10):continue
        theme=scene_at(s)
        density={'forest':.91,'rocky':.78,'exposed':.22,'ruined':.65,
                 'marsh':.49,'fortified':.40}[theme]
        for side in (-1,1):
            if rng.random()>density: continue
            dist=rng.uniform(3.5,22.0)
            off=side*dist
            ground=exterior_prop_ground(s,off,masks,segments,ground_cache)
            if ground is None:continue
            # Preserve coherent BIOMES instead of equal tree/rock probabilities
            # everywhere. No preloaded world; theme can be recomputed per chunk.
            choice=rng.random()
            if theme=='rocky':choice=.75+choice*.25
            elif theme=='ruined':choice=.29+choice*.25 if choice<.74 else .80+choice*.20
            elif theme=='forest':choice=choice*.70
            elif theme=='exposed':choice=.49+choice*.51
            scale=rng.uniform(.7,1.35)
            if choice<.28:  # standing tree, with three grounded silhouettes
                height=rng.uniform(5.0,10.4)*scale
                lean=(rng.random()-.5)*.50*scale
                top=(ground[0]+lean,ground[1]+height,ground[2]-lean*.35)
                trunk_radius=(.10+.06*rng.random())*scale
                bark_mat=13 if rng.random()<.52 else 14
                # Real bark trunk and branches, not a floating crown sprite.
                bar(mesh,ground,top,trunk_radius,bark_mat,8 if dist<12 else 6,1.0)
                species=rng.random()
                if theme=='forest':species*=.79  # coherent woods, more conifers
                if species<.42:
                    conifer_canopy(mesh,ground,height,
                                  rng.uniform(1.1,1.9)*scale,rng)
                elif species<.85:
                    for k in range(3):
                        yy=ground[1]+height*(.49+.12*k)
                        ang=rng.uniform(-math.pi,math.pi)
                        reach=(.8+1.2*rng.random())*scale
                        branch_end=(ground[0]+math.cos(ang)*reach,yy+(.3+.6*rng.random())*scale,
                                    ground[2]+math.sin(ang)*reach)
                        bar(mesh,(ground[0],yy,ground[2]),branch_end,
                            .035*scale+.028*scale*rng.random(),bark_mat,6,.75)
                    broadleaf_canopy(mesh,top,scale,rng)
                else:
                    # Dead stripped trees: bark, asymmetrical bare branches,
                    # occasional snapped tops. Distinct silhouettes in ruins.
                    for k in range(4):
                        yy=ground[1]+height*(.32+.14*k)
                        ang=rng.uniform(-math.pi,math.pi)
                        reach=(.45+1.45*rng.random())*scale
                        end=(ground[0]+math.cos(ang)*reach,yy+(.12+.55*rng.random())*scale,
                             ground[2]+math.sin(ang)*reach)
                        bar(mesh,(ground[0],yy,ground[2]),end,
                            (.035+.02*rng.random())*scale,bark_mat,6,.65)
            elif choice<.51: # snapped and splintered trunk
                height=rng.uniform(1.7,4.8)*scale
                top=(ground[0]+rng.uniform(-.27,.27),ground[1]+height,ground[2]+rng.uniform(-.27,.27))
                bar(mesh,ground,top,.16*scale,13 if rng.random()<.5 else 14,12 if dist<12 else 8,.8)
                for k in range(2):
                    t=.55+.18*k; ay=ground[1]+height*t
                    end=(ground[0]+rng.uniform(-1.4,1.4)*scale,ay+rng.uniform(.1,.6),
                         ground[2]+rng.uniform(-1.4,1.4)*scale)
                    bar(mesh,(ground[0],ay,ground[2]),end,.06*scale,13,9,.6)
                # broken top gets a smaller angled splinter
                bar(mesh,top,(top[0]+.22*scale,top[1]+.3*scale,top[2]-.16*scale),.06*scale,16,8,.5)
            elif choice<.72: # bush, different size + number of clumps
                for k in range(rng.randint(2,4)):
                    ang=rng.uniform(-math.pi,math.pi)
                    rr=rng.uniform(.05,.6)*scale
                    blob(mesh,(ground[0]+math.cos(ang)*rr,ground[1]+.45*scale,ground[2]+math.sin(ang)*rr),
                         (.64*scale,.58*scale,.72*scale),6,11 if dist<12 else 8,5 if dist<12 else 3,rng.random()*5)
            else: # surface stone / small boulder: fractured grey chunks
                rad=rng.uniform(.20,.72)*scale
                rock_fragment(mesh,(ground[0],ground[1],ground[2]),rad,
                              int(s*13+off*7),7 if rng.random()<.95 else 1,
                              (1.0,rng.uniform(.72,1.06),rng.uniform(.78,1.25)))
                if rad>.46 and rng.random()>.48:
                    angle=rng.random()*math.tau; d=rad*(.7+.45*rng.random())
                    sx=ground[0]+math.cos(angle)*d
                    sz=ground[2]+math.sin(angle)*d
                    if masks is None or not soil_over_excavation(sx,sz,masks):
                        sy=(_exterior_y_world(sx,sz,segments,ground_cache)
                            if segments is not None else ground[1])
                        rock_fragment(mesh,(sx,sy,sz),rad*(.34+.22*rng.random()),
                                      int(s*19+off*11)+5,7)
    # Old field wire — less uniform: more posts, slight sag and irregular barbs.
    for base in range(int(start//26)*26,int((start+CHUNK_LEN)//26)*26+1,26):
        if not start<=base<start+CHUNK_LEN-8:continue
        if near_junction(base,12):continue
        side=-1 if noise(base,77)<.5 else 1
        offs=side*(rng.uniform(3.9,6.4))
        posts=[]
        count=4 if noise(base,78)>.34 else 5
        for j in range(count):
            ss=base+j*rng.uniform(2.4,3.1)
            p=exterior_prop_ground(ss,offs+rng.uniform(-.3,.3),
                                   masks,segments,ground_cache)
            if p is None:continue
            top=(p[0]+rng.uniform(-.14,.14),p[1]+(.88+0.42*rng.random()),p[2]+rng.uniform(-.14,.14))
            bar(mesh,p,top,.042 if j%2 else .048,5,5,.5)
            posts.append((p,top))
        for i in range(len(posts)-1):
            for level in (.42,.73,1.01):
                a,b=posts[i],posts[i+1]
                q1=tuple(a[0][k]*(1-level)+a[1][k]*level for k in range(3))
                q2=tuple(b[0][k]*(1-level)+b[1][k]*level for k in range(3))
                mid=((q1[0]+q2[0])*.5,(q1[1]+q2[1])*.5-.05-rng.random()*.07,(q1[2]+q2[2])*.5)
                bar(mesh,q1,mid,.010,7,4,.6)
                bar(mesh,mid,q2,.010,7,4,.6)
                for j in range(1,4):
                    t=j/4
                    wire_point=tuple(q1[k]*(1-t)+q2[k]*t for k in range(3))
                    for barb in (-1,1):
                        end=(wire_point[0]+.07*barb, wire_point[1]+.03+.02*rng.random(), wire_point[2]-.05*barb)
                        bar(mesh,wire_point,end,.011,7,3,.25)


def battlefield_debris_chunk(mesh,start,masks=None,segments=None,ground_cache=None):
    """Clusters of destroyed field debris and shattered trees.

    A small fixed set of local mesh instances is recreated from global seed
    positions; nothing grows with distance travelled. Keep the same authentic
    tree distribution in scenery_chunk (requested as a KNOWN GOOD).
    """
    first=max(0,int(math.floor((start-3.0)/7.2))-1)
    last=int(math.ceil((start+CHUNK_LEN-3.0)/7.2))+2
    for i in range(first,last):
        center_s=3.0+7.2*i+2.1*(noise(i,4001)-.5)
        if not start<=center_s<start+CHUNK_LEN:continue
        if near_junction(center_s,6.0):continue
        theme=scene_at(center_s)
        density={'forest':.50,'ruined':.96,'rocky':.70,'exposed':.66,
                 'fortified':.49,'marsh':.67}[theme]
        if noise(i,4002)>density:continue
        side=-1 if noise(i,4003)<.5 else 1
        rng=random.Random(SEED*3101+i*8309)
        anchor_offset=side*(5.+29.*noise(i,4004))
        # Larger fragmented patches; no repeating two straight rows.
        fragments=2+int(noise(i,4005)*5)
        for j in range(fragments):
            s=center_s+rng.uniform(-2.4,2.4)
            off=anchor_offset+rng.uniform(-3.7,3.7)
            # Offscreen/outside current generation interval => next chunk
            # regenerates a local half independently. Avoid renderer popping.
            if not start<=s<start+CHUNK_LEN or abs(off)<3.0:continue
            point=exterior_prop_ground(s,off,masks,segments,ground_cache)
            if point is None:continue
            kind=rng.random()
            if kind<.36:
                r=.14+.48*rng.random()
                rock_fragment(mesh,(point[0],point[1],point[2]),r,
                              i*17+j,7 if kind<.32 else 1,
                              (1.0,rng.uniform(.65,1.0),rng.uniform(.75,1.25)))
            elif kind<.73:
                # Scattered snapped splinters / collapsed supports.
                length=.6+2.3*rng.random()
                end=(point[0]+rng.uniform(-1,1)*length,
                     point[1]+rng.uniform(.07,.27),
                     point[2]+rng.uniform(-1,1)*length)
                beam_world(mesh,(point[0],point[1]+.06,point[2]),end,
                           .035+.085*rng.random(),13 if rng.random()<.46 else 14)
            else:
                # Soft bags with stitched fabric and a grounded base.
                for k in range(rng.randrange(1,4)):
                    position=(point[0]+rng.uniform(-.28,.28),point[1]-.02,
                              point[2]+rng.uniform(-.28,.28))
                    sandbag_mesh(mesh,position,.40,.29,.16,i*71+j*5+k,
                                 heading=rng.uniform(-math.pi,math.pi))
        if theme=='ruined' and rng.random()>.80 and abs(anchor_offset)>=13.0:
            ruined_outpost(mesh,center_s,anchor_offset,i,masks=masks,
                           segments=segments,ground_cache=ground_cache)


# --- V1.7 scenic construction, determined by route metres rather than frames ---
# Themes are deliberately long stretches so the world reads as places, not random props.
SCENES = ('fortified', 'exposed', 'ruined', 'marsh', 'forest', 'rocky')
def scene_at(s):
    idx=max(0,int(s//145))
    # staggered but deterministic world zones
    return SCENES[(idx*5 + idx//3)%len(SCENES)]


@dataclass(frozen=True)
class TrenchObstacle:
    """One seeded world object is shared by rendering and local navigation.

    `detour` is a signed lateral target in route metres. A zero detour means
    a step-over/cosmetic object; never misrepresent such a prop as a hard wall.
    """
    index: int
    s: float
    kind: str
    lateral: float
    length: float
    width: float
    height: float
    detour: float = 0.0


def paired_cover_at(index):
    """One real grounded crate beside SOME paired enemies, 360m apart.

    Both static scene mesh and LOS/nav queries call this deterministic layout.
    Not a floating animation-only bunker or an invisible protected state.
    """
    partner=encounter_partner_spec(index)
    if partner is None:return None
    s=partner.route_m-1.25
    if near_junction(s,6.0) or tunnel_drop(s)<-.10:
        return None
    side=-1 if partner.lateral<0 else 1
    # Leave a walkable central lane and respect the excavated bank width.
    minimum_width=min(wall_width(s+d,side) for d in (-.6,0.,.6))
    if minimum_width<.88:return None
    width=.41
    lateral=side*min(abs(partner.lateral),minimum_width-width*.5-.06)
    return TrenchObstacle(-200000-index,s,'crate',lateral,.67,width,.98,
                           -side*.30)


def paired_covers_in_range(start,end):
    """O(visible distance / 360m), deterministic across chunk boundaries."""
    if end<start:return []
    first=max(0,math.floor((start-520.)/360.)-1)
    last=max(0,math.ceil((end-515.)/360.)+1)
    found=[]
    for k in range(first,last+1):
        ob=paired_cover_at(4+5*k)
        if ob is not None and start<=ob.s<=end:
            found.append(ob)
    return found


def trench_obstacle(index):
    """Deterministic obstruction with measured walking clearance.

    Skips junctions and actual Z/W turns. Never place an impassable crate in
    a tight excavation; objects are not randomly teleported on chunk reload.
    """
    if index < 0:
        return None
    s = 15.0 + OBSTACLE_SPACING*index + (noise(index,401)-.5)*4.0
    # Reserved side-wall tactical cover replaces overlapping ordinary
    # clutter. There must never be two interpenetrating collision objects.
    nearby_pair=int(round((s-517.2)/360.0))
    if nearby_pair>=0 and abs(s-(517.2+360.*nearby_pair))<2.25:
        return None
    if tunnel_drop(s)<-.10:
        return None  # avoid unseen raised obstacles inside sloping tunnel
    j=nearest_junction(s)
    if j is not None and abs(j[0]-s)<9.0:
        return None
    near=int(round((s-BEND_FIRST)/BEND_SPACING))
    if near >= 0 and abs(bend_marker(near)-s)<8.0:
        return None
    choice=noise(index,402)
    kind=('crate' if choice<.26 else
          'rubble' if choice<.48 else
          'timber' if choice<.69 else
          'sandbags' if choice<.84 else 'puddle')
    lateral=(noise(index,403)-.5)*.17
    if kind in ('crate','rubble'):
        large=kind=='crate' and noise(index,450)>.73
        width=(.73 if large else .54) if kind=='crate' else .62
        length=(1.16 if large else .72) if kind=='crate' else .95
        height=(1.13 if large else .58) if kind=='crate' else .42
        if large:
            # Half-cover crate to one side, leaving a verified escape lane.
            wall_side=1 if noise(index,451)>.5 else -1
            available=min(wall_width(s+d,wall_side) for d in (-1.,0.,1.))
            lateral=wall_side*(available-width*.5-.075)
        # Eye/camera may follow either side if there is genuine lateral room.
        # Account for changing width a few metres before and after obstacle.
        clearance=.245
        # Use the exact eye-position limits from SoldierBrain. We must not
        # plan a detour the steering controller later clamps into the prop.
        right_limit=min(cover_distance(s+d,1) for d in (-3.,-1.,0.,1.,3.))
        left_limit=min(cover_distance(s+d,-1) for d in (-3.,-1.,0.,1.,3.))
        right=lateral+width*.5+clearance
        left=lateral-width*.5-clearance
        possibilities=[]
        if right<right_limit-.035: possibilities.append((right,right_limit-right))
        if left>-left_limit+.035: possibilities.append((left,left_limit+left))
        if not possibilities: return None
        # Prefer the widest available lane, not necessarily the same side.
        possibilities.sort(key=lambda option:option[1],reverse=True)
        detour=possibilities[0][0]
    else:
        width=min(1.32,2*min(wall_width(s,1),wall_width(s,-1))-.36) if kind=='timber' else (.78 if kind=='sandbags' else .86)
        length=.24 if kind=='timber' else (.70 if kind=='sandbags' else 1.30)
        height=.22 if kind=='timber' else (.29 if kind=='sandbags' else .02)
        detour=0.0
    return TrenchObstacle(index,s,kind,lateral,length,width,height,detour)


def obstacles_in_range(start, end):
    """Stable lookup across GPU chunk boundaries and save/reload cycles."""
    first=max(0,math.floor((start-17.0)/OBSTACLE_SPACING)-1)
    last=max(0,math.ceil((end-13.0)/OBSTACLE_SPACING)+1)
    result=[]
    for i in range(first,last+1):
        ob=trench_obstacle(i)
        if ob is not None and start<=ob.s<=end:
            result.append(ob)
    result.extend(paired_covers_in_range(start,end))
    return result


def obstacle_steering(s):
    """Smooth walk-around path, zero before and after each obstruction."""
    offset=0.0
    for ob in obstacles_in_range(s-4.0,s+4.0):
        if not ob.detour:continue
        d=s-ob.s
        if -3.9<d<-1.0:
            factor=eased((d+3.9)/2.9)
        elif -1.0<=d<=1.0:
            factor=1.0
        elif 1.0<d<3.9:
            factor=1.0-eased((d-1.0)/2.9)
        else:factor=0.
        offset+=ob.detour*factor
    return offset


def sandbag_mesh(mesh, foot, length, width, height, seed_index, heading=0.0,
                 cloth=19):
    """Grounded compressed fabric sack with asymmetric silhouette and stitched lip.

    FOOT is the ground/bag-bottom point in world XYZ, not the bag centre.
    Does NOT register extra collision: placements reuse existing visual footprints.
    One octagonal ring per height level; no floating spherical "sandbags".
    """
    rng=random.Random(SEED*1723+int(seed_index)*6011)
    cx,cy,cz=foot
    length*=.91+.15*rng.random()
    width*=.90+.13*rng.random()
    height*=.89+.14*rng.random()
    hx=length*.5;hz=width*.5
    c=math.cos(heading);d=math.sin(heading)
    outline=((-0.72,-1.),(.72,-1.),(1.,-.62),(1.,.62),
             (.72,1.),(-.72,1.),(-1.,.62),(-1.,-.62))
    rings=[]
    for level,expansion,scale_h in ((.03,.83,0.),(.35,1.06,.38),(.81,.94,.86)):
        ring=[]
        for i,(u,v) in enumerate(outline):
            wiggle=1.+.055*math.sin(i*2.7+seed_index*.41)
            x=u*hx*expansion*wiggle
            z=v*hz*expansion*wiggle
            # A lumpy sewn surface, not a precise perfect octagonal prism.
            y=cy+height*scale_h+height*.025*math.sin(i*2.13+seed_index*.7)
            ring.append((cx+x*c-z*d,y,cz+x*d+z*c))
        rings.append(ring)
    for k in (0,1):
        for i in range(8):
            j=(i+1)%8
            mesh.quad(rings[k][j],rings[k][i],rings[k+1][i],rings[k+1][j],cloth,
                      ((i/8.,k*.42),(j/8.,k*.42),(j/8.,(k+1)*.42),(i/8.,(k+1)*.42)))
    crown=(cx,cy+height*.89,cz)
    base=(cx,cy+height*.015,cz)
    for i in range(8):
        j=(i+1)%8
        mesh.quad(crown,rings[2][j],rings[2][i],crown,cloth,
                  ((.5,.5),(i/8.,.91),(j/8.,.91),(.5,.5)))
        mesh.quad(base,rings[0][i],rings[0][j],base,cloth,
                  ((.5,.5),(j/8.,.03),(i/8.,.03),(.5,.5)))
    # Narrow raised stitching along the sack's long top seam.
    seam=.028*width
    def pt(u,v):
        return (cx+u*c-v*d,cy+height*.90,cz+u*d+v*c)
    mesh.quad(pt(-hx*.69,-seam),pt(hx*.69,-seam),
              pt(hx*.69,seam),pt(-hx*.69,seam),9,
              ((0.,0.),(length,0.),(length,.09),(0.,.09)))


def hanging_fabric(mesh, s, side, seed_index, span=1.9, drop=0.85):
    """Draped, lightly wind-animated patch attached to a trench bank.

    Follows each bank's actual X/Z curve at the fabric height. Anchored along
    the top, with wavy / sometimes torn free hem. Does not protrude into the
    soldier's collision corridor. Wind is applied on GPU only to material 26.
    """
    rng=random.Random(SEED*2269+int(seed_index)*983)
    samples=10
    heights=6
    center_h=wall_height(s,side)
    top=min(center_h-.14,1.94+.22*rng.random())
    drop=min(drop,max(.32,top-.55))
    points=[]
    for k in range(heights+1):
        v=k/heights
        row=[]
        for j in range(samples+1):
            u=j/samples
            ss=s+(u-.5)*span
            bank_h=wall_height(ss,side)
            y=min(top,bank_h-.08)-drop*v
            # Slight sag and gathered folds, even before wind animation.
            y-=.045*v*math.sin(u*math.tau*1.7+seed_index*.37)
            if k==heights:
                y-=.05*noise(seed_index*31+j,7310)
            h=max(.05,min(bank_h-.04,y))
            offset=side*(wall_offset(ss,side,h,bank_h)-.075
                         -.018*math.sin(u*math.tau*3.+seed_index*.51)*v)
            row.append(world(ss,offset,h))
        points.append(row)
    for k in range(heights):
        v0=k/heights;v1=(k+1)/heights
        for j in range(samples):
            u0=j/samples;u1=(j+1)/samples
            mesh.quad(points[k][j],points[k][j+1],
                      points[k+1][j+1],points[k+1][j],26,
                      ((u0,v0),(u1,v0),(u1,v1),(u0,v1)))
    # Actual fixture pegs, not part of the fluttering fabric geometry.
    for j in (0,samples):
        ss=s+((j/samples)-.5)*span
        beam_world(mesh,points[0][j],(points[0][j][0],points[0][j][1]+.09,
                                         points[0][j][2]),.013,10)


def torn_camouflage_net(mesh, top_left, top_right, drop, seed_index):
    """Sparse gapped woven net slung over ruined wall debris.

    Holes are *actual geometry holes*, not transparency planes. Staggered
    threads loosely sag. Layer 26 adds subtle cloth motion in the vertex GPU.
    """
    rng=random.Random(SEED*4279+int(seed_index)*1717)
    cols=8
    rows=4
    def point(u,v):
        x=top_left[0]*(1-u)+top_right[0]*u
        z=top_left[2]*(1-u)+top_right[2]*u
        y=top_left[1]*(1-u)+top_right[1]*u-drop*v
        y-=.14*v*math.sin(u*math.pi)
        # Rotate the net gently out of the rigid wall plane.
        z+=.09*v*(.7+.3*math.sin(u*math.tau))
        return (x,y,z)
    for i in range(cols+1):
        u=i/cols
        if 0<i<cols and rng.random()<.09:continue
        thickness=.010+.013*rng.random()
        for j in range(rows):
            v0=j/rows;v1=(j+1)/rows
            if rng.random()<.06:continue
            mesh.quad(point(max(0,u-thickness),v0),point(min(1,u+thickness),v0),
                      point(min(1,u+thickness),v1),point(max(0,u-thickness),v1),26,
                      ((u,v0),(u+.015,v0),(u+.015,v1),(u,v1)))
    for j in range(rows+1):
        v=j/rows
        for i in range(cols):
            u0=i/cols;u1=(i+1)/cols
            if rng.random()<.10:continue
            thickness=.009+.011*rng.random()
            mesh.quad(point(u0,max(0,v-thickness)),point(u1,max(0,v-thickness)),
                      point(u1,min(1,v+thickness)),point(u0,min(1,v+thickness)),26,
                      ((u0,v),(u1,v),(u1,v+.01),(u0,v+.01)))


def ruined_wall_panel(mesh, center, length, height, heading, seed_index):
    """Fractured stone/brick wall with actual missing tiles and window voids.

    This creates a thick, disconnected rubble silhouette, not a uniform grey
    box or a billboard. The bottom is sunk 10 cm into terrain at its anchor.
    """
    rng=random.Random(SEED*881+int(seed_index)*191)
    cx,cy,cz=center
    cy-=.10
    along=(math.cos(heading),math.sin(heading))
    normal=(-along[1],along[0])
    count=7;levels=5
    cell_w=length/count
    cell_h=height/levels
    # Two open columns by two open rows make a real window; no behind-wall fill.
    aperture=2+rng.randrange(3)
    doorway=1 if rng.random()<.43 else -1
    top_profile=[max(1,int((.60+.43*rng.random())*levels)) for _ in range(count)]
    top_profile[0]=max(2,top_profile[0]-1)
    top_profile[-1]=max(1,top_profile[-1]-2)
    for ix in range(count):
        for iy in range(levels):
            if iy>=top_profile[ix]:continue
            if ix in (aperture,aperture+1) and iy in (2,3):continue
            if ix==doorway and iy<=2:continue
            if iy>1 and rng.random()<(.17 if iy<4 else .33):continue
            if ix in (0,count-1) and iy>1 and rng.random()<.23:continue
            span=.97 if iy<2 else rng.uniform(.70,.99)
            xlocal=(-length*.5+(ix+.5)*cell_w)+rng.uniform(-.024,.024)
            zlocal=rng.uniform(-.020,.020)
            px=cx+along[0]*xlocal+normal[0]*zlocal
            pz=cz+along[1]*xlocal+normal[1]*zlocal
            py=cy+iy*cell_h
            wall_h=cell_h*(.90 if iy<2 else rng.uniform(.65,.98))
            thickness=.17+.11*rng.random()
            stone_material=7 if rng.random()<.64 else 2 if rng.random()<.67 else 1
            # Local quadrilateral block; only world XY vertex data is emitted.
            half_x=cell_w*span*.5
            half_z=thickness*.5
            def v(u,w,dy):
                return (px+u*along[0]+w*normal[0],
                        py+dy,
                        pz+u*along[1]+w*normal[1])
            a=v(-half_x,-half_z,0.);b=v(half_x,-half_z,0.)
            c=v(half_x,half_z,0.);d=v(-half_x,half_z,0.)
            e=v(-half_x,-half_z,wall_h);f=v(half_x,-half_z,wall_h)
            g=v(half_x,half_z,wall_h);h=v(-half_x,half_z,wall_h)
            uv=((0.,0.),(span,0.),(span,wall_h),(0.,wall_h))
            for p0,p1,p2,p3 in ((a,e,f,b),(d,c,g,h),(b,c,g,f),
                                (d,a,e,h),(e,h,g,f),(d,a,b,c)):
                mesh.quad(p0,p1,p2,p3,stone_material,uv)
    return length


def ruined_outpost(mesh, s, offset, index, masks=None, segments=None,
                   ground_cache=None):
    """Distant wrecked masonry, supported on the real world-grid height."""
    rng=random.Random(SEED*4709+int(index)*1207)
    if abs(offset)<13.0:
        return
    base=exterior_prop_ground(s,offset,masks,segments,ground_cache)
    if base is None:return
    if masks is not None:
        # Includes distant bends: route-local lateral distance alone cannot
        # guarantee an outpost stays clear of a later nearby route segment.
        x,_,z=base
        half=4.5
        patch=((x-half,z-half),(x+half,z-half),
               (x+half,z+half),(x-half,z+half))
        if ground_patch_intersects_excavation(patch,masks):
            return
    width=2.4+2.3*rng.random()
    height=1.7+1.4*rng.random()
    angle=.4+rng.random()*math.tau
    ruined_wall_panel(mesh,base,width,height,angle,index*11)
    if rng.random()<.43:
        # Tattered net fastened to ruined masonry. Geometry is non-blocking.
        ca=math.cos(angle);sa=math.sin(angle)
        a=(base[0]-ca*width*.32,base[1]+height*.68,base[2]-sa*width*.32)
        b=(base[0]+ca*width*.32,base[1]+height*.73,base[2]+sa*width*.32)
        torn_camouflage_net(mesh,a,b,.68,index*17)
    if rng.random()<.82:
        dx=math.cos(angle)*width*.40
        dz=math.sin(angle)*width*.40
        center=(base[0]+dx,base[1],base[2]+dz)
        ruined_wall_panel(mesh,center,width*(.53+.23*rng.random()),
                          height*(.57+.35*rng.random()),angle+math.pi*.5,index*11+1)
    # Localized collapsed masonry; the outer world grid itself is unchanged.
    for j in range(7+rng.randrange(7)):
        dx=rng.uniform(-width*.70,width*.70)
        dz=rng.uniform(-width*.70,width*.70)
        size=.10+.35*rng.random()
        x,z=base[0]+dx,base[2]+dz
        if masks is not None and soil_over_excavation(x,z,masks):
            continue
        y=(_exterior_y_world(x,z,segments,ground_cache)
           if segments is not None else base[1])
        rock_fragment(mesh,(x,y-.015,z),size,
                      index*53+j,7,(1.0,.63+.30*rng.random(),1.0))
    if rng.random()<.55:
        a=(base[0]-width*.28,base[1]+.14,base[2]-.18)
        b=(base[0]+width*.16,base[1]+.25,base[2]+.75)
        beam_world(mesh,a,b,.043,8)


def wood_crate(mesh,s,lateral,y,length,width,height,weathered=15):
    """Timber-box silhouette with individual planks, structural strips and straps.

    Same bounding footprint as local_box -> existing SoldierBrain steering valid.
    The geometry is decorative; this does NOT introduce unregistered blockers.
    """
    local_box(mesh,s,lateral,y,length,width,height,weathered)
    # Raised slats on visible sides; very thin to keep silhouette legible.
    for k in range(3):
        h=y+.045+(k+.5)*(height-.09)/3.
        for side in (-1,1):
            # Outside side walls, parallel with the corridor.
            local_box(mesh,s,lateral+side*(width*.5+.009),h,length*.96,.014,.024,18)
        for end in (-1,1):
            local_box(mesh,s+end*(length*.5+.008),lateral,h,.015,width*.91,.022,18)
    # Corner braces darker than planking, making cube read as a crate.
    for side in (-1,1):
        for end in (-1,1):
            local_box(mesh,s+end*length*.45,lateral+side*width*.45,y-.01,.040,.038,height+.03,13)
    local_box(mesh,s,lateral,y+height-.025,length*.97,width*.97,.035,16)


def draw_trench_obstacles(mesh,start):
    """Small reusable primitives; higher-quality GLB assets can replace them."""
    for ob in obstacles_in_range(start,start+CHUNK_LEN-0.0001):
        s,c=ob.s,ob.lateral
        rng=random.Random(SEED*89+ob.index*113)
        if ob.kind=='crate':
            wood_crate(mesh,s,c,floor_y(s,c)-.04,ob.length,ob.width,ob.height,
                       15 if rng.random()<.6 else 18)
            # Dark corner straps and raised edge rails make silhouette readable.
            for ds in (-ob.length*.37,ob.length*.37):
                local_box(mesh,s+ds,c,floor_y(s,c)+ob.height-.04,.055,ob.width+.025,.045,10)
            for dc in (-ob.width*.37,ob.width*.37):
                local_box(mesh,s,c+dc,floor_y(s,c)+ob.height-.04,ob.length,.040,.045,8)
        elif ob.kind=='rubble':
            for j in range(9):
                ds=rng.uniform(-ob.length*.42,ob.length*.42)
                dc=rng.uniform(-ob.width*.22,ob.width*.22)
                r=rng.uniform(.095,.17)
                ss=s+ds;lat=c+dc
                p=world(ss,lat,floor_y(ss,lat))
                rock_fragment(mesh,p,r,ob.index*19+j,7 if j%3 else 1,
                              (1.0,.82,.92))
            # A broken board poking out of the earth heap.
            beam_world(mesh,world(s-.32,c-.13,floor_y(s-.32,c-.13)+.10),
                       world(s+.18,c+.22,floor_y(s+.18,c+.22)+.38),.047,8)
        elif ob.kind=='timber':
            p=world(s,c-ob.width*.5,floor_y(s,c-ob.width*.5)+.13)
            q=world(s,c+ob.width*.5,floor_y(s,c+ob.width*.5)+.13)
            beam_world(mesh,p,q,.115,8)
            beam_world(mesh,world(s+.27,c-ob.width*.25,floor_y(s+.27,c)+.06),
                       world(s-.26,c+ob.width*.30,floor_y(s-.26,c)+.08),.045,8)
        elif ob.kind=='sandbags':
            for row in range(2):
                for j in range(3-row):
                    dc=(j-(2-row)/2)*.25
                    ss=s+(row-.5)*.15
                    y=floor_y(ss,c+dc)+row*.135
                    foot=world(ss,c+dc,y)
                    sandbag_mesh(mesh,foot,.40,.27,.16,ob.index*41+row*5+j,
                                 heading=math.pi*.5-route(ss)[2]+.13*(rng.random()-.5))
        else:  # shallow water; no collision requirement
            local_box(mesh,s,c,floor_y(s,c)+.005,ob.length,ob.width,.010,12)
    # Decorative, NON-colliding fragments stay close to the dirt wall. Their
    # placement is independent of obstruction decisions / streaming order.
    for index in range(int(start//11)-1,int((start+CHUNK_LEN)//11)+2):
        if index<0:continue
        ss=7.0+11.0*index+(noise(index,715)-.5)*3.0
        if not start <= ss < start+CHUNK_LEN:continue
        if near_junction(ss,5.0):continue
        side=1 if noise(index,716)>.5 else -1
        inset=max(.65,wall_width(ss,side)-.115)
        lateral=side*inset
        r=.10+.065*noise(index,717)
        rock_fragment(mesh,world(ss,lateral,floor_y(ss,lateral)),r,
                      index*19,7,(1.0,.85,1.2))
        if noise(index,718)>.38:
            a=world(ss-.32,lateral,.065)
            b=world(ss+.48,lateral-side*.045,.082)
            beam_world(mesh,a,b,.039,8)


def box(mesh, center, dims, material, uv_scale=1.0):
    """Axis-aligned 3-D model prototype; real GLB props can replace later."""
    x,y,z=center;dx,dy,dz=(v*.5 for v in dims)
    p=[(x-dx,y-dy,z-dz),(x+dx,y-dy,z-dz),(x+dx,y+dy,z-dz),(x-dx,y+dy,z-dz),
       (x-dx,y-dy,z+dz),(x+dx,y-dy,z+dz),(x+dx,y+dy,z+dz),(x-dx,y+dy,z+dz)]
    t=((0,0),(uv_scale,0),(uv_scale,uv_scale),(0,uv_scale))
    for a,b,c,d in ((0,3,2,1),(4,5,6,7),(0,4,7,3),(1,2,6,5),(3,7,6,2),(0,1,5,4)):
        mesh.quad(p[a],p[b],p[c],p[d],material,t)


def local_box(mesh, s, lateral, y, length, width, height, material):
    """Rectangular object aligned to route tangent, with world-stable UV."""
    def pt(ds,dl,h): return world(s+ds,lateral+dl,y+h)
    a=length*.5;b=width*.5;h=height
    # quad on each side; prevents flat timber silhouettes
    p=[pt(-a,-b,0),pt(a,-b,0),pt(a,b,0),pt(-a,b,0),
       pt(-a,-b,h),pt(a,-b,h),pt(a,b,h),pt(-a,b,h)]
    wood=material in (8,13,14,15,16,18)
    # Local coordinates: identify long grain and a stable cross-grain direction.
    local=((-a,-b,0),(a,-b,0),(a,b,0),(-a,b,0),
           (-a,-b,h),(a,-b,h),(a,b,h),(-a,b,h))
    grain=max(range(3),key=lambda i:(length,width,height)[i])
    for ids in ((0,1,2,3),(4,7,6,5),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)):
        if wood:
            spans=[max(local[i][j] for i in ids)-min(local[i][j] for i in ids) for j in range(3)]
            along=grain if spans[grain]>.0001 else max(range(3),key=lambda j:spans[j])
            across=max((j for j in range(3) if j!=along),key=lambda j:spans[j])
            # Photographed plank sheets cover ~1.6m across and ~2.4m along grain.
            uv=tuple((local[i][across]/1.6+s*.137,local[i][along]/2.4) for i in ids)
        else:
            uv=((0,0),(length,0),(length,width),(0,width))
        mesh.quad(*(p[i] for i in ids),material,uv)


def beam_world(mesh, a, b, thickness=.10, material=8):
    # Only sizeable logs receive rounded cross-sections. Thin wires stay cheap.
    bar(mesh,a,b,thickness,material,facets=11 if thickness>=.06 else 7,repeat=.76)



def oil_lantern(mesh, flame, scale=1.0):
    """Small framed oil hurricane lantern, not a glowing sphere.

    The warm point light uses this *same* flame position. Transparent glass
    is intentionally avoided until we have correct alpha depth sorting.
    """
    cx,cy,cz=flame
    unit=.075*scale
    # Basal kerosene reservoir and raised hinged cap.
    blob(mesh,(cx,cy-.12*scale,cz),(.082*scale,.060*scale,.077*scale),10,12,4)
    bar(mesh,(cx,cy-.11*scale,cz),(cx,cy+.12*scale,cz),.014*scale,10,8)
    blob(mesh,(cx,cy+.122*scale,cz),(.069*scale,.030*scale,.069*scale),10,12,3)
    blob(mesh,(cx,cy,cz),(.038*scale,.066*scale,.041*scale),11,10,5)
    for k in range(4):
        ang=math.tau*k/4+.2
        dx, dz=unit*math.cos(ang),unit*math.sin(ang)
        bar(mesh,(cx+dx,cy-.064*scale,cz+dz),
                 (cx+dx*.83,cy+.11*scale,cz+dz*.83),.010*scale,10,6)
    # Carrying handle: curved blackened metal arc.
    points=[]
    for k in range(7):
        ang=math.pi*k/6
        points.append((cx+.049*math.cos(ang)*scale,
                       cy+(.15+.055*math.sin(ang))*scale,cz))
    for a,b in zip(points,points[1:]):bar(mesh,a,b,.009*scale,10,6)


def junction_decorations(mesh,index,side):
    """Seeded entrance dressing, deliberately outside the playable centre.

    Supports the existing T/Y/X *local* topology; no new false connectivity.
    The walls and branching route remain authoritative geometry.
    """
    marker=junction_marker(index)
    kind=junction_kind(index)
    s=marker-1.65
    flank=side
    offset=flank*max(.65,wall_width(s,flank)-.08)
    ground=floor_y(s,0.0)
    hh=wall_height(s,flank)
    head=min(hh-.12,1.92)
    if head<1.35:return
    post=world(s,offset,ground+.015)
    end=world(s,offset,head)
    if kind=='X':
        # Crossing: twin weather-beaten direction boards fixed near the lip.
        beam_world(mesh,post,end,.044,13)
        for y in (head-.25,head-.43):
            box(mesh,world(s,offset-flank*.045,y),(.56,.12,.075),
                15 if int(y*9)%2 else 8)
    elif kind=='Y':
        # Fork: angled brace reaches along one wall, not across the walking lane.
        beam_world(mesh,post,end,.045,14)
        beam_world(mesh,world(s-.55,offset,ground+.35),
                   world(s,offset,head-.11),.033,8)
        box(mesh,world(s,offset-flank*.045,head-.22),(.37,.16,.065),15)
    else:
        # Side entrance: either a single worn label or an empty broken frame.
        beam_world(mesh,post,end,.037,13)
        if noise(index,14982)>.32:
            box(mesh,world(s,offset-flank*.043,head-.22),(.42,.16,.055),8)
        else:
            beam_world(mesh,world(s-.29,offset,head-.33),
                       world(s+.26,offset,head-.43),.028,15)


def dugout_style(index):
    """Deterministic visual style of a roofed side shelter; no global registry."""
    return ('bunk','supply','aid','signal')[int(noise(index,14819)*4)]


def dugout_furnishings(mesh,index,side,outer):
    """Seeded shelter furnishings. Keep the centre clear up to search depth.

    Big crates at the far wall are future enemy cover, NOT nav blockers.
    SoldierBrain currently stops before them; later graph/collision work must
    register these as actual cover and obstacle objects before allowing entry.
    """
    if branch_type(index)!='dugout':return
    rng=random.Random(SEED*11003+index*187+side*351)
    def spot(along,across,up):return branch_point(index,side,along,across,up)
    # V3.10 environmental identity: damp depressions and scattered splinters
    # in every roofed dugout. Low-poly, seed-stable; neither new nav blockers
    # nor invisible shot colliders. Openings and corner geometry stay intact.
    floor_level=floor_y(junction_marker(index),0.0)
    for patch_id,along in enumerate((2.9,6.3,9.8)):
        if along > outer-2.15 or noise(index*37+patch_id,19482)<.24:
            continue
        side_offset=(noise(index*43+patch_id,19483)-.5)*.48
        radius=.22+.23*noise(index*43+patch_id,19484)
        y=floor_level+.018
        # Genuinely flat wet-earth stains follow the dugout floor. Layer 12
        # is a visual puddle, explicitly NON_BALLISTIC in the raycast.
        mesh.quad(spot(along-radius,side_offset,y),
                  spot(along,side_offset-radius*.57,y),
                  spot(along+radius,side_offset,y),
                  spot(along,side_offset+radius*.57,y),12,
                  ((0.,.5),(.5,0.),(1.,.5),(.5,1.)))
        if patch_id != 1:
            # Two short snapped battens; not a corridor-wide railway sleeper.
            a=spot(along-.27,side_offset+.42,floor_level+.025)
            b=spot(along+.24,side_offset+.31,floor_level+.037)
            beam_world(mesh,a,b,.023,15 if patch_id==0 else 18)
    if outer>5.5:
        section=2.5
        flank=-1 if noise(index,19501)<.5 else 1
        edge=flank*(min(branch_half_width(index,section+d) for d in (-.5,0,.5))-.18)
        if abs(edge)-.14>.68:
            def repair_spot(x,z,y):
                return spot(section+x,edge+z,floor_level+y-.006)
            fieldwork_prop(mesh,repair_spot,{'bunk':3,'supply':0,'aid':1,'signal':2}[dugout_style(index)],index+70000)
    for section in (3.4,5.6,7.4):
        if section>outer-2.1:continue
        flank=(-1 if rng.random()<.5 else 1)
        # Supply piles and military kit pressed against an excavated side wall.
        lateral=flank*(.82+.045*rng.random())
        choice=rng.randrange(4)
        if choice==0:
            anchor=spot(section,lateral,.23)
            box(mesh,anchor,(.56,.43,.48),15)
            box(mesh,spot(section,lateral,.47),(.60,.055,.49),18)
            for dz in (-.21,.21):
                box(mesh,spot(section,lateral+dz,.32),(.045,.40,.03),10)
        elif choice==1:
            # Rolled blankets / sandbag bedding.
            for j in range(3):
                p=spot(section+(j-1)*.24,lateral,.11+rng.random()*.04)
                blob(mesh,p,(.21,.12,.20),19,13,5,j*1.2)
            bar(mesh,spot(section-.26,lateral,.26),spot(section+.28,lateral,.28),.021,8,8)
        elif choice==2:
            # Timber pallet / improvised short shelf.
            for end in (-1,1):
                beam_world(mesh,spot(section+end*.42,lateral,.05),
                           spot(section+end*.42,lateral,.48),.035,14)
            for j in range(3):
                box(mesh,spot(section+(j-1)*.24,lateral,.46),(.20,.045,.45),
                    (15,16,18)[j])
            box(mesh,spot(section,lateral,.59),(.35,.22,.23),19)
        else:
            # Spent bedding and rope coil, not a raw grey cube.
            box(mesh,spot(section,lateral,.07),(.8,.10,.38),8)
            blob(mesh,spot(section,lateral,.17),(.31,.11,.18),9,13,5,section)
            for j in range(5):
                a=j*math.tau/5
                p=spot(section+math.sin(a)*.13,lateral+math.cos(a)*.10,.24)
                q=spot(section+math.sin(a+.9)*.13,lateral+math.cos(a+.9)*.10,.24)
                bar(mesh,p,q,.012,10,5)
    # Four actual room identities. Keep everything flush to the side wall,
    # away from SoldierBrain's centre corridor and the curved entrance.
    # All props are deterministic for this index and remain chunk-local.
    style=dugout_style(index)
    anchor=min(outer-2.35,9.15)
    if style=='bunk':
        # Timber bunk with rolled blanket: near-wall bed, not a world-space cube.
        for dx in (-.62,.62):
            for end in (-.37,.37):
                beam_world(mesh,spot(anchor+dx,.76+end*.02,.04),
                           spot(anchor+dx,.76+end*.02,.54),.038,13)
        for xx in (-.6,-.20,.20,.60):
            box(mesh,spot(anchor+xx,.75,.46),(.29,.07,.62),15)
        box(mesh,spot(anchor+.20,.75,.55),(.68,.10,.55),19)
        blob(mesh,spot(anchor-.48,.75,.66),(.22,.095,.22),19,10,4,2.)
    elif style=='supply':
        # Neat, half-empty shelves and a small pile of supplies.
        for h in (.52,1.05):
            box(mesh,spot(anchor,.78,h),(.96,.075,.32),14)
        for end in (-.41,.41):
            beam_world(mesh,spot(anchor+end,.91,.10),
                       spot(anchor+end,.91,1.40),.036,13)
        for n in range(3):
            box(mesh,spot(anchor-.32+n*.31,.78,.68),(.22,.19,.25),8)
    elif style=='aid':
        # Improvised first-aid corner: stained cloth on a cot and kit chest.
        box(mesh,spot(anchor,.77,.25),(1.20,.14,.68),15)
        box(mesh,spot(anchor,.76,.36),(1.03,.075,.60),19)
        box(mesh,spot(anchor-.34,.82,.58),(.24,.29,.26),9)
        box(mesh,spot(anchor-.34,.62,.74),(.12,.038,.16),18)
        box(mesh,spot(anchor-.34,.62,.74),(.035,.10,.16),18)
    else:
        # Message board and small communications shelf with bulky handset.
        box(mesh,spot(anchor,.91,1.29),(1.20,.80,.055),8)
        box(mesh,spot(anchor,.72,.66),(.87,.075,.29),13)
        box(mesh,spot(anchor+.20,.70,.80),(.26,.15,.22),10)
        for j in range(5):
            box(mesh,spot(anchor-.45+j*.19,.88,1.30),(.10,.13,.027),
                15 if j%2 else 19)
    # Large worn cover / supply crates are deliberately behind inspection stop.
    # Further back => available as *visual* future enemy cover.
    cover_along=outer-1.15
    # Substitute the existing rear cover pile, beyond the inspection stop;
    # never add an appliance to the playable passage or a room entrance.
    if index==0 or (index>2 and noise(index,82931)<.025):
        def appliance(x,z,y):return spot(cover_along+x,-.22+z,floor_level+y)
        fieldwork_prop(mesh,appliance,9,index)
        def stores(x,z,y):return spot(cover_along+x,.51+z,floor_level+y)
        fieldwork_prop(mesh,stores,8,index)
        def refuse(x,z,y):return spot(cover_along-.95+x,-.70+z,floor_level+y)
        fieldwork_prop(mesh,refuse,5,index+809)
        def meal(x,z,y):return spot(cover_along-.80+x,.67+z,floor_level+y)
        fieldwork_prop(mesh,meal,6,index+910)
    else:
        center=spot(cover_along,.08,.52)
        box(mesh,center,(1.08,1.06,.73),16)
        for y in (.22,.47,.76):
            box(mesh,spot(cover_along,.08,y),(.075,.028,.76),18)
        for dx in (-.45,.45):
            box(mesh,spot(cover_along+dx,.08,.55),(.055,.99,.08),13)
    # Battered hanging rags / equipment on the wall; offset remains near bank.
    if outer>10.5:
        # Formerly a beige, floating oval: an actual scrap of hanging canvas.
        # Fixed upper edge, modest folds and a torn free hem, attached to a bar.
        rng2=random.Random(SEED*3487+index*197+side*13)
        along0,along1=8.36,9.04
        top_y=1.69
        bar(mesh,spot(along0,-.97,top_y),spot(along1,-.97,top_y),.024,13,9)
        count=5
        for j in range(count):
            a=along0+(along1-along0)*j/count
            b=along0+(along1-along0)*(j+1)/count
            va=.47+.16*noise(index*43+j,1931)
            vb=.47+.16*noise(index*43+j+1,1931)
            # Near-wall flap, always tied to the actual top edge.
            mesh.quad(spot(a,-.965,top_y),spot(b,-.965,top_y),
                      spot(b,-.927,top_y-vb),spot(a,-.927,top_y-va),26,
                      ((j/count,0.),((j+1)/count,0.),((j+1)/count,1.),(j/count,1.)))


def lantern_side(k):
    return 1 if noise(k,1882)>.5 else -1


def branch_point(index, side, along, across=0.0, height=1.5):
    """Shared geometric anchor for dugout decorations and nearby lights."""
    marker=junction_marker(index)
    x,z,h=route(marker)
    near=max(wall_width(marker,side),wall_width(marker-2,side),
             wall_width(marker+2,side))-.18
    depth=near+along
    shift=across+branch_center_shift(index,along)
    return (x+side*math.cos(h)*depth+math.sin(h)*shift,
            height,
            z-side*math.sin(h)*depth+math.cos(h)*shift)


def nearby_lanterns(s, radius=35.0):
    """Exactly the same seeded lamps as the geometry; no global lamp cache."""
    first=max(0,int(math.floor((s-radius-16)/37)))
    last=int(math.ceil((s+radius-16)/37))+1
    result=[]
    for n in range(first,last+1):
        k=16+n*37
        if abs(k-s)>radius or scene_at(k)!='fortified': continue
        side=lantern_side(k)
        w=wall_width(k,side)
        h=min(1.8,wall_height(k,side)-.1)
        result.append(world(k,side*(w-.45),h-.18))
    # Three actual wall lamps per covered main-route tunnel. Bounded lookup.
    base=max(0,int((s-radius-280.)//504)-1)
    for cycle in range(base,base+4):
        for local in (27.,35.,43.):
            m=280.+cycle*504.+local
            if abs(m-s)<radius:
                wall=wall_width(m,-1)
                result.append(world(m,-(wall-.26),1.66+tunnel_drop(m)))
    # Distant branches are not tracked. Only local, still-visible dugouts
    # contribute a warm accent light around their entrance.
    candidate=nearest_junction(s)
    if candidate is not None and abs(candidate[0]-s)<radius:
        i=int(candidate[1])
        if branch_type(i)=='dugout':
            for side in junction_sides(i):
                result.append(branch_point(i,side,5.0,-.78,1.72))
    return result[:8]

def environment_chunk(mesh,start):
    """Rugged trench life: timber revetment, duckboards, sandbags, puddles,
    crates, lanterns, signs. All positioned in true world coordinates."""
    rng=random.Random(SEED*5501+round(start/CHUNK_LEN)*7919)
    # Alternate reinforced and bare earth segments. No objects across side
    # branches or on the soldier's walking centreline except flat duckboards.
    for k in range(int(start)+1,int(start+CHUNK_LEN)-1):
        s=float(k)
        theme=scene_at(s)
        near_branch=near_junction(s,5)
        if near_branch: continue
        # Patchy plank paths: incomplete parallel strips, shifted seams, mud
        # gaps and different widths. Avoid identical railway sleepers.
        if tunnel_drop(s)>-.10 and theme in ('fortified','ruined','marsh','forest') and k%3==0:
            patch=int(s//21.)
            lining,_=lining_section(s)
            if noise(patch,2100)>(.24 if lining==0 else .48 if lining==1 else .78):
                across=min(.85, min(wall_width(s,-1),wall_width(s,1))-.18)
                count=3 if noise(patch,2101)<.48 else 4
                for lane in range(count):
                    if noise(k*13+lane,2102)<(.10 if theme=='fortified' else .24):continue
                    lane_offset=-across+(lane+.5)*2*across/count
                    ds=(noise(k*17+lane,2103)-.5)*.38
                    length=1.45+.72*noise(k*17+lane,2104)
                    wid=2*across/count-.025
                    y=floor_y(s+ds,lane_offset)+.005
                    local_box(mesh,s+ds,lane_offset,y,length,wid,.032+.015*noise(k+lane,2105),
                              (15,16,18,8)[int(noise(k*17+lane,2106)*4)])
        if tunnel_drop(s)>-.10 and theme in ('marsh','ruined') and k%5==2:
            # dark reflective-looking wet mud patches, slightly above the floor
            local_box(mesh,s,rng.uniform(-.42,.42),floor_y(s,0.)+.014,rng.uniform(.7,1.65),
                      rng.uniform(.42,.98),.012,12)
        if theme in ('fortified','ruined','forest') and k%3==0:
            for side in (-1,1):
                rng.random()  # preserve downstream existing decoration RNG
        # Posts, sills and lining are now one construction in wall_dressing_chunk.
        if tunnel_drop(s)>-.10 and theme in ('fortified','ruined') and k%23==11:
            side=1 if rng.random()<.5 else -1
            w=wall_width(s,side)
            local_box(mesh,s,side*(w-.28),-.035,.8,.42,.58,9) # crate, side-mounted
        if tunnel_drop(s)>-.10 and theme in ('fortified','ruined') and k%19==9:
            side=1 if rng.random()<.5 else -1
            h=wall_height(s,side)
            # sandbag mound on the wall's lip, not in player path
            for r in range(2):
                for q in range(3-r):
                    ss=s+(q-(2-r)*.5)*.45+.06*(rng.random()-.5)
                    lateral=side*(wall_width(ss,side)+.14+(.025 if r else 0.))
                    base=world(ss,lateral,wall_height(ss,side)+.02+r*.157)
                    sandbag_mesh(mesh,base,.54,.36,.18,k*79+r*5+q,
                                 heading=math.pi*.5-route(ss)[2]+.15*(rng.random()-.5))
        if tunnel_drop(s)>-.10 and theme=='fortified' and k%37==16:
            side=lantern_side(k)
            w=wall_width(s,side)
            h=min(1.8,wall_height(s,side)-.1)
            # lantern on an actual metal bracket, with emissive inner bulb
            foot=world(s,side*(w-.08),h)
            edge=world(s,side*(w-.42),h+.08)
            beam_world(mesh,foot,edge,.025,10)
            base=world(s,side*(w-.44),h-.30)
            box(mesh,base,(.18,.055,.18),10)
            flame=world(s,side*(w-.45),h-.18)
            oil_lantern(mesh,flame,.90)
    # A few patchy, muddy drapes, never a full corridor of identical canvas.
    # Positions are seed-stable and never obstruct the soldier's route.
    first=int(start//29)-1
    last=int((start+CHUNK_LEN)//29)+2
    for i in range(max(0,first),last):
        ss=14.0+29.0*i+.95*(noise(i,6801)-.5)
        if not start<=ss<start+CHUNK_LEN:continue
        if near_junction(ss,6.5) or tunnel_open(ss)>.05:continue
        if scene_at(ss) not in ('fortified','ruined','forest'):continue
        if noise(i,6802)<.32:continue
        side=-1 if noise(i,6803)<.5 else 1
        hanging_fabric(mesh,ss,side,i,span=1.1+1.5*noise(i,6804),
                       drop=.48+.48*noise(i,6805))
    # Ruined-zone outposts are generated separately in battlefield_debris_chunk.

def fieldwork_prop(mesh, spot, family, seed):
    """Five small repair/supply silhouettes in a .94 x .28m envelope.

    spot maps local metres onto the existing floor/room frame. Solid faces use
    real wood/metal layers and therefore participate in the existing shot mesh.
    No per-item render objects, collision registration or global RNG state.
    """
    rng=random.Random(SEED*917+seed*1051)
    def plank(x,z,y,length,width,height,layer=18):
        corners=[spot(x+dx,z+dz,y+dy) for dx,dz,dy in
                 ((-length/2,-width/2,0),(length/2,-width/2,0),
                  (length/2,width/2,0),(-length/2,width/2,0),
                  (-length/2,-width/2,height),(length/2,-width/2,height),
                  (length/2,width/2,height),(-length/2,width/2,height))]
        for indices in ((0,4,5,1),(1,5,6,2),(2,6,7,3),
                        (3,7,4,0),(4,7,6,5),(0,1,2,3)):
            pts=[corners[k] for k in indices]
            # Metre-scaled grain on each face, including thin end grain.
            u=math.dist(pts[0],pts[1])*2
            v=math.dist(pts[1],pts[2])*2
            mesh.quad(*pts,layer,((0,0),(u,0),(u,v),(0,v)))
    if family>=5:
        first_vertex=len(mesh.vertices)
        # Modern field supplies share the existing bounded prop frame and batch.
        # All filled volumes retain solid material semantics (including bags).
        def rounded_rings(rings, layer, phase=0.):
            n=12
            for row in range(len(rings)-1):
                def point(j,k):
                    y,rx,rz,shift=rings[k]
                    a=math.tau*(j%n)/n
                    crease=1.+.12*math.sin(a*5+phase+k*.7)
                    return spot(shift+rx*math.cos(a)*crease,rz*math.sin(a)*crease,y)
                for j in range(n):
                    mesh.quad(point(j,row),point(j+1,row),point(j+1,row+1),point(j,row+1),
                              layer,((j/n,row/3),((j+1)/n,row/3),((j+1)/n,(row+1)/3),(j/n,(row+1)/3)))
        if family==5:  # flattened base, asymmetric shoulders, gathered plastic neck
            h=.29+.16*(seed%2)+.035*rng.random()
            rounded_rings(((0.,0.,0.,0.),(.012,.22,.10,0.),
                           (h*.40,.23,.10,-.016),(h*.78,.18,.10,.025),
                           (h,.026,.023,.055),(h+.025,.014,.012,.055)),24,seed)
            plank(.055,0,h,.065,.025,.018,10)
            rounded_rings(((h+.025,.012,.01,.055),(h+.075,.052,.025,.07),
                           (h+.08,0.,0.,.07)),24,seed)
        elif family==6:  # folded foil meal pouch, crushed carton, no invented labels
            for j in range(3):
                x=(j-1)*.26;z=rng.uniform(-.035,.035)
                layer=9 if j!=1 else 10
                for k in range(3):
                    a=x-.10+k*.067;b=a+.067
                    ya=.008+(.022 if k==1 else .004)
                    yb=.008+(.022 if k+1==1 else .004)
                    mesh.quad(spot(a,z-.085,ya),spot(b,z-.085,yb),
                              spot(b,z+.085,yb+.012),spot(a,z+.085,ya+.012),layer,
                              ((k/3,0),((k+1)/3,0),((k+1)/3,1),(k/3,1)))
                plank(x,z-.08,.012,.20,.012,.014,10)
            plank(.29,.015,.0,.18,.16,.075,9)
            plank(.29,.015,.075,.18,.018,.013,24)
        elif family==7:  # ribbed polymer transport case with metal catches
            plank(0,0,0,.62,.25,.29,24)
            plank(0,0,.29,.65,.27,.035,24)
            for x in (-.25,-.08,.09,.25):
                plank(x,-.12,.035,.025,.015,.22,10)
            for x in (-.21,.21):plank(x,-.12,.24,.045,.014,.075,10)
            bar(mesh,spot(-.085,-.12,.20),spot(.085,-.12,.20),.014,10,6)
        elif family==8:  # squat jerrycan with a genuinely open carry handle
            plank(0,0,0,.29,.21,.32,24)
            plank(0,0,.32,.27,.20,.045,24)
            for x in (-.10,.10):plank(x,0,.365,.035,.045,.07,24)
            plank(0,0,.425,.23,.045,.027,24)
            bar(mesh,spot(-.09,-.065,.36),spot(-.09,-.065,.39),.032,10,8)
            for x in (-.08,.08):plank(x,-.112,.06,.018,.012,.22,10)
        elif family==9:  # 60cm dryer: recessed drum, lint vent, fascia, no soap drawer
            plank(0,0,.035,.60,.60,.78,10)
            plank(0,0,.815,.62,.62,.035,10)
            # Front faces toward negative local x. Annular door and recessed dark drum.
            def ring(a,r,depth):return spot(-.306-depth,math.cos(a)*r,.43+math.sin(a)*r)
            for j in range(24):
                a=j*math.tau/24;b=(j+1)*math.tau/24
                uv=((0,0),(1,0),(1,1),(0,1))
                mesh.quad(ring(a,.225,.042),ring(b,.225,.042),ring(b,.177,.042),ring(a,.177,.042),10,uv)
                mesh.quad(ring(a,.177,.042),ring(b,.177,.042),ring(b,.155,.006),ring(a,.155,.006),24,uv)
                mesh.quad(ring(a,.155,.005),ring(b,.155,.005),spot(-.311,0,.43),spot(-.311,0,.43),24,uv)
            plank(-.318,0,.70,.025,.56,.09,9)
            bar(mesh,spot(-.34,.17,.746),spot(-.36,.17,.746),.033,10,12)
            plank(-.339,-.10,.725,.013,.13,.036,24)
            for j in range(7):plank(-.32,-.18+j*.057,.11,.025,.032,.065,24)
            plank(-.365,-.19,.41,.035,.028,.10,10)
            # Drum baffles distinguish a vented tumble dryer from a laundry washer.
            for z in (-.07,.07):plank(-.322,z,.37,.018,.016,.11,10)
            for x in (-.24,.24):
                for z in (-.24,.24):plank(x,z,0,.055,.055,.035,24)
        # One solid atlas layer: enamel, creased plastic, kraft and olive plastic.
        # Inset each tile to avoid mip bleed. No change to old material IDs.
        for i in range(first_vertex,len(mesh.vertices),9):
            layer=int(mesh.vertices[i+8])
            tile=(0 if layer==10 else 1 if family==5 else 3) if layer!=9 else 2
            if layer in (9,24) or (family==9 and layer==10):
                u,v=mesh.vertices[i+6:i+8]
                mesh.vertices[i+6]=.02+(tile%2)*.5+min(1.,max(0.,u))*.46
                mesh.vertices[i+7]=.52-(tile//2)*.5+min(1.,max(0.,v))*.46
                mesh.vertices[i+8]=27.
        return
    if family==0:  # open slatted ammunition tray, actual hollow interior
        length=.54+.12*rng.random()
        for j in range(3):plank(0,(j-1)*.077,0,length,.071,.026,15)
        for side in (-1,1):
            for row in range(2):
                plank(0,side*.115,.028+row*.087,length,.022,.074,18)
            plank(side*(length/2-.012),0,.026,.024,.22,.18,16)
            plank(side*length*.34,-.129,.04,.025,.012,.13,10)
        # Loose lid resting diagonally across one end, leaving contents visible.
        plank(-length*.29,0,.207,.11,.26,.022,16)
    elif family==1:  # dented open pail, inner wall and rolled rim, no fake cap
        n=10
        def ring(j,up,inner=False):
            angle=math.tau*(j%n)/n
            r=(.092 if up==0 else .122)*(1-.12*math.sin(j*2.7+seed)**4)
            if inner:r-=.009
            return spot(math.cos(angle)*r,math.sin(angle)*r,.018+up)
        for j in range(n):
            k=(j+1)%n
            uv=((j/n,0),(k/n,0),(k/n,.36),(j/n,.36))
            mesh.quad(ring(j,0),ring(k,0),ring(k,.27),ring(j,.27),10,uv)
            mesh.quad(ring(k,0,True),ring(j,0,True),ring(j,.27,True),ring(k,.27,True),10,uv)
            mesh.quad(ring(j,.27),ring(k,.27),ring(k,.27,True),ring(j,.27,True),10,uv)
            mesh.quad(spot(0,0,.018),ring(k,0),ring(j,0),spot(0,0,.018),10,uv)
        for j in range(8):
            t=j*math.pi/8;u=(j+1)*math.pi/8
            bar(mesh,spot(.119*math.cos(t),0,.26+.14*math.sin(t)),
                spot(.119*math.cos(u),0,.26+.14*math.sin(u)),.007,10,6)
    elif family==2:  # field wire spool, wood cheeks with exposed winding
        for y in (.012,.174):
            bar(mesh,spot(0,0,y),spot(0,0,y+.018),.118,16,10)
        bar(mesh,spot(0,0,.03),spot(0,0,.175),.066,13,8)
        for row in range(6):
            for j in range(10):
                t=j*math.tau/10;u=(j+1)*math.tau/10
                y=.043+row*.022
                bar(mesh,spot(.080*math.cos(t),.080*math.sin(t),y),
                    spot(.080*math.cos(u),.080*math.sin(u),y+.002),.009,10,6)
    elif family==3:  # short broken duckboard repair, missing and split slats
        for z in (-.085,.085):plank(0,z,0,.85,.035,.032,13)
        for j in range(6):
            if j==2:continue
            width=.14 if j==3 else .25
            plank((j-2.5)*.14,.025 if j==3 else 0,.032,
                  .10+.015*rng.random(),width,.025,15 if j%2 else 16)
        plank(.02,-.09,.057,.29,.028,.012,18)
    else:  # paired repaired shoring feet with mismatched battens and iron nails
        for x in (-.29,.29):
            plank(x,.055,0,.072,.10,.43+.08*rng.random(),13)
        for row in range(3):
            plank(.015*(row-1),-.012,.09+row*.125,.77-.05*rng.random(),.043,.085,
                  18 if row!=1 else 16)
            for x in (-.29,.29):
                bar(mesh,spot(x,-.037,.13+row*.125),
                    spot(x,-.044,.13+row*.125),.009,10,6)


def fieldwork_layout(start):
    """Global anchor ownership; whole props leave a >=1.36m central lane."""
    for index in range(max(0,int((start-9)//6.7)-1),int((start+CHUNK_LEN)/6.7)+1):
        s=9.+index*6.7+(noise(index,1001)-.5)*3.2
        if not start<=s<start+CHUNK_LEN:continue
        if near_junction(s,6.) or obstacles_in_range(s-1.5,s+1.5):continue
        if noise(index,1004)<.18:continue
        side=-1 if noise(index//3,1002)<.5 else 1
        # Check the complete footprint at several heights against the actual
        # bank, including its undercut. Narrow bends get no new decoration.
        bank=min(wall_offset(ss,side,floor_y(ss,0)+dy,wall_height(ss,side))
                 for ss in (s-.5,s-.25,s,s+.25,s+.5) for dy in (0.,.25,.55))
        edge=bank-.17
        if edge-.14<.68:continue
        family=int(noise(index,1003)*5)
        yield s,side*edge,family,index


def modern_fieldwork_layout(start):
    """Reuse safe anchors, checking the *whole* meal/refuse cluster footprint."""
    for s,edge,old,index in fieldwork_layout(start):
        selected=index in (2,3,4,5,27) if s<200 else index%13 in (2,3,4)
        if not selected:continue
        if near_junction(s+.35,6.6) or obstacles_in_range(s-.55,s+1.15):continue
        side=1 if edge>0 else -1
        bank=min(wall_offset(t,side,floor_y(t,0)+y,wall_height(t,side))
                 for t in (s-.5,s,s+.5,s+1.15) for y in (0.,.3,.6))
        family={2:5,3:8,4:5,5:7,27:5}.get(index,5+index%4)
        lining=min(lining_bank(t,side) for t in (s-.5,s,s+.5,s+1.15))
        # Include the existing diagonal braces, not only the soil envelope.
        inset=.335 if family==7 else .315
        edge=side*min(abs(edge),bank-.17,lining-inset)
        if abs(edge)-.14<.68:continue
        yield s,edge,family,index


def trench_detail_props(mesh, start):
    """Baked, grounded wall-edge repairs; also follows covered passage ramps."""
    clusters={p[3]:p for p in modern_fieldwork_layout(start)}
    for s,edge,family,index in fieldwork_layout(start):
        if index in clusters:s,edge,family,index=clusters[index]
        def spot(x,z,y):
            return world(s+x,edge+z,floor_y(s+x,edge+z)+y-.006)
        # Quiet stretches retain existing field repairs; activity clusters are
        # globally anchored and repeat sparsely beyond the introductory 200m.
        if index in clusters:
            fieldwork_prop(mesh,spot,family,index)
            def litter(x,z,y):return spot(x+.72,z,y)
            fieldwork_prop(mesh,litter,6,index+19000)
        else:
            fieldwork_prop(mesh,spot,family,index)

def lining_section(s):
    """Render-only roughly 14–34m construction runs: dense, wattle, repair, sparse, earth, heavy.

    Shared by wall lining and existing duckboards; never used by navigation.
    Stable global metres, independent of chunk size and build order.
    """
    zone=max(0,int(s//24.))
    def boundary(i):return i*24.+(noise(i,28112)-.5)*10. if i else 0.
    if zone and s<boundary(zone):zone-=1
    elif s>=boundary(zone+1):zone+=1
    style=(zone+int(noise(0,28101)*3))%6
    return style,zone


def lining_bank(s,side):
    """Inner soil envelope for straight, grounded framing instead of warped posts."""
    ground=floor_y(s,0.)
    top=min(2.5,wall_height(s,side)-ground-.10)
    return min(wall_offset(s,side,ground+top*j/12,wall_height(s,side))
               for j in range(13))


def wall_dressing_layout(start):
    """Connected three-metre bays with explicit end/portal and obstacle clearance."""
    first=max(0,int((start-1.5)//3.)-1)
    last=int((start+CHUNK_LEN)/3.)+1
    for i in range(first,last):
        s=1.5+i*3.
        if not start<=s<start+CHUNK_LEN:continue
        style,zone=lining_section(s)
        crossing=nearest_junction(s)
        for side in (-1,1):
            # Wattle is a local repair on one bank, never a symmetric fence.
            if style==1 and side>0 and noise(i//2,28117)<.48:continue
            # Only the excavated side needs a gap; the opposite bank can retain
            # its lining. Include the full bay and an extra door-jamb margin.
            if crossing is not None and abs(crossing[0]-s)<4.6 and side in junction_sides(crossing[1]):continue
            if style==2:
                centre=zone*24.+12.
                preferred=1 if wall_width(centre,1)>=wall_width(centre,-1) else -1
                if side!=preferred:continue
            samples=tuple(s+d for d in (-1.55,-1.,-.5,0.,.5,1.,1.55))
            height=min(wall_height(t,side)-floor_y(t,0)-.16 for t in samples)
            cap=(2.35,1.18 if side>0 else 1.55,1.65,1.85,1.30,2.40)[style]
            height=min(height,cap*(.92+.08*math.sin(s*.19+side)))
            if height<.65:continue
            width=min(lining_bank(t,side) for t in samples)
            # Scaled thickness in narrower reaches, never move the actual bank.
            if width<.80:continue
            blocked=any(ob.lateral*side>0 and abs(ob.s-s)<ob.length*.5+1.7
                        for ob in obstacles_in_range(s-3.,s+3.))
            if blocked:continue
            yield s,side,height,i*2+(side>0)


def wall_dressing_chunk(mesh,start):
    """Bank-hugging plank/wattle repairs, hanging tools, rope and exposed roots.

    Existing earth triangles remain intact. Solid surfaces use wood/metal;
    only hanging fabric uses the existing non-ballistic animated cloth layer.
    """
    for s,side,height,seed in wall_dressing_layout(start):
        style,zone=lining_section(s)
        # Section stock/age, six-metre repair campaigns, then small component
        # tolerances. All keys use global metres; no mutable/random state.
        age=noise(zone,28501)
        campaign=int(s//6.)
        wear=noise(campaign+side*1703,28502)
        stock=16 if age>.60 else 18
        repair=wear>.64
        height*=.82+.18*noise(campaign,28503)
        span=.90+.09*noise(campaign,28504)
        middle=(noise(campaign,28505)-.5)*.40
        banks={}
        room=min(lining_bank(s+d,side) for d in (-1.55,-1.,-.5,0.,.5,1.,1.55))
        thickness=min(1.,(room-.72)/.19)
        def point(x,y,inset=.065):
            t=s+x*span
            if t not in banks:banks[t]=lining_bank(t,side)
            off=side*(banks[t]-inset*thickness)
            # Board feet, posts and sill share the actual floor at their offset.
            return world(t,off,floor_y(t,off)+y)
        def strip(x0,y0,x1,y1,width,layer=18,depth=.075):
            # Flat split timber with real thickness, rather than round rods.
            dx,dy=x1-x0,y1-y0
            length=math.hypot(dx*span,dy)
            ux,uy=-dy/length*width*.5/span,dx*span/length*width*.5
            coords=((x0+ux,y0+uy),(x0-ux,y0-uy),
                    (x1-ux,y1-uy),(x1+ux,y1+uy))
            front=[point(x,y,depth) for x,y in coords]
            board_depth=.028+.014*noise(campaign,28506)
            back=[point(x,y,depth-board_depth) for x,y in coords]
            offset=(s+x0)*.137
            uv=((offset,0),(offset+width/1.6,0),
                (offset+width/1.6,length/2.4),(offset,length/2.4))
            mesh.quad(*front,layer,uv)
            mesh.quad(*reversed(back),layer,uv)
            for j in range(4):
                k=(j+1)%4
                mesh.quad(front[j],back[j],back[k],front[k],layer,
                          ((0,0),(.07,0),(.07,length),(0,length)))
        # Each bay is a built frame: embedded feet, bottom sleeper, connected
        # cap and posts. No unrelated free-standing poles on the same wall.
        for x in (-1.46,1.46):
            strip(x,-.045,x,height+.015,.14 if style==5 else .085,stock,depth=.16)
            # Short transverse anchors connect framing to soil without turning
            # each narrow upright into an implausibly deep, tapered wooden slab.
            for yy in (height*.66,):
                anchor=point(x,yy,.12)
                t=s+x*span
                bank=wall_offset(t,side,anchor[1],wall_height(t,side))+.025
                bar(mesh,anchor,world(t,side*bank,anchor[1]),.035,stock,6)
        strip(-1.50,.035,1.50,.035,.095,stock,depth=.15)
        if style not in (1,4) and (style!=3 or repair):
            strip(-1.50,height-.17,1.50,height-.17,.095 if style==5 else .075,stock,depth=.15)
        if style in (0,2,3):
            # A broad patch of close-set boards with split, uneven top edges.
            count=(5+int(wear*3)) if style==3 else (12+int(age*4))
            for j in range(count):
                x=(j-(count-1)*.5)*(2.77/(count-1))
                top=height-.025-.10*noise(campaign*19+j//2,27402)
                if repair and j in (count//2,count//2+1):top-=.14
                bottom=-.025
                if style==2 and j in (3,11):bottom=height*.22
                # A section shares its timber stock; occasional repairs are
                # visible, not a different bright material on every board.
                layer=stock
                # One contiguous repair, not alternating bright/dark stripes.
                if repair and count//3<=j<count//3+3:layer=15
                spacing=2.77/(count-1)
                width=(.19 if style==3 else spacing*.91)*(.94+.10*noise(campaign*13+j,28507))
                lean=.018*(noise(campaign,28508)-.5)
                strip(x,bottom,x+lean,top,width,layer)
            if style==2:
                strip(-1.42,.14,1.42,height-.20,.095,15 if repair else stock,depth=.18)
            else:
                strip(-1.50,height*(.40+.12*wear),1.50,height*(.40+.12*wear),.075,stock,depth=.15)
        elif style==4:
            # Open earth with one low retaining rail and deeply seated uprights.
            strip(-1.50,.23+.12*wear,1.50,.23+.12*wear,.12+.035*age,stock,depth=.12)
        elif style==5:
            # Heavy shoring: broad horizontal retaining boards, a middle post
            # and two braces terminating at the common ground sill.
            pitch=.23+.055*age
            for row in range(max(3,int(height/pitch))):
                yy=.12+row*pitch
                end=1.44-(.16 if repair and row%3==1 else 0.)
                layer=15 if repair and row in (1,2) else stock
                strip(-1.44,yy,end,yy+.015*(wear-.5),pitch*.85,layer,depth=.075)
            strip(middle,-.045,middle,height,.14,stock,depth=.16)
            strip(-1.40,.12,middle-.10,height-.20,.12,stock,depth=.18)
            if age>.60 or repair:
                strip(1.40,.12,middle+.10,height-.20,.12,stock,depth=.18)
        else:
            # Slender round withies weave around stakes, not flat fence rails.
            for x in (-.75,0.,.75):
                strip(x,-.035,x,height,.045,14,depth=.12)
            rows=max(5,int(height/(.14+.045*age)))
            for row in range(rows):
                base=.09+row*(height-.17)/rows
                for k in range(8):
                    # A small worn patch between stakes; continuous earth behind.
                    if wear>.55 and row>rows*.62 and 2<=k<=3:continue
                    def twig(j):
                        x=-1.5+j*.375
                        y=base+(.018+.018*wear)*math.sin(x*2.1+row*.9+campaign)
                        depth=.105+.035*math.cos(x*math.tau/.75+row*math.pi)
                        return point(x,y,depth)
                    bar(mesh,twig(k),twig(k+1),.025+.009*age,14,6,repeat=.8)
        # Three repairs alternate large, recognizable silhouettes at eye level.
        kit=seed%8==3 and room>1.06 and height>1.3
        if kit and style==0:
            # Rope coiled on a wall peg: sagging elliptical loops, not a torus.
            bar(mesh,point(.30,height*.83,.11),point(.30,height*.83,.25),.025,10,6)
            for loop in range(2):
                for j in range(12):
                    t=j*math.tau/12;u=(j+1)*math.tau/12
                    rx=.20+loop*.018;ry=.32+loop*.022
                    cy=height*.83-ry
                    bar(mesh,point(.30+rx*math.sin(t),cy+ry*math.cos(t),.22+loop*.014),
                        point(.30+rx*math.sin(u),cy+ry*math.cos(u),.22+loop*.014),.012,19,6)
        elif kit and style==2:
            # Entrenching spade: D grip, wooden shaft and a shaped steel blade.
            y=height*.84
            bar(mesh,point(-.2,y-.16,.24),point(.02,.39,.24),.022,15,7)
            for a,b in (((-.29,y),(-.12,y)),((-.29,y),(-.25,y-.16)),
                        ((-.12,y),(-.16,y-.16)),((-.25,y-.16),(-.16,y-.16))):
                bar(mesh,point(*a,.24),point(*b,.24),.014,10,6)
            blade=[(-.12,.44),(.16,.44),(.14,.18),(.02,.12),(-.10,.18)]
            center=point(.02,.29,.25)
            for j in range(5):
                mesh.quad(center,point(*blade[j],.24),point(*blade[(j+1)%5],.24),center,10,
                          ((.5,.5),(0,0),(1,1),(.5,.5)))
            # Hooks clearly attach the tool to the boards.
            for yy in (.45,y-.16):
                bar(mesh,point(-.13,yy,.11),point(-.13,yy,.27),.012,10,6)
        elif kit and style==1:
            hanging_fabric(mesh,s,side,seed+81000,span=1.15,drop=min(.70,height*.44))
        # A small exposed root fragment remains embedded in the ACTUAL soil,
        # not in the flat construction plane where it could float above boards.
        if style==2 and seed%6==0:
            def root(t,y):
                h=wall_height(t,side)
                up=min(h-.03,floor_y(t,0)+y)
                return world(t,side*(wall_offset(t,side,up,h)-.004),up)
            bar(mesh,root(s,height+.13),root(s+.12,height-.04),.012,14,6)


def trench_floor_litter(mesh, start):
    """Tiny stones and compacted clumps, kept close to walls and off nav path."""
    for i in range(max(0,int(start//3.3)-1),int((start+CHUNK_LEN)//3.3)+2):
        s=1.15+i*3.3+.35*(noise(i,1802)-.5)
        if not start<=s<start+CHUNK_LEN or near_junction(s,3.5):continue
        side=-1 if noise(i,1803)>.5 else 1
        offset=side*min(.88,wall_width(s,side)-.15)
        if noise(i,1804)>.22 and tunnel_drop(s)>-.10:
            cluster=1+int(noise(i,1806)*3)
            for k in range(cluster):
                ss=s+.12*(k-(cluster-1)*.5)
                edge_off=offset+side*(.02+.08*noise(i*7+k,1807))
                r=.032+.05*noise(i*5+k,1805)
                rock_fragment(mesh,world(ss,edge_off,floor_y(ss,edge_off)),r,
                              i*5+k,7,(1.0,.72,1.18))
                if noise(i*11+k,1808)>.72:
                    blob(mesh,world(ss+.05,edge_off-side*.03,floor_y(ss,edge_off)+r*.18),
                         (r*.95,r*.42,r*.75),8,9,4,i*.47+k)


def fallen_wood_layout(start):
    """Seed-stable small sticks/logs BESIDE the walkable centreline.

    Returns (route_m, lateral_m, length_m, radius_m, skew, seed_key).
    Low-profile scenery is not a new collider or AI navigation obstacle.
    """
    first=max(0,int((start-1.4)//5.7)-1)
    last=int((start+CHUNK_LEN-1.4)//5.7)+2
    for i in range(first,last):
        s=1.4+5.7*i+(noise(i,8111)-.5)*2.0
        if not start<=s<start+CHUNK_LEN:continue
        if near_junction(s,5.5) or obstacles_in_range(s-1.4,s+1.4):continue
        theme=scene_at(s)
        if noise(i,8112)>({'ruined':.81,'forest':.72,'fortified':.55,
                            'rocky':.35,'marsh':.48,'exposed':.39}[theme]):continue
        side=-1 if noise(i,8113)<.5 else 1
        # Entire cylinder (including uneven caps) is outside +-0.58m of
        # the route centre; loose twigs have no new collision geometry.
        limit=wall_width(s,side)-.22
        if limit<.70:continue
        lateral=side*(.70+max(0.,limit-.70)*noise(i,8114))
        radius=.022+.050*noise(i,8115)
        length=.35+1.35*noise(i,8116)
        skew=(noise(i,8117)-.5)*.35
        yield (s,lateral,length,radius,skew,i)


def trench_fallen_wood_chunk(mesh,start):
    """Grounded bark, snapped stock and twig clusters along the wall edges.

    Kept away from corners, scheduled obstacles and the centre walking strip.
    One baked mesh, no per-frame transforms or extra OpenGL draw calls.
    """
    for s,lateral,length,radius,skew,seed in fallen_wood_layout(start):
        direction=1 if noise(seed,8120)>.5 else -1
        s0=s-length*.5;s1=s+length*.5
        off0=lateral+skew*.5;off1=lateral-skew*.5
        side=-1 if lateral<0 else 1
        # Clamp near both ends: no wood through irregular trench banks.
        off0=side*min(abs(off0),wall_width(s0,side)-.14)
        off1=side*min(abs(off1),wall_width(s1,side)-.14)
        floor0=floor_y(s0,off0)
        floor1=floor_y(s1,off1)
        p=world(s0,off0,floor0+radius*.78)
        q=world(s1,off1,floor1+radius*(.81+.09*noise(seed,8121)))
        layer=13 if noise(seed,8122)<.65 else 14
        bar(mesh,p,q,radius,layer,6 if radius<.044 else 8,.52)
        # A few shorter snapped side shoots, some pieces have no twigs.
        if radius>.041 and noise(seed,8123)>.42:
            for k in range(1+int(noise(seed,8124)*2)):
                t=.18+.43*noise(seed*2+k,8125)
                pp=tuple(p[j]*(1-t)+q[j]*t for j in range(3))
                twig_len=.16+.26*noise(seed*3+k,8126)
                twig_s=s0+length*t+.12*direction
                twig_off=side*min(wall_width(twig_s,side)-.15,
                                  abs(lateral)+twig_len*.3)
                qq=world(twig_s,twig_off,floor_y(twig_s,twig_off)+.035)
                bar(mesh,pp,qq,.012,layer,6,.28)


def make_chunk(start):
    mesh=MeshBuilder()
    masks=world_excavation_masks(start)
    # Share the exact world-X/Z terrain projection between roofs and props.
    segments=_terrain_route_segments(start)
    edge_height_cache={}
    stride=SEGMENT*.5
    normal_cache={}
    def earth_normal(s,y,side=0):
        key=(s,y,side)
        if key not in normal_cache:normal_cache[key]=earth_surface_normal(s,y,side)
        return normal_cache[key]
    for i in range(int(CHUNK_LEN/stride)):
        s0=start+i*stride;s1=s0+stride
        # Real excavated floor, anchored to BOTH banks. The previous fixed
        # 3.84m-wide floor protruded through walls whenever the trench was
        # narrow (~1.55m). It produced false overlapping horizontal layers.
        # A 4cm overlap at each actual bank edge avoids light leaks.
        def floor_side_edge(s,side):
            h=wall_height(s,side)
            return side*(wall_offset(s,side,floor_y(s,0.),h)+.035)
        if s0==int(s0):
            floor_end=s0+SEGMENT
            lo0,hi0=floor_side_edge(s0,-1),floor_side_edge(s0,1)
            lo1,hi1=floor_side_edge(floor_end,-1),floor_side_edge(floor_end,1)
            for transverse in range(10):
                t0=transverse/10.;t1=(transverse+1)/10.
                a0=smooth(lo0,hi0,t0);b0=smooth(lo0,hi0,t1)
                a1=smooth(lo1,hi1,t0);b1=smooth(lo1,hi1,t1)
                mesh.quad(world(s0,a0,floor_y(s0,a0)),
                          world(floor_end,a1,floor_y(floor_end,a1)),
                          world(floor_end,b1,floor_y(floor_end,b1)),
                          world(s0,b0,floor_y(s0,b0)),21,
                          ((a0,s0*.84),(a1,floor_end*.84),(b1,floor_end*.84),(b0,s0*.84)),
                          normals=(earth_normal(s0,a0),earth_normal(floor_end,a1),
                                   earth_normal(floor_end,b1),earth_normal(s0,b0)))
        for side in (-1,1):
            w0,w1=wall_width(s0,side),wall_width(s1,side)
            h0,h1=wall_height(s0,side),wall_height(s1,side)
            # Open the side wall where an actual T junction branches off.
            crossing=nearest_junction((s0+s1)*.5)
            cut=(crossing is not None and abs((s0+s1)*.5-crossing[0])<2.0
                 and side in junction_sides(crossing[1]))
            def layers(s,h):
                # Non-parallel geological strata; vertical segmentation permits
                # rough excavation faces instead of flat corridor slabs.
                return [floor_y(s,0.)-.06,h*(.20+.035*math.sin(s*.053+side)),
                        h*(.49+.040*math.sin(s*.073-side)),
                        h*(.77+.028*math.sin(s*.045+side)),h]
            a,b=layers(s0,h0),layers(s1,h1)
            if not cut:
                mats=surface_material((s0+s1)*.5,side,0)
                for band,mat in enumerate(mats):
                    for j in range(5):
                        t0=j/5.; t1=(j+1)/5.
                        y00=smooth(a[band],a[band+1],t0)
                        y01=smooth(a[band],a[band+1],t1)
                        y10=smooth(b[band],b[band+1],t0)
                        y11=smooth(b[band],b[band+1],t1)
                        # Real uneven cross section, different at each level.
                        p0=world(s0,side*wall_offset(s0,side,y00,h0),y00)
                        p1=world(s0,side*wall_offset(s0,side,y01,h0),y01)
                        p2=world(s1,side*wall_offset(s1,side,y11,h1),y11)
                        p3=world(s1,side*wall_offset(s1,side,y10,h1),y10)
                        mesh.quad(p0,p1,p2,p3,20,
                                  ((s0*1.12,y00*1.2),(s0*1.12,y01*1.2),
                                   (s1*1.12,y11*1.2),(s1*1.12,y10*1.2)),
                                  normals=(earth_normal(s0,y00,side),earth_normal(s0,y01,side),
                                           earth_normal(s1,y11,side),earth_normal(s1,y10,side)))
            # Close the subcell cutout seam with a narrow rounded earth shoulder.
            # Same top anchor as the wall, overlapping the actual exterior grid.
            # Never bridge a branch mouth or covered tunnel footprint.
            if not cut and max(tunnel_open(s0),tunnel_open(s1))<.01:
                def shoulder(ss,t):
                    h=wall_height(ss,side)
                    lip=wall_offset(ss,side,h,h)
                    reach=.58+.09*math.sin(ss*.71+side)
                    x,z=sides(ss,side*(lip+reach*t))
                    outer=_exterior_y_world(x,z,segments,edge_height_cache)
                    y=smooth(h,outer-.025,eased(t))
                    return (x,y,z)
                for j in range(4):
                    ta,tb=j/4.,(j+1)/4.
                    pts=(shoulder(s0,ta),shoulder(s1,ta),shoulder(s1,tb),shoulder(s0,tb))
                    mesh.quad(*pts,20,tuple((p[0],p[2]) for p in pts))
            # Outdoor terrain is no longer swept through bends along route
            # offsets: see build_exterior_terrain below. The near-bank earth
            # and the underground roof shell stay generated as before.

        # Sparse exposed sleeper remnants, not evenly spaced rail ties.
        if s0==int(s0) and int(s0)%9==0 and noise(int(s0)//9,990)>.35 and tunnel_open(s0)<.25:
            lo,hi=world(s0,-min(.91,wall_width(s0,-1)-.1),floor_y(s0,0)+.026),world(s0,min(.91,wall_width(s0,1)-.1),floor_y(s0,0)+.026)
            lo2,hi2=world(s0+.16,-min(.91,wall_width(s0+.16,-1)-.1),floor_y(s0+.16,0)+.026),world(s0+.16,min(.91,wall_width(s0+.16,1)-.1),floor_y(s0+.16,0)+.026)
            mesh.quad(lo,hi,hi2,lo2,2,((0,0),(2.1,0),(2.1,.18),(0,.18)))
    # Longitudinal covered passages: real roof, descending then climbing dirt
    # floor. Portal edges connect to the bank lip rather than floating mid-air.
    for k in range(int(start),int(start+CHUNK_LEN)):
        a,b=float(k),float(k+1)
        if min(tunnel_open(a),tunnel_open(b))<.01:continue
        ha=(wall_height(a,-1),wall_height(a,1))
        hb=(wall_height(b,-1),wall_height(b,1))
        wa=(wall_width(a,-1)+.13,wall_width(a,1)+.13)
        wb=(wall_width(b,-1)+.13,wall_width(b,1)+.13)
        roof_v_a=roof_v_b=0.0
        for j in range(12):
            ra=-1+2*j/12;rb=-1+2*(j+1)/12
            def roof_shoulders(ss,heights):
                # Guaranteed headroom from the actual (potentially ramped)
                # floor, INCLUDING the shoulders near the wall. The old
                # tunnel directly reused bank heights, which can fall to 1m
                # and clip a standing player's head in later tunnel cycles.
                minimum=floor_y(ss,0.)+2.16
                exposure=tunnel_open(ss)
                return tuple(smooth(h,max(h,minimum),exposure) for h in heights)
            def arch(ss,rr,heights):
                h0,h1=roof_shoulders(ss,heights)
                edge=smooth(h0,h1,(rr+1)*.5)
                # Different tunnel cycles have different arch proportions;
                # smooth variation cannot create seams between 80m chunks.
                cycle=int(max(0.,ss-280.)//504.)
                wobble=(.07+.09*noise(cycle,14231))*math.sin(ss*.39+cycle*1.7)
                arch_rise=.34+.24*noise(cycle,14232)+wobble
                crown=max(h0,h1)+arch_rise
                curve=.60+.22*noise(cycle,14233)
                return edge+(crown-edge)*pow(max(0.,1-rr*rr),curve)
            def ap(ss,rr,ww,heights,extra=0.):
                half=ww[0] if rr<0 else ww[1]
                return world(ss,half*rr,arch(ss,rr,heights)+extra)
            p0,p1,p2,p3=ap(a,ra,wa,ha),ap(b,ra,wb,hb),ap(b,rb,wb,hb),ap(a,rb,wa,ha)
            next_a=roof_v_a+math.dist(p0,p3)*1.12
            next_b=roof_v_b+math.dist(p1,p2)*1.12
            def roof_normal(ss,rr):
                def rp(t,r):
                    return ap(t,r,(wall_width(t,-1)+.13,wall_width(t,1)+.13),
                              (wall_height(t,-1),wall_height(t,1)))
                e=.02
                return surface_normal(rp(ss-e,rr),rp(ss+e,rr),
                                      rp(ss,max(-1.,rr-e)),rp(ss,min(1.,rr+e)),(0.,-1.,0.))
            mesh.quad(p0,p1,p2,p3,20,
                      ((a*1.12,roof_v_a),(b*1.12,roof_v_b),
                       (b*1.12,next_b),(a*1.12,next_a)),
                      normals=(roof_normal(a,ra),roof_normal(b,ra),
                               roof_normal(b,rb),roof_normal(a,rb)))
            roof_v_a,roof_v_b=next_a,next_b
            # Top of tunnel: continuous thin overburden skin, not ordinary
            # outdoor terrain crossing THROUGH the interior arch.
            mesh.quad(ap(a,ra,wa,ha,.24),ap(a,rb,wa,ha,.24),
                      ap(b,rb,wb,hb,.24),ap(b,ra,wb,hb,.24),3,
                      ((a*.16,ra*.36),(a*.16,rb*.36),
                       (b*.16,rb*.36),(b*.16,ra*.36)))
        # Sculpt the shell into the real outdoor grid rather than ending the
        # overburden midair at a shoulder 13cm beyond the trench wall. The
        # narrow grid mask extends 85cm beyond the lip; the bank ends slightly
        # farther out, at the ACTUAL X/Z ground height (not a guessed hill).
        for side in (-1,1):
            idx=0 if side<0 else 1
            def bank_pt(ss,heights,widths,far):
                half=widths[idx]+(1.25 if far else 0.)
                x,z=sides(ss,side*half)
                if far:
                    y=_exterior_y_world(x,z,segments,edge_height_cache)-.035
                else:
                    y=roof_shoulders(ss,heights)[idx]+.24
                return (x,y,z)
            mesh.quad(bank_pt(a,ha,wa,False),bank_pt(b,hb,wb,False),
                      bank_pt(b,hb,wb,True),bank_pt(a,ha,wa,True),3,
                      ((a,0.),(b,0.),(b,1.),(a,1.)))
        # Join elevated ceiling shoulders back down to the real excavated
        # wall tops. Without these panels the higher roof would float above
        # shallow earthen banks, producing new light leaks at the sides.
        for side in (-1,1):
            idx=0 if side<0 else 1
            top_a=roof_shoulders(a,ha)[idx]
            top_b=roof_shoulders(b,hb)[idx]
            base_a=ha[idx];base_b=hb[idx]
            if max(top_a-base_a,top_b-base_b)>.001:
                across_a=side*wa[idx];across_b=side*wb[idx]
                mesh.quad(world(a,across_a,base_a),world(b,across_b,base_b),
                          world(b,across_b,top_b),world(a,across_a,top_a),20,
                          ((a,base_a),(b,base_b),(b,top_b),(a,top_a)))
        if (k-280)%504 in (27,35,43):
            m=float(k)
            foot=world(m,-(wall_width(m,-1)-.08),1.70+tunnel_drop(m))
            flame=world(m,-(wall_width(m,-1)-.26),1.66+tunnel_drop(m))
            beam_world(mesh,foot,flame,.023,10)
            oil_lantern(mesh,flame,.90)
        # V3.9.7: distinct service/ruined/damp variants per seeded tunnel.
        # Attached to existing bank, never crossing the walkable centre lane.
        sector=int(math.floor((a-280.)/504.))
        tunnel_style=int(noise(sector,14901)*3)
        if (k-280)%504 in (30,40):
            flank=1 if noise(sector,14902)>.5 else -1
            offset=flank*(wall_width(a,flank)-.19)
            if tunnel_style==0:  # prepared passage: utility rack
                local_box(mesh,a,offset,floor_y(a,0.)+.38,.68,.19,.56,14)
                local_box(mesh,a,offset,floor_y(a,0.)+.78,.58,.12,.07,15)
            elif tunnel_style==1:  # ruined: fallen boards and short wall props
                local_box(mesh,a,offset,floor_y(a,0.)+.14,.77,.21,.28,8)
                bar(mesh,world(a-.5,offset,floor_y(a,0.)+.29),
                    world(a+.4,offset,floor_y(a,0.)+.38),.040,13,6)
            else:  # wetter shelter: marked crates and a rusted tin
                local_box(mesh,a,offset,floor_y(a,0.)+.20,.51,.28,.39,16)
                local_box(mesh,a,offset,floor_y(a,0.)+.47,.46,.31,.035,10)
        # Different roof sectors keep their own reinforcement rhythm: close
        # timber frames, sparse decayed supports, or mixed makeshift ribs.
        sector=int(math.floor((a-280.)/504.))
        support_stride=(3,4,6)[int(noise(sector,14251)*3)]
        if k%support_stride==0:
            # Broken shoring frames suggest underground excavation.
            for side in (-1,1):
                h=wall_height(a,side)
                pos=world(a,side*(wall_width(a,side)-.09),floor_y(a,0))
                tip=world(a,side*(wall_width(a,side)-.09),h-.1)
                beam_world(mesh,pos,tip,.067,13 if k%8 else 14)
            for j in range(8):
                r0=-1+2*j/8.;r1=-1+2*(j+1)/8.
                def pt(r):
                    return world(a,r*(wa[0] if r<0 else wa[1]),
                                 arch(a,r,ha)-.045)
                beam_world(mesh,pt(r0),pt(r1),.053,15 if k%8 else 13)

    # Seeded T/Y/X side passages. Cave roof and banks are real 3D surfaces;
    # never draw a large rectangular room or an exposed square cutout.
    for marker in range(JUNCTION_FIRST, int(start+CHUNK_LEN)+1, JUNCTION_SPACING):
        if not start <= marker < start+CHUNK_LEN:
            continue
        branch_idx=(marker-JUNCTION_FIRST)//JUNCTION_SPACING
        x,z,h=route(marker)
        right=(math.cos(h),-math.sin(h))
        forward=(math.sin(h),math.cos(h))
        outer=float(branch_length(branch_idx))
        roofed=branch_type(branch_idx)=='dugout'
        for branch_side in junction_sides(branch_idx):
            junction_decorations(mesh,branch_idx,branch_side)
            sideheight=max(1.58,wall_height(marker,branch_side))
            near_edge=max(wall_width(marker,branch_side),
                          wall_width(marker-2,branch_side),
                          wall_width(marker+2,branch_side))-.18
            def bp(along, across, height):
                dist=near_edge+along
                shifted=across+branch_center_shift(branch_idx,along)
                return (x+branch_side*right[0]*dist+forward[0]*shifted,height,
                        z+branch_side*right[1]*dist+forward[1]*shifted)

            def hw(along):return branch_half_width(branch_idx,along)
            # Asymmetric, hand-excavated cross sections: roof and walls share
            # the same seed-stable profiles at every metre boundary.
            def shoulder(along):
                blend=eased((along-1.0)/3.0)
                return (1.55+.07*math.sin(along*.47+branch_idx)
                        +blend*.075*math.sin(along*.81+branch_idx*1.71))
            def crown(along):
                blend=eased((along-1.0)/3.0)
                base=max(2.32,min(2.62,sideheight+.60))
                return base+.05*math.sin(along*.51+branch_idx)+blend*(
                    .12*math.sin(along*.38+branch_idx*2.23)
                    +.07*math.sin(along*1.03+branch_idx*.93))
            def vaulted(along,ratio):
                # ratio:-1..+1. A broken irregular ellipse/cave arch.
                r=max(-1.0,min(1.0,ratio))
                return shoulder(along)+(crown(along)-shoulder(along))*pow(max(0.0,1.0-r*r),.62)

            for n in range(int(outer)):
                a=float(n);b=a+1.0
                wa,wb=hw(a),hw(b)
                fa=floor_y(marker,0.)+.025*math.sin(a*.79+branch_idx*.73)
                fb=floor_y(marker,0.)+.025*math.sin(b*.79+branch_idx*.73)
                mesh.quad(bp(a,-wa,fa-.004),bp(a,wa,fa-.004),
                          bp(b,wb,fb-.004),bp(b,-wb,fb-.004),21,
                          ((a*.81,-wa),(a*.81,wa),(b*.81,wb),(b*.81,-wb)))
                # Narrow, irregular walls: not a box with four flat sides.
                for flank in (-1,1):
                    ya=shoulder(a) if roofed else sideheight+.08*math.sin(a*.3+flank)
                    yb=shoulder(b) if roofed else sideheight+.08*math.sin(b*.3+flank)
                    for j in range(4):
                        t0=j/4.;t1=(j+1)/4.
                        aa=wa+(.035*math.sin(a*.9+j*1.4+flank))*t0
                        ab=wa+(.035*math.sin(a*.9+(j+1)*1.4+flank))*t1
                        ba=wb+(.035*math.sin(b*.9+j*1.4+flank))*t0
                        bb=wb+(.035*math.sin(b*.9+(j+1)*1.4+flank))*t1
                        mesh.quad(bp(a,flank*aa,fa-.05+(ya-fa+.05)*t0),
                                  bp(a,flank*ab,fa-.05+(ya-fa+.05)*t1),
                                  bp(b,flank*bb,fb-.05+(yb-fb+.05)*t1),
                                  bp(b,flank*ba,fb-.05+(yb-fb+.05)*t0),20,
                                  ((a,t0*ya),(a,t1*ya),(b,t1*yb),(b,t0*yb)))
                if roofed and n==5:
                    # One enclosed, wall-mounted lantern and a tiny shelf.
                    # Same location is passed to GPU lighting via nearby_lanterns.
                    anchor=bp(a,-.78,1.68)
                    bracket=bp(a,-.78,1.88)
                    beam_world(mesh,anchor,bracket,.018,10)
                    oil_lantern(mesh,bp(a,-.78,1.72),.78)
                    box(mesh,bp(a,-.78,1.52),(.19,.06,.21),8)
                if roofed:
                    # Curved vaulted roof: 10 subdivisions, with uneven timber
                    # beams to suggest hand-excavated dugouts.
                    for j in range(10):
                        ra=-1.+2*j/10.; rb=-1.+2*(j+1)/10.
                        # Wind inward normals toward the cave floor; an
                        # excavated ceiling should not face the bright sun.
                        mesh.quad(bp(a,wa*ra,vaulted(a,ra)),
                                  bp(b,wb*ra,vaulted(b,ra)),
                                  bp(b,wb*rb,vaulted(b,rb)),
                                  bp(a,wa*rb,vaulted(a,rb)),20,
                                  ((a,j*.24),(b,j*.24),
                                   (b,(j+1)*.24),(a,(j+1)*.24)))
                    # Earth bank *outside* the vaulted roof, covering its
                    # side openings and reconnecting to continuous terrain.
                    # The V2.9 roof was only one surface; terrain crossed
                    # through it farther down the dugout (floating earth).
                    for flank in (-1,1):
                        aw,bw=hw(a),hw(b)
                        roof_a=bp(a,flank*aw,shoulder(a))
                        roof_b=bp(b,flank*bw,shoulder(b))
                        for t0,t1 in ((0.,.5),(.5,1.)):
                            # A FULL-WIDTH soil wing at along=0 was drawn
                            # into the OPEN main trench at X junctions. Grow
                            # it smoothly after the branch portal instead.
                            wing_a=1.5*eased(a/4.0)
                            wing_b=1.5*eased(b/4.0)
                            w0=aw+t0*wing_a;w1=aw+t1*wing_a
                            v0=bw+t0*wing_b;v1=bw+t1*wing_b
                            # Soil rests on terrain, no disconnected slab.
                            # Side bank must meet the .22m outer roof skin
                            # exactly: previous +.14 produced an 8cm crack.
                            y00=smooth(shoulder(a)+.22,sideheight+.04,t0)
                            y01=smooth(shoulder(a)+.22,sideheight+.04,t1)
                            y10=smooth(shoulder(b)+.22,sideheight+.04,t0)
                            y11=smooth(shoulder(b)+.22,sideheight+.04,t1)
                            mesh.quad(bp(a,flank*w0,y00),bp(a,flank*w1,y01),
                                      bp(b,flank*v1,y11),bp(b,flank*v0,y10),3,
                                      ((a,t0),(a,t1),(b,t1),(b,t0)))
                        # Keep the portal open. Extended outer skirts in the
                        # first metres can cross the main trench itself near
                        # an X junction; let the original bank handle its lip.
                        if a>=4.0:
                            # The historical 1.5m outer earth wing STOPPED in
                            # midair (sideheight is not the real terrain height).
                            # Connect its far lip to the absolute world-grid ground
                            # outside the cutout, with another sloping earth strip.
                            shell_a=bp(a,flank*(aw+1.50),sideheight+.04)
                            shell_b=bp(b,flank*(bw+1.50),sideheight+.04)
                            def grid_attachment(along,width):
                                # In sharp X/Y junctions the terrain cutout from
                                # ANOTHER nearby bank may extend beyond 1.58m.
                                # Find the first genuine land point with a small
                                # margin instead of joining a floating earth wing
                                # to yet another masked-out part of the world.
                                extra=2.28
                                for k in range(18):
                                    extra=2.28+.45*k
                                    px,_,pz=bp(along,flank*(width+extra),0.)
                                    pad=.15
                                    patch=((px-pad,pz-pad),(px+pad,pz-pad),
                                           (px+pad,pz+pad),(px-pad,pz+pad))
                                    if not ground_patch_intersects_excavation(patch,masks):
                                        break
                                px,_,pz=bp(along,flank*(width+extra),0.)
                                py=_exterior_y_world(px,pz,segments,edge_height_cache)
                                return (px,py-.035,pz)
                            near_grid_a=grid_attachment(a,aw)
                            near_grid_b=grid_attachment(b,bw)
                            mesh.quad(shell_a,shell_b,near_grid_b,near_grid_a,3,
                                      ((a,0.),(b,0.),(b,1.),(a,1.)))
                            # Soil has VOLUME below the surface, not an unsupported
                            # suspended horizontal wing. These hidden vertical
                            # skirts close visibility from inside the side tunnel.
                            depth=min(floor_y(marker,0.)-.45,
                                      near_grid_a[1]-.40,near_grid_b[1]-.40)
                            mesh.quad(shell_a,shell_b,
                                      (shell_b[0],depth,shell_b[2]),
                                      (shell_a[0],depth,shell_a[2]),20,
                                      ((a,0.),(b,0.),(b,1.),(a,1.)))
                    # Rough outer shell above cave roof; thickness .22m.
                    for j in range(10):
                        ra=-1.+2*j/10.; rb=-1.+2*(j+1)/10.
                        mesh.quad(bp(a,wa*ra,vaulted(a,ra)+.22),
                                  bp(a,wa*rb,vaulted(a,rb)+.22),
                                  bp(b,wb*rb,vaulted(b,rb)+.22),
                                  bp(b,wb*ra,vaulted(b,ra)+.22),3,
                                  ((a,ra),(a,rb),(b,rb),(b,ra)))
                    frame_stride=(3,4,5)[int(noise(branch_idx,14271)*3)]
                    if n>=1 and (n-1)%frame_stride==0:
                        # Branch-specific structural rhythm; all posts follow
                        # the exact seed-specific vault profile, never a lid.
                        for flank in (-1,1):
                            top=bp(a,flank*wa,vaulted(a,flank))
                            bottom=bp(a,flank*(wa-.045),0.0)
                            beam_world(mesh,bottom,top,.065,13 if n%2 else 14)
                        for j in range(6):
                            ra=-1.+2*j/6.;rb=-1.+2*(j+1)/6.
                            a1=bp(a,wa*ra,vaulted(a,ra)-.045)
                            b1=bp(a,wa*rb,vaulted(a,rb)-.045)
                            beam_world(mesh,a1,b1,.05,15 if n%2 else 13)
                # A few seeded uneven floor boards inside buried sections.
                if roofed and n%3==1 and n>2:
                    width=max(.9,2*min(wa,wb)-.17)
                    ypl=floor_y(marker,0.)+.017
                    mesh.quad(bp(a,-width*.5,ypl),bp(a,width*.5,ypl),
                              bp(a+.58,width*.5,ypl),bp(a+.58,-width*.5,ypl),15,
                              ((0,0),(width,0),(width,.58),(0,.58)))
            dugout_furnishings(mesh,branch_idx,branch_side,outer)
            # The end is sealed by an arched cave face; no square end wall.
            end_width=hw(outer)
            if roofed:
                for j in range(10):
                    r0=-1.+2*j/10.;r1=-1.+2*(j+1)/10.
                    y0=vaulted(outer,r0);y1=vaulted(outer,r1)
                    mesh.quad(bp(outer,r0*end_width,-.04),
                              bp(outer,r0*end_width,y0),
                              bp(outer,r1*end_width,y1),
                              bp(outer,r1*end_width,-.04),20,
                              ((r0*end_width,0),(r0*end_width,y0),
                               (r1*end_width,y1),(r1*end_width,0)))
            else:
                mesh.quad(bp(outer,-end_width,-.04),
                          bp(outer,-end_width,sideheight),
                          bp(outer,end_width,sideheight),
                          bp(outer,end_width,-.04),20,
                          ((-end_width,0),(-end_width,sideheight),
                           (end_width,sideheight),(end_width,0)))
    # A world-space heightfield replaces the old folded 145m-long ribbons.
    # Excavation masks still protect the playable trench and the dugout roof.
    prop_ground_cache=edge_height_cache
    build_exterior_terrain(mesh,start,masks,segments)
    scenery_chunk(mesh,start,masks,segments,prop_ground_cache)
    battlefield_debris_chunk(mesh,start,masks,segments,prop_ground_cache)
    rim_vegetation_chunk(mesh,start,masks,segments,prop_ground_cache)
    environment_chunk(mesh,start)
    draw_trench_obstacles(mesh,start)
    trench_detail_props(mesh,start)
    wall_dressing_chunk(mesh,start)
    trench_floor_litter(mesh,start)
    trench_fallen_wood_chunk(mesh,start)
    return mesh.as_array()


def gait_targets(mode, speed, stride, travelled, strength):
    """Target camera micro-motion for one LEFT+RIGHT footfall pair.

    Distance-driven, small and deliberately stabilized: this describes the
    viewpoint, not a skeleton animation. No GUI/OpenGL dependency.
    """
    if stride<=0.0 or strength<=0.0:
        return 0.0,0.0,0.0
    full_cycle=max(.70,2.0*stride)
    phase=2.0*math.pi*travelled/full_cycle
    crouched=mode in (3,4)
    lateral_amp=(.0009 if crouched else .0016 if speed<1.8 else
                 .0022 if speed<2.8 else .0028)
    bob_amp=(.0015 if crouched else .0024 if speed<1.8 else .0038)
    lateral=lateral_amp*strength*math.sin(phase)
    bob=bob_amp*strength*(.55+.45*math.cos(2.0*phase))
    roll=(.0005 if crouched else .0010)*strength*math.sin(phase-.4)
    return lateral,bob,roll


def turn_heading_toward(current, desired, dt, max_rate):
    """Frame-rate-independent shortest-angle first-person head/body rotation.

    Unlike a frame-count-based step, a 400 FPS renderer cannot rotate the
    viewpoint faster than max_rate radians per real elapsed second.
    """
    if not all(math.isfinite(v) for v in (current, desired, dt, max_rate)):
        raise ValueError('turn inputs must be finite')
    dt=max(0.0, float(dt))
    rate=max(0.0, float(max_rate))
    delta=math.atan2(math.sin(desired-current), math.cos(desired-current))
    step=max(-rate*dt,min(rate*dt,delta))
    return current+step


def perspective(fov,aspect,near,far):
    f=1/math.tan(math.radians(fov)/2)
    return np.array([[f/aspect,0,0,0],[0,f,0,0],[0,0,-(far+near)/(far-near),-2*far*near/(far-near)],[0,0,-1,0]],dtype='f4')

def view_from(position,target,roll=0.):
    # OpenGL right-handed: view looks along negative Z.
    eye=np.asarray(position,dtype='f4'); focus=np.asarray(target,dtype='f4')
    f=focus-eye; f/=np.linalg.norm(f)
    up=np.array([math.sin(roll), math.cos(roll),0],dtype='f4')
    s=np.cross(f,up);s/=np.linalg.norm(s)
    u=np.cross(s,f)
    result=np.eye(4,dtype='f4');result[0,:3]=s;result[1,:3]=u;result[2,:3]=-f
    result[:3,3]= -result[:3,:3]@eye
    return result

class SoldierBrain:
    """Deterministic tactical traversal controller.

    Decisions own body travel, stance, heading, eye scan and lean.
    Rendering only interpolates the requested pose. This is tactical staging,
    not geometric visibility/raycasting or an enemy detector.
    """
    def __init__(self):
        self.state='main'
        self.phase='avancerar'
        self.visited=set()
        self.checked_bends=set()
        self.bend_index=None
        self.bend_side=0
        self.bend_stop=0.0
        self.branch_index=None
        self.branch_side=0
        self.branch_progress=0.0
        self.investigate_depth=0.0
        # Only observed evidence triggers an entry. There are currently NO
        # off-trunk enemies in EnemyDirector, so random seed is not a threat.
        # This bounded per-run memory is part of the EXISTING SoldierBrain.
        self.branch_evidence={}
        self.last_branch_decision=None
        self.routine_rooms_cleared=0  # successful physical traversal, not guesses
        # A branch with no local evidence gets ONE brief, real corner check,
        # followed by a watched pass. This is not a room-clear/search action.
        self.passing_branch=None
        self.wait=0.0
        self.inspect_duration=0.0
        self.look_offset=0.0
        self.roll=0.0
        self.peek_shift=0.0
        self.peek_forward=0.0
        self.forward_target=0.0
        self.heading_offset=0.0
        self.heading_actual=None
        self.turn_preference=0.0
        self.look_target=0.0
        self.roll_target=0.0
        self.shift_target=0.0
        self.stance=5
        self.visit_count=0
        self._last_distance=0.0
        self.corner_progress=0.0
        self.corner_length=0.0
        self.corner_cumulative=(0.0,)
        self.corner_points=()
        self.corner_outbound=True
        self.obstacle_active=None
        self.speed_factor=1.0
        self.step_lift=0.0
        self.gaze_pitch=0.0
        # Interrupt overlay, not an alternate route/nav FSM. During a blast
        # the current turn/branch is frozen; no progress/visited state is lost.
        self.blast_timer=0.0
        self.blast_level=0
        self.blast_distance=float('inf')
        self.post_blast_caution=0.0
        # Incoming-fire overlay: no reset or replacement of navigation FSM.
        self.fire_timer=0.0
        self.fire_recent=0.0
        self.fire_count=0
        self.fire_offset=0.0
        self.combat_offset=0.0
        self.combat_lane_target=0.0
        self.combat_gaze=0.0
        self.fire_side=1
        self.fire_urgency=0

    def interrupt_for_blast(self,level,distance_metres):
        """Suspend whatever navigation action is underway, without teleporting.

        A second impact in a salvo can escalate or extend the interruption.
        Never resets visited branches, corner progress, or enemy awareness.
        """
        level=int(level)
        if not 0<=level<=3:
            raise ValueError('invalid blast reaction level')
        if level==0:return False
        self.blast_level=max(self.blast_level,level)
        self.blast_distance=min(self.blast_distance,float(distance_metres))
        duration={1:.48,2:1.15,3:1.90}[level]
        self.blast_timer=max(self.blast_timer,duration)
        self.post_blast_caution=max(self.post_blast_caution,1.4)
        self.phase=('hör en explosion' if self.blast_level==1 else
                    'kastar sig ned i skydd' if self.blast_level==3 else
                    'hukande i skydd vid explosion')
        return True

    def update_contact_lane(self,dt,distance,enemy_lateral=None):
        """Body-space wall lane in a confirmed encounter, never aim assistance.

        Only this existing brain updates navigation. Soft temporal response
        avoids camera snapping; final clearance uses _main_offset().
        """
        if enemy_lateral is None or self.state!='main' or self.blast_timer>0.0:
            target=0.0
        else:
            target=tactical_cover_lane(enemy_lateral,distance,wall_width)
        # The established obstacle detour and corner path have priority over
        # optional combat framing. Never cancel collision/steering to produce
        # a prettier shot. Interleaved trench geometry remains authoritative.
        if abs(obstacle_steering(distance))>.07 or near_junction(distance,6.0):
            target=0.0
        self.combat_lane_target=target
        alpha=1.0-math.exp(-max(0.0,min(.08,dt))*1.55)
        self.combat_offset=smooth(self.combat_offset,target,alpha)
        # Human shoulder scan is coarse and small, never follows enemy XY.
        yaw_goal=(.22 if target>.01 else -.22 if target<-.01 else 0.0)
        self.combat_gaze=smooth(self.combat_gaze,yaw_goal,
            1.0-math.exp(-max(0.0,min(.08,dt))*1.7))

    def receive_fire(self, from_side=1, near_miss=True):
        """React to an ACTUAL hostile shot, never to sound-only ambience."""
        self.fire_side=1 if from_side>=0 else -1
        self.fire_timer=max(self.fire_timer,2.3)
        self.fire_recent=0.0
        self.fire_count=min(8,self.fire_count+1)
        self.fire_urgency=2 if self.fire_count>=3 else 1
        self.phase='beskjuten: söker skydd vid jordväggen'

    def update_fire_cover(self,dt,distance,enemy_alive):
        """Slide toward an actual navigable wall lane; no teleport/side walls."""
        self.fire_timer=max(0.0,self.fire_timer-dt)
        self.fire_recent+=dt
        if not enemy_alive and self.fire_timer<=0:
            self.fire_count=max(0,self.fire_count-1)
        if self.fire_timer>0.0 and enemy_alive and self.state=='main':
            desired=self.fire_side*cover_distance(distance,self.fire_side)*.83
        else:
            desired=0.0
        self.fire_offset=smooth(self.fire_offset,desired,1.0-math.exp(-dt*4.2))

    def combat_posture(self,enemy_visible):
        """Recommendation only; the existing FSM retains route ownership."""
        if self.blast_timer>0.0 or self.fire_timer<=0.0:
            return None
        if not enemy_visible:
            self.phase='beskjuten: söker fri sikt från bättre skydd'
            return 4  # Walk crouched until line of sight is useful.
        if self.fire_urgency>=2 and self.fire_recent<.65:
            self.phase='beskjuten: kastar sig ner för att undvika elden'
            return 2
        self.phase='beskjuten: knästående vid väggen, ger skottläge'
        return 1

    def reset(self):
        self.__init__()

    def _reassess_after_blast(self):
        """Reconsider unsafe observations from the CURRENT position.

        A half-finished corner is kept geometrically intact (its Bezier
        progress cannot be restarted without teleporting). Once stationary,
        a new peek is required instead of blindly trusting the old sightline.
        """
        if self.state in ('observe_settle','observe_peek','observe_assess'):
            self._start('observe_settle',.24,'kontrollerar sidogången på nytt efter nedslag')
        elif self.state in ('edge','peek','retract','assess'):
            self._start('edge',.24,'kontrollerar öppningen på nytt efter nedslag')
        elif self.state in ('bend_settle','bend_peek','bend_retract','bend_assess'):
            self._start('bend_settle',.24,'bedömer hörnet på nytt efter nedslag')
        elif self.state in ('kneel_scan','dugout_clear'):
            self._start('dugout_clear',.65,'genomsöker skyddsrummet igen efter nedslag')
        else:
            # The existing FSM recalculates nearby bends/obstacles, even if
            # the prior progress lies inside a partially completed corner.
            self.phase='omvärderar läget efter explosion'

    def report_branch_evidence(self,index,kind,confidence=1.0):
        """Record an explicit, externally observed reason to search a branch.

        This is a hook for later branch NPCs/sound localisation, NOT a seeded
        chance of imaginary hidden enemies. Main-route gunfire alone does not
        establish that a particular branch is occupied.
        """
        if kind not in ('seen_enemy','localized_sound','movement','friendly_report'):
            raise ValueError('unsupported branch evidence source')
        confidence=float(confidence)
        if not math.isfinite(confidence) or not 0.0<=confidence<=1.0:
            raise ValueError('invalid branch evidence confidence')
        index=int(index)
        if index<0:return False
        if len(self.branch_evidence)>=24 and index not in self.branch_evidence:
            oldest=next(iter(self.branch_evidence))
            self.branch_evidence.pop(oldest,None)
        prior=self.branch_evidence.get(index)
        if prior is None or confidence>prior[1]:
            self.branch_evidence[index]=(kind,confidence)
        return True

    def _should_enter_branch(self,index):
        signal=self.branch_evidence.get(index)
        return bool(signal and signal[1]>=.65)

    def _routine_room_check(self,index):
        """Occasional non-combat search; never implies an invisible enemy.

        Deterministic and sparse. Tutorial branches / early tested encounters
        remain unchanged. Calls the EXISTING real side-branch movement FSM.
        """
        return bool(index>=6 and index%6==2 and branch_type(index)=='dugout')

    def _depth_to_check(self,index):
        """Prototype estimate only: not physical visibility."""
        length=float(branch_length(index))
        if branch_type(index)=='dugout':
            # The end is hidden beneath an arch, so walking in briefly is sane.
            return min(length-2.1, 6.4 + 1.9*noise(index,1179))
        bend=abs(branch_center_shift(index,length))
        if bend < .43:
            return 0.0 if noise(index,155)<.7 else max(CORNER_DEPTH+0.6,min(length-2.,1.3+noise(index,157)*1.8))
        return max(CORNER_DEPTH+0.6,min(length-2.0,2.4+noise(index,159)*3.9))

    def _start(self,state,duration,phase):
        self.state=state
        self.wait=max(.02,float(duration))
        self.inspect_duration=self.wait
        self.phase=phase

    def _close_to_junction(self,distance):
        i=max(0,math.ceil((distance-JUNCTION_FIRST-0.00001)/JUNCTION_SPACING))
        return i,junction_marker(i)

    def _passing_glance(self,distance):
        """Keep watching the opening after a stopped corner check.

        This envelope uses distance instead of elapsed time, so the soldier
        cannot snap the head forward while accelerating past the portal.
        Lack of evidence is NOT proof that the room is empty.
        """
        if self.passing_branch is None or self.state != 'main':
            return 0.0
        marker=junction_marker(self.passing_branch)
        offset=distance-marker
        if offset < -6.5:
            return 0.0
        if offset <= -BRANCH_SAFE_OBSERVE_BACK:
            return eased((offset+6.5)/(6.5-BRANCH_SAFE_OBSERVE_BACK))
        if offset <= 0.35:
            return 1.0
        if offset < 2.50:
            return 1.0-eased((offset-0.35)/2.15)
        self.passing_branch=None
        return 0.0

    def _pose_for_state(self):
        side=self.bend_side if self.state.startswith('bend_') else self.branch_side
        # Look (radians), head tilt (radians), side lean (metres), forward lean (metres).
        targets={
            'main':(0.,0.,0.,0.),
            # Unconfirmed branch: short stationary check, then watch the
            # opening as the body continues down the main trench.
            'observe_approach':(side*.12,0.,0.,0.),
            'observe_settle':(side*.30,-side*.008,0.,0.),
            'observe_peek':(side*1.06,-side*.020,side*.055,.060),
            'observe_assess':(side*1.02,-side*.012,side*.035,.035),
            'approach':(side*.12,0.,0.,0.),
            'edge':(side*.22,-side*.013,side*.07,.035),
            'peek':(side*.89,-side*.060,side*.17,.18),
            'retract':(side*.33,-side*.015,side*.04,.025),
            'assess':(side*.19,0.,side*.02,0.),
            'branch_rest':(side*.12,0.,side*.035,0.),
            'move_to_mouth':(side*.26,0.,side*.12,0.),
            'turn_in':(0.,0.,0.,0.),
            'dash':(0.,0.,0.,0.),
            'kneel_scan':(0.,-side*.018,0.,0.),
            'explore':(0.,0.,0.,0.),
            'dugout_clear':(0.,0.,0.,0.),
            'turn_back':(0.,0.,0.,0.),
            'backtrack':(0.,0.,0.,0.),
            'turn_main':(0.,0.,0.,0.),
            'rejoin':(0.,0.,0.,0.),
            'bend_approach':(side*.10,0.,0.,0.),
            'bend_settle':(side*.15,0.,0.,0.),
            'bend_peek':(side*.66,-side*.035,side*.13,.13),
            'bend_retract':(side*.18,-side*.010,side*.035,0.),
            'bend_assess':(side*.20,0.,0.,0.),
        }
        self.look_target,self.roll_target,self.shift_target,self.forward_target=targets.get(self.state,(0.,0.,0.,0.))
        if self.state=='main' and self.passing_branch is not None:
            amount=self._passing_glance(self._last_distance)
            side=junction_side(self.passing_branch) if self.passing_branch is not None else 0
            # Look to the SIDE, not spin the entire body into the branch.
            # Tiny lean only: large shifts move the first-person eye through
            # timber/earth, even though the body's planned path is clear.
            # A side branch is roughly 90 degrees from the main trench.
            # The previous 0.52rad glance barely showed the opening at all.
            self.look_target=side*1.15*amount
            self.roll_target=-side*.010*amount
            self.shift_target=side*.010*amount
            self.forward_target=.008*amount
        if self.state=='dugout_clear':
            # Observe right/left and back to centre while kneeling. The head
            # moves; the body and walking path do not snap sideways.
            phase=1.0-self.wait/max(.1,self.inspect_duration)
            self.look_target=.33*math.sin(phase*math.tau)*(.75+.25*noise(self.branch_index,2931))
            self.roll_target=-self.look_target*.035

    def _tick_pose(self,dt):
        # Small, time-based angular and shoulder changes; never snap.
        alpha=1-math.exp(-dt*4.0)
        # Rate-limit head motion independently of framerate.  Exponential
        # smoothing alone can still produce a violent first-frame head snap.
        desired=smooth(self.look_offset,self.look_target,alpha)
        turn_step=max(0.,dt)*1.20
        self.look_offset+=max(-turn_step,min(turn_step,desired-self.look_offset))
        self.roll=smooth(self.roll,self.roll_target,alpha)
        self.peek_shift=smooth(self.peek_shift,self.shift_target,alpha)
        self.peek_forward=smooth(self.peek_forward,self.forward_target,alpha)

    def _branch_position(self,progress):
        """Branch centreline matches the excavated floor centre, including near edge."""
        marker=junction_marker(self.branch_index)
        x,z,h=route(marker)
        side=self.branch_side
        near=max(wall_width(marker,side),wall_width(marker-2,side),
                 wall_width(marker+2,side))-.18
        lateral=side*(near+progress)
        shifted=branch_center_shift(self.branch_index,progress)
        return (x+math.cos(h)*lateral+math.sin(h)*shifted,
                z-math.sin(h)*lateral+math.cos(h)*shifted)

    def _junction_cover(self, s, marker, side):
        """Smooth wall-hugging lane from approach through departure.

        The offset is also evaluated in 'main', BEFORE SoldierBrain first
        registers the junction, so the camera never jumps into cover.
        """
        if s < marker-7.0 or s > marker+4.5:
            return 0.0
        if s < marker-BRANCH_SAFE_OBSERVE_BACK:
            blend=eased((s-(marker-7.0))/(7.0-BRANCH_SAFE_OBSERVE_BACK))
        elif s <= marker+1.25:
            blend=1.0
        else:
            blend=1.0-eased((s-(marker+1.25))/3.25)
        return side*cover_distance(s,side)*blend

    def _bend_cover(self,s,index,side):
        marker=bend_marker(index)
        if s<marker-7.0 or s>marker+10.: return 0.0
        if s<marker-2.2: blend=eased((s-(marker-7.))/4.8)
        elif s<marker+5.: blend=1.0
        else: blend=1.0-eased((s-(marker+5.))/5.0)
        return side*cover_distance(s,side)*.78*blend

    def _main_offset(self,s):
        """Compose overlapping cover lanes smoothly, never jump between them.

        A Z/W bend can lie within metres of a T junction. Giving either
        lane hard precedence made the camera teleport sideways at that seam.
        """
        offset=0.0
        i,mark=self._close_to_junction(s)
        if i not in self.visited and mark-7.<=s<=mark+1.25:
            offset+=self._junction_cover(s,mark,junction_side(i))
        if self.branch_index is not None:
            previous=junction_marker(self.branch_index)
            if previous-7.<=s<=previous+4.5:
                offset+=self._junction_cover(s,previous,self.branch_side)
        base=max(0,int((s-BEND_FIRST)//BEND_SPACING))
        for k in range(max(0,base-1),base+2):
            m=bend_marker(k)
            if m-7.<=s<=m+10.:
                direction=bend_direction(k)
                if abs(direction)>.42:
                    offset+=self._bend_cover(s,k,1 if direction>0 else -1)
        # Reuse the same seeded obstacle target as the rendered geometry.
        # Obstacle sites are kept outside bend/junction manoeuvre regions.
        offset += obstacle_steering(s)
        # Combat cover is an additive, smoothly time-filtered lateral request;
        # the final clearance clamp is shared with ordinary steering.
        offset += self.fire_offset + self.combat_offset
        if abs(offset)<1.e-9:
            return 0.0
        side=1 if offset>0 else -1
        return max(-cover_distance(s,side),min(cover_distance(s,side),offset))

    def _curve_controls(self,outbound):
        """Quarter-turn joins the *same wall-side lane* used by the approach."""
        marker=junction_marker(self.branch_index)
        a=marker-CORNER_MAIN_OFFSET
        exit_s=marker+1.25
        entrance=route(a)
        start=(entrance[0]+math.cos(entrance[2])*self._main_offset(a),
               entrance[1]-math.sin(entrance[2])*self._main_offset(a))
        join=route(exit_s)
        finish=(join[0]+math.cos(join[2])*self._main_offset(exit_s),
                join[1]-math.sin(join[2])*self._main_offset(exit_s))
        # Tangents toward the opening and out of it; no centreline jump.
        guide_out=route(marker-.04)
        g_out=(guide_out[0]+math.cos(guide_out[2])*self._main_offset(marker-.04),
               guide_out[1]-math.sin(guide_out[2])*self._main_offset(marker-.04))
        guide_in=route(marker+.32)
        g_in=(guide_in[0]+math.cos(guide_in[2])*self._main_offset(marker+.32),
              guide_in[1]-math.sin(guide_in[2])*self._main_offset(marker+.32))
        branch0=self._branch_position(0.0)
        branch1=self._branch_position(CORNER_DEPTH)
        if outbound:
            return (start,g_out,branch0,branch1)
        return (branch1,branch0,g_in,finish)

    @staticmethod
    def _bezier(points,t):
        a,b,c,d=points;u=1.-t
        return (u*u*u*a[0]+3*u*u*t*b[0]+3*u*t*t*c[0]+t*t*t*d[0],
                u*u*u*a[1]+3*u*u*t*b[1]+3*u*t*t*c[1]+t*t*t*d[1])

    @staticmethod
    def _bezier_heading(points,t):
        a,b,c,d=points;u=1.-t
        dx=3*u*u*(b[0]-a[0])+6*u*t*(c[0]-b[0])+3*t*t*(d[0]-c[0])
        dz=3*u*u*(b[1]-a[1])+6*u*t*(c[1]-b[1])+3*t*t*(d[1]-c[1])
        return math.atan2(dx,dz)

    def _start_corner(self,outbound):
        self.corner_outbound=outbound
        self.corner_points=self._curve_controls(outbound)
        length=[0.]
        last=self._bezier(self.corner_points,0.)
        # Arc-length lookup: travelled metres, not a timed canned animation.
        for k in range(1,41):
            current=self._bezier(self.corner_points,k/40.)
            length.append(length[-1]+math.hypot(current[0]-last[0],current[1]-last[1]))
            last=current
        self.corner_cumulative=tuple(length)
        self.corner_length=length[-1]
        self.corner_progress=0.

    def _corner_pose(self):
        progress=min(self.corner_length,max(0.,self.corner_progress))
        arr=self.corner_cumulative
        idx=1
        while idx<len(arr)-1 and arr[idx]<progress:idx+=1
        f=(progress-arr[idx-1])/max(1.e-6,arr[idx]-arr[idx-1])
        t=((idx-1)+f)/(len(arr)-1)
        x,z=self._bezier(self.corner_points,t)
        return x,z,self._bezier_heading(self.corner_points,t)

    def choose(self,distance,dt):
        dt=max(0.,min(.08,dt))
        self._last_distance=distance
        self.speed_factor=1.0
        self.obstacle_active=None
        self.step_lift=0.0
        self.gaze_pitch=0.0
        if self.blast_timer>0.0:
            self.blast_timer=max(0.0,self.blast_timer-dt)
            self.speed_factor=0.0
            # Drop the previous peek/lean gradually; no camera snapping.
            self.look_target=0.0
            self.roll_target=0.0
            self.shift_target=0.0
            self.forward_target=0.0
            self._tick_pose(dt)
            if self.blast_timer>0.0:
                self.phase=('hör en explosion' if self.blast_level==1 else
                            'kastar sig ned i skydd' if self.blast_level==3 else
                            'hukande i skydd vid explosion')
                return 2 if self.blast_level==3 else 1 if self.blast_level==2 else 0
            self.blast_level=0
            self.blast_distance=float('inf')
            self._reassess_after_blast()
        else:
            self.post_blast_caution=max(0.,self.post_blast_caution-dt)
        # The soldier brain, not the renderer, decides how long each
        # preparation step lasts. The order is observable and testable.
        if self.state=='main':
            # Treat *every* new generated Z/W bend as unknown ground.
            # A bend only becomes safe after an actual stop/peek/retract.
            first=max(0,int((distance-BEND_FIRST)//BEND_SPACING))
            for k in (first,first+1):
                bm=bend_marker(k)
                delta=bend_direction(k)
                if (k not in self.checked_bends and abs(delta)>.82
                        and min(wall_height(bm,-1),wall_height(bm,1))>1.45
                        and bm-7.0<=distance<bm+1.0):
                    self.bend_index=k
                    self.bend_side=1 if delta>0 else -1
                    self.bend_stop=bm-2.2
                    self._start('bend_approach',.0,'bromsar inför osäkert hörn')
                    break
            if self.state=='main':
                i,marker=self._close_to_junction(distance)
            else:
                i,marker=-1,-10000
            if i not in self.visited and distance>=marker-6.5 and distance<marker+.15:
                self.visited.add(i)
                self.branch_index=i
                self.branch_side=junction_side(i)
                self.branch_progress=0.
                self.visit_count+=1
                if self._should_enter_branch(i):
                    # Preserve the full, cautious FSM for ACTUAL evidence.
                    self.passing_branch=None
                    self._start('approach',.35+noise(i,201)*.45,
                                'bromsar före misstänkt sidogång')
                else:
                    # A short stop is essential: see INTO the opening before
                    # committing to pass. No evidence means no full search.
                    self.passing_branch=None
                    self._start('observe_approach',.12,
                                'närmar sig sidogången, beredd att kika in')
        elif self.wait>0:
            self.wait=max(0.,self.wait-dt)
            if self.wait==0.:
                i=self.branch_index
                if self.state=='observe_approach':
                    if distance < junction_marker(i)-BRANCH_SAFE_OBSERVE_BACK-.01:
                        self.wait=.10  # approach advances to the actual wall edge
                    else:
                        self._start('observe_settle',.25,
                                    'stannar i skydd vid sidogångens hörn')
                elif self.state=='observe_settle':
                    self._start('observe_peek',1.85,
                                'vrider blicken, håller uppsikt och lyssnar i sidogången')
                elif self.state=='observe_peek':
                    self._start('observe_assess',.35,
                                'bedömer det han faktiskt kunnat observera')
                elif self.state=='observe_assess':
                    # Real evidence overrides routine curiosity. A seeded
                    # empty-room check is a BEHAVIOR choice, not enemy evidence.
                    actual_evidence=self._should_enter_branch(i)
                    routine=self._routine_room_check(i)
                    if actual_evidence or routine:
                        self.last_branch_decision=(i,'investigate' if actual_evidence else 'routine_search')
                        self.investigate_depth=self._depth_to_check(i)
                        if self.investigate_depth<.15:
                            self._start('branch_rest',.40,
                                        'sidogången kan kontrolleras från öppningen')
                        else:
                            self._start('move_to_mouth',.12,
                                ('lokaliserad aktivitet — undersöker sidogången'
                                 if actual_evidence else
                                 'rutinkontrollerar ett skyddsrum utan larm'))
                    else:
                        self.last_branch_decision=(i,'observe_only')
                        self.passing_branch=i
                        self.state='main'
                        self.wait=0.0
                        self.phase='fortsätter med blicken kvar i sidogången'
                elif self.state=='bend_settle':
                    self._start('bend_peek',.30+noise(self.bend_index,311)*.19,
                                'kikar försiktigt runt nytt hörn')
                elif self.state=='bend_peek':
                    self._start('bend_retract',.15+noise(self.bend_index,312)*.13,
                                'drar tillbaka huvudet')
                elif self.state=='bend_retract':
                    self._start('bend_assess',.12+noise(self.bend_index,313)*.17,
                                'bedömer passage')
                elif self.state=='bend_assess':
                    self.checked_bends.add(self.bend_index)
                    self.state='main'
                    self.phase='fortsätter försiktigt runt hörnet'
                elif self.state=='approach':
                    if distance < junction_marker(i)-BRANCH_SAFE_OBSERVE_BACK-.01:
                        self.wait=.10
                    else:
                        self._start('edge',.3+noise(i,202)*.35,'ställer sig vid väggen')
                elif self.state=='edge':
                    self._start('peek',.42+noise(i,203)*.33,'kikar snabbt runt hörnet')
                elif self.state=='peek':
                    self._start('retract',.22+noise(i,204)*.22,'drar tillbaka huvudet')
                elif self.state=='retract':
                    self._start('assess',.35+noise(i,205)*.45,'bedömer gången')
                elif self.state=='assess':
                    if not (self._should_enter_branch(i) or self._routine_room_check(i)):
                        # Observation only: no evidence of enemy in a SIDE
                        # tunnel, no reason to clear it on every encounter.
                        # Stay in protected position, breathe, then resume.
                        self.last_branch_decision=(i,'observe_only')
                        self._start('branch_rest',.55+.55*noise(i,2207),
                                    'sidogången verkar tom — lyssnar och hämtar andan')
                    else:
                        actual_evidence=self._should_enter_branch(i)
                        self.last_branch_decision=(i,'investigate' if actual_evidence else 'routine_search')
                        self.investigate_depth=self._depth_to_check(i)
                        if self.investigate_depth<.15:
                            self._start('branch_rest',.4,'sidogången redan överblickbar')
                        else:
                            self._start('move_to_mouth',.12,
                                ('misstänkt aktivitet — går in och undersöker' if actual_evidence
                                 else 'söker igenom ett rum på rutin'))
                elif self.state=='branch_rest':
                    self.state='main'
                    self.phase='fortsätter huvudgången efter avsökning'
                elif self.state=='kneel_scan':
                    self.state='explore'
                    self.phase='undersöker bortre hörnet'
                elif self.state=='dugout_clear':
                    self.state='turn_back'
                    self.phase='återvänder efter genomsökning'
                elif self.state=='rejoin':
                    self.state='main'
                    self.phase='fortsätter längs huvudgången'
        self._pose_for_state()
        self._tick_pose(dt)
        if self.state=='main':
            # Look/slow down for an actual obstruction, not random idle scans.
            nearby=obstacles_in_range(distance-1.5,distance+4.1)
            ob=min(nearby,key=lambda obj:abs(obj.s-distance)) if nearby else None
            if ob is not None and -1.5 <= ob.s-distance <= 4.1:
                self.obstacle_active=ob
                proximity=ob.s-distance
                if ob.kind in ('crate','rubble'):
                    self.speed_factor=.55 if proximity<2.1 else .78
                    self.phase='går runt '+('låda' if ob.kind=='crate' else 'ras/bråte')
                elif ob.kind=='puddle':
                    self.speed_factor=.7
                    self.phase='passerar vatten och lera'
                else:
                    self.speed_factor=.49 if proximity<1.3 else .70
                    self.phase='kliver över '+('stock' if ob.kind=='timber' else 'sandsäckar')
                    d=distance-ob.s
                    if abs(d)<.72:
                        # A single controlled rise/step, not a sudden teleport.
                        self.step_lift=.078*math.sin(math.pi*eased((d+.72)/1.44))
                    self.gaze_pitch=-.055*eased((4.1-proximity)/4.1)
                return 5  # walk, not a crouch animation for every obstruction
            bend=abs(math.atan2(math.sin(route(distance+8)[2]-route(distance)[2]),
                                math.cos(route(distance+8)[2]-route(distance)[2])))
            self.phase='avancerar försiktigt' if bend>.45 else 'avancerar'
            if self.passing_branch is not None and self._passing_glance(distance)>0.0:
                # The STOP and peek have happened already. Carry the gaze
                # through the portal; do not whip it forward immediately.
                self.speed_factor=min(self.speed_factor,.88)
                self.gaze_pitch=-.025*self._passing_glance(distance)
                self.phase='går förbi och håller uppsikt in i sidogången'
                return 6  # purposeful brisk walking after the safety check
            return 4 if bend>.45 else (5 if bend>.17 else 7)
        if self.state in ('bend_approach','bend_settle','bend_peek','bend_retract','bend_assess'):
            return 4 if self.state=='bend_approach' else 1
        if self.state in ('observe_approach','observe_settle','observe_peek','observe_assess'):
            return 5 if self.state=='observe_approach' else 0
        if self.state in ('approach','edge','peek','retract','assess','branch_rest','move_to_mouth','turn_in','turn_back','turn_main','rejoin'):
            return 4 if self.state in ('approach','move_to_mouth','turn_in','turn_main') else (1 if self.state in ('edge','peek','retract','assess') else 0)
        if self.state=='dash':return 8
        if self.state in ('kneel_scan','dugout_clear'):return 1
        if self.state=='explore':return 4
        if self.state=='backtrack':return 5
        return 0

    def advance(self,distance,movement):
        # A blast can interrupt even a partial corner traversal: keep both
        # main route distance and branch/corner progress exactly unchanged.
        if self.blast_timer>0.0:
            return distance,0.0
        # Distance on the trunk is separate from travelled side-branch distance.
        if self.state=='bend_approach':
            travel=min(movement,max(0.,self.bend_stop-distance))
            if distance+travel>=self.bend_stop-.015:
                self._start('bend_settle',.12+noise(self.bend_index,310)*.15,
                            'stannar i skydd vid hörnet')
            return distance+travel,travel
        if self.state=='main':
            i,marker=self._close_to_junction(distance)
            if i not in self.visited and distance<marker:
                movement=min(movement,max(0.,marker-BRANCH_SAFE_OBSERVE_BACK-.05-distance))
            return distance+movement,movement
        if self.state in ('approach','observe_approach'):
            marker=junction_marker(self.branch_index)
            travel=min(movement,max(0.,marker-BRANCH_SAFE_OBSERVE_BACK-distance))
            return distance+travel,travel
        if self.state=='move_to_mouth':
            marker=junction_marker(self.branch_index)
            travel=min(movement,max(0.,marker-CORNER_MAIN_OFFSET-distance))
            if distance+travel>=marker-CORNER_MAIN_OFFSET-.012:
                self._start_corner(True)
                self.state='turn_in'
                self.phase='rundar hörnet i rörelse'
            return distance+travel,travel
        if self.state=='turn_in':
            travel=min(movement,max(0.,self.corner_length-self.corner_progress))
            self.corner_progress+=travel
            if self.corner_progress>=self.corner_length-.012:
                self.corner_progress=self.corner_length
                self.branch_progress=CORNER_DEPTH
                self.state='dash'
                self.phase='tar sig snabbt vidare i sidogången'
            return distance,travel
        if self.state=='dash':
            stop=min(self.investigate_depth,max(1.9,min(3.3,self.investigate_depth*.6)))
            travel=min(movement,max(0.,stop-self.branch_progress))
            self.branch_progress+=travel
            if self.branch_progress>=stop-.015:
                self._start('kneel_scan',.75+noise(self.branch_index,206)*.8,'går ned på knä och söker av')
            return distance,travel
        if self.state=='explore':
            travel=min(movement,max(0.,self.investigate_depth-self.branch_progress))
            self.branch_progress+=travel
            if self.branch_progress>=self.investigate_depth-.015:
                if branch_type(self.branch_index)=='dugout':
                    if self.last_branch_decision==(self.branch_index,'routine_search'):
                        self.routine_rooms_cleared+=1
                    self._start('dugout_clear',1.45+noise(self.branch_index,2932)*.7,
                                'knästående söker av skyddsrummet')
                else:
                    self.state='turn_back'
                    self.wait=0.0
                    self.phase='vänder kroppen kontrollerat'
            return distance,travel
        if self.state=='turn_back':
            if self._heading_aligned(distance,goal='back'):
                self.state='backtrack'
                self.phase='går tillbaka till korsningen'
            return distance,0.
        if self.state=='backtrack':
            travel=min(movement,max(0.,self.branch_progress-CORNER_DEPTH))
            self.branch_progress-=travel
            if self.branch_progress<=CORNER_DEPTH+.015:
                self.branch_progress=CORNER_DEPTH
                self._start_corner(False)
                self.state='turn_main'
                self.phase='rundar tillbaka till huvudgången'
            return distance,travel
        if self.state=='turn_main':
            travel=min(movement,max(0.,self.corner_length-self.corner_progress))
            self.corner_progress+=travel
            if self.corner_progress>=self.corner_length-.012:
                self.corner_progress=self.corner_length
                distance=junction_marker(self.branch_index)+1.25
                self._start('rejoin',.35+noise(self.branch_index,207)*.3,
                            'kontrollerar huvudgången')
            return distance,travel
        return distance,0.

    def _desired_heading(self,distance,goal=None):
        if self.state in ('turn_in','turn_main') and self.corner_points:
            return self._corner_pose()[2]
        if goal=='main' or self.state in ('main','approach','edge','peek','retract','assess',
                                          'observe_approach','observe_settle','observe_peek','observe_assess',
                                          'move_to_mouth','rejoin',
                                          'bend_approach','bend_settle','bend_peek','bend_retract','bend_assess'):
            return route(distance+3.2)[2]
        i=self.branch_index
        main_heading=route(junction_marker(i))[2]
        direction=self.branch_side*math.pi/2
        if goal=='back' or self.state in ('turn_back','backtrack'):
            return main_heading+direction+math.pi
        return main_heading+direction

    def _heading_aligned(self,distance,goal):
        if self.heading_actual is None:return False
        target=self._desired_heading(distance,goal)
        error=math.atan2(math.sin(target-self.heading_actual),math.cos(target-self.heading_actual))
        return abs(error)<.12

    def camera_pose(self,main_distance):
        if self.state in ('turn_in','turn_main') and self.corner_points:
            return self._corner_pose()
        if self.state in ('main','approach','edge','peek','retract','assess',
                                          'observe_approach','observe_settle','observe_peek','observe_assess',
                                          'move_to_mouth','rejoin',
                          'bend_approach','bend_settle','bend_peek','bend_retract','bend_assess'):
            x,z,h=route(main_distance)
            lateral=self._main_offset(main_distance)
            return (x+math.cos(h)*lateral,
                    z-math.sin(h)*lateral,
                    self._desired_heading(main_distance))
        x,z=self._branch_position(self.branch_progress)
        return x,z,self._desired_heading(main_distance)

def build_enemy_sculpture(mesh, shape, material):
    """Build reusable *fitted* body parts once, never per frame.

    The five shapes use smooth multi-ring silhouettes rather than chains of
    round balls/boxes. Coordinate systems are compatible with part_model and
    with the metres used by the shared actor_joints pose and hit regions.
    """
    if shape=='jacket':
        # Defined y, left/right radius, front/back radius, offset from spine.
        rings=((-0.39,.205,.139,0.),(-.29,.230,.165,-.002),
               (-.09,.280,.184,.01),(.18,.330,.192,.015),
               (.29,.299,.170,-.01),(.34,.160,.107,-.014))
    elif shape=='helmet':
        rings=((-0.065,.218,.224,.008),(-.043,.238,.244,.010),
               (.006,.227,.228,0.),(.069,.194,.185,-.014),
               (.124,.126,.127,-.015),(.149,.025,.030,-.018))
    elif shape=='face':
        # Human jaw narrower below cheekbones; rear skull remains rounded.
        rings=((-0.178,.068,.086,-.019),(-.137,.096,.103,-.002),
               (-.070,.126,.117,.005),(.021,.145,.145,0.),
               (.115,.129,.126,-.008),(.164,.075,.078,-.017),
               (.179,.014,.020,-.025))
    elif shape=='limb':
        rings=((-0.50,.79,.83,0.),(-.29,.94,1.00,.018),
               (-.03,1.01,1.02,-.008),(.27,.95,.93,.008),
               (.50,.84,.81,0.))
    elif shape=='boot':
        # Shape local XY/Z; sole, raised toe cap, ankle and laced instep.
        rings=((-1.0,.86,.79,.27),(-.76,.94,1.05,.33),
               (-.43,.96,.99,.26),(.16,.82,.72,-.12),
               (.78,.71,.57,-.37),(1.0,.55,.42,-.41))
    else:
        raise ValueError('unknown enemy sculpture: '+str(shape))
    facets=16 if shape not in ('boot','limb') else 12
    def point(ring,index):
        y,rx,rz,shift=ring
        angle=math.tau*(index%facets)/facets
        return (math.cos(angle)*rx,y,shift+math.sin(angle)*rz)
    for lower,upper in zip(rings,rings[1:]):
        v0=lower[0];v1=upper[0]
        for j in range(facets):
            k=(j+1)%facets
            mesh.quad(point(lower,j),point(upper,j),point(upper,k),point(lower,k),
                      material,((j/facets,v0),(j/facets,v1),
                                (k/facets,v1),(k/facets,v0)))
    low,high=rings[0],rings[-1]
    for j in range(facets):
        k=(j+1)%facets
        mesh.quad((0,low[0],low[3]),point(low,k),point(low,j),(0,low[0],low[3]),
                  material,((.5,.5),(k/facets,0),(j/facets,0),(.5,.5)))
        mesh.quad((0,high[0],high[3]),point(high,j),point(high,k),(0,high[0],high[3]),
                  material,((.5,.5),(j/facets,1),(k/facets,1),(.5,.5)))


class EnemyRenderer:
    """GPU-resident shared meshes for all enemies: ONE mesh per shape/material.

    No per-frame OBJ loads and NO per-frame vertex buffer uploads. Each body
    part is drawn with a transform matrix using the game's existing materials
    and lighting. This is still procedural: true GLB skeletal animation is NEXT.
    """
    def __init__(self,ctx):
        self.ctx=ctx
        self.program=ctx.program(vertex_shader=ACTOR_VERT,fragment_shader=ACTOR_FRAG)
        self.meshes={}
        for shape,material in (
            ('oval',10),('oval',11),('oval',23),('oval',24),('oval',25),
            ('cube',10),('cube',24),('jacket',23),('helmet',24),
            ('face',25),('boot',10),('limb',23),
            ('cylinder',10),('cylinder',13),('cylinder',23),('cylinder',24),('cylinder',25)
        ):
            data=_rig_shape_template(shape,material)
            buffer=ctx.buffer(data.tobytes())
            vao=ctx.vertex_array(self.program,[(buffer,'3f 3f 2f 1f',
                                     'in_pos','in_normal','in_uv','in_layer')])
            self.meshes[(shape,material)]=(buffer,vao)
        self.program['materials']=0
        self.program['material_normals']=1
        self.program['sun_direction'].value=SUN_DIRECTION
        self.program['moon_direction'].value=MOON_DIRECTION
        try:self.lamp_uniform=self.program['lamp_positions']
        except KeyError:self.lamp_uniform=self.program['lamp_positions[0]']

    @staticmethod
    def part_model(pos,heading,part):
        """Model = world translation × actor orientation × bone/mesh transform."""
        c,s=math.cos(heading),math.sin(heading)
        model=np.eye(4,dtype='f4')
        model[:3,:3]=np.array(((c,0.,s),(0.,1.,0.),(-s,0.,c)),dtype='f4')
        model[:3,3]=pos
        local=np.eye(4,dtype='f4')
        if part[0] in ('cylinder','limb'):
            a=np.array(part[2],dtype='f4'); b=np.array(part[3],dtype='f4')
            direction=b-a
            length=float(np.linalg.norm(direction))
            if length<1e-6:return model
            ey=direction/length
            helper=np.array((0.,0.,1.),dtype='f4') if abs(ey[2])<.95 else np.array((1.,0.,0.),dtype='f4')
            ex=np.cross(ey,helper);ex/=max(1.e-6,float(np.linalg.norm(ex)))
            ez=np.cross(ex,ey)
            r=part[4]
            local[:3,:3]=np.column_stack((ex*r,ey*length,ez*r))
            local[:3,3]=(a+b)*.5
        else:
            scale=np.diag(np.asarray(part[3],dtype='f4'))
            if len(part)>4:
                # Float = legacy death pitch; tuple = head yaw / full local
                # orientation. OpenGL receives same rigid matrix as before.
                angles=part[4]
                pitch,yaw,roll=((float(angles),0.,0.) if isinstance(angles,(float,int))
                                else tuple(float(v) for v in angles))
                cp,sp=math.cos(pitch),math.sin(pitch)
                cy,sy=math.cos(yaw),math.sin(yaw)
                cr,sr=math.cos(roll),math.sin(roll)
                rx=np.array(((1.,0.,0.),(0.,cp,-sp),(0.,sp,cp)),dtype='f4')
                ry=np.array(((cy,0.,sy),(0.,1.,0.),(-sy,0.,cy)),dtype='f4')
                rz=np.array(((cr,-sr,0.),(sr,cr,0.),(0.,0.,1.)),dtype='f4')
                local[:3,:3]=ry@rx@rz@scale
            else:
                local[:3,:3]=scale
            local[:3,3]=part[2]
        return model@local

    def collect_instances(self,agents,position_at,heading_at):
        """Compute the same animated actor transforms for camera and shadow passes.

        NEVER move hitboxes/AI from this renderer: only share matrices to
        avoid costly per-pass Python recomputation of ~70 body/weapon parts.
        """
        instances=[]
        for agent in agents:
            pos=position_at(agent.spec)
            heading=actor_facing(agent,heading_at(agent.route_m))
            for part in enemy_parts(agent):
                instances.append(((part[0],part[1]),self.part_model(pos,heading,part)))
        return instances

    def draw(self,agents,position_at,heading_at,projection_view,eye,
             fog_strength,normal_enabled,lights,instances=None):
        prg=self.program
        prg['mvp'].write(np.asarray(projection_view,dtype='f4').T.tobytes())
        prg['camera_position'].value=eye
        prg['fog_strength'].value=fog_strength
        prg['use_pbr_normals'].value=int(normal_enabled)
        prg['lamp_count'].value=len(lights)
        if lights:
            self.lamp_uniform.write(np.asarray(lights+[(0.,0.,0.)]*(8-len(lights)),dtype='f4').tobytes())
        if instances is None:
            instances=self.collect_instances(agents,position_at,heading_at)
        for key,model in instances:
            vao=self.meshes[key][1]
            prg['model'].write(model.T.tobytes())
            vao.render(moderngl.TRIANGLES)

    def release(self):
        for vbo,vao in self.meshes.values():vao.release();vbo.release()
        self.program.release()



def _rig_shape_template(shape,material):
    """Shared CPU sculpt template for the one-time V3.9.3 GPU rig bind."""
    mesh=MeshBuilder()
    if shape=='oval':blob(mesh,(0.,0.,0.),(1.,1.,1.),material,18,9,1.4)
    elif shape=='cube':
        # Padded carriers/pouches and machined receiver edges, within the
        # ORIGINAL box bounds. Shared by live skinning and legacy fallback.
        radius=.16 if material==24 else .05
        core=1.-radius
        grid=(-1.,-core,core,1.)
        for axis in range(3):
            u=(axis+1)%3;v=(axis+2)%3
            for sign in (-1.,1.):
                for i in range(3):
                    for j in range(3):
                        points=[];normals=[];uv=[]
                        for du,dv in ((0,0),(1,0),(1,1),(0,1)):
                            q=np.zeros(3);q[axis]=sign;q[u]=grid[i+du];q[v]=grid[j+dv]
                            centre=np.clip(q,-core,core)
                            n=q-centre;n/=np.linalg.norm(n)
                            points.append(tuple(centre+radius*n));normals.append(tuple(n))
                            uv.append(((q[u]+1)*.5,(q[v]+1)*.5))
                        if sign<0:
                            points.reverse();normals.reverse();uv.reverse()
                        mesh.quad(*points,material,uv,normals=normals)
    elif shape in ('jacket','helmet','face','boot','limb'):
        build_enemy_sculpture(mesh,shape,material)
    else:bar(mesh,(0.,-.5,0.),(0.,.5,0.),1.,material,12,1.)
    data=mesh.as_array()
    if shape in ('oval','jacket','helmet','face','limb'):
        # Weld NORMALS only across the UV seam: no movement of anatomy,
        # no topology/hit-region changes and no extra tessellation.
        _,inverse=np.unique(np.round(data[:,:3],6),axis=0,return_inverse=True)
        triangles=data[:,:3].reshape(-1,3,3)
        area=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        sums=np.zeros((inverse.max()+1,3),dtype='f4')
        np.add.at(sums,inverse,np.repeat(area,3,axis=0))
        lengths=np.linalg.norm(sums,axis=1,keepdims=True)
        smooth=sums/np.maximum(lengths,1e-8)
        valid=lengths[inverse,0]>1e-8
        data[valid,3:6]=smooth[inverse[valid]]
    return data


class StartupScreen:
    """Actual OpenGL startup card, rendered before expensive texture/meshing.

    All loading labels are shown on the same GL window; this avoids a second
    window/context and works with the existing projector placement.
    """
    def __init__(self,ctx):
        self.ctx=ctx
        self.font=pygame.font.Font(None,34)
        self.small=pygame.font.Font(None,23)
        self.program=ctx.program(vertex_shader='''#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 uv;
void main(){uv=in_uv;gl_Position=vec4(in_pos,0.0,1.0);}
''',fragment_shader='''#version 330
uniform sampler2D tex;
in vec2 uv;
out vec4 color;
void main(){color=texture(tex,uv);}
''')
        verts=np.array([-1,-1,0,0, 1,-1,1,0, 1,1,1,1,
                        -1,-1,0,0, 1,1,1,1, -1,1,0,1],dtype='f4')
        self.vbo=ctx.buffer(verts.tobytes())
        self.vao=ctx.vertex_array(self.program,[(self.vbo,'2f 2f','in_pos','in_uv')])
        self.program['tex'].value=0

    def show(self,label,progress):
        # SDL event pumping prevents OS 'not responding' during CPU meshing.
        pygame.event.pump()
        ww,hh=pygame.display.get_surface().get_size()
        card=pygame.Surface((ww,hh),pygame.SRCALPHA,32)
        card.fill((16,22,25,255))
        title=self.font.render('SKYTTEGRAV  |  LADDAR',True,(220,217,194))
        card.blit(title,(max(12,(ww-title.get_width())//2),hh//2-78))
        detail=self.small.render(label,True,(194,204,191))
        card.blit(detail,(max(12,(ww-detail.get_width())//2),hh//2-23))
        left=max(22,int(ww*.14)); width=max(40,ww-left*2)
        pygame.draw.rect(card,(52,61,56),(left,hh//2+23,width,15))
        pygame.draw.rect(card,(162,166,119),(left,hh//2+23,int(width*max(0.,min(1.,progress))),15))
        texture=self.ctx.texture((ww,hh),4,pygame.image.tostring(card,'RGBA',True))
        texture.filter=(moderngl.LINEAR,moderngl.LINEAR)
        try:
            self.ctx.disable(moderngl.DEPTH_TEST)
            texture.use(location=0)
            self.vao.render(moderngl.TRIANGLES)
            pygame.display.flip()
        finally:
            texture.release()

    def release(self):
        self.vao.release();self.vbo.release();self.program.release()


def generate_chunk_array(start, seed=DEFAULT_SEED):
    """Explicit seed crosses spawn boundary; no inherited GL/world state."""
    set_world_seed(seed)
    return make_chunk(float(start))


class Trench:
    def __init__(self):
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION,3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION,3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE,24)
        pygame.display.set_mode((W,H),pygame.OPENGL|pygame.DOUBLEBUF|pygame.RESIZABLE)
        self._projector_window=None
        try:
            from pygame._sdl2.video import Window
            self._projector_window=Window.from_display_module()
            px,py=self._projector_window.position
            self._projector_window.position=(px,py-WINDOW_RAISE_PIXELS)
            print(f'Projektorfönster: flyttat {WINDOW_RAISE_PIXELS} px upp (F9/F10 justerar)')
        except Exception as exc:
            print(f'Projektorfönster: automatisk förflyttning stöds inte här ({exc}); dra manuellt')
        pygame.display.set_caption('Oändlig skyttegrav — riktig 3D / ModernGL')
        self.ctx=moderngl.create_context(require=330)
        loading=StartupScreen(self.ctx)
        loading.show('Startar OpenGL och shaders ...',.06)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.prog=self.ctx.program(vertex_shader=VERT,fragment_shader=FRAG)
        self.sky_prog=self.ctx.program(vertex_shader=SKYVERT,fragment_shader=SKYFRAG)
        self.sky_buffer=self.ctx.buffer(np.array([-1,-1,3,-1,-1,3],dtype='f4').tobytes())
        self.sky_vao=self.ctx.vertex_array(self.sky_prog,[(self.sky_buffer,'2f','xy')])
        self.sky_prog['sun_direction'].value=SUN_DIRECTION
        self.sky_prog['moon_direction'].value=MOON_DIRECTION
        self.sky_prog['tan_half_fov'].value=math.tan(math.radians(CAMERA_VERTICAL_FOV)*.5)
        loading.show('Förbereder texturer och normalmappar ...',.18)
        self.texture,self.normal_texture,self.has_pbr_normals=self.load_textures()
        loading.show('Laddar fiender, utrustning och effekter ...',.52)
        self.prog['materials']=0
        self.prog['material_normals']=1
        self.prog['use_pbr_normals']=int(self.has_pbr_normals)
        self.fog_strength=FOG_DENSITY
        self.prog['fog_strength'].value=self.fog_strength
        for prg in (self.prog,):
            prg['sun_direction'].value=SUN_DIRECTION
            prg['moon_direction'].value=MOON_DIRECTION
        self.prog['time_seconds'].value=0.0
        self.prog['lamp_count'].value=0
        # Driver/ModernGL versions disagree on array uniform member names.
        # Query the program once; never throw KeyError on the first draw call.
        try:self.lamp_uniform=self.prog['lamp_positions']
        except KeyError:self.lamp_uniform=self.prog['lamp_positions[0]']
        self.enemies=EnemyDirector(SEED)
        self.arcade=ArcadeRun(_seed=SEED)
        self.death_screen=DeathScreen(self.ctx)
        self.death_age=0.0
        self.last_player_eye=None
        self.player_visibility={}
        self.enemy_shots_on_last_update=()
        self.player_can_shoot={}
        self.visible_opportunity_seconds={}
        self.hostile_shot_counts={}
        # Sight queries cost milliseconds in dense trench geometry. Sample
        # perception at a bounded rate, but recheck WORLD LOS on each shot.
        self.perception_timer=0.0
        self.footstep_alert_timer=0.0
        # Prefer GPU skinning: ONE draw call per 16-bone soldier rather than
        # ~60 distinct body pieces per soldier per lighting pass. Keep the
        # previous renderer available if a driver rejects the skinning shaders.
        try:
            self.enemy_renderer=SkeletalEnemyRenderer(
                self.ctx,_rig_shape_template,EnemyRenderer.part_model,
                ACTOR_FRAG,SUN_DIRECTION,MOON_DIRECTION)
            print(f'V3.9.5: GPU-skelett 16 ben, tre utrustningsvarianter, '
                  f'{self.enemy_renderer.gpu_bytes/1024/1024:.1f} MiB resident')
        except Exception as exc:
            print(f'GPU-skelett kan inte startas, legacy-soldat används: '
                  f'{type(exc).__name__}: {exc}',file=sys.stderr)
            self.enemy_renderer=EnemyRenderer(self.ctx)
        tactical_path=ROOT/'assets'/'characters'/'tactical'/'soldier.glb'
        if tactical_path.is_file():
            try:
                tactical=TacticalEnemyRenderer(self.ctx,tactical_path,SUN_DIRECTION,MOON_DIRECTION)
                tactical.ground_height=self.enemy_visual_ground
                self.enemy_renderer.release()
                self.enemy_renderer=tactical
                print('LIVE: taktisk soldat aktiv (befintlig rörelse- och träffmekanik)')
            except Exception as exc:
                print(f'Taktisk soldat kunde inte laddas; reservsoldat kvar: {exc}',file=sys.stderr)
        self.pose_lab_index=-1
        self.pose_lab_dummy=EnemyAgent(EnemySpec(-999,5.0,0.0,'cover'))
        self.pose_lab_controller=PoseTransition()
        self.pose_lab_dummy.pose_controller=self.pose_lab_controller
        # Optional generic GLB experiment remains F1-only. The fitted tactical
        # live skin above uses existing visual bones; hitboxes remain in EnemyDirector.
        self.glb_renderer=None
        model_path=ROOT/'assets'/'characters'/'soldier.glb'
        if model_path.is_file():
            try:
                self.glb_renderer=GLBSkinnedRenderer(
                    self.ctx,model_path,SUN_DIRECTION,MOON_DIRECTION)
                print('V3.9.7: riggad GLB laddad för F1-testläge:',model_path)
            except Exception as exc:
                print('GLB kan inte användas; tidigare soldat kvar:',
                      type(exc).__name__,str(exc),file=sys.stderr)
        else:
            print('V3.9.7: valfri assets/characters/soldier.glb saknas; '
                  'F1 använder den aktiva soldatrenderaren.')
        # Separate renderer: depth-only draw never affects geometry, AI or hitboxes.
        # Fall back to V3.9.1 lighting on OpenGL driver/resource failure.
        self.shadow_renderer=None
        try:
            self.shadow_renderer=ShadowRenderer(self.ctx,SUN_DIRECTION,MOON_DIRECTION)
            self.shadow_renderer.setup_material(self.prog)
            self.shadow_renderer.setup_material(self.enemy_renderer.program)
            if self.glb_renderer is not None:
                self.shadow_renderer.setup_material(self.glb_renderer.program)
            print('V3.9.2: skuggkartor för sol, måne och närmaste lampa (F12 AV/PÅ)')
        except Exception as exc:
            print(f'Skuggor kunde inte startas: {type(exc).__name__}: {exc}',file=sys.stderr)
            self.prog['shadow_mode'].value=0
            self.enemy_renderer.program['shadow_mode'].value=0
        # World-only VFX: explosions never alter terrain/AI/hit resolution.
        # Renderer allocates GPU geometry ONCE and reuses it every frame.
        self.battlefield=BattlefieldEventDirector(seed=SEED)
        self.battlefield_renderer=BattlefieldEffectRenderer(self.ctx)
        self.hit_notification=''
        self.hit_notification_until=0.
        self.enemy_death_pause=0.0
        self.last_camera_viewproj=None
        self.last_enemy_frame=None
        try:pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_CROSSHAIR)
        except Exception:pass
        self.chunks={}
        # Immutable exact world-triangle ballistic occluders for rendered chunks.
        self.shot_chunk_meshes={}
        self.shot_mesh_cache=OrderedDict()  # pairs with existing bounded raw mesh cache
        self.last_world_shot_meshes=()
        self.last_shot_cover=None
        # Main-thread-only, bounded CPU mesh cache. Saves expensive rebuilds
        # when scrubbing backwards and forwards through the same regions.
        self.mesh_cache=OrderedDict()
        self.chunk_failures={}
        self.chunk_retry_after={}
        self.debug_jump_target=None
        self.debug_jump_requested_at=0.0
        self.debug_jump_stream_chunk=None
        self.debug_jump_saved_chunks=set()
        self.todo=queue.Queue(maxsize=16)
        self.ready=queue.Queue(maxsize=16)
        self.pending=set()
        self.generation=0
        self.world_seed=SEED
        self.stream_events=deque(maxlen=4096)
        self.stream_seen=set()
        self.stream_diagnostics=os.environ.get('SKYTTEGRAV_STREAM_TRACE','0')=='1'
        self.worker_stop=threading.Event()
        self.worker_current_chunk=None
        self.worker_last_ms=0.0
        # A separate spawned PROCESS prevents seconds of procedural Python
        # mesh generation from contending with rendering on the same GIL.
        # Set SKYTTEGRAV_MESH_WORKER=thread to compare/fall back at runtime.
        self.worker_uses_process=(os.environ.get('SKYTTEGRAV_MESH_WORKER','process')
                                  .strip().lower()!='thread')
        target=(self._chunk_worker_process if self.worker_uses_process
                else self._chunk_worker)
        print('Terrängbygge:', 'separat process (spawn)' if self.worker_uses_process
              else 'äldre Python-tråd')
        self.worker=threading.Thread(target=target,daemon=True,name='trench-mesh-builder')
        self.worker.start()
        self.chunk_upload_ms=0.0
        self.frame_max_ms=0.0
        self.frame_mean_ms=0.0
        # F11 diagnostics are OFF by default: no additional GPU work, only
        # phase wall-times sampled when investigating stutter on real hardware.
        self.profile_enabled=False
        self.profile_samples=[]
        self.profile_transition_ms=[]
        self.profile_steady_ms=[]
        self.profile_last_report=time.perf_counter()
        self.framing_samples=[]
        self.framing_sample_timer=0.0
        # Retain the initial *CPU* mesh (~a few MB) so R can restore a fully
        # visible starting scene without synchronous multi-second meshing.
        # Its VBO may still be evicted on long walks; the geometry is immutable.
        loading.show('Genererar skyttegraven: startområdet ...',.65)
        self.home_chunk_data=make_chunk(0.0)
        self.chunks[0]=self.chunk(0,self.home_chunk_data)
        for i in range(1,4):
            loading.show(f'Ritar kartan och laddar sektion {i+1}/4 ...',.68+i*.09)
            self.chunks[i]=self.chunk(i)
        loading.show('Klart - öppnar skyttegraven ...',1.0)
        loading.release()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.distance=0.;self.speed=0.;self.mode=5;self.auto=True
        self.brain=SoldierBrain()
        self.auto_timer=0.;self.step_distance=0.;self.look=0.;self.height=1.55
        self.gait_strength=0.0;self.last_moved=0.0
        self.gait_lateral=0.0;self.gait_vertical=0.0;self.gait_roll=0.0
        self.elapsed=0.;self.paused=False;self.fps=0.;self.ui_timer=0.
        self.demo_mode=False
        self.demo_shot_lock=threading.RLock()
        self.demo_shot_epoch=0
        self.demo_shot_ready=True
        self.demo_target_agent=None
        self.demo_pending=False
        self.demo_targets=()
        self.demo_notice=''
        self.demo_keys=set()
        self.demo_resume_frame=False
        self.demo_results=deque(maxlen=32)
        self.obstacle_camera_lift=0.0
        self.terrain_eye_base=floor_y(0.,0.)
        self.view_heading=None;self.last_dt=1/60;self.camera_roll=0.0
        self.last=time.monotonic();self.running=True
        self.frame_size=(W,H)
    def load_textures(self):
        # Texture-array layers are stable: the original first 13 retain their IDs.
        # Extra DETAIL layers have local 1K fallback images in this delta.
        base_names=['clay.png','gravel.png','loam.png','humus.png','mud.png',
                    'bark.png','foliage.png','stone.png','timber.png','canvas.png',
                    'metal.png','flame.png','wet.png']
        detail_names=['pine_bark','bark_willow','wood_planks','wood_planks_grey',
                      'mossy_rock','wood_planks_dirt','sandbag_mud']
        # Earth originals from V1.9 + six NEW optional Poly Haven originals.
        ph_overrides={0:'brown_mud',1:'roots',2:'mud_forest',3:'mud_forest',4:'brown_mud_02',
                      8:'wood_planks_dirt',
                      13:'pine_bark',14:'bark_willow',15:'wood_planks',
                      16:'wood_planks_grey',17:'mossy_rock',18:'wood_planks_dirt'}
        ph_folder=ROOT/'assets'/'polyhaven'
        detail_folder=ROOT/'assets'/'detail'
        frames=[];normals=[];have_normals=False
        loaded_color=0;loaded_normal=0;loaded_local_normals=0
        blank_normal=bytes([128,128,255])*(TEXTURE_SIZE*TEXTURE_SIZE)
        # IDs 20..22 are shader-only terrain IDs, NEVER indexed. Reserve
        # placeholder array layers before 23..25, the soldier's reusable skin.
        enemy_names={23:'uniform',24:'webbing',25:'skin'}
        for index in range(28):
            if index==27:
                fallback=detail_folder/'field_supplies_atlas.png'
            elif index in enemy_names:
                fallback=ROOT/'assets'/'enemies'/(enemy_names[index]+'.png')
            elif index>=20:
                fallback=ROOT/'assets'/base_names[0]
            else:
                fallback=(ROOT/'assets'/base_names[index] if index<13 else
                          detail_folder/(detail_names[index-13]+'.png'))
            asset=ph_overrides.get(index)
            color=ph_folder/(asset+'_diff.png') if asset else None
            normal=ph_folder/(asset+'_nor_gl.png') if asset else None
            source=color if color is not None and color.is_file() else fallback
            if color is not None and source==color:loaded_color+=1
            with Image.open(source) as im:
                frames.append(im.convert('RGB').resize((TEXTURE_SIZE,TEXTURE_SIZE),Image.BICUBIC)
                              .transpose(Image.FLIP_TOP_BOTTOM).tobytes())
            chosen_normal=None
            if normal is not None and source==color and normal.is_file():
                chosen_normal=normal
                loaded_normal+=1
            elif index in enemy_names and source==fallback:
                # Matched locally authored OpenGL normal maps for soldier cloth.
                # Zero extras are required to boot an existing V3.9.1 install.
                soldier_normal=ROOT/'assets'/'enemies'/(enemy_names[index]+'_nor_gl.png')
                if soldier_normal.is_file():
                    chosen_normal=soldier_normal
                    loaded_local_normals+=1
            elif 13<=index<20 and source==fallback:
                # Never pair a downloaded photographic albedo with an unrelated
                # procedural fallback normal map.
                local_n=detail_folder/(detail_names[index-13]+'_nor_gl.png')
                if local_n.is_file():
                    chosen_normal=local_n
                    loaded_local_normals+=1
            if chosen_normal is not None:
                with Image.open(chosen_normal) as im:
                    normals.append(im.convert('RGB').resize((TEXTURE_SIZE,TEXTURE_SIZE),Image.BICUBIC)
                                   .transpose(Image.FLIP_TOP_BOTTOM).tobytes())
                have_normals=True
            else:
                normals.append(blank_normal)
        tex=self.ctx.texture_array((TEXTURE_SIZE,TEXTURE_SIZE,len(frames)),3,data=b''.join(frames))
        ntex=self.ctx.texture_array((TEXTURE_SIZE,TEXTURE_SIZE,len(normals)),3,data=b''.join(normals))
        for item in (tex,ntex):
            item.repeat_x=True;item.repeat_y=True
            item.build_mipmaps()
            item.filter=(moderngl.LINEAR_MIPMAP_LINEAR,moderngl.LINEAR)
            try:item.anisotropy=16.0
            except Exception:pass
        print(f'Material: {TEXTURE_SIZE}px; Poly Haven {loaded_color} colors / {loaded_normal} normals; '
              f'{loaded_local_normals} bundled detail normals; 28 layers total')
        return tex,ntex,have_normals

    def _remember_mesh(self,index,data):
        """Remember only a few immutable CPU meshes, never OpenGL objects."""
        self.mesh_cache[index]=data
        self.mesh_cache.move_to_end(index)
        while len(self.mesh_cache)>MESH_CACHE_CHUNKS:
            old,_=self.mesh_cache.popitem(last=False)
            if hasattr(self,'shot_mesh_cache'):
                self.shot_mesh_cache.pop(old,None)

    def chunk(self,index,data=None):
        # Worker threads prepare collision triangles while generating geometry;
        # the GL thread NEVER has to do ~20ms CPU collision preparation in a
        # live frame. Legacy ndarray values in existing tests remain valid.
        precomputed=None
        if isinstance(data,tuple) and len(data)==2 and isinstance(data[1],WorldShotMesh):
            data,precomputed=data
        if data is None:
            data=self.mesh_cache.get(index)
            if data is None:
                data=make_chunk(index*CHUNK_LEN)
        if precomputed is None and hasattr(self,'shot_mesh_cache'):
            precomputed=self.shot_mesh_cache.get(index)
        if precomputed is None:
            precomputed=WorldShotMesh(data)
        vbo=self.ctx.buffer(data.tobytes())
        vao=self.ctx.vertex_array(self.prog,[(vbo,'3f 3f 2f 1f','in_pos','in_normal','in_uv','in_layer')])
        self._remember_mesh(index,data)
        if hasattr(self,'shot_chunk_meshes'):
            self.shot_chunk_meshes[index]=precomputed
        if hasattr(self,'shot_mesh_cache'):
            self.shot_mesh_cache[index]=precomputed
        self._stream_event('gpu_ready',index,vertices=len(data))
        return (vbo,vao,len(data))

    def _stream_event(self,phase,index,**details):
        if not getattr(self,'stream_diagnostics',False):return
        record=dict(phase=phase,index=index,time=time.perf_counter(),
            distance=getattr(self,'distance',0.),generation=self.generation,
            seed=getattr(self,'world_seed',SEED),**details)
        self.stream_events.append(record)
        import json
        print('STREAM '+json.dumps(record),flush=True)

    def _stream_frame(self,phase):
        if not getattr(self,'stream_diagnostics',False):return
        for index in sorted(self.chunks):
            key=(self.generation,index,phase,id(self.chunks[index][0]))
            if key not in self.stream_seen:
                self.stream_seen.add(key);self._stream_event(phase,index)

    def _publish_chunk_result(self,generation,index,data):
        """Never drop completed work because the bounded delivery queue is full."""
        while not self.worker_stop.is_set() and generation==self.generation:
            try:
                self.ready.put((generation,index,data),timeout=.15)
                return True
            except queue.Full:
                continue
        return False

    def _chunk_worker_process(self):
        """Build heavy Python meshes OUTSIDE the render process and its GIL.

        Keep original queued, generation-tagged delivery path. The spawned
        child cannot inherit SDL/OpenGL state as Linux fork() would do.
        The immutable ballistic mesh is then prepared here in the existing
        worker thread; only the GL upload remains on the main thread.
        """
        try:
            pool=ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'))
        except Exception as exc:
            print(f'MESH WORKER: process start failed; using thread: {exc}',file=sys.stderr)
            return self._chunk_worker()
        try:
            while not self.worker_stop.is_set():
                try:generation,index=self.todo.get(timeout=.15)
                except queue.Empty:continue
                if generation!=self.generation:continue
                start=time.perf_counter()
                self.worker_current_chunk=index
                try:
                    job_seed=getattr(self,'world_seed',SEED)
                    self._stream_event('cpu_start',index,job_seed=job_seed)
                    task=pool.submit(generate_chunk_array,index*CHUNK_LEN,job_seed)
                    data=None
                    while not self.worker_stop.is_set():
                        try:
                            data=task.result(timeout=.15)
                            break
                        except FutureTimeout:
                            continue
                    if data is None:break
                    result=(data,WorldShotMesh(data))
                    self._stream_event('cpu_ready',index,job_generation=generation,job_seed=job_seed)
                except Exception as exc:
                    # A broken spawn environment must not leave an empty world.
                    # Retry the current job in the original worker, then keep
                    # using the thread path until the game is restarted.
                    print(f'MESH WORKER: process error for {index}: {exc}; '
                          'switching to thread fallback',file=sys.stderr)
                    if generation==self.generation:
                        try:self.todo.put_nowait((generation,index))
                        except queue.Full:self.pending.discard(index)
                    self.worker_current_chunk=None
                    return self._chunk_worker()
                finally:
                    self.worker_last_ms=(time.perf_counter()-start)*1000.0
                    self.worker_current_chunk=None
                if generation==self.generation:
                    self._publish_chunk_result(generation,index,result)
        finally:
            pool.shutdown(wait=False,cancel_futures=True)

    def _chunk_worker(self):
        while not self.worker_stop.is_set():
            try: generation,index=self.todo.get(timeout=.15)
            except queue.Empty:continue
            if generation!=self.generation:
                continue
            try:
                with WORLD_BUILD_LOCK:
                    if generation!=self.generation:continue
                    job_seed=SEED
                    self._stream_event('cpu_start',index,job_seed=job_seed)
                    data=make_chunk(index*CHUNK_LEN)
                self._stream_event('cpu_ready',index,job_generation=generation,job_seed=job_seed)
                # CPU-only; no OpenGL access in worker. Mesh tuple and precise
                # ballistic soup are passed atomically to the render thread.
                data=(data,WorldShotMesh(data))
            except Exception as exc:
                print(f'Chunk {index} mesh generation failed: {exc}',file=sys.stderr)
                data=None
            if generation==self.generation:
                self._publish_chunk_result(generation,index,data)

    def _invalidate_chunk_generation(self):
        """Cancel queued jobs, not GPU geometry; workers never touch OpenGL."""
        self.generation+=1
        self.pending.clear()
        self.chunk_failures.clear()
        self.chunk_retry_after.clear()
        for work in (self.todo,self.ready):
            while True:
                try:work.get_nowait()
                except queue.Empty:break

    def _jump_required_chunks(self,route_metres):
        core=max(0,int(math.floor(route_metres/CHUNK_LEN)))
        return {i for i in (core-1,core,core+1) if i>=0}

    def _jump_ready_progress(self):
        target=self.debug_jump_target
        if target is None:return (0,0)
        required=self._jump_required_chunks(target)
        return (len(required.intersection(self.chunks)),len(required))

    def _deliver_ready_chunks(self,wanted):
        """Perform at most one GPU upload each frame; preserve generation IDs."""
        uploaded=0
        for _ in range(UPLOADS_PER_FRAME):
            try: gen,index,data=self.ready.get_nowait()
            except queue.Empty:break
            if gen!=self.generation:continue
            self.pending.discard(index)
            if data is None:
                self.chunk_failures[index]=self.chunk_failures.get(index,0)+1
                if self.chunk_failures[index]>=JUMP_MESH_RETRIES:
                    self.chunk_retry_after[index]=time.monotonic()+CHUNK_FAILURE_BACKOFF_SECONDS
                    print(f'CHUNK ERROR: {index} failed repeatedly; holding world and retrying after cooldown',file=sys.stderr)
                continue
            if index not in wanted or index in self.chunks:
                continue
            start=time.perf_counter()
            self.chunks[index]=self.chunk(index,data)
            self.chunk_upload_ms=(time.perf_counter()-start)*1000
            uploaded+=1
        return uploaded

    def _load_cached_chunks(self,wanted,priority,uploaded):
        """Fast GPU-only cache restore under the same per-frame upload budget."""
        for index in priority:
            if uploaded>=UPLOADS_PER_FRAME:break
            if index in wanted and index not in self.chunks and index not in self.pending:
                data=self.mesh_cache.get(index)
                if data is not None:
                    start=time.perf_counter()
                    self.chunks[index]=self.chunk(index,data)
                    self.chunk_upload_ms=(time.perf_counter()-start)*1000
                    uploaded+=1
        return uploaded

    def _schedule_missing_chunks(self,priority):
        for i in priority:
            if i in self.chunks or i in self.pending or i in self.mesh_cache:
                continue
            if self.chunk_failures.get(i,0)>=JUMP_MESH_RETRIES:
                if time.monotonic()<self.chunk_retry_after.get(i,float('inf')):
                    continue
                self.chunk_failures.pop(i,None)
                self.chunk_retry_after.pop(i,None)
                print(f'CHUNK RETRY: bygg om sektion {i} efter timeout')
            try:self.todo.put_nowait((self.generation,i))
            except queue.Full:break
            self.pending.add(i)
            self._stream_event('requested',i)

    def _evict_chunks_except(self,wanted):
        for i in list(self.chunks):
            if i not in wanted:
                vbo,vao,_=self.chunks.pop(i)
                if hasattr(self,'shot_chunk_meshes'):
                    self.shot_chunk_meshes.pop(i,None)
                vao.release();vbo.release()

    def _finish_debug_jump(self,target):
        """Only change camera/AI after the landing's real 3D meshes are ready."""
        self.distance=target
        self.speed=0.0
        self.step_distance=target
        self.last_moved=0.0
        self.gait_strength=0.0
        self.gait_lateral=self.gait_vertical=self.gait_roll=0.0
        self.camera_roll=0.0
        self.height=1.55
        self.view_heading=None
        self.look=0.0
        self.obstacle_camera_lift=0.0
        self.terrain_eye_base=floor_y(target,0.0)
        self.brain.reset()
        self.enemies.reset()
        self.arcade.reset()  # clear previous combat before reproducible debug test
        self.death_age=0.0
        self.last_player_eye=None
        self.player_visibility.clear()
        self.player_can_shoot.clear()
        self.visible_opportunity_seconds.clear()
        self.hostile_shot_counts.clear()
        self.perception_timer=0.0
        self.footstep_alert_timer=0.0
        self.enemy_death_pause=0.0
        self.last_enemy_frame=None
        self.last_camera_viewproj=None
        self.auto=True
        self.mode=5
        self.paused=False
        self.last_dt=0.0
        self.debug_jump_target=None
        self.debug_jump_stream_chunk=None
        self.debug_jump_saved_chunks.clear()
        print(f'DEBUG: landat på {target:.1f} m med färdig terräng (samma seed)')

    def keep_chunks(self):
        if getattr(self,'demo_pending',False):return
        """Normal prefetch or atomic debug landing; never display an empty jump."""
        target=self.debug_jump_target
        if target is not None:
            # Autorepeated keys adjust the queued destination without moving
            # the camera or repeatedly invalidating background mesh work.
            required=self._jump_required_chunks(target)
            if time.monotonic()-self.debug_jump_requested_at<JUMP_KEY_SETTLE_SECONDS:
                saved=self.debug_jump_saved_chunks | required
                self._deliver_ready_chunks(saved)
                return
            core=int(target//CHUNK_LEN)
            if required.issubset(self.chunks):
                self._finish_debug_jump(target)
            else:
                if self.debug_jump_stream_chunk!=core:
                    # One new generation per settled landing CHUNK, not key.
                    self._invalidate_chunk_generation()
                    self.debug_jump_stream_chunk=core
                    print(f'DEBUG: förbereder landning {target:.0f} m, chunk {core}')
                wanted=self.debug_jump_saved_chunks | required
                priority=[core,core+1]+([core-1] if core>0 else [])
                # Preserve old visible world until landing is complete.
                self._evict_chunks_except(wanted)
                uploaded=self._deliver_ready_chunks(wanted)
                self._load_cached_chunks(wanted,priority,uploaded)
                if required.issubset(self.chunks):
                    self._finish_debug_jump(target)
                else:
                    self._schedule_missing_chunks(priority)
                    return

        current=max(0,math.floor(self.distance/CHUNK_LEN))
        wanted=set(range(max(0,current-RETAIN_BEHIND),current+PREFETCH_AHEAD+1))
        priority=sorted(wanted,key=lambda x:(abs(x-current),x))
        uploaded=self._deliver_ready_chunks(wanted)
        self._evict_chunks_except(wanted)
        self._load_cached_chunks(wanted,priority,uploaded)
        self._schedule_missing_chunks(priority)

    def debug_jump(self,delta):
        """Queue an atomic debug scrub; keys can be autorepeated to any range.

        The camera and game progress remain fixed until real geometry for the
        entire landing neighbourhood has arrived. Only debug controls use this.
        """
        base=self.distance if self.debug_jump_target is None else self.debug_jump_target
        target=max(0.0,base+float(delta))
        if abs(target-base)<1e-6:return target
        if self.debug_jump_target is None:
            self.debug_jump_saved_chunks=set(self.chunks)
        self.debug_jump_target=target
        self.debug_jump_requested_at=time.monotonic()
        print(f'DEBUG: begärt hopp till {target:.0f} m; väntar på terräng')
        return target

    def enemy_position(self,spec):
        if spec.exterior_side != 0:
            director=getattr(self,'enemies',None)
            agent=director.active.get(spec.enemy_id) if director is not None else None
            route_m=agent.route_m if agent is not None else spec.route_m
            # Same hill function used for the actual exterior terrain mesh.
            # A small additional clearance avoids sinking into the rim and
            # leaves the actor above the bank, never inside the trench.
            side=spec.exterior_side
            offset=side*(wall_width(route_m,side)+spec.exterior_clearance)
            return world_on_ground(route_m,offset)
        if spec.branch_index is not None:
            # Real branch-floor pose, also used for frozen hit spheres.
            # Side tunnels are geometry, not fake screen offsets.
            index=spec.branch_index
            depth=spec.branch_depth
            floor=floor_y(junction_marker(index),0.) + .025*math.sin(
                  depth*.79+index*.73)
            return branch_point(index,spec.branch_side,depth,0.,floor)
        # Render, effects and shot-frozen hit centres share the SAME dynamic
        # route position. No visual/physical position divergence when running.
        director=getattr(self,'enemies',None)
        agent=director.active.get(spec.enemy_id) if director is not None else None
        route_m=(agent.route_m if agent is not None else spec.route_m)
        lateral=spec.lateral+(agent.lateral_offset if agent is not None else 0.0)
        if agent is not None and not agent.dead:
            # Navigation belongs to the deterministic route/obstacle system,
            # not to animation root motion. Follow camera steering, then allow
            # for a soldier's BODY width rather than only the camera eye.
            lateral+=obstacle_steering(route_m)
            for ob in obstacles_in_range(route_m-3.2,route_m+3.2):
                if (ob.kind not in ('crate','rubble') or not ob.detour or
                    (agent is not None and ob.index==agent.cover_obstacle_id)):
                    continue
                # Standing human shoulder/pack half-width ~0.32m. An actor
                # must clear the actual rendered obstacle, not pass its centre.
                safe=ob.width*.5+.34
                sign=1 if ob.detour>ob.lateral else -1
                desired=ob.lateral+sign*safe
                lo=-(wall_width(route_m,-1)-.36)
                hi=wall_width(route_m,1)-.36
                if not lo<=desired<=hi:
                    alternate=ob.lateral-sign*safe
                    desired=alternate if lo<=alternate<=hi else max(lo,min(hi,desired))
                travel=abs(route_m-ob.s)
                outer=ob.length*.5+2.3
                inner=ob.length*.5+.18
                influence=eased((outer-travel)/max(.01,outer-inner))
                lateral+=(desired-lateral)*influence
        # All tactical displacements are bounded by the SAME corridor geometry.
        side=1 if lateral>=0 else -1
        limit=max(.0,wall_width(route_m,side)-.36)
        lateral=max(-limit,min(limit,lateral))
        x,z=sides(route_m,lateral)
        return (x,floor_y(route_m,lateral),z)

    def enemy_visual_ground(self,agent,x,z):
        """Floor samples for corpse presentation only; never changes enemy_position."""
        spec=agent.spec
        if spec.branch_index is not None:
            origin=np.asarray(branch_point(spec.branch_index,spec.branch_side,0.,0.,0.))
            tip=np.asarray(branch_point(spec.branch_index,spec.branch_side,1.,0.,0.))
            axis=(tip-origin)[[0,2]];axis/=max(1e-8,np.linalg.norm(axis))
            depth=float(np.dot(np.array((x,z))-origin[[0,2]],axis))
            return floor_y(junction_marker(spec.branch_index),0.)+.025*math.sin(depth*.79+spec.branch_index*.73)
        # Reuse the world's route projection and height functions locally.
        metres=np.arange(math.floor(agent.route_m)-4,math.floor(agent.route_m)+5,dtype=float)
        coords=np.asarray([route(float(t))[:2] for t in metres])
        delta=coords[1:]-coords[:-1]
        segments=(metres[:-1],coords[:-1],delta,np.maximum(1e-7,np.sum(delta*delta,axis=1)))
        ss,lateral,_=_nearest_terrain_route(x,z,segments)
        return (hill_y(ss,spec.exterior_side,lateral) if spec.exterior_side else floor_y(ss,lateral))

    def _enemy_shot_occluded(self,agent,t,origin,direction,route_at_shot=None,
                             enemy_route_at_shot=None,meshes_at_shot=None):
        """Conservative 2D trench-wall/obstruction proxy for V3.3.

        This is NOT final triangle-exact LOS. It prevents shooting through
        dirt around a corner; true world-mesh depth picking is NEXT.
        """
        if t<=0:return True
        # V3.10: player bullets use actual PRESENTED world triangles. We do
        # not allow the older coarse corridor cylinder to override a visible
        # opening in the excavated bank. Foliage / hanging cloth do not block.
        meshes=(getattr(self,'last_world_shot_meshes',()) if meshes_at_shot is None
                else meshes_at_shot)
        if meshes:
            obstruction=trace_world(origin,direction,t,meshes)
            if obstruction is not None:
                self.last_shot_cover=cover_name(obstruction.layer)
                return True
            return False
        # A branch enemy cannot use the main-route fallback: that proxy is
        # not valid inside actual side excavation. Better refuse the shot.
        if agent.spec.branch_index is not None:return True
        # Conservative fallback only if there is no loaded collision mesh.
        # Geometry along the main corridor: use closest local path point.
        start=max(0.,(self.distance if route_at_shot is None else route_at_shot)-3.0)
        frozen_enemy_route=(agent.route_m if enemy_route_at_shot is None
                            else enemy_route_at_shot)
        end=frozen_enemy_route+3.0
        route_points=[(sides(float(s),0.),float(s))
                      for s in np.arange(start,end+.51,.5)]
        for k in range(1,max(1,int(t/.30))):
            q=k*.30
            if q>t-.20:break
            x=origin[0]+direction[0]*q
            y=origin[1]+direction[1]*q
            z=origin[2]+direction[2]*q
            # Keep closest segment; do not use a world-Z assumption on bends.
            best_dist=1e20;best_s=None;best_across=0.
            for j in range(len(route_points)-1):
                (ax,az),sa=route_points[j]
                (bx,bz),sb=route_points[j+1]
                dx,dz=bx-ax,bz-az
                lam=max(0.,min(1.,((x-ax)*dx+(z-az)*dz)/max(1e-10,dx*dx+dz*dz)))
                cx,cz=ax+dx*lam,az+dz*lam
                d=math.hypot(x-cx,z-cz)
                if d<best_dist:
                    best_dist=d;best_s=sa+(sb-sa)*lam
                    best_across=(x-cx)*math.cos(route(best_s)[2])-(z-cz)*math.sin(route(best_s)[2])
            if best_s is None:continue
            side=-1 if best_across<0 else 1
            # Cylinder camera and enemy head may be close to walls, allow 7cm.
            if best_dist>wall_width(best_s,side)+.07 and y < wall_height(best_s,side)-.02:
                return True
        # Main-route box obstructions: ray may not pass straight through a crate.
        for ob in obstacles_in_range((self.distance if route_at_shot is None else route_at_shot)-1.,frozen_enemy_route+1.):
            if ob.kind not in ('crate','rubble'):continue
            # Conservative broad proxy, bounded to the obstacle's height.
            for k in range(1,max(1,int(t/.20))):
                q=k*.20
                if q>t-.20:break
                x=origin[0]+direction[0]*q
                y=origin[1]+direction[1]*q
                z=origin[2]+direction[2]*q
                cx,cz=sides(ob.s,ob.lateral)
                if math.hypot(x-cx,z-cz)<ob.width*.47 and floor_y(ob.s,ob.lateral)<y<floor_y(ob.s,ob.lateral)+ob.height:
                    return True
        return False

    def submit_game_hit(self,x,y,source='mouse_simulated',frame=None):
        # Serialize mode/reset transitions with callback dispatch. Check before
        # waiting: a trigger received during MARKERA must never wait for resume.
        epoch=getattr(self,'demo_shot_epoch',0)
        if getattr(self,'demo_pending',False) or not getattr(self,'demo_shot_ready',True):return None
        with self._manual_lock():
            if epoch!=getattr(self,'demo_shot_epoch',0):return None
            return self._submit_game_hit(x,y,source,frame)

    def _submit_game_hit(self,x,y,source='mouse_simulated',frame=None):
        if getattr(self,'demo_mode',False):
            self.begin_demo_shot()
            return None
        if not getattr(self,'demo_shot_ready',True):return None
        """Render-time game pixels -> frozen 3D intersection -> game event.

        Skjutbanan will supply hit.game_x/game_y from its verified physical
        pipeline and an associated shot-frozen presentation frame. It MUST
        NOT call this before physical verification or map camera pixels here.
        Standalone mouse/T keys are marked as simulated input only.
        """
        if self.debug_jump_target is not None:
            return None  # never fire against a frame from the OLD world
        if getattr(self,'pose_lab_index',-1)>=0:return None  # F1 diagnostic is not gameplay
        frame=self.last_enemy_frame if frame is None else frame
        if frame is None or not self.arcade.alive:return None
        actor_routes=dict(frame.actor_routes)
        self.last_shot_cover=None
        result=self.enemies.resolve_frozen_hit(float(x),float(y),frame,
            lambda agent,t,origin,direction:self._enemy_shot_occluded(
                agent,t,origin,direction,route_at_shot=frame.route_m,
                enemy_route_at_shot=actor_routes.get(agent.spec.enemy_id),
                meshes_at_shot=getattr(frame,'world_shot_meshes',None)))
        # Any player gunfire can be heard by main-route patrols, regardless
        # of hit/miss. Hearing never fakes a physical hit or eliminates anyone.
        self.arcade.player_shot(result)
        # A nearby ALREADY ENGAGED opponent can seek a real mesh-backed
        # crate. Hearing player fire does not reveal any hidden enemy to us.
        cover_signal=getattr(self.enemies,'react_to_player_fire',None)
        reactions=(cover_signal(self.distance,obstacles_in_range,
                   fallen_enemy_id=result[0] if result else None)
                   if callable(cover_signal) else ())
        if reactions:print(f'FIENDER SÖKER SKYDD: {reactions}')
        heard=self.enemies.hear_shot(self.distance)
        if heard:print(f'FIENDER HÖR SKOTT: {heard}')
        if result:
            self.hit_notification=f'TRÄFF: FIENDE {result[0]} – {result[1]} (ELIMINERAD)'
            self.enemy_death_pause=max(self.enemy_death_pause,1.25)
            print(f'{source}: {self.hit_notification}')
        else:
            # A non-lethal round can still strike the REAL rendered wall,
            # timber, stone or sandbag. Never fabricate an impact near the
            # crosshair; require an actual ray/triangle intersection first.
            impact=None
            meshes=getattr(frame,'world_shot_meshes',None)
            if meshes is None:meshes=getattr(self,'last_world_shot_meshes',())
            if meshes:
                origin,direction=ray_from_screen(float(x),float(y),
                    frame.width,frame.height,frame.view_projection)
                impact=trace_world(origin,direction,min(90.0,FAR),meshes)
                if impact is not None:
                    label=cover_name(impact.layer)
                    point=tuple(origin[i]+direction[i]*impact.distance for i in range(3))
                    splash='wood' if label=='TRÄ' else 'stone' if label in ('STEN','METALL') else 'earth'
                    self.battlefield.bullet_splash(point,material=splash)
                    self.last_shot_cover=label
            if self.last_shot_cover:
                self.hit_notification=f'SKOTT I {self.last_shot_cover}'
                print(f'{source}: {self.hit_notification}')
            else:
                self.hit_notification='MISS'
                print(f'{source}: MISS – ingen synlig 3D-träffzon träffades')
        self.hit_notification_until=time.monotonic()+2.2
        return result

    def submit_verified_hit_event(self,hit,frame):
        """Future Skjutbanan adapter. The caller owns PHYSICAL verification.

        hit is the canonical HitEvent with game_x/game_y (viewport-local).
        The supplied frame MUST be the frozen snapshot from the SHOT instant.
        This prototype deliberately does not wire itself into Skjutbanan yet.
        """
        if getattr(hit,'is_simulated',False):
            raise ValueError('Simulated HitEvent is not a verified physical shot')
        return self.submit_game_hit(hit.game_x,hit.game_y,
                                    source='physical_verified',frame=frame)

    def _manual_lock(self):
        return self.__dict__.setdefault('demo_shot_lock',threading.RLock())

    def _manual_mouse_gate(self,blocked):
        # SDL rejects pending mouse triggers at ingress, including Event.post.
        # Already-fetched events are discarded by handle's existing batch guard.
        method=getattr(getattr(pygame,'event',None),'set_blocked' if blocked else 'set_allowed',None)
        if method is not None:method(pygame.MOUSEBUTTONDOWN)

    def _manual_cancel(self):
        """Shutdown/death cancels an unscored mark; never invent a verdict."""
        with self._manual_lock():
            self.demo_shot_epoch=getattr(self,'demo_shot_epoch',0)+1
            self.demo_shot_ready=False
            self.demo_pending=False
            self.demo_targets=()
            self.demo_target_agent=None
            self.demo_presented_frame=None
            self._manual_mouse_gate(True)

    def _manual_valid_ids(self,frame,eye=None):
        if frame is None:return ()
        return tuple(sorted({idx for idx,_ in frame.actor_routes
            if idx in self.enemies.active and not self.enemies.active[idx].dead
            and (enemy_shown_to_player_mesh(frame,idx,eye)
                 if eye is not None and getattr(frame,'world_shot_meshes',None)
                 else self.player_can_shoot.get(idx,False))}))

    def set_manual_mode(self,enabled):
        with self._manual_lock():
            if self.demo_pending:return False
            preferred=self._manual_valid_ids(self.last_enemy_frame,getattr(self,'last_player_eye',None))
            self.enemies.set_manual_presentation(enabled,self.distance,preferred)
            self.demo_mode=bool(enabled)
            self.demo_shot_epoch=getattr(self,'demo_shot_epoch',0)+1
            self.demo_shot_ready=False
            self.demo_notice=''
            self.demo_target_agent=None
            # Do not accept a shot against the old multi-target presentation.
            self._manual_mouse_gate(True)
            return True

    def _manual_frame_presented(self):
        # Called only after flip: callback targets always come from a presented
        # frame, never the new frame being assembled by the renderer.
        self.demo_presented_frame=self.last_enemy_frame
        self.demo_presented_eye=getattr(self,'last_player_eye',None)
        if not self.demo_pending:
            self.demo_shot_ready=True
            self._manual_mouse_gate(False)

    def begin_demo_shot(self):
        """One atomic trigger. Arrivals during MARKERA are dropped, not queued."""
        epoch=getattr(self,'demo_shot_epoch',0)
        if self.demo_pending or not getattr(self,'demo_shot_ready',True):return False
        frame=getattr(self,'demo_presented_frame',self.last_enemy_frame)
        eye=getattr(self,'demo_presented_eye',getattr(self,'last_player_eye',None))
        with self._manual_lock():
            if (epoch!=getattr(self,'demo_shot_epoch',0) or self.demo_pending
                or not getattr(self,'demo_shot_ready',True) or self.paused
                or not self.demo_mode or not self.running or not self.arcade.alive or self.debug_jump_target is not None
                or self.pose_lab_index>=0):return False
            self.demo_targets=self._manual_valid_ids(frame,eye)
            self.demo_target_agent=(self.enemies.active[self.demo_targets[0]]
                                    if len(self.demo_targets)==1 else None)
            self.demo_pending=True
            self.demo_shot_epoch=epoch+1
            self.demo_notice=('Mål '+str(self.demo_targets[0]) if len(self.demo_targets)==1
                              else 'Inget entydigt mål: SPACE avvisas; M avslutar som bom')
            self._manual_mouse_gate(True)
            self.last_dt=0.
            return True

    def decide_demo_shot(self,hit):
        with self._manual_lock():return self._decide_demo_shot(hit)

    def _decide_demo_shot(self,hit):
        """Explicit operator result, with no fabricated coordinates or hit region."""
        if not self.demo_pending:return False
        result=None
        if hit:
            if len(self.demo_targets)!=1:
                self.demo_notice='TRÄFF AVVISAD: inget entydigt mål. M = BOM'
                return False
            idx=self.demo_targets[0];agent=self.enemies.active.get(idx)
            if agent is None or agent.dead or agent is not getattr(self,'demo_target_agent',None):
                self.demo_notice='Målet saknas. M = BOM'
                return False
            if not agent.eliminate('manual_demo'):return False
            self.enemies.eliminated.add(idx)
            result=(idx,'manual_demo')
            self.enemies.last_hit=result
            self.enemy_death_pause=max(self.enemy_death_pause,1.25)
        self.arcade.player_shot(result)
        react=getattr(self.enemies,'react_to_player_fire',None)
        if callable(react):react(self.distance,obstacles_in_range,
                                fallen_enemy_id=result[0] if result else None)
        self.enemies.hear_shot(self.distance)
        self.demo_results.append({'source':'manual_demo','verdict':'hit' if hit else 'miss',
                                  'target_id':result[0] if result else None})
        self.hit_notification='MANUELL TRÄFF' if hit else 'MANUELL BOM'
        self.hit_notification_until=time.monotonic()+2.2
        self.last_shot_cover=None
        self.demo_shot_epoch=getattr(self,'demo_shot_epoch',0)+1
        self.demo_shot_ready=False
        self.demo_target_agent=None
        self.demo_pending=False
        self.demo_notice='';self.demo_targets=()
        self.demo_resume_frame=True;self.last_dt=0.
        self.last=time.monotonic()
        return True

    def restart_world(self,new_seed=None):
        with self._manual_lock():return self._restart_world(new_seed)

    def _restart_world(self,new_seed=None):
        """Existing full restart, with optional explicit seed and world-cache reset."""
        global SEED
        if getattr(self,'demo_pending',False):
            self.demo_notice='Bedöm skottet med SPACE eller M före omstart.'
            return False
        self.debug_jump_target=None
        self.debug_jump_stream_chunk=None
        self.debug_jump_saved_chunks.clear()
        # A full deterministic restart: cancel old prefetch jobs,
        # reset first-person pose, and make the start geometry visible.
        self._invalidate_chunk_generation()
        if new_seed is not None and int(new_seed)!=SEED:
            with WORLD_BUILD_LOCK:
                set_world_seed(new_seed)
                self.world_seed=SEED
            self._evict_chunks_except(set())
            self.mesh_cache.clear();self.shot_mesh_cache.clear()
            self.shot_chunk_meshes.clear();self.last_world_shot_meshes=()
            self.last_shot_cover=None
            if self.shadow_renderer is not None:
                for _,vao in self.shadow_renderer.world_vaos.values():vao.release()
                self.shadow_renderer.world_vaos.clear()
            for name in ('_alive_palette','_fallen'):
                cache=getattr(self.enemy_renderer,name,None)
                if cache is not None:cache.clear()
            self.battlefield.seed=SEED
            self.arcade._seed=SEED
            loading=StartupScreen(self.ctx)
            try:
                for index in range(4):
                    loading.show(f'Ny bana, seed {SEED}: sektion {index+1}/4',index/4.)
                    data=make_chunk(index*CHUNK_LEN)
                    if index==0:self.home_chunk_data=data
                    self.chunks[index]=self.chunk(index,data)
            finally:
                loading.release();self.ctx.enable(moderngl.DEPTH_TEST)
        self.distance=0.0
        self.elapsed=0.0
        self.step_distance=0.0
        self.speed=0.0
        self.mode=5
        self.auto=True
        self.paused=False
        self.look=0.0
        self.height=1.55
        self.gait_strength=0.0
        self.last_moved=0.0
        self.gait_lateral=0.0
        self.gait_vertical=0.0
        self.gait_roll=0.0
        self.camera_roll=0.0
        self.brain.reset()
        self.view_heading=None
        self.obstacle_camera_lift=0.0
        self.terrain_eye_base=floor_y(0.,0.)
        self.enemies.world_seed=SEED
        self.enemies.reset()
        self.arcade.reset()
        self.death_age=0.0
        self.last_player_eye=None
        self.player_visibility.clear()
        self.player_can_shoot.clear()
        self.visible_opportunity_seconds.clear()
        self.hostile_shot_counts.clear()
        self.pose_lab_index=-1
        self.perception_timer=0.0
        self.footstep_alert_timer=0.0
        self.battlefield.reset()
        self.last_camera_viewproj=None
        self.last_enemy_frame=None
        self.hit_notification=''
        self.hit_notification_until=0.0
        self.enemy_death_pause=0.0
        self.frame_max_ms=0.0
        self.frame_mean_ms=0.0
        self.chunk_upload_ms=0.0
        if 0 not in self.chunks:
            self.chunks[0]=self.chunk(0,self.home_chunk_data)
        self.demo_shot_epoch=getattr(self,'demo_shot_epoch',0)+1
        self.demo_shot_ready=False
        self.demo_target_agent=None
        self.demo_presented_frame=None
        self._manual_mouse_gate(True)
        self.demo_targets=();self.demo_notice='';self.demo_results.clear()
        self.demo_resume_frame=True
        self.last=time.monotonic()
        print(f'OMSTART: seed {SEED}')
        return True

    def handle(self):
        # Drop triggers from the entire batch containing a pending/accepted shot.
        # Nothing received during MARKERA is replayed after the decision.
        demo_batch_blocked=getattr(self,'demo_pending',False)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                self.running=False;self._manual_cancel();return
            if event.type==pygame.VIDEORESIZE:
                self.frame_size=(max(400,event.w),max(260,event.h))
            if event.type==pygame.KEYUP and hasattr(self,'demo_keys'):
                self.demo_keys.discard(event.key)
            if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
                if getattr(self,'demo_mode',False):
                    if not demo_batch_blocked:
                        demo_batch_blocked=self.begin_demo_shot()
                    continue
                if self.arcade.alive and not self.paused:
                    self.submit_game_hit(*event.pos,source='mouse_simulated')
                continue
            if event.type!=pygame.KEYDOWN:continue
            if event.key==pygame.K_ESCAPE:
                self.running=False;self._manual_cancel();return
            if event.key in (pygame.K_b,pygame.K_SPACE,pygame.K_m,pygame.K_r,pygame.K_s) and hasattr(self,'demo_keys'):
                repeated=bool(getattr(event,'repeat',False)) or event.key in self.demo_keys
                self.demo_keys.add(event.key)
                if (getattr(self,'demo_mode',False) or event.key in (pygame.K_b,pygame.K_r,pygame.K_s)) and repeated:continue
            if getattr(self,'demo_pending',False):
                if event.key in (pygame.K_r,pygame.K_s):
                    self.demo_notice='Bedöm skottet med SPACE eller M före omstart.'
                if event.key in (pygame.K_SPACE,pygame.K_m):
                    self.decide_demo_shot(event.key==pygame.K_SPACE)
                continue
            if demo_batch_blocked:continue
            if event.key==pygame.K_b:
                self.set_manual_mode(not self.demo_mode)
                demo_batch_blocked=True
                continue
            if getattr(self,'demo_mode',False) and event.key in (pygame.K_t,pygame.K_SPACE,pygame.K_m):continue
            if event.key==pygame.K_t:
                if self.arcade.alive and not self.paused:
                    ww,hh=pygame.display.get_surface().get_size()
                    self.submit_game_hit(ww//2,hh//2,source='keyboard_simulated')
            elif event.key==pygame.K_ESCAPE:self.running=False
            elif event.key in (pygame.K_F9, pygame.K_F10):
                if self._projector_window is not None:
                    x,y=self._projector_window.position
                    try:
                        self._projector_window.position=(x,y+(-12 if event.key==pygame.K_F9 else 12))
                    except Exception as exc:
                        print(f'Kunde inte flytta fönstret: {exc}')
                        self._projector_window=None
            elif event.key==getattr(pygame,'K_F1',-111):
                entering=self.pose_lab_index<0
                self.pose_lab_index+=1
                if self.pose_lab_index>=len(POSE_LAB):self.pose_lab_index=-1
                status=(POSE_LAB[self.pose_lab_index] if self.pose_lab_index>=0 else 'AV')
                if entering:
                    # A fresh lab session never starts in yesterday's final
                    # death pose. Subsequent F1 presses blend from the actual
                    # previously displayed skeleton without a visual snap.
                    self.pose_lab_controller=PoseTransition()
                    self.pose_lab_dummy.pose_controller=self.pose_lab_controller
                    if self.glb_renderer is not None:
                        self.glb_renderer.reset_preview()
                elif self.pose_lab_index>=0:
                    self.pose_lab_controller.request(status,self.pose_lab_dummy)
                print('V3.9.7 F1 SOLDATKOMMANDO:',status)
            elif event.key==getattr(pygame,'K_F12',-120):
                if self.shadow_renderer is not None:
                    self.shadow_renderer.enabled=not self.shadow_renderer.enabled
                    print('F12: skuggor', 'PÅ' if self.shadow_renderer.enabled else 'AV')
            elif event.key==pygame.K_F11:
                self.profile_enabled=not self.profile_enabled
                self.profile_samples.clear()
                self.profile_transition_ms.clear()
                self.profile_steady_ms.clear()
                self.framing_samples.clear()
                self.framing_sample_timer=0.0
                self.profile_last_report=time.perf_counter()
                print('F11: bildruteprofilering', 'PÅ' if self.profile_enabled else 'AV')
            elif event.key==pygame.K_F3:
                self.battlefield.trigger('near_shell',self.distance,world_on_ground)
                print('Battlefield: nära visuellt nedslag (F3), test av skyddsreaktion')
            elif event.key==pygame.K_F4:
                self.battlefield.trigger('flyby',self.distance,world_on_ground)
                print('Battlefield VFX: testpassage med flyg (F4)')
            elif event.key==pygame.K_F5:
                self.battlefield.trigger('barrage',self.distance,world_on_ground)
                print('Battlefield VFX: test av artillerisalva (F5)')
            elif event.key==pygame.K_F6:
                self.battlefield.trigger('shell',self.distance,world_on_ground)
                print('Battlefield VFX: test av enskilt nedslag (F6)')
            elif event.key==pygame.K_F2:
                self.battlefield.enabled=not self.battlefield.enabled
                print(f'Battlefield VFX slump-events: {"PÅ" if self.battlefield.enabled else "AV"}')
            elif event.key==pygame.K_F7:
                self.fog_strength=max(0.,self.fog_strength-.2)
                print(f'Dimma: {self.fog_strength:.1f} (F7 mindre, F8 mer)')
            elif event.key==pygame.K_F8:
                self.fog_strength=min(3.,self.fog_strength+.2)
                print(f'Dimma: {self.fog_strength:.1f} (F7 mindre, F8 mer)')
            elif event.key==pygame.K_UP:self.debug_jump(25.0)
            elif event.key==pygame.K_DOWN:self.debug_jump(-25.0)
            elif event.key in (pygame.K_SPACE,pygame.K_p):self.paused=not self.paused
            elif event.key in (pygame.K_r,pygame.K_s):
                seed=None
                if event.key==pygame.K_s:
                    seed=SEED
                    while seed==SEED:seed=random.SystemRandom().randrange(1,2**31)
                self.restart_world(seed)
                demo_batch_blocked=True
            elif event.key==pygame.K_0:self.auto=True
            elif pygame.K_1<=event.key<=pygame.K_9:
                self.auto=False;self.mode=event.key-pygame.K_1
    def update(self,dt):
        if getattr(self,'demo_mode',False):
            with self._manual_lock():return self._update_game(dt)
        return self._update_game(dt)

    def _update_game(self,dt):
        if getattr(self,'demo_pending',False) and not self.arcade.alive:
            self._manual_cancel()
        if getattr(self,'demo_pending',False) or getattr(self,'demo_resume_frame',False):
            self.demo_resume_frame=False
            self.last_dt=0.
            return
        if self.debug_jump_target is not None:
            # A staging jump freezes gameplay, hit timers and virtual movement.
            # The old terrain stays rendered while its replacement is built.
            self.last_dt=0.0
            return
        # Camera has historically used the __init__ default (1/60) forever.
        # At 400 FPS this made heading/lean converge far too quickly, while
        # speed/position used real dt. Keep visual and navigation clocks equal.
        self.last_dt=0.0 if self.paused else max(0.0,min(.055,float(dt)))
        if getattr(self,'pose_lab_index',-1)>=0:
            self.elapsed+=dt
            self.pose_lab_dummy.total_time+=dt
            self.pose_lab_dummy.state_time+=dt
            self.pose_lab_controller.advance(dt)
            return
        if self.paused:return
        if not self.arcade.alive:
            self.death_age+=dt
            return
        self.elapsed+=dt
        keys=pygame.key.get_pressed()
        self.look+=(float(keys[pygame.K_d])-float(keys[pygame.K_a]))*dt*.75
        self.look*=math.exp(-1.7*dt)
        # Compute sight from the virtual first-person BODY, not the projector.
        # Reuse last presented eye position; next rendered frame refreshes it.
        eye=(self.last_player_eye if self.last_player_eye is not None else
             self._initial_player_eye())
        body=player_body(eye,self.height)
        geo=dict(route=route,sides=sides,wall_width=wall_width,
                 wall_height=wall_height,floor_y=floor_y,
                 obstacles_in_range=obstacles_in_range)
        self.perception_timer-=dt
        perception_due=self.perception_timer<=0.0
        if perception_due:
            # The CPU ray tests run around 15 times/second, not once per
            # 300–400 FPS render. AI pose/aim continues updating every frame.
            self.perception_timer=.065
            self.player_visibility.clear()
            self.player_can_shoot.clear()
        # The opportunity comes from the LAST PRESENTED frame, never an
        # imagined future pose. Remove expired IDs on each perception tick.
        if perception_due:
            active_ids={idx for idx,a in self.enemies.active.items() if not a.dead}
            for idx in list(self.visible_opportunity_seconds):
                if idx not in active_ids:self.visible_opportunity_seconds.pop(idx,None)
            for idx in active_ids:
                actor=self.enemies.active.get(idx)
                if actor is not None and off_trunk(actor.spec):
                    shown=enemy_shown_to_player_mesh(self.last_enemy_frame,idx,eye)
                    self.player_can_shoot[idx]=shown
                    if shown and self.auto and actor.spec.branch_index is not None:
                        # Only a side-room sentry contributes room evidence.
                        self.brain.report_branch_evidence(
                            actor.spec.branch_index,'seen_enemy',1.0)
                else:
                    self.player_can_shoot[idx]=enemy_shown_to_player(
                        self.last_enemy_frame,idx,eye,geometry=geo)
        for idx,shown in self.player_can_shoot.items():
            prev=self.visible_opportunity_seconds.get(idx,0.0)
            self.visible_opportunity_seconds[idx]=(
                min(8.0,prev+dt) if shown else max(0.0,prev-dt*.65))
        def visibility(agent):
            if agent.spec.enemy_id not in self.player_visibility:
                px,ground,pz=self.enemy_position(agent.spec)
                origin=(px,ground+1.48,pz)
                visible=(mesh_clear_body(origin,body,self.last_world_shot_meshes)
                         if off_trunk(agent.spec) else
                         can_see_body(origin,body,agent.route_m,self.distance,**geo))
                self.player_visibility[agent.spec.enemy_id]=visible
            if not self.player_visibility[agent.spec.enemy_id]:
                return False
            # Initial discovery requires the soldier to face the player.
            # An already-engaged actor retains its contact while turning.
            if agent.state in ('idle','investigating','alert'):
                pos=self.enemy_position(agent.spec)
                facing=actor_facing(agent,route(agent.route_m)[2])
                return enemy_faces_player(pos,eye,facing)
            return True
        self.enemies.update(dt,self.distance,can_see_player=visibility,
                            branch_side_at=junction_side)
        for enemy_id in self.enemies.gunshots_this_frame:
            enemy=self.enemies.active.get(enemy_id)
            if enemy is None or enemy.dead:continue
            px,ground,pz=self.enemy_position(enemy.spec)
            # Never allow a sampled perception cache to authorize lethal fire.
            # Always re-evaluate geometry at the actual hostile-shot event.
            fresh_eye=self.enemy_position(enemy.spec)
            current_body=player_body(eye,self.height)
            from_enemy=(fresh_eye[0],fresh_eye[1]+1.48,fresh_eye[2])
            shots_visible=(mesh_clear_body(from_enemy,current_body,
                          self.last_world_shot_meshes)
                          if off_trunk(enemy.spec) else
                          can_see_body(from_enemy,current_body,
                                       enemy.route_m,self.distance,**geo))
            clear=bool(shots_visible)
            self.player_visibility[enemy_id]=shots_visible
            cfg=difficulty_at(self.distance)
            count=self.hostile_shot_counts.get(enemy_id,0)
            self.hostile_shot_counts[enemy_id]=count+1
            # Lethal contact requires BOTH world LOS and a real prior on-screen
            # opportunity, sufficient response time AND warning rounds.
            fair=(self.player_can_shoot.get(enemy_id,False) and
                  self.visible_opportunity_seconds.get(enemy_id,0.0)>=cfg.reaction_grace and
                  count>=cfg.warning_shots)
            died=self.arcade.enemy_shot(enemy_id,clear_sight=clear,
                distance_metres=math.hypot(px-eye[0],pz-eye[2]),
                exposure=min(1.0,len(shots_visible)/3.0),
                allow_damage=fair,progress_metres=self.distance)
            if clear and self.auto:
                # Shelter response belongs to SoldierBrain, independent of
                # whether the random hit/miss roll happened to be lethal.
                self.brain.receive_fire(from_side=-1 if enemy.spec.lateral>=0 else 1)
            if clear and not died:
                # This is a bounded visual impact CUE near a bank, NOT
                # simulated bullet penetration or persistent terrain damage.
                nearby=tuple(ob for ob in obstacles_in_range(
                    self.distance-.8,self.distance+2.6)
                    if ob.kind in ('crate','rubble','sandbags','timber'))
                if nearby:
                    ob=min(nearby,key=lambda item:abs(item.s-self.distance))
                    bx,bz=sides(ob.s,ob.lateral)
                    by=floor_y(ob.s,ob.lateral)+min(.74,max(.12,ob.height*.70))
                    splash=MATERIALS[ob.kind].splash
                else:
                    flank=-1 if (enemy_id+self.arcade.incoming_shots)%2 else 1
                    bank_offset=flank*max(.52,wall_width(self.distance,flank)-.08)
                    bx,bz=sides(self.distance+1.15,bank_offset)
                    by=floor_y(self.distance+1.15,bank_offset)+.72
                    splash='earth'
                self.battlefield.bullet_splash((bx,by,bz),material=splash)
            if died:
                self.speed=0.0
                self.hit_notification='DU ÄR DÖD'
                print('ARCADE:',self.arcade.status())
                return
        # Footsteps only reach nearby patrols. Gunshots retain a much larger
        # hearing distance. This prevents the first enemy sprinting from 32m
        # toward the player while still invisibly behind the first bank.
        self.footstep_alert_timer=max(0.0,self.footstep_alert_timer-dt)
        if self.speed>=2.1 and self.footstep_alert_timer==0.0:
            alerted=self.enemies.hear_shot(self.distance,max_range=12.0)
            if alerted:print(f'FIENDER HÖR SPRINGSTEG: {alerted}')
            self.footstep_alert_timer=1.4
        self.enemy_death_pause=max(0.0,self.enemy_death_pause-dt)
        # Battlefield geometry remains visual-only. Actual DETONATIONS emit
        # a one-frame actor cue; never trigger reaction on artillery scheduling.
        # Covered tunnels/dugouts suppress new random outdoor events.
        outside=(tunnel_drop(self.distance)>-.10 and
                 (not self.auto or self.brain.state in
                  ('main','approach','edge','peek','retract','assess',
                   'observe_approach','observe_settle','observe_peek','observe_assess',
                   'move_to_mouth','branch_rest','rejoin','bend_approach',
                   'bend_settle','bend_peek','bend_retract','bend_assess')))
        detonations=self.battlefield.update(dt,self.distance,world_on_ground,
                                            allow_events=outside)
        if detonations:
            # Use camera position in WORLD X/Z, not route distance: the soldier
            # may be exploring a side dugout while shells land outside.
            soldier_x,soldier_z,_=self.brain.camera_pose(self.distance) if self.auto else route(self.distance)
            for impact in detonations:
                bx,_,bz=impact.position
                dist=math.hypot(soldier_x-bx,soldier_z-bz)
                level=blast_reaction_level(dist,impact.strength)
                if self.auto and level:
                    self.brain.interrupt_for_blast(level,dist)
                self.enemies.react_to_blast(impact.position,impact.strength,
                                            blast_reaction_level,self.enemy_position)
        if self.auto:
            self.mode=self.brain.choose(self.distance,dt)
            threat=self.enemies.current_threat(self.distance)
            if threat is not None and self.distance>=threat.route_m-8.05:
                if self.brain.blast_timer>0.0:
                    if self.brain.blast_level==3:self.mode=2
                    elif self.brain.blast_level==2:self.mode=1
                elif self.player_can_shoot.get(threat.spec.enemy_id,False):
                    self.mode=1
                    self.brain.phase='KONTAKT: FIENDE I SYNLIGT SKOTTLÄGE'
                else:
                    # Enemies behind a bend are not a stationary kill-clock.
                    # The soldier keeps seeking a real shooting opportunity.
                    self.mode=4
                    self.brain.phase='söker bättre skjutvinkel förbi skydd'
            # Tactical flank is a real, smoothly-moving camera/body lane.
            # Enemy centre is measured in the ENEMY's own route coordinate
            # frame, even when the main trench is strongly curved.
            if threat is not None:
                ex,_,ez=self.enemy_position(threat.spec)
                cx,cz,ch=route(threat.route_m)
                enemy_side=(ex-cx)*math.cos(ch)-(ez-cz)*math.sin(ch)
            else:
                enemy_side=None
            self.brain.update_contact_lane(dt,self.distance,enemy_side)
            self.brain.update_fire_cover(dt,self.distance,threat is not None)
            response=self.brain.combat_posture(
                bool(threat and self.player_can_shoot.get(threat.spec.enemy_id,False)))
            if response is not None and threat is not None and self.brain.blast_timer<=0:
                self.mode=response
            if self.brain.post_blast_caution>0.0 and self.mode>=5:
                self.mode=4  # cautious resumption; preserves real route/nav state
            if self.enemy_death_pause>0.0 and self.brain.blast_timer<=0.0:
                self.mode=1  # let the player SEE the collapse before advancing
        name,target,height,stride=MOVES[self.mode]
        # SoldierBrain plans tempo; render loop only follows the target speed.
        if self.auto:
            target *= self.brain.speed_factor
        self.speed=smooth(self.speed,target,1-math.exp(-dt*(1.25 if target>self.speed else 2.6)))
        advanced=self.speed*dt
        if self.auto and (self.brain.blast_timer>0.0 or self.enemy_death_pause>0.0):
            advanced=0.0  # do not drift through wall/corner while crouching
        if self.auto:
            if self.brain.state=='main':
                threat=self.enemies.current_threat(self.distance)
                if threat is not None and not self.player_can_shoot.get(threat.spec.enemy_id,False):
                    advanced=min(advanced,max(0.0,threat.route_m-3.5-self.distance))
                else:
                    advanced=self.enemies.clamp_advance(self.distance,advanced)
            self.distance,moved=self.brain.advance(self.distance,advanced)
            self.step_distance+=moved
            self.last_moved=moved
        else:
            self.distance+=advanced;self.step_distance+=advanced
            self.last_moved=advanced
        self.gait_strength=smooth(self.gait_strength,
                                  min(1., self.last_moved / max(0.0001,dt) / .75),
                                  1-math.exp(-dt*(7.0 if self.last_moved>0.0001 else 5.0)))
        stance_rate=7.3 if self.auto and self.brain.blast_timer>0.0 else 3.2
        self.height=smooth(self.height,height,1-math.exp(-dt*stance_rate))
        lift=self.brain.step_lift if self.auto else 0.0
        self.obstacle_camera_lift=smooth(self.obstacle_camera_lift,lift,1-math.exp(-dt*7.5))
        self.arcade.step(dt,self.distance)

    def _initial_player_eye(self):
        px,pz,_=(self.brain.camera_pose(self.distance) if self.auto
                  else route(self.distance))
        return (px,floor_y(self.distance)+self.height,pz)

    def camera(self):
        x,z,desired=self.brain.camera_pose(self.distance) if self.auto else route(self.distance)
        # A human head cannot instantaneously reverse 180 degrees.
        if self.view_heading is None:
            self.view_heading=desired
        dt=min(.10,max(.0,self.last_dt))
        # In first person this IS the player's head. Clamp turn speed in
        # real elapsed seconds, not in artificial 60Hz steps per draw call.
        turn_rate=(1.15 if self.auto and self.brain.state=='turn_back' else
                   1.75 if self.auto else 2.25)
        self.view_heading=turn_heading_toward(self.view_heading,desired,dt,turn_rate)
        if self.auto:self.brain.heading_actual=self.view_heading
        heading=self.view_heading
        stride=MOVES[self.mode][3]
        targets=gait_targets(self.mode,self.speed,stride,self.step_distance,
                             self.gait_strength)
        gait_alpha=1.-math.exp(-dt*6.0)
        self.gait_lateral=smooth(self.gait_lateral,targets[0],gait_alpha)
        self.gait_vertical=smooth(self.gait_vertical,targets[1],gait_alpha)
        self.gait_roll=smooth(self.gait_roll,targets[2],gait_alpha)
        # Apply the current frame's stabilized lean, not the previous one.
        roll_target=(self.brain.roll if self.auto else 0.)+self.gait_roll
        self.camera_roll=smooth(self.camera_roll,roll_target,
                                1.-math.exp(-dt*4.0))
        lateral=self.gait_lateral
        bob=self.gait_vertical
        breath=(.0036+.002*min(1.,self.speed/3))*math.sin(self.elapsed*1.55)
        x+=math.cos(heading)*lateral
        z-=math.sin(heading)*lateral
        # SoldierBrain owns both the head lean and the forward/side peek.
        # All components settle smoothly when the inspection ends.
        if self.auto:
            shoulder=self.brain.peek_shift
            forward_lean=self.brain.peek_forward
            x+=math.cos(heading)*shoulder+math.sin(heading)*forward_lean
            z-=math.sin(heading)*shoulder-math.cos(heading)*forward_lean
        # Hunched movement keeps the head *low and slightly forward*, rather
        # than rocking the whole camera sideways like upright running.
        crouch=max(0.,min(1.,(1.42-self.height)/.40))
        x+=math.sin(heading)*.075*crouch
        z+=math.cos(heading)*.075*crouch
        # Head follows actual displaced floor, with low-pass motion so small
        # ruts never masquerade as abrupt stepwise footfalls. Dugouts use their
        # own near-level floor, shared with floor mesh at the entrance.
        ground_target=(floor_y(self.distance,0.) if not self.auto or
                       self.brain.state in ('main','approach','edge','peek','retract','assess',
                                            'observe_approach','observe_settle','observe_peek','observe_assess',
                                            'move_to_mouth','branch_rest','rejoin','bend_approach',
                                            'bend_settle','bend_peek','bend_retract','bend_assess')
                       else floor_y(junction_marker(self.brain.branch_index),0.)
                       if self.brain.branch_index is not None else floor_y(self.distance,0.))
        self.terrain_eye_base=smooth(self.terrain_eye_base,ground_target,
                                      1.-math.exp(-dt*5.2))
        eye=(x,self.height+bob+breath+self.obstacle_camera_lift+self.terrain_eye_base,z)
        heading=heading+self.look+(self.brain.look_offset+self.brain.combat_gaze
                                   if self.auto else 0.)
        eye_y=eye[1]
        pitch=self.brain.gaze_pitch if self.auto else 0.0
        aim=(x+math.sin(heading)*12,
             eye_y-.085-.14*crouch+math.tan(pitch)*12,
             z+math.cos(heading)*12)
        return eye,aim

    def _rig_lab_position(self,spec):
        if spec.enemy_id==-999:
            ss=self.distance+5.1
            x,z=sides(ss,0.0)
            return (x,floor_y(ss,0.),z)
        return self.enemy_position(spec)

    def draw(self):
        if getattr(self,'demo_mode',False):
            with self._manual_lock():return self._draw_game()
        return self._draw_game()

    def _draw_game(self):
        ww,hh=pygame.display.get_surface().get_size()
        self.ctx.viewport=(0,0,ww,hh)
        self.ctx.clear(.18,.23,.29,1.0,depth=1.0)
        eye,aim=self.camera()
        self.last_player_eye=eye
        # Sun and sky have no translational parallax. Only orientation matters.
        forward=np.asarray(aim,dtype='f4')-np.asarray(eye,dtype='f4')
        forward/=max(1e-8,float(np.linalg.norm(forward)))
        nominal_up=np.array((math.sin(self.camera_roll),math.cos(self.camera_roll),0.0),dtype='f4')
        right=np.cross(forward,nominal_up)
        right/=max(1e-8,float(np.linalg.norm(right)))
        up=np.cross(right,forward)
        self.sky_prog['sky_right'].value=tuple(float(v) for v in right)
        self.sky_prog['sky_up'].value=tuple(float(v) for v in up)
        self.sky_prog['sky_forward'].value=tuple(float(v) for v in forward)
        self.sky_prog['view_aspect'].value=ww/hh
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.sky_vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)
        mat=perspective(CAMERA_VERTICAL_FOV,ww/hh,.045,FAR) @ view_from(eye,aim,self.camera_roll)
        self.last_camera_viewproj=mat.copy()
        self.last_enemy_frame=self.enemies.capture_frame(
            mat,ww,hh,self.distance,self.enemy_position,lambda ss:route(ss)[2])
        # F11 only: sampled ACTUAL projected positions; diagnostic never feeds
        # aim, spawns, camera, or physical/game-space hit calculations.
        if self.profile_enabled and self.elapsed>=self.framing_sample_timer:
            self.framing_sample_timer=self.elapsed+.33
            for agent in self.enemies.active.values():
                if (agent.dead or abs(agent.route_m-self.distance)>35.0
                        or not self.player_can_shoot.get(agent.spec.enemy_id,False)):continue
                x,y,z=self.enemy_position(agent.spec)
                clip=mat@np.array([x,y+1.1,z,1.0],dtype='f4')
                if clip[3]<=.001:continue
                ndc=clip[:3]/clip[3]
                if -1<=ndc[0]<=1 and -1<=ndc[1]<=1 and -1<=ndc[2]<=1:
                    self.framing_samples.append((float(ndc[0])*.5+.5,
                                                 float(ndc[1])*.5+.5))
                    if len(self.framing_samples)>400:self.framing_samples.pop(0)
        self.frame_size=(ww,hh)
        self.prog['mvp'].write(mat.T.tobytes())
        self.prog['time_seconds'].value=float(self.elapsed)
        self.prog['camera_position'].value=eye
        self.prog['fog_strength'].value=self.fog_strength
        # Eight nearest deterministic lights; GPU only gets a bounded payload.
        emitters=nearby_lanterns(self.distance,35.0)
        self.prog['lamp_count'].value=len(emitters)
        if emitters:
            # ModernGL array members may be exposed via base name or [0].
            self.lamp_uniform.write(np.asarray(emitters+[(0.,0.,0.)]*(8-len(emitters)),dtype='f4').tobytes())
        # Draw the SAME world and animated actor pose into the light depth maps.
        # Bounded to current/adjacent chunks; F12 isolates GPU cost in F11.
        if getattr(self,'pose_lab_index',-1)>=0:
            dummy=self.pose_lab_dummy
            dummy.route_m=self.distance+5.1
            dummy.pose_preview=POSE_LAB[self.pose_lab_index]
            visible=[dummy]
            position_at=self._rig_lab_position
        else:
            visible=[a for a in self.enemies.active.values()
                     if abs(a.route_m-self.distance)<65.]
            position_at=self.enemy_position
        # Generic GLB stays preview-only; the fitted tactical skin is already
        # selected as enemy_renderer during normal startup.
        active_renderer=(self.glb_renderer if self.glb_renderer is not None
                         and self.pose_lab_index>=0 else self.enemy_renderer)
        instances=(active_renderer.collect_instances(visible,position_at,
                         lambda ss:route(ss)[2]) if visible else [])
        if self.shadow_renderer is not None:
            shadow=self.shadow_renderer
            if shadow.enabled:
                try:
                    shadow.draw_maps(self.chunks,active_renderer,instances,eye,emitters,
                        (ww,hh),max(0,math.floor(self.distance/CHUNK_LEN)),self.elapsed)
                    self._stream_frame('shadow_draw')
                except Exception as exc:
                    # Restore old lighting instead of crashing on unsupported GL.
                    print(f'Skuggor inaktiverade efter renderfel: {type(exc).__name__}: {exc}',file=sys.stderr)
                    shadow.enabled=False
            shadow.bind_material(self.prog)
            shadow.bind_material(active_renderer.program)
        self.texture.use(location=0)
        self.normal_texture.use(location=1)
        # Freeze the occluders of the frame being presented. Async streaming
        # may evict GPU chunks on later frames, but these CPU refs stay valid
        # for the image whose actor hit spheres were also frozen above.
        self.last_world_shot_meshes=tuple(self.shot_chunk_meshes[i]
            for i in sorted(self.chunks) if i in self.shot_chunk_meshes)
        # Immutable pose + immutable world references in ONE displayed-frame
        # snapshot. Delayed shot resolution may never use later streamed walls.
        self.last_enemy_frame=replace(self.last_enemy_frame,
                                      world_shot_meshes=self.last_world_shot_meshes)
        for i in sorted(self.chunks):self.chunks[i][1].render()
        self._stream_frame('main_draw')
        # All actors share GPU-resident geometry and materials; one set of
        # instance transforms per frame. World depth buffer occludes enemies.
        if visible:
            active_renderer.draw(visible,position_at,
                                     lambda ss:route(ss)[2],mat,eye,
                                     self.fog_strength,self.has_pbr_normals,
                                     emitters,instances=instances)
        # VFX on top of the real 3D depth-buffer (not baked into terrain).
        self.battlefield_renderer.draw(self.battlefield,mat,eye,right,up)
        if not self.arcade.alive and self.pose_lab_index<0:
            self.death_screen.draw(self.death_age,self.arcade,ww,hh)
        mode='MANUELL' if self.demo_mode else 'NORMAL'
        lines=((mode+f' | B = läge | S = ny bana | seed {SEED}',) if not self.demo_pending else
               ('MANUELL | MARKERA','SPACE = TRÄFF    M = BOM',self.demo_notice))
        self.death_screen.draw_status(lines,ww,hh)
        pygame.display.flip()
        self._manual_frame_presented()
    def _record_profile(self,phase_ms,transition=False):
        """Compare pose switches versus steady motion, plus mesh activity."""
        self.profile_samples.append(tuple(phase_ms))
        if transition:
            self.profile_transition_ms.append(phase_ms[-1])
        else:
            self.profile_steady_ms.append(phase_ms[-1])
        now=time.perf_counter()
        if now-self.profile_last_report<4.0:return
        batches=self.profile_samples
        if not batches:return
        labels=('events','update','chunks','draw','total')
        lines=[]
        for i,label in enumerate(labels):
            seq=sorted(row[i] for row in batches)
            lines.append(f'{label} avg={sum(seq)/len(seq):.1f} '
                         f'p95={seq[int((len(seq)-1)*.95)]:.1f} max={seq[-1]:.1f}ms')
        thread_work=getattr(self,'worker_current_chunk',None)
        worker_mode=('process' if getattr(self,'worker_uses_process',False) else 'thread')
        world_info=f'worker={worker_mode} active={thread_work} '
        world_info+=f'last_build={getattr(self,"worker_last_ms",0.0):.0f}ms '
        world_info+=f'gpu_upload={getattr(self,"chunk_upload_ms",0.0):.1f}ms'
        def percentile95(values):
            if not values:return 'n/a'
            ordered=sorted(values)
            return f'{ordered[int((len(ordered)-1)*.95)]:.1f}ms (n={len(ordered)})'
        print(f'PROFILE {self.distance:.0f}m state={self.brain.state} '
              f'mode={MOVES[self.mode][0]} | '+ ' | '.join(lines)+' | '+world_info)
        if self.framing_samples:
            counts=[0,0,0]
            for xx,yy in self.framing_samples:
                counts[min(2,int(xx*3.0))]+=1
            print('PROFILE ENEMY PROJECTED X left/middle/right: '+str(tuple(counts))
                  +' (samples='+str(len(self.framing_samples))
                  +'; LOS-filtered projection, not distinct shots or physical accuracy)')
            self.framing_samples.clear()
        print('PROFILE MOTION CPU total p95: transition='
              +percentile95(self.profile_transition_ms)
              +' steady='+percentile95(self.profile_steady_ms)
              +' | GPU frametime: test F12 skuggor AV/PÅ separat')
        self.profile_samples.clear()
        self.profile_transition_ms.clear()
        self.profile_steady_ms.clear()
        self.profile_last_report=now

    def run(self):
        try:
            while self.running:
                now=time.monotonic()
                raw_dt=max(0.00001,now-self.last)
                dt=min(.055,raw_dt)
                self.last=now
                if self.profile_enabled:
                    p0=time.perf_counter()
                    self.handle();p1=time.perf_counter()
                    motion_before=(self.brain.state,self.mode)
                    self.update(dt);p2=time.perf_counter()
                    motion_after=(self.brain.state,self.mode)
                    self.keep_chunks();p3=time.perf_counter()
                    self.draw();p4=time.perf_counter()
                    self._record_profile(((p1-p0)*1000.,(p2-p1)*1000.,
                                          (p3-p2)*1000.,(p4-p3)*1000.,
                                          (p4-p0)*1000.),
                                         transition=(motion_before!=motion_after))
                else:
                    self.handle();self.update(dt);self.keep_chunks();self.draw()
                # Never cap the DISPLAYED FPS / max-frame value to 55ms.
                # Simulation dt is capped for safety but spikes are real.
                fps=1/max(.001,raw_dt)
                self.fps=.9*self.fps+.1*fps if self.fps else fps
                ms=raw_dt*1000
                self.frame_max_ms=max(ms,self.frame_max_ms)
                self.frame_mean_ms=.95*self.frame_mean_ms+.05*ms if self.frame_mean_ms else ms
                self.ui_timer+=dt
                if self.ui_timer>.3:
                    self.ui_timer=0
                    name=MOVES[self.mode][0]
                    notice=(' | '+self.hit_notification if time.monotonic()<self.hit_notification_until else '')
                    vfx_status='PÅ' if self.battlefield.enabled else 'AV'
                    jump_note=''
                    if self.debug_jump_target is not None:
                        have,total=self._jump_ready_progress()
                        jump_note=f' | LADDAR HOPP {self.debug_jump_target:.0f} m ({have}/{total} sektioner) – väntar säkert'
                    pygame.display.set_caption(f'3D SKYTTEGRAV V3.10.3 | {self.arcade.status()} | {self.distance:06.0f} m | {self.brain.phase if self.auto else name} | {name} {self.speed:.1f} m/s | {self.fps:.0f} FPS | max {self.frame_max_ms:.0f}ms · upload {self.chunk_upload_ms:.0f}ms | F3 NÄRA F4 FLYG F5 ARTILLERI F6 GRANAT · F2 VFX {vfx_status} | ↑↓ 25m · MUSKLICK SKJUT · T SIKTE MITT · 0 AUTO{jump_note}{notice} | F1 SKELETT {POSE_LAB[self.pose_lab_index] if self.pose_lab_index>=0 else "AV"} | F12 SKUGGOR')
                pygame.time.wait(1)
        finally:
            for vbo,vao,_ in self.chunks.values():vao.release();vbo.release()
            self.worker_stop.set();self.worker.join(timeout=1.0)
            if self.shadow_renderer is not None:self.shadow_renderer.release()
            self.enemy_renderer.release()
            if self.glb_renderer is not None:self.glb_renderer.release()
            self.battlefield_renderer.release()
            self.death_screen.release()
            self.normal_texture.release();self.texture.release();self.prog.release();self.sky_prog.release()
            pygame.quit()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Den oändliga skyttegraven')
    parser.add_argument('--seed',type=int,default=DEFAULT_SEED,
                        help='Återskapa en bana; standard är 1944')
    set_world_seed(parser.parse_args().seed)
    try:Trench().run()
    except Exception as error:
        print(f'3D-test could not start: {type(error).__name__}: {error}',file=sys.stderr)
        raise
