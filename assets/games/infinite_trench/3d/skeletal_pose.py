"""V3.9.3 experimental, reproducible humanoid bone pose library.

This is rigid single-influence GPU skeletal skinning of a procedural bind mesh,
NOT an imported artist-rigged GLB or a physics/collision solver. Root translation
continues to be owned by EnemyAgent.route_m / the existing route and obstacles.
No animation root motion is ever added to world position.
"""
from __future__ import annotations
import math
from functools import lru_cache
from dataclasses import dataclass
import numpy as np
from enemy_system import EnemyAgent,EnemySpec
from enemy_system import actor_joints

BONES = (
    'pelvis','spine','head',
    'left_upper_arm','left_forearm','left_hand',
    'right_upper_arm','right_forearm','right_hand',
    'left_thigh','left_shin','left_foot',
    'right_thigh','right_shin','right_foot','weapon',
)
assert len(BONES) == 16

POSE_LAB = ('stand','walk','run','crouch','kneel','prone',
            'supine_sleep','fear_curl','aim','fire','spade_attack','dead')


def _joints(agent, pose):
    """Local metre anchors; animation is sampled without mutating game logic."""
    j = actor_joints(agent)
    out = {key:np.asarray(j[key],dtype=np.float64).copy() for key in ('pelvis','torso','head')}
    for key in ('hips','knees','ankles','feet'):
        out[key]=[np.asarray(x,dtype=np.float64).copy() for x in j[key]]
    crouch=float(j['crouch'])
    movement=float(j['move'])
    cycle=float(j['cycle'])
    t=float(agent.total_time)
    # The diagnostic mode must preview distinct human locomotion speeds even
    # though the dummy never advances along the route. Gameplay remains owned
    # by actual travelled metres, never by a looping preview timer.
    lab_locomotion=pose in ('walk','run')
    if lab_locomotion:
        movement=.70 if pose=='walk' else 1.0
        cycle=t*(5.8 if pose=='walk' else 10.8)
    elif pose=='stand':
        movement=0.0
    elif pose=='base':
        movement=float(j['move'])
        # Speed is measured by EnemyAgent.update from route displacement.
        # Walking and running therefore have different swing amplitudes.
        pace=min(1.0,max(0.,float(getattr(agent,'gait_speed_mps',0.0)))/2.8)
        movement*=.60+.40*pace

    if pose in ('walk','run','stand','base'):
        # Breathing and quiet postural sway are present even without movement;
        # asymmetry stops a living soldier from resembling a frozen mannequin.
        breath=.011*math.sin(t*2.05+agent.spec.enemy_id*.36)
        sway=.016*math.sin(t*.81+agent.spec.enemy_id*.73)
        out['pelvis'][0]+=sway*.55
        out['torso'][0]+=sway
        out['head'][0]+=sway*1.15
        for key in ('torso','head'):
            out[key][1]+=breath
        if lab_locomotion:
            # Large stride and knee clearance at a run, deliberate heel lift
            # at a walk. The 16-bone bind mesh itself stays GPU resident.
            run=pose=='run'
            stride=.29 if run else .17
            rise=.19 if run else .075
            bounce=.030 if run else .012
            out['pelvis'][1]=.88+bounce*(1.-math.cos(2.*cycle))
            out['torso'][1]=1.15+bounce*(1.-math.cos(2.*cycle))+breath
            out['head'][1]=1.67+bounce*(1.-math.cos(2.*cycle))+breath
            out['torso'][2]+=.10 if run else .025
            out['head'][2]+=.07 if run else .018
            for i,side in enumerate((-1.,1.)):
                w=math.sin(cycle+math.pi*i)
                out['hips'][i]=np.array((side*.16,out['pelvis'][1]-.015,.025))
                out['knees'][i]=np.array((side*.17,.49+rise*.30*max(0.,w),
                                          .10+(.095 if run else .055)*w))
                out['ankles'][i]=np.array((side*.175,.165+rise*max(0.,w),
                                           .02+stride*w))
                out['feet'][i]=np.array((side*.175,.108+rise*max(0.,w),
                                         .12+stride*w))
    if pose=='crouch':
        drop=.19
        out['pelvis'][1]-=drop
        out['torso'][1]-=.28
        out['head'][1]-=.31
        for i in range(2):
            out['hips'][i][1]-=drop
            out['knees'][i][1]-=.14
            out['knees'][i][2]+=.17
            out['ankles'][i][2]+=.10
            out['feet'][i][2]+=.10
    if pose=='kneel':
        out['pelvis'][:]=(0.,.62,.02)
        out['torso'][:]=(0.,.94,.07)
        out['head'][:]=(0.,1.43,.09)
        for i,s in enumerate((-1,1)):
            out['hips'][i][:]=(s*.17,.59,.02)
        out['knees'][0][:]=(-.18,.14,-.04)
        out['ankles'][0][:]=(-.17,.14,-.33)
        out['feet'][0][:]=(-.17,.12,-.38)
        out['knees'][1][:]=(.18,.51,.42)
        out['ankles'][1][:]=(.18,.15,.55)
        out['feet'][1][:]=(.18,.12,.66)
    if pose in ('prone','supine_sleep'):
        back=pose=='supine_sleep'
        out['pelvis'][:]=(0.,.23,-.41)
        out['torso'][:]=(0.,.32 if not back else .22,.02)
        out['head'][:]=(0.,.39 if not back else .24,.61)
        for i,s in enumerate((-1,1)):
            out['hips'][i][:]=(s*.16,.24,-.42)
            out['knees'][i][:]=(s*.17,.16,-.76)
            out['ankles'][i][:]=(s*.175,.13,-1.10)
            out['feet'][i][:]=(s*.175,.11,-1.21)
    if pose=='dead':
        # Unlike 'prone', a corpse is NOT braced to sight a rifle. Weight
        # settles unevenly: one knee folds, the neck loses support, the arms
        # relax to either side. Deterministic mirror variation costs no RNG.
        side=1.0 if agent.spec.enemy_id%2 else -1.0
        out['pelvis'][:]=(side*.04,.255,-.34)
        out['torso'][:]=(-side*.06,.32,.005)
        out['head'][:]=(-side*.22,.30,.52)
        for i,s in enumerate((-1.,1.)):
            out['hips'][i][:]=(s*.17,.245,-.34)
            if s==side:
                out['knees'][i][:]=(s*.21,.225,-.52)
                out['ankles'][i][:]=(s*.30,.13,-.88)
                out['feet'][i][:]=(s*.30,.11,-.99)
            else:
                out['knees'][i][:]=(s*.18,.16,-.72)
                out['ankles'][i][:]=(s*.18,.13,-1.08)
                out['feet'][i][:]=(s*.18,.11,-1.20)
    if pose=='fear_curl':
        out['pelvis'][:]=(0.,.37,-.13)
        out['torso'][:]=(0.,.57,.20)
        out['head'][:]=(0.,.83,.34)
        for i,s in enumerate((-1,1)):
            out['hips'][i][:]=(s*.16,.35,-.13)
            out['knees'][i][:]=(s*.17,.17,.38)
            out['ankles'][i][:]=(s*.16,.12,.39)
            out['feet'][i][:]=(s*.16,.11,.52)
    # Other endpoints are derived from posture so arms stay attached to torso.
    torso=out['torso'];head=out['head']
    up=(head-torso)
    length=np.linalg.norm(up)
    up=up / max(length,1.e-6)
    out['neck']=torso + up*.30
    out['shoulders']=[];out['elbows']=[];out['hands']=[]
    for i,s in enumerate((-1,1)):
        shoulder=torso+np.array((s*.30,.22 if pose not in ('prone','supine_sleep','dead') else .05,-.015))
        elbow=torso+np.array((s*.34,-.06,.08))
        hand=torso+np.array((s*.115,-.13,.31))
        if pose=='prone':
            shoulder=torso+np.array((s*.30,0.,.11))
            elbow=torso+np.array((s*.36,-.1,.35))
            hand=torso+np.array((s*.14,-.13,.62))
        elif pose=='dead':
            # Arms spill away from the weapon rather than forming the
            # suspicious-looking two-handed binocular/rifle pose.
            turn=1.0 if agent.spec.enemy_id%2 else -1.0
            shoulder=torso+np.array((s*.30,0.,.07))
            if s==turn:
                elbow=torso+np.array((s*.48,-.14,-.12))
                hand=torso+np.array((s*.58,-.19,-.19))
            else:
                elbow=torso+np.array((s*.42,-.12,.21))
                hand=torso+np.array((s*.49,-.19,.34))
        elif pose=='supine_sleep':
            shoulder=torso+np.array((s*.31,0.,.04))
            elbow=torso+np.array((s*.52,-.02,-.06))
            hand=torso+np.array((s*.54,-.04,-.24))
        elif pose=='fear_curl':
            shoulder=torso+np.array((s*.27,.12,.03))
            elbow=torso+np.array((s*.20,.07,.28))
            hand=head+np.array((s*.13,-.07,.12))
        elif pose=='spade_attack':
            shoulder=torso+np.array((s*.29,.22,-.01))
            elbow=torso+np.array((s*.29,.47,-.13))
            hand=torso+np.array((s*.14,.68,.09+ .35*math.sin(agent.total_time*6.)))
        if pose in ('aim','fire','base'):
            # A shouldered rifle: right hand at trigger/stock, left hand well
            # forward under the fore-end; the sight line lies below the eye.
            # The normal AI channels use the same *continuous* aim_fraction.
            aim=1.0 if pose in ('aim','fire') else max(0.,min(1.,float(agent.aim_fraction())))
            shoulder_elbow=torso+np.array((s*.32,.10,.19))
            shouldered_hand=torso+np.array((s*.115,.32 if i==0 else .27,.60 if i==0 else .34))
            elbow=elbow*(1.-aim)+shoulder_elbow*aim
            hand=hand*(1.-aim)+shouldered_hand*aim
            if pose=='base' and aim>.95:
                # Avoid an exaggerated barrel sway while sighting.
                hand+=np.array((0.,.003*math.sin(t*1.8+agent.spec.enemy_id),0.))
        if pose in ('walk','run','base') and movement>.02:
            # Arms counter-rotate relative to the legs; weapon aim suppresses
            # this swing so fighting actors do not wave the gun around.
            aim=float(agent.aim_fraction()) if pose=='base' else 0.0
            swing=(.045 if pose=='walk' else .085 if pose=='run' else .065*movement)
            wave=math.sin(cycle+math.pi*i)
            swing*=max(0.0,1.0-aim)
            elbow=elbow+np.array((0.,.023*wave,swing*wave))
            hand=hand+np.array((0.,.020*wave,swing*wave*1.12))
        if pose in ('fire','base'):
            # Short shoulder impulse; the hands and gun recoil together. Fire
            # in the pose lab repeats at a legible, fixed demonstration cadence.
            if pose=='fire':
                phase=t%1.05
                kick=.088*math.exp(-phase*20.)
            elif agent.state=='firing':
                kick=.088*math.exp(-max(0.0,agent.state_time)*19.)
            else:
                kick=0.0
            elbow=elbow+np.array((0.,kick*.12,-kick*.60))
            hand=hand+np.array((0.,kick*.19,-kick))
        out['shoulders'].append(shoulder)
        out['elbows'].append(elbow)
        out['hands'].append(hand)
    out['weapon_anchor']=(out['hands'][0]+out['hands'][1])*.5 + np.array((0.,.0,.12))
    if pose=='dead':
        # Rifle drops independently of the wrists onto the trench floor.
        side=1.0 if agent.spec.enemy_id%2 else -1.0
        out['weapon_anchor'][:]=(side*.40,.16,.34)
    return out


def _orientation(a,b):
    """Shortest rotation from bind segment to posed segment (3x3)."""
    x=np.asarray(b[0],dtype=np.float64)-np.asarray(a[0],dtype=np.float64)
    y=np.asarray(b[1],dtype=np.float64)-np.asarray(a[1],dtype=np.float64)
    nx=np.linalg.norm(x);ny=np.linalg.norm(y)
    if nx<1e-8 or ny<1e-8:return np.eye(3)
    x=x/nx;y=y/ny
    v=np.cross(x,y);c=float(np.dot(x,y));s=float(np.linalg.norm(v))
    if s<1e-8:
        if c>0:return np.eye(3)
        axis=np.cross(x,np.array((1.,0.,0.)) if abs(x[0])<.8 else np.array((0.,0.,1.)))
        axis/=np.linalg.norm(axis)
        return 2*np.outer(axis,axis)-np.eye(3)
    skew=np.array(((0,-v[2],v[1]),(v[2],0,-v[0]),(-v[1],v[0],0)))
    return np.eye(3)+skew+skew@skew*((1-c)/(s*s))


def _matrix(anchor0,anchor1,tip0=None,tip1=None,twist=0.0):
    a=np.asarray(anchor0,dtype=np.float64);b=np.asarray(anchor1,dtype=np.float64)
    if tip0 is not None:
        r=_orientation((a,tip0),(b,tip1))
    else:r=np.eye(3)
    if twist:
        ct,st=math.cos(twist),math.sin(twist)
        yrot=np.array(((ct,0.,st),(0.,1.,0.),(-st,0.,ct)))
        r=r@yrot
    mat=np.eye(4,dtype=np.float64)
    mat[:3,:3]=r
    mat[:3,3]=b-r@a
    return mat.astype('f4')


@lru_cache(maxsize=1)
def _bind_joints():
    reference=EnemyAgent(EnemySpec(0,32.0,-.18,'unaware'))
    return _joints(reference,'stand')


def bone_matrices(agent,pose='auto'):
    """One 16-matrix palette per actor, evaluated once per frame (GPU skinning)."""
    if pose=='auto':
        # Existing smooth EnemyAgent channels are authoritative in gameplay;
        # only explicit lab poses bypass them. No abrupt FSM pose jumps.
        pose='dead' if agent.dead else 'auto_normal'
    # Bind pose remains identical to enemy_parts() neutral skeleton.
    base=_bind_joints()
    # Restore neutral gait, if any, for bind joints by independent stand sample.
    target=_joints(agent,pose) if pose!='auto_normal' else _joints(agent,'base')
    falling=1.0
    if pose=='dead' and agent.dead:
        falling=max(0.,min(1.,float(agent.state_time)/1.18))
        falling=falling*falling*(3.0-2.0*falling)
        start=_joints(agent,'base')
        for name in ('pelvis','torso','head','neck'):
            target[name]=start[name]*(1.0-falling)+target[name]*falling
        for name in ('hips','knees','ankles','feet','shoulders','elbows','hands'):
            target[name]=[start[name][i]*(1.0-falling)+target[name][i]*falling
                          for i in range(2)]
        # Let go shortly after impact; the gun is not attached to the hands
        # during the final settle, but its movement remains continuous.
        release=max(0.,min(1.,(float(agent.state_time)-.15)/.75))
        release=release*release*(3.-2.*release)
        target['weapon_anchor']=start['weapon_anchor']*(1.-release)+target['weapon_anchor']*release
    mats=[]
    mats.append(_matrix(base['pelvis'],target['pelvis']))
    mats.append(_matrix(base['torso'],target['torso'],base['neck'],target['neck'],
                        twist=math.pi if pose=='supine_sleep' else 0.0))
    head_turn=getattr(agent,'head_look',0.)
    if pose=='stand':
        # In the idle lab, a live sentry occasionally checks his surroundings.
        # Gameplay obtains this rotation from EnemyAgent's perception state.
        head_turn=.065*math.sin(agent.total_time*.9+agent.spec.enemy_id*.61)
    if pose=='dead':
        side=1.0 if agent.spec.enemy_id%2 else -1.0
        head_turn=head_turn*(1.-falling)+side*1.15*falling
    mats.append(_matrix(base['head'],target['head'],base['neck'],target['neck'],
                        twist=math.pi if pose=='supine_sleep' else head_turn))
    for i in range(2):
        mats.append(_matrix(base['shoulders'][i],target['shoulders'][i],base['elbows'][i],target['elbows'][i]))
        mats.append(_matrix(base['elbows'][i],target['elbows'][i],base['hands'][i],target['hands'][i]))
        mats.append(_matrix(base['hands'][i],target['hands'][i]))
    for i in range(2):
        mats.append(_matrix(base['hips'][i],target['hips'][i],base['knees'][i],target['knees'][i]))
        mats.append(_matrix(base['knees'][i],target['knees'][i],base['ankles'][i],target['ankles'][i]))
        mats.append(_matrix(base['feet'][i],target['feet'][i]))
    # Dropped gun stays low and points diagonally away from the body.
    gun_turn=0.
    if pose=='dead':
        side=1.0 if agent.spec.enemy_id%2 else -1.0
        gun_turn=side*.62*(release if agent.dead else 1.)
    mats.append(_matrix(base['weapon_anchor'],target['weapon_anchor'],twist=gun_turn))
    result=np.asarray(mats,dtype='f4')
    # Calibrated from the actual 3 V3.9.2 bind meshes, not camera bob.
    # Correct the individual pieces that would otherwise clip below y=0.
    # Boots stay planted; never translate the full actor up into the air.
    # Special postures are currently available in the F1 pose laboratory;
    # no pose correction affects normal game navigation or hit geometry.
    grounding={
      'crouch':{10:.073,13:.073},
      'kneel':{10:.058},
      'prone':{1:.19,9:.28,10:.25,12:.28,13:.25,15:.15},
      'supine_sleep':{1:.26,3:.09,4:.06,6:.09,7:.06,
                      9:.28,10:.25,12:.28,13:.25,15:.15},
      'fear_curl':{10:.24,13:.24},
      'dead':{1:.19,4:.095,7:.095,9:.28,10:.25,12:.28,13:.25,15:.18},
    }
    for joint,lift in grounding.get(pose,{}).items():
        result[joint,1,3]+=lift*(falling if pose=='dead' else 1.)
    if pose=='auto_normal':
        # Smooth standing -> crouched animations can flex lower legs through
        # the local ground plane by up to ~5 cm: correct only the shin bones.
        lift=.065*float(agent.crouch_fraction())
        for idx in (10,13):result[idx,1,3]+=lift
    assert result.shape==(len(BONES),4,4)
    return result


def selected_visual_pose(agent,override=None):
    return 'auto' if override is None else override


# V3.9.6: diagnostic command-driven pose transitions. These matrices are
# visualization-only; game navigation and shot-frozen collision authority stay
# with EnemyAgent/EnemyDirector. No CPU mesh buffers are modified per frame.

def _rotation_quat(r):
    """Stable proper rotation -> unit quaternion (w,x,y,z)."""
    r=np.asarray(r,dtype=np.float64)
    trace=float(np.trace(r))
    if trace>0.0:
        t=math.sqrt(max(0.,trace+1.))*2.
        q=(.25*t,(r[2,1]-r[1,2])/t,(r[0,2]-r[2,0])/t,(r[1,0]-r[0,1])/t)
    else:
        i=int(np.argmax(np.diag(r)))
        if i==0:
            t=math.sqrt(max(1.e-12,1.+r[0,0]-r[1,1]-r[2,2]))*2.
            q=((r[2,1]-r[1,2])/t,.25*t,(r[0,1]+r[1,0])/t,(r[0,2]+r[2,0])/t)
        elif i==1:
            t=math.sqrt(max(1.e-12,1.+r[1,1]-r[0,0]-r[2,2]))*2.
            q=((r[0,2]-r[2,0])/t,(r[0,1]+r[1,0])/t,.25*t,(r[1,2]+r[2,1])/t)
        else:
            t=math.sqrt(max(1.e-12,1.+r[2,2]-r[0,0]-r[1,1]))*2.
            q=((r[1,0]-r[0,1])/t,(r[0,2]+r[2,0])/t,(r[1,2]+r[2,1])/t,.25*t)
    a=np.asarray(q,dtype=np.float64)
    return a/max(1.e-12,float(np.linalg.norm(a)))


def _quat_rotation(q):
    w,x,y,z=q
    return np.array(((1.-2.*(y*y+z*z),2.*(x*y-z*w),2.*(x*z+y*w)),
                     (2.*(x*y+z*w),1.-2.*(x*x+z*z),2.*(y*z-x*w)),
                     (2.*(x*z-y*w),2.*(y*z+x*w),1.-2.*(x*x+y*y))),dtype=np.float32)


def interpolate_bone_matrices(source,target,alpha):
    """Rigid 16-bone transform blend; no stretching limbs via matrix lerp."""
    a=np.asarray(source,dtype=np.float32)
    b=np.asarray(target,dtype=np.float32)
    if a.shape!=b.shape or a.shape!=(len(BONES),4,4):
        raise ValueError('Incompatible bone palettes')
    amount=max(0.,min(1.,float(alpha)))
    if not math.isfinite(float(alpha)):raise ValueError('non-finite transition')
    if amount<=0:return a.copy()
    if amount>=1:return b.copy()
    out=np.zeros_like(a)
    out[:,3,3]=1.
    out[:,:3,3]=a[:,:3,3]*(1.-amount)+b[:,:3,3]*amount
    for i in range(len(BONES)):
        qa=_rotation_quat(a[i,:3,:3]); qb=_rotation_quat(b[i,:3,:3])
        d=float(np.dot(qa,qb))
        if d<0:qb=-qb;d=-d
        if d>.9995:q=qa*(1.-amount)+qb*amount
        else:
            angle=math.acos(max(-1.,min(1.,d)))
            sin_angle=math.sin(angle)
            q=(qa*math.sin((1.-amount)*angle)+qb*math.sin(amount*angle))/sin_angle
        q=q/max(1.e-12,float(np.linalg.norm(q)))
        out[i,:3,:3]=_quat_rotation(q)
    return out


@dataclass
class PoseTransition:
    """F1 demo controller: commands produce transitions, not hard pose switches.

    The same frozen palette is handed to both depth-shadow and color pass.
    Production enemies continue using EnemyAgent's existing live pose channels.
    """
    pose: str='stand'
    duration: float=.44
    elapsed: float=1.0
    source: np.ndarray | None=None

    def sample(self,agent):
        target=bone_matrices(agent,self.pose)
        if self.source is None or self.elapsed>=self.duration:
            return target
        t=max(0.,min(1.,self.elapsed/max(.001,self.duration)))
        smooth=t*t*(3.-2.*t)
        return interpolate_bone_matrices(self.source,target,smooth)

    def request(self,pose,agent):
        if pose not in POSE_LAB:raise ValueError('unknown diagnostic pose')
        if pose==self.pose:return
        self.source=self.sample(agent).copy()  # interrupt from what was displayed
        self.pose=pose
        slow={'kneel':.73,'prone':.93,'supine_sleep':1.00,
              'fear_curl':.83,'dead':1.02,'spade_attack':.48}
        self.duration=slow.get(pose,.42)
        self.elapsed=0.

    def advance(self,dt):
        if not math.isfinite(float(dt)) or dt<0:raise ValueError('invalid delta time')
        self.elapsed=min(self.duration,max(0.,self.elapsed+float(dt)))
        if self.elapsed>=self.duration:self.source=None
