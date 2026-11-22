"""Offline display palette and the supplied camera-response proxy; never a detector feature.

All RGB values are projector bytes. Predicted DN is an engineering proxy, not
measured camera brightness or a calibrated probability of hole detection.
"""
from __future__ import annotations
import numpy as np
from PIL import Image

MODEL_WEIGHTS=np.array([16.,115.,35.])
EXPORT_FLOOR=128.  # margin above the requested 120-DN proxy at alpha-composited edges
PALETTE={'vase':(18,225,175),'jar':(245,225,35),'bowl':(25,215,245),
         'bottle':(30,235,85),'lamp':(250,210,40)}
HIT_COLOR=(255,212,45)
HIT_PALETTE={'jar':(25,220,250),'lamp':(25,220,250)}


def camera_level(rgb):
    values=np.asarray(rgb,dtype=np.float64)/255.
    return 37.+np.sum(MODEL_WEIGHTS*np.clip(values,0,1)**2.2,axis=-1)


def ensure_floor(rgb,floor=EXPORT_FLOOR):
    values=np.clip(np.rint(np.asarray(rgb,dtype=np.float64)),0,255)  # solve after byte quantization
    r=values[...,0]/255.;b=values[...,2]/255.
    required=np.maximum(0.,(floor-37.-16.*r**2.2-35.*b**2.2)/115.)
    if np.any(required>1):raise ValueError('Palette floor cannot be reached with green alone')
    values[...,1]=np.maximum(values[...,1],np.ceil(255.*required**(1/2.2)))
    return np.clip(np.rint(values),0,255).astype(np.uint8)


def colorize(image,kind=None,*,part=None,hit=False):
    image=image.convert('RGBA');data=np.asarray(image).copy();rgb=data[:,:,:3].astype(float)
    level=(rgb*np.array([.2126,.7152,.0722])).sum(axis=2)
    visible=data[:,:,3]>=128
    if not visible.any():return image
    low,high=np.percentile(level[visible],[3,97]);span=max(1.,high-low)
    shade=np.clip((level-low)/span,0,1)
    if kind is None:
        # Neutral 194..214 is darker than the original chalk-white room. Gray
        # 140..170 cannot meet the supplied 120-DN camera proxy (about 81..105 DN).
        gray=194.+20.*shade
        result=np.repeat(gray[:,:,None],3,axis=2)
    else:
        base=HIT_PALETTE.get(kind,HIT_COLOR) if hit else ((250,230,30) if part in ('flower','lid') else PALETTE.get(kind,(210,225,180)))
        result=np.asarray(base,dtype=float)[None,None,:]*(.80+.23*shade[:,:,None])
    data[:,:,:3]=ensure_floor(result)
    return Image.fromarray(data)


def stats(image):
    a=np.asarray(image.convert('RGBA'));valid=a[:,:,3]>=128;rgb=a[:,:,:3][valid]
    dn=camera_level(rgb)
    return dict(pixels=int(valid.sum()),predicted_dn_min=float(dn.min()),
                predicted_dn_p05=float(np.quantile(dn,.05)),predicted_dn_median=float(np.median(dn)),
                predicted_dn_max=float(dn.max()),
                mean_rgb=[float(x) for x in rgb.mean(axis=0)],
                mean_channel_span=float((rgb.max(axis=1).astype(float)-rgb.min(axis=1)).mean()))


def protect_render(image, floor=EXPORT_FLOOR):
    """Keep Cycles shading/specular colors; minimally lift toward white for DN floor.

    No texture or synthetic highlights added. Alpha is unchanged. Integer safety
    search protects the floor after RGB quantization. Not camera calibration.
    """
    a=np.array(image.convert('RGBA'))
    # Exact RGB palette lookup avoids re-solving identical transparent/background
    # colors hundreds of thousands of times. No sampling or approximate colors.
    shape=a.shape[:2]
    colors,inverse=np.unique(a[:,:,:3].reshape(-1,3),axis=0,return_inverse=True)
    rgb=colors.astype(float);need=camera_level(rgb)<floor
    if not need.any():return Image.fromarray(a)
    lo=np.zeros(len(rgb));hi=np.ones_like(lo)
    for _ in range(16):
        mid=(lo+hi)*.5
        trial=rgb+(255.-rgb)*mid[:,None]
        ok=camera_level(np.floor(trial))>=floor
        hi=np.where(ok,mid,hi);lo=np.where(ok,lo,mid)
    lifted=np.floor(rgb+(255.-rgb)*hi[:,None])
    protected=np.where(need[:,None],lifted,rgb).clip(0,255).astype(np.uint8)
    a[:,:,:3]=protected[inverse].reshape(*shape,3)
    return Image.fromarray(a)


def hit_render(image,kind):
    """Contrasting partial-hit state retaining the underlying rendered highlights."""
    a=np.array(image.convert('RGBA'))
    target=np.asarray(HIT_PALETTE.get(kind,HIT_COLOR),dtype=float)
    a[:,:,:3]=np.rint(.45*a[:,:,:3].astype(float)+.55*target).astype(np.uint8)
    return protect_render(Image.fromarray(a))
