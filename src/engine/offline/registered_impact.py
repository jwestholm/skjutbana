"""RESEARCH_ONLY registered local residual measurements; never live authority.

All coordinates are canonical PRE-camera pixels. Crops carry their camera origin.
The existing V2 translation estimator aligns CURRENT back to PRE, not vice versa.
No labels, target identity, source weights or candidate scores enter these features.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import cv2
import numpy as np
from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2, DEFAULT_CONFIG


@dataclass
class RegisteredPair:
    pre: np.ndarray
    post: np.ndarray
    aligned: np.ndarray
    origin: tuple[int,int]
    registration: dict
    global_offset: float


def prepare_pair(pre, post, *, origin=(0,0), translation=None):
    if pre.ndim!=2 or pre.shape!=post.shape: raise ValueError('PRE and POST must share the same grayscale camera crop')
    roi=np.full(pre.shape,255,np.uint8)
    if translation is None:
        # Same estimator, self-bias correction and acceptance gates as V2.
        _, bias=CandidateGeneratorV2._register_current(None,pre,pre,roi=roi,cfg=DEFAULT_CONFIG)
        aligned,reg=CandidateGeneratorV2._register_current(None,pre,post,roi=roi,cfg=DEFAULT_CONFIG,
            registration_bias=(bias['raw_dx'],bias['raw_dy']))
    else:
        dx,dy=map(float,translation)
        aligned=cv2.warpAffine(post.astype(np.float32),np.float32([[1,0,-dx],[0,1,-dy]]),(pre.shape[1],pre.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        reg=dict(dx=dx,dy=dy,applied=True,response=None,origin='explicit_known_translation')
    offset=float(np.median(pre.astype(np.float32)-aligned.astype(np.float32)))
    return RegisteredPair(pre,post,aligned,tuple(origin),reg,offset)


def sample(image, xy, origin, radius=16):
    x,y=xy[0]-origin[0],xy[1]-origin[1]
    if x-radius<0 or y-radius<0 or x+radius>=image.shape[1] or y+radius>=image.shape[0]:
        raise ValueError('Candidate patch crosses recorded crop boundary')
    # Convert only the small source patch; preserve fractional camera position.
    x0=int(math.floor(x-radius));y0=int(math.floor(y-radius))
    local=image[y0:int(math.ceil(y+radius))+1,x0:int(math.ceil(x+radius))+1].astype(np.float32)
    return cv2.getRectSubPix(local,(2*radius+1,2*radius+1),(float(x-x0),float(y-y0)))


def affine_ring(pre, post, ring):
    """Robust paired affine fit outside the impact center; flat rings use offset."""
    x=pre[ring].astype(float);y=post[ring].astype(float)
    gain=1.;offset=float(np.median(y-x));weights=np.ones(len(x))
    if np.std(x)>1:
        for _ in range(4):
            mx=np.average(x,weights=weights);my=np.average(y,weights=weights)
            variance=np.average((x-mx)**2,weights=weights)
            if variance<=1: break
            gain=float(np.average((x-mx)*(y-my),weights=weights)/variance)
            offset=float(np.median(y-gain*x))
            residual=y-(gain*x+offset)
            sigma=max(1.,1.4826*float(np.median(np.abs(residual-np.median(residual)))))
            weights=np.minimum(1.,1.5*sigma/np.maximum(np.abs(residual),1e-6))
    return gain,offset


def residual_features(residual):
    n=residual.shape[0];yy,xx=np.mgrid[:n,:n];r=np.hypot(xx-n//2,yy-n//2)
    center=r<=4;ring=(r>=8)&(r<=12);disk=r<=12
    absres=np.abs(residual);dark=np.maximum(residual,0);bright=np.maximum(-residual,0)
    sigma=max(1.,1.4826*float(np.median(np.abs(residual[ring]-np.median(residual[ring])))))
    out=dict(center_abs=float(absres[center].mean()),center_dark=float(dark[center].mean()),center_bright=float(bright[center].mean()),
        ring_abs=float(absres[ring].mean()),ring_dark=float(dark[ring].mean()),ring_bright=float(bright[ring].mean()),ring_sigma=sigma,
        compact=float(absres[center].mean()-absres[ring].mean()),signed_contrast=float(residual[center].mean()-residual[ring].mean()),
        dark_contrast=float(dark[center].mean()-dark[ring].mean()),concentration=float(absres[center].sum()/max(absres[disk].sum(),1e-6)),
        polarity=float(dark[center].sum()/max(absres[center].sum(),1e-6)),peak=float(absres[center].max()))
    for inner,outer in ((0,2),(2,4),(4,8),(8,12)):
        m=(r>=inner)&(r<=outer);out[f'radial_{inner}_{outer}']=float(absres[m].mean())
    energy=absres[disk];prob=energy/max(energy.sum(),1e-6)
    out['entropy']=float(-np.sum(prob*np.log(np.maximum(prob,1e-12)))/np.log(len(prob)))
    mask=((absres>3*sigma)&disk).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    candidates=[i for i in range(1,count) if np.any((labels==i)&center)]
    chosen=max(candidates,key=lambda i:float(absres[labels==i].sum()),default=0)
    component=(labels==chosen) if chosen else np.zeros_like(mask,dtype=bool)
    area=int(component.sum());out['component_area']=area
    out['component_excess']=float(np.maximum(absres[component]-3*sigma,0).sum())
    out['component_dark']=float(dark[component].sum())
    if area:
        contours,_=cv2.findContours(component.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        perimeter=sum(cv2.arcLength(c,True) for c in contours)
        contour_area=sum(cv2.contourArea(c) for c in contours)
        out['component_circularity']=float(4*math.pi*contour_area/max(perimeter**2,1))
        coordinates=np.column_stack(np.nonzero(component));eig=np.linalg.eigvalsh(np.cov(coordinates.T)) if area>2 else [0,0]
        out['component_elongation']=float((eig[-1]+1)/(eig[0]+1))
    else:out.update(component_circularity=0.,component_elongation=0.)
    return out


def extract(pair, xy, *, return_patches=False):
    p=sample(pair.pre,xy,pair.origin);q=sample(pair.post,xy,pair.origin);a=sample(pair.aligned,xy,pair.origin)
    yy,xx=np.mgrid[:p.shape[0],:p.shape[1]];r=np.hypot(xx-p.shape[1]//2,yy-p.shape[0]//2);ring=(r>=8)&(r<=16)
    residual=p-a;ring_offset=float(np.median(residual[ring]));gain,offset=affine_ring(p,a,ring)
    residuals=dict(raw=p-q,registered=residual,global_median=residual-pair.global_offset,
        ring_median=residual-ring_offset,ring_affine=gain*p+offset-a)
    values={f'{mode}_{k}':v for mode,res in residuals.items() for k,v in residual_features(res).items()}
    values.update(registration_dx=pair.registration['dx'],registration_dy=pair.registration['dy'],
        registration_applied=pair.registration['applied'],global_offset=pair.global_offset,ring_offset=ring_offset,ring_affine_gain=gain,ring_affine_offset=offset,
        pre_mean=float(p[r<=4].mean()),post_mean=float(a[r<=4].mean()),pre_contrast=float(p[ring].mean()-p[r<=4].mean()),post_contrast=float(a[ring].mean()-a[r<=4].mean()))
    if return_patches:
        dx=pair.registration['dx'] if pair.registration['applied'] else 0
        dy=pair.registration['dy'] if pair.registration['applied'] else 0
        display_pre=cv2.warpAffine(p,np.float32([[1,0,dx],[0,1,dy]]),(p.shape[1],p.shape[0]),borderMode=cv2.BORDER_REPLICATE)
        return values,dict(pre=p,registered_pre=display_pre,post=q,aligned_post=a,abs_registered=np.abs(residual),**residuals)
    return values
