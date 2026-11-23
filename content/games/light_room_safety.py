"""Local display safety around active targets. No detector/measurement authority.

Only prepared on a layout change, never while a shot query is in flight.
"""
import numpy as np
import pygame
import cv2

TARGET_MARGIN=12
FEATHER=32
FLOOR_DN=124.


def camera_proxy(rgb):
    return 37.+np.sum(np.array([16.,115.,35.])*(np.asarray(rgb,dtype=float)/255.)**2.2,axis=-1)


def local_background(source,items):
    """Preserve every byte outside target AABBs + margin + feather.

    Silhouettes and their immediate miss neighbourhood are protected.
    AABBs are only a fallback for tool fixtures without rendered alpha. Neither the mask nor real hole history vetoes a shot.
    """
    rgb=pygame.surfarray.array3d(source);w,h=source.get_size();weight=np.zeros((w,h),np.float32)
    for item in items:
        g=item['geometry'];x0=max(0,int(g.x)-TARGET_MARGIN-FEATHER);x1=min(w,int(np.ceil(g.right))+TARGET_MARGIN+FEATHER)
        y0=max(0,int(g.y)-TARGET_MARGIN-FEATHER);y1=min(h,int(np.ceil(g.bottom))+TARGET_MARGIN+FEATHER)
        mask=item.get('display_mask')
        if mask is None:
            xx=np.arange(x0,x1)[:,None];yy=np.arange(y0,y1)[None,:]
            dx=np.maximum(np.maximum(g.x-TARGET_MARGIN-xx,xx-(g.right+TARGET_MARGIN)),0)
            dy=np.maximum(np.maximum(g.y-TARGET_MARGIN-yy,yy-(g.bottom+TARGET_MARGIN)),0)
            a=np.clip(1.-np.maximum(dx,dy)/FEATHER,0,1)
        else:
            region=np.zeros((y1-y0,x1-x0),np.uint8)
            gx,gy=round(g.x),round(g.y);left=max(x0,gx);top=max(y0,gy)
            right=min(x1,gx+mask.shape[0]);bottom=min(y1,gy+mask.shape[1])
            region[top-y0:bottom-y0,left-x0:right-x0]=mask[left-gx:right-gx,top-gy:bottom-gy].T
            distance=cv2.distanceTransform(1-region,cv2.DIST_L2,5).T
            a=np.clip((TARGET_MARGIN+FEATHER-distance)/FEATHER,0,1)
        weight[x0:x1,y0:y1]=np.maximum(weight[x0:x1,y0:y1],a)
    # Solve a neutral mix only for affected dark pixels; no global gamma/exposure.
    need=(weight>0)&(camera_proxy(rgb)<FLOOR_DN)
    colors=rgb[need].astype(float);low=np.zeros(len(colors));high=np.ones(len(colors))
    for _ in range(14):
        mid=(low+high)*.5
        ok=camera_proxy(np.floor(colors+(255.-colors)*mid[:,None]))>=FLOOR_DN
        high=np.where(ok,mid,high);low=np.where(ok,low,mid)
    lift=np.floor(colors+(255.-colors)*high[:,None]);a=weight[need,None]
    rgb[need]=np.ceil(colors+(lift-colors)*a).clip(0,255).astype(np.uint8)
    return pygame.surfarray.make_surface(rgb),weight
