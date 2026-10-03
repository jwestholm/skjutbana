"""Bounded, read-only calibration diagnostics. Never installs a mapping.

Explicit geometry report input only; no physical-session discovery or hit labels.
Fits compare the same observed full-camera points in camera pixels. Spatial
holdouts and fixed-model capture transfer are separate from training residuals.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from src.engine.camera.aruco_calibrator import ArucoCalibrator, MARKER_LAYOUT
from automation.physical_foundation_check import write_new


MODELS = ('FORWARD_LS', 'CAMERA_LS', 'MAGSAC', 'QUADRATIC_RESIDUAL')


def project(matrix, points):
    points=np.asarray(points,np.float64).reshape(-1,2)
    q=np.c_[points,np.ones(len(points))]@np.asarray(matrix).T
    if np.any(np.abs(q[:,2])<1e-12):raise ValueError('projective pole')
    return q[:,:2]/q[:,2,None]


def summary(residual):
    error=np.linalg.norm(np.asarray(residual),axis=1)
    return dict(n=len(error),mean=float(np.mean(error)),median=float(np.median(error)),
        max=float(np.max(error)),within_1=int(sum(error<=1)),within_2=int(sum(error<=2)),
        within_3=int(sum(error<=3)),units='full_camera_pixels')


def basis(client, viewport):
    x,y,w,h=viewport
    p=(np.asarray(client)-[x+w/2,y+h/2])/[w/2,h/2]
    u,v=p.T
    return np.c_[np.ones(len(p)),u,v,u*u,u*v,v*v]


def fit_model(name, client, camera, viewport):
    """Fixed methods/configs. Homographies have true inverses; polynomial does not
    provide a runtime inverse and is explicitly ineligible for installation.
    """
    cv2.setRNGSeed(0)
    if name in ('FORWARD_LS','MAGSAC'):
        method=0 if name=='FORWARD_LS' else cv2.USAC_MAGSAC
        H,mask=cv2.findHomography(np.asarray(camera,np.float64),np.asarray(client,np.float64),method,3.)
        matrix=np.linalg.inv(H) if H is not None else None
    else:
        matrix,mask=cv2.findHomography(np.asarray(client,np.float64),np.asarray(camera,np.float64),0)
    if matrix is None or not np.isfinite(matrix).all():raise ValueError('invalid diagnostic fit')
    model=dict(name=name,client_to_camera=matrix.tolist(),viewport=list(viewport),
        returned_mask_count=int(mask.sum()),live_installable=False)
    if name=='QUADRATIC_RESIDUAL':
        coeff,_,rank,_=np.linalg.lstsq(basis(client,viewport),camera-project(matrix,client),rcond=None)
        if rank<6:raise ValueError('rank deficient quadratic holdout')
        model.update(residual_coefficients=coeff.tolist(),inverse_status='NOT_DEFINED_RESEARCH_ONLY')
    else:model['camera_to_client']=np.linalg.inv(matrix).tolist()
    return model


def predict(model, client):
    result=project(model['client_to_camera'],client)
    if 'residual_coefficients' in model:
        result+=basis(client,model['viewport'])@np.asarray(model['residual_coefficients'])
    return result


def min_line_radius(points):
    """Exact minimum enclosing-strip half-width for a 2D point set.

    A homography maps each straight input row to a straight output line. If this
    lower bound exceeds 3px, no homography can put every observed row point within
    3px, regardless of fitting objective. It is a bound on observations, not proof
    of which optical/surface process displaced them.
    """
    points=np.asarray(points,np.float64)
    hull=cv2.convexHull(points.astype(np.float32)).reshape(-1,2).astype(np.float64)
    if len(hull)<3:return dict(radius=0.,normal=None,line_offset=None)
    edges=np.roll(hull,-1,axis=0)-hull
    normals=np.c_[-edges[:,1],edges[:,0]]
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    values=points@normals.T;width=np.ptp(values,axis=0);i=int(np.argmin(width))
    return dict(radius=float(width[i]/2),normal=normals[i].tolist(),
        line_offset=float((values[:,i].max()+values[:,i].min())/2))


def row_column_bounds(ids, points):
    records=[]
    for axis in (0,1):
        for value in sorted(set(MARKER_LAYOUT[i][axis] for i in ids)):
            indices=[j for j,i in enumerate(ids) if MARKER_LAYOUT[i][axis]==value]
            if len(indices)<3:continue
            records.append(dict(axis=('column_u','row_v')[axis],layout_value=value,
                marker_ids=[ids[j] for j in indices],**min_line_radius(np.asarray(points)[indices])))
    return records


def diagonal_center(corners):
    p=np.c_[np.asarray(corners),np.ones(4)]
    center=np.cross(np.cross(p[0],p[2]),np.cross(p[1],p[3]))
    if abs(center[2])<1e-12:raise ValueError('degenerate marker')
    return center[:2]/center[2]


def inspect_frames(root, report, ids):
    """Reuse live detector; compare one standard subpixel refinement, not a sweep."""
    cal=ArucoCalibrator(__import__('pygame').Rect(report['context']['viewport']))
    sub=ArucoCalibrator(cal.viewport_rect)
    sub._aruco_params.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
    sub._aruco_detector=cv2.aruco.ArucoDetector(sub._aruco_dict,sub._aruco_params)
    frames=[];all_corners=[]
    for i,saved in enumerate(report['frames']):
        path=root/f'geometry_frame_{i+1:02d}.npy'
        if path.is_symlink():raise ValueError('symlink frame forbidden')
        im=np.load(path,allow_pickle=False)
        corners,found,_=cal.detect_markers(im)
        refined,ref_ids,_=sub.detect_markers(im)
        original={int(mid):p[0] for mid,p in zip(found.flatten(),corners)}
        refined={int(mid):p[0] for mid,p in zip(ref_ids.flatten(),refined)}
        if set(original)!=set(ids) or set(refined)!=set(ids):raise ValueError('marker coverage changed')
        stored={p['marker_id']:p['observed_camera_xy'] for p in saved['points']}
        if any(np.linalg.norm(original[mid].mean(0)-stored[mid])>.001 for mid in ids):
            raise ValueError('live redetection differs from saved report; do not mix versions')
        rows=[];gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        gradient=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
        for mid in ids:
            p=original[mid];d=diagonal_center(p);rp=refined[mid]
            edge=[]
            for a,b in zip(p,np.roll(p,-1,axis=0)):
                xy=np.linspace(a,b,25).round().astype(int)
                edge.extend(gradient[xy[:,1],xy[:,0]].tolist())
            rows.append(dict(marker_id=mid,corners=p.tolist(),mean=p.mean(0).tolist(),
                diagonal=d.tolist(),subpixel_diagonal=diagonal_center(rp).tolist(),
                diagonal_minus_mean_px=float(np.linalg.norm(d-p.mean(0))),
                subpixel_minus_default_px=float(np.linalg.norm(diagonal_center(rp)-p.mean(0))),
                edge_gradient_mean=float(np.mean(edge)),convex=bool(cv2.isContourConvex(p)),
                oriented_area=float(cv2.contourArea(p,oriented=True))))
        frames.append(dict(frame=i+1,markers=rows,
            default_diagonal_bounds=row_column_bounds(ids,np.array([r['diagonal'] for r in rows])),
            subpixel_diagonal_bounds=row_column_bounds(ids,np.array([r['subpixel_diagonal'] for r in rows]))))
        all_corners.append(original)
    # Original observed pixels, labels only. No enhanced/repainted marker content.
    im=np.load(root/'geometry_frame_01.npy',allow_pickle=False)
    montage=np.full((480,1200,3),245,np.uint8)
    for j,mid in enumerate(ids):
        p=all_corners[0][mid];lo=np.floor(p.min(0)-8).astype(int);hi=np.ceil(p.max(0)+8).astype(int)
        crop=im[lo[1]:hi[1],lo[0]:hi[0]];x=(j%8)*150;y=(j//8)*160
        montage[y:y+128,x:x+144]=cv2.resize(crop,(144,128))
        cv2.putText(montage,f'ID {mid}',(x+3,y+148),0,.5,(0,0,0),1,cv2.LINE_AA)
    return frames,montage


def vector_field(camera, residual, ids, title):
    """Standalone OpenCV diagnostic plot; arrows are 8x measured camera residual."""
    image=np.full((900,1700,3),250,np.uint8)
    def pixel(xy):return tuple(np.round((np.asarray(xy)-[1500,950])*[1.05,1.]).astype(int)+[100,60])
    for x in range(1600,3001,200):
        a=pixel([x,1000]);b=pixel([x,1700]);cv2.line(image,a,b,(220,220,220),1)
        cv2.putText(image,str(x),(a[0]-15,820),0,.5,(70,70,70),1)
    for y in range(1000,1701,100):
        a=pixel([1550,y]);b=pixel([3000,y]);cv2.line(image,a,b,(220,220,220),1)
        cv2.putText(image,str(y),(15,a[1]+5),0,.5,(70,70,70),1)
    for p,d,mid in zip(camera,residual,ids):
        color=(30,130,20) if np.linalg.norm(d)<=3 else (20,30,210)
        a=pixel(p);b=pixel(p+8*d);cv2.arrowedLine(image,a,b,color,2,cv2.LINE_AA,tipLength=.15)
        cv2.circle(image,a,4,(0,0,0),-1);cv2.putText(image,str(mid),(a[0]+7,a[1]-7),0,.5,(0,0,0),1,cv2.LINE_AA)
    cv2.putText(image,title,(25,28),0,.65,(0,0,0),1,cv2.LINE_AA)
    cv2.putText(image,'Camera X / Y down. Arrows = observed - predicted, magnified 8x. Green <=3px. NOT a correction.',(25,865),0,.6,(0,0,0),1,cv2.LINE_AA)
    return image


def run(report_path, output):
    report_path=Path(report_path).resolve();root=report_path.parent
    if 'S03' in str(report_path):raise ValueError('forbidden input')
    report=json.loads(report_path.read_text())
    if report.get('schema')!='foundation-geometry-1' or not report.get('measurement_valid_for_calibrated_setup'):
        raise ValueError('matched physical geometry report required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    paths=[report_path]+[root/f'geometry_frame_{i+1:02d}.npy' for i in range(len(report['frames']))]
    manifest=json.loads((root/'hashes.json').read_text())
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if any(manifest.get(p.name)!=hashes[str(p)] for p in paths):raise ValueError('capture manifest mismatch')
    write_new(output/'input_hashes.json',hashes)
    cal=report['context']['calibration'];ids=sorted(cal['matched_ids']);index={mid:i for i,mid in enumerate(cal['matched_ids'])}
    camera=np.array([cal['camera_points'][index[i]] for i in ids]);client=np.array([cal['viewport_points'][index[i]] for i in ids]);viewport=cal['viewport_rect']
    calibrator=ArucoCalibrator(__import__('pygame').Rect(viewport))
    if not np.array_equal(client,[calibrator.rendered_marker_positions()[i] for i in ids]):raise ValueError('rendered correspondences differ')
    fresh=np.array([[next(p['observed_camera_xy'] for p in f['points'] if p['marker_id']==i) for i in ids] for f in report['frames']])
    current=dict(name='CURRENT',client_to_camera=cal['inverse_homography'],camera_to_client=cal['homography'])
    models={'CURRENT':current,**{n:fit_model(n,client,camera,viewport) for n in MODELS}}
    results={};tables=[]
    for name,model in models.items():
        predicted=predict(model,client);residual=camera-predicted
        if name=='CURRENT':
            for mid,c,p,d in zip(ids,camera,client,residual):
                tables.append(dict(marker_id=mid,layout_u=MARKER_LAYOUT[mid][0],layout_v=MARKER_LAYOUT[mid][1],
                    client_x=p[0],client_y=p[1],camera_x=c[0],camera_y=c[1],dx=d[0],dy=d[1],error=float(np.linalg.norm(d))))
        result=dict(stored_fit=summary(residual),fresh_fixed_model=summary((fresh-predicted).reshape(-1,2)),model=model)
        if 'camera_to_client' in model:result['max_roundtrip_client_px']=float(np.max(np.linalg.norm(project(model['camera_to_client'],predicted)-client,axis=1)))
        else:result['max_roundtrip_client_px']=None
        if name!='CURRENT':
            held=[]
            for j in range(len(ids)):
                train=np.arange(len(ids))!=j;m=fit_model(name,client[train],camera[train],viewport)
                held.append(camera[j]-predict(m,client[j:j+1])[0])
            result['leave_one_marker_out']=summary(held)
            columns=[]
            for u in sorted(set(MARKER_LAYOUT[i][0] for i in ids)):
                test=np.array([MARKER_LAYOUT[i][0]==u for i in ids]);m=fit_model(name,client[~test],camera[~test],viewport)
                columns.append(dict(layout_u=u,**summary(camera[test]-predict(m,client[test]))))
            result['held_out_columns']=columns
            first=fit_model(name,client,fresh[0],viewport)
            result['fresh_frame1_refit']=summary(fresh[0]-predict(first,client))
            result['fresh_frames2_5_fixed_after_frame1_refit']=summary((fresh[1:]-predict(first,client)).reshape(-1,2))
        results[name]=result
        cv2.imwrite(str(output/f'vectors_{name}.png'),vector_field(camera,residual,ids,name+' | stored calibration residuals'))
    for name,key in [('by_id',lambda r:r['marker_id']),('by_layout',lambda r:(r['layout_v'],r['layout_u'])),('by_camera_location',lambda r:(r['camera_y'],r['camera_x']))]:
        with (output/f'residuals_{name}.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(tables[0]));writer.writeheader();writer.writerows(sorted(tables,key=key))
    inspection,montage=inspect_frames(root,report,ids)
    cv2.imwrite(str(output/'observed_marker_montage.png'),montage)
    cv2.imwrite(str(output/'vectors_fresh_CURRENT.png'),vector_field(fresh.mean(0),fresh.mean(0)-predict(current,client),ids,'CURRENT | latest five-frame mean, no alignment offset'))
    repeat=fresh-fresh.mean(0)
    write_new(output/'results.json',dict(models=results,stored_line_bounds=row_column_bounds(ids,camera),
        fresh_line_bounds=[row_column_bounds(ids,f) for f in fresh],
        repeatability=summary(repeat.reshape(-1,2)),mean_fresh_minus_stored=(fresh.mean((0,1))-camera.mean(0)).tolist(),
        per_marker_fresh_shift=[dict(marker_id=i,shift=(fresh[:,j].mean(0)-camera[j]).tolist(),std=fresh[:,j].std(0).tolist()) for j,i in enumerate(ids)],
        marker_inspection=inspection,physical_PASS_claim=False,production_modified=False))
    write_new(output/'provenance.json',dict(command=sys.argv,head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        status=subprocess.check_output(['git','status','--short'],text=True),opencv=cv2.__version__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixed_config='No sweep. Two LS objectives, default MAGSAC at existing 3-client-px threshold, one quadratic residual diagnostic.',
        spaces='oriented full Camera <-> client pixels; layout fractions are NOT surveyed physical Board Space',
        limitations='Same setup repeats and spatial holdouts only; no independent physical plane or lens calibration; polynomial has no validated inverse.'))
    if hashes!={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}:raise ValueError('inputs changed')
    write_new(output/'output_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    print(json.dumps({n:{k:v for k,v in r.items() if k not in ('model','held_out_columns')} for n,r in results.items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.report,args.output)
