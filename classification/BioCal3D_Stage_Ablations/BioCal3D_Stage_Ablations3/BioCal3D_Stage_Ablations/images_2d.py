"""Deterministic colour-preserving orthographic images with explicit mesh mapping."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from study_data import geometry,plane_frame,digest,signature,write_json,write_csv

RENDER_VERSION='orthographic-rgb-1'



def to_grayscale(rgb):
    # ITU-R BT.601 luma on encoded RGB; all three raw channels are identical.
    y=np.rint(np.asarray(rgb,dtype=np.float32)@np.array([.299,.587,.114],np.float32)).clip(0,255).astype(np.uint8)
    return np.repeat(y[...,None],3,axis=2)

def coverage_mask(mask,height,width):
    return np.asarray(Image.fromarray(mask.astype(np.float32)).resize((width,height),Image.Resampling.BOX),dtype=np.float32)

def pool_numpy(features,coverage):
    f=np.asarray(features,dtype=np.float32);w=np.asarray(coverage,dtype=np.float32)
    if f.ndim!=3 or f.shape[1:]!=w.shape or w.sum()<=0:raise ValueError('Spatial features and valid ROI coverage do not match.')
    valid=w>=.05
    if not valid.any():raise ValueError('No feature cells cover the sample.')
    return np.concatenate([(f*w).sum((1,2))/w.sum(),np.where(valid[None],f,-np.inf).max((1,2))]).astype(np.float32)

def _raster_vectorized(pixels,vertices,faces,colors,size,background):
    rgb=np.full((size,size,3),background,np.uint8);depth=np.full((size,size),-np.inf,np.float32)
    ids=np.full((size,size),-1,np.int32);bary=np.zeros((size,size,3),np.float32)
    for k,face in enumerate(faces):
        t=pixels[face];xmin=max(0,int(np.ceil(t[:,0].min())));xmax=min(size-1,int(np.floor(t[:,0].max())))
        ymin=max(0,int(np.ceil(t[:,1].min())));ymax=min(size-1,int(np.floor(t[:,1].max())))
        if xmin>xmax or ymin>ymax:continue
        x0,y0=t[0];x1,y1=t[1];x2,y2=t[2];den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(den)<1e-12:continue
        yy,xx=np.mgrid[ymin:ymax+1,xmin:xmax+1]
        a=((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/den
        b=((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/den;c=1-a-b
        inside=(a>=-1e-7)&(b>=-1e-7)&(c>=-1e-7)
        z=a*vertices[face[0],2]+b*vertices[face[1],2]+c*vertices[face[2],2]
        hit=inside&(z>depth[ymin:ymax+1,xmin:xmax+1])
        if not hit.any():continue
        iy,ix=np.where(hit);iy+=ymin;ix+=xmin;weights=np.stack([a[hit],b[hit],c[hit]],1)
        weights=np.maximum(weights,0);weights/=weights.sum(1,keepdims=True)
        rgb[iy,ix]=np.rint(weights@colors[face]*255).clip(0,255).astype(np.uint8)
        depth[iy,ix]=z[hit];ids[iy,ix]=k;bary[iy,ix]=weights
    return rgb,ids,bary

def _raster_loops(pixels,vertices,faces,colors,size,background):
    """Optional numba implementation; identical barycentric/z-buffer convention."""
    rgb=np.full((size,size,3),background,np.uint8);depth=np.full((size,size),-np.inf,np.float32)
    ids=np.full((size,size),-1,np.int32);bary=np.zeros((size,size,3),np.float32)
    for k in range(len(faces)):
        f0,f1,f2=faces[k];x0,y0=pixels[f0];x1,y1=pixels[f1];x2,y2=pixels[f2]
        den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(den)<1e-12:continue
        xmin=max(0,int(np.ceil(min(x0,x1,x2))));xmax=min(size-1,int(np.floor(max(x0,x1,x2))))
        ymin=max(0,int(np.ceil(min(y0,y1,y2))));ymax=min(size-1,int(np.floor(max(y0,y1,y2))))
        for y in range(ymin,ymax+1):
            for x in range(xmin,xmax+1):
                a=((y1-y2)*(x-x2)+(x2-x1)*(y-y2))/den;b=((y2-y0)*(x-x2)+(x0-x2)*(y-y2))/den;c=1-a-b
                if min(a,b,c)<-1e-7:continue
                z=a*vertices[f0,2]+b*vertices[f1,2]+c*vertices[f2,2]
                if z<=depth[y,x]:continue
                a=max(a,0);b=max(b,0);c=max(c,0);total=a+b+c;a/=total;b/=total;c/=total
                for ch in range(3):rgb[y,x,ch]=int(min(255,max(0,np.rint(255*(a*colors[f0,ch]+b*colors[f1,ch]+c*colors[f2,ch])))))
                depth[y,x]=z;ids[y,x]=k;bary[y,x,0]=a;bary[y,x,1]=b;bary[y,x,2]=c
    return rgb,ids,bary

_accelerated=None
def rasterize(pixels,vertices,faces,colors,size,background,accelerate=True):
    global _accelerated
    if accelerate:
        try:
            if _accelerated is None:
                from numba import njit
                _accelerated=njit(cache=True)(_raster_loops)
            return _accelerated(pixels,vertices,faces,colors,size,background)
        except ImportError:pass
    return _raster_vectorized(pixels,vertices,faces,colors,size,background)

def render(row,size,field_mm,background=127,accelerate=True):
    vertices,faces,normals,colors,area,fn=geometry(row)
    if colors is None or not colors.max()>0:raise ValueError(f'{row["scan_id"]}: genuine vertex RGB missing. Repair prepared geometry; black colour substitution is not allowed.')
    center,_=plane_frame(vertices,faces,area)
    # Fixed physical field for every scan. No per-sample zoom or artificial lighting.
    uv=np.column_stack([(vertices[:,0]-center[0])*size/field_mm+size/2-.5,
                        (center[1]-vertices[:,1])*size/field_mm+size/2-.5])
    if uv.min()<-.50001 or uv.max()>size-.49999:raise ValueError(f'{row["scan_id"]}: geometry exceeds {field_mm} mm field. Choose a larger --field-mm for the whole experiment; no silent cropping.')
    rgb,ids,bary=rasterize(uv,vertices,faces,colors,size,background,accelerate);valid=ids>=0
    if valid.sum()<32:raise ValueError('Too little projected sample area; check mm units and canonical orientation.')
    yy,xx=np.where(valid);world=(vertices[faces[ids[valid]]]*bary[valid,:,None]).sum(1)
    ux=(world[:,0]-center[0])*size/field_mm+size/2-.5;vy=(center[1]-world[:,1])*size/field_mm+size/2-.5
    error=float(np.max(np.hypot(ux-xx,vy-yy)))
    if error>.002:raise ValueError(f'Pixel-to-mesh reprojection failed: {error} pixels.')
    return rgb,ids,bary,{'center_mm':center.tolist(),'field_mm':field_mm,'size':size,'background':background,
                         'valid_pixels':int(valid.sum()),'max_reprojection_error_pixels':error,
                         'coordinate_convention':'x right, world y up/image row down, nearest visible surface is maximum world z',
                         'source_geometry_sha256':row['geometry_sha256']}

def prepare_images(rows,output,size=448,field_mm=16.,background=127,accelerate=True):
    output=Path(output);records=[]
    implementation=signature([digest(__file__),digest(Path(__file__).with_name('study_data.py'))])
    for i,row in enumerate(rows,1):
        folder=output/'images'/row['scanner']/row['sample_name'];folder.mkdir(parents=True,exist_ok=True)
        spec=signature([RENDER_VERSION,implementation,row['geometry_sha256'],size,field_mm,background])
        meta_path=folder/'metadata.json';paths=[folder/n for n in ['rgb.png','gray.png','valid_mask.png','maps.npz']]
        reuse=meta_path.exists() and all(p.is_file() for p in paths)
        if reuse:
            meta=json.loads(meta_path.read_text());reuse=meta.get('signature')==spec
            if reuse:reuse=all(meta.get('file_hashes',{}).get(p.name)==digest(p) for p in paths)
        if not reuse:
            rgb,ids,bary,meta=render(row,size,field_mm,background,accelerate)
            Image.fromarray(rgb).save(folder/'rgb.png');Image.fromarray(to_grayscale(rgb)).save(folder/'gray.png')
            Image.fromarray(np.uint8(ids>=0)*255).save(folder/'valid_mask.png')
            np.savez_compressed(folder/'maps.npz',face_index=ids,barycentric=bary,center_mm=np.array(meta['center_mm']),field_mm=np.array(field_mm),source_geometry_sha256=np.array(row['geometry_sha256']))
            meta.update(signature=spec,render_version=RENDER_VERSION,file_hashes={p.name:digest(p) for p in paths})
            write_json(meta_path,meta)
        records.append(dict(row,image_dir=str(folder),image_signature=spec,image_rgb_sha256=meta['file_hashes']['rgb.png'],
                            image_gray_sha256=meta['file_hashes']['gray.png'],valid_mask_sha256=meta['file_hashes']['valid_mask.png']))
        if i==1 or i%16==0 or i==len(rows):print(f'Images {i}/{len(rows)}: {row["scanner"]} {row["sample_name"]}',flush=True)
    write_csv(output/'image_manifest.csv',records)
    return records

def read_image(row,mode):
    if mode not in ['rgb','gray']:raise ValueError(mode)
    rgb=np.array(Image.open(Path(row['image_dir'])/(mode+'.png')).convert('RGB'))
    mask=np.array(Image.open(Path(row['image_dir'])/'valid_mask.png'))>0
    return rgb,mask
