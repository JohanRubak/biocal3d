"""Project X-AnyLabeling polygons onto the mesh stored in BioCal3D maps.npz.
Install: python -m pip install numpy pillow matplotlib plotly
Run on one sample folder containing maps.npz, metadata.json and rgb.json:
  python biocal3d_project_labels_3d.py --sample-dir "C:/path/to/sample" --open
Optional: --maps custom.npz --annotation custom.json --metadata custom.json
--reviewed-background: explicitly certify the whole ROI was checked; untouched
valid pixels become no-plaque. Otherwise untouched pixels remain unknown.

Writes interactive 3D HTML, 2D QC, labelled mesh arrays, labels.npz compatible
with the ablation script, and an ASCII PLY with separate face label properties.
Face labels use surface-area-weighted pixel voting; vertex labels use
barycentric weighted voting. These are discretizations, not new ground truth.
Unobserved elements remain unknown. Exact labelled 3D pixel locations are also
saved, preserving fine boundaries that cannot be represented by coarse faces.
Displayed RGB is reconstructed from projected pixels; original vertex RGB is
not stored in maps.npz. Original vertices and face indices are never changed.
Test identity is retained. No model fitting or change to split assignments.
"""
from pathlib import Path
import argparse,json,webbrowser
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def raster_labels(annotation,shape,valid,reviewed=False):
    data=json.loads(Path(annotation).read_text())
    h,w=shape
    if (data.get('imageHeight'),data.get('imageWidth'))!=(h,w):
        raise ValueError('Annotation image dimensions differ from maps; do not resize.')
    labels=np.full(shape,-1,dtype=np.int8)
    if reviewed: labels[valid]=0
    groups={1:[],0:[],-1:[]};counts={}
    names={'plaque':1,'no plaque':0,'no_plaque':0,'background':0,'clean':0,'uncertain':-1,'ignore':-1}
    for item in data['shapes']:
        name=item['label'].strip().lower()
        if name not in names: raise ValueError('Unknown annotation label: '+name)
        target=names[name];groups[target].append(item);counts[name]=counts.get(name,0)+1
    # Explicit no-plaque overrides plaque; uncertain overrides all other classes.
    for target in [1,0,-1]:
        for item in groups[target]:
            canvas=Image.new('L',(w,h));draw=ImageDraw.Draw(canvas)
            pts=[tuple(map(float,p)) for p in item['points']]
            kind=item.get('shape_type','polygon')
            if not np.isfinite(np.asarray(pts)).all(): raise ValueError('Nonfinite polygon coordinates')
            if kind=='polygon' and len(pts)>=3: draw.polygon(pts,fill=1)
            elif kind=='rectangle' and len(pts)==2: draw.rectangle(pts,fill=1)
            else: raise ValueError(f'Unsupported shape {kind}; export polygon annotations or a mask first.')
            labels[(np.asarray(canvas)>0)&valid]=target
    return labels,data,counts


def votes_to_labels(votes):
    total=votes.sum(1);fraction=np.divide(votes[:,1],total,out=np.full(len(total),np.nan),where=total>0)
    label=np.full(len(total),-1,dtype=np.int8); label[total>0]=(fraction[total>0]>=.5).astype(np.int8)
    return label,fraction,total


def project(z,labels):
    vertices=z['vertices'];faces=z['faces'];tid=z['triangle_id'];bary=z['barycentric'];valid=z['valid'].astype(bool)
    if faces.ndim!=2 or faces.shape[1]!=3 or faces.min()<0 or faces.max()>=len(vertices): raise ValueError('Invalid faces')
    if tid.shape!=labels.shape or bary.shape!=(*labels.shape,3): raise ValueError('Invalid raster mapping shapes')
    ids=tid[valid];weights=bary[valid].astype(float)
    if ids.min()<0 or ids.max()>=len(faces) or not np.isfinite(weights).all() or not np.allclose(weights.sum(1),1,atol=1e-4): raise ValueError('Invalid triangle/barycentric correspondence')
    if weights.min() < -1e-4: raise ValueError('Negative barycentric weights')
    points=np.einsum('ni,nij->nj',weights,vertices[faces[ids]])
    lab=labels[valid];area=z['pixel_surface_area_mm2'][valid].astype(float)
    if not np.isfinite(area).all() or (area<=0).any(): raise ValueError('Invalid pixel areas')
    face_votes=np.zeros((len(faces),2));vertex_votes=np.zeros((len(vertices),2))
    for cls in [0,1]:
        keep=lab==cls
        np.add.at(face_votes[:,cls],ids[keep],area[keep])
        for j in range(3): np.add.at(vertex_votes[:,cls],faces[ids[keep],j],area[keep]*weights[keep,j])
    fl,ff,ft=votes_to_labels(face_votes);vl,vf,vt=votes_to_labels(vertex_votes)
    # Approximate RGB from observed surface pixels without replacing source colour.
    rgb=z['rgb'][valid];fv=np.zeros((len(faces),3));fa=np.zeros(len(faces))
    np.add.at(fa,ids,area)
    for j in range(3): np.add.at(fv[:,j],ids,area*rgb[:,j])
    face_rgb=np.divide(fv,fa[:,None],out=np.full_like(fv,.5),where=fa[:,None]>0)
    observed=np.zeros(len(faces));np.add.at(observed,ids,area)
    unknown=np.zeros(len(faces));np.add.at(unknown,ids[lab<0],area[lab<0])
    return dict(vertices=vertices,faces=faces,face_labels=fl,face_plaque_fraction=ff,
        face_known_area_mm2=ft,face_observed_area_mm2=observed,face_unknown_area_mm2=unknown,
        vertex_labels=vl,vertex_plaque_fraction=vf,vertex_known_weight=vt,
        face_rgb_reconstructed=face_rgb,pixel_xyz=points,pixel_labels=lab,
        pixel_triangle_id=ids,pixel_barycentric=weights,pixel_area_mm2=area)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sample-dir',type=Path,default=Path('.'))
    p.add_argument('--maps',type=Path);p.add_argument('--annotation',type=Path);p.add_argument('--metadata',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--reviewed-background',action='store_true');p.add_argument('--open',action='store_true')
    a=p.parse_args();root=a.sample_dir
    mapfile=a.maps or root/'maps.npz';annotation=a.annotation or root/'rgb.json';meta=a.metadata or root/'metadata.json'
    with np.load(mapfile,allow_pickle=False) as bundle: z={k:bundle[k] for k in bundle.files}
    identity=json.loads(meta.read_text());valid=z['valid'].astype(bool)
    labels,data,counts=raster_labels(annotation,valid.shape,valid,a.reviewed_background)
    result=project(z,labels);out=a.output or root/'_labels_3d';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'labelled_geometry.npz',**result)
    np.savez_compressed(out/'labels.npz',labels=labels,verified=np.array(bool(data.get('checked',False) or a.reviewed_background)),
        **{k:np.array(str(identity[k])) for k in ['split','scanner','sample_name','physical_id']})
    # PNG value 255 encodes unknown, NPZ uses -1.
    Image.fromarray(np.where(labels<0,255,labels).astype(np.uint8)).save(out/'labels.png')
    rgb=z['rgb'];overlay=rgb.copy();palette=np.array([[.12,.65,1],[1,.15,.05]])
    for cls in [0,1]: overlay[labels==cls]=.5*rgb[labels==cls]+.5*palette[cls]
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    axes[0].imshow(rgb);axes[0].set_title('Original projected RGB')
    axes[1].imshow(overlay);axes[1].set_title('Plaque red; no-plaque blue; unknown unchanged')
    axes[2].imshow(labels,vmin=-1,vmax=1,cmap=matplotlib.colors.ListedColormap(['gray','dodgerblue','red']));axes[2].set_title('Unknown gray / no-plaque blue / plaque red')
    for ax in axes: ax.axis('off')
    fig.tight_layout();fig.savefig(out/'qc_2d.png',dpi=170);plt.close(fig)
    v=result['vertices'];f=result['faces'];label=result['face_labels']
    colors=['#999999' if x<0 else ('#2299ff' if x==0 else '#ff3311') for x in label]
    fig=make_subplots(rows=1,cols=3,specs=[[{'type':'scene'}]*3],subplot_titles=['Surface RGB (reconstructed)','Triangle labels (majority known-pixel vote)','Exact plaque pixels on 3D surface'])
    base=dict(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],flatshading=True,lighting=dict(ambient=1,diffuse=0,specular=0))
    fig.add_trace(go.Mesh3d(**base,facecolor=['rgb(%d,%d,%d)'%tuple(c) for c in np.uint8(np.clip(result['face_rgb_reconstructed'],0,1)*255)],name='RGB'),row=1,col=1)
    fig.add_trace(go.Mesh3d(**base,facecolor=colors,name='Face labels'),row=1,col=2)
    fig.add_trace(go.Mesh3d(**base,color='lightgray',opacity=.35,name='Mesh'),row=1,col=3)
    pts=result['pixel_xyz'][result['pixel_labels']==1]
    # Deterministic display subsampling only; all projected points retained in NPZ.
    pts=pts[::max(1,int(np.ceil(len(pts)/50000)))]
    pts=pts+z['basis'][:,2]*.002
    fig.add_trace(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode='markers',marker=dict(size=2,color='red'),name='Plaque pixels'),row=1,col=3)
    normal=z['basis'][:,2]*2.2;up=z['basis'][:,1]
    camera=dict(eye=dict(zip(['x','y','z'],normal)),up=dict(zip(['x','y','z'],up)))
    for i in range(1,4): fig.update_scenes(aspectmode='data',camera=camera,row=1,col=i)
    fig.update_layout(title=f"{identity['scanner']} {identity['sample_name']} | {identity['split']} | unknown surfaces are gray",height=650)
    fig.write_html(out/'labels_3d.html',include_plotlyjs=True)
    with (out/'labelled_mesh.ply').open('w') as stream:
        stream.write(f'ply\nformat ascii 1.0\nelement vertex {len(v)}\nproperty double x\nproperty double y\nproperty double z\nelement face {len(f)}\nproperty list uchar int vertex_indices\nproperty int label\nproperty float plaque_fraction\nend_header\n')
        for point in v: stream.write(' '.join(map(str,point))+'\n')
        for face,l,frac in zip(f,label,result['face_plaque_fraction']): stream.write('3 '+' '.join(map(str,face))+f' {l} {frac}\n')
    # Validate 3D locations by reprojecting using the stored plane coordinates.
    local=(result['pixel_xyz']-z['origin'])@z['basis'];h,w=valid.shape;fov=float(z['fov_mm'])
    xx=(local[:,0]/fov+.5)*w-.5;yy=(.5-local[:,1]/fov)*h-.5
    rr,cc=np.nonzero(valid);error=np.sqrt((xx-cc)**2+(yy-rr)**2)
    summary={**identity,'annotation_checked':bool(data.get('checked',False)), 'reviewed_background':a.reviewed_background,
        'annotation_shapes':counts,'plaque_pixels':int((labels==1).sum()),'no_plaque_pixels':int((labels==0).sum()),
        'unknown_valid_pixels':int(((labels<0)&valid).sum()),'unobserved_faces':int((result['face_observed_area_mm2']==0).sum()),
        'faces_with_unknown_pixels':int((result['face_unknown_area_mm2']>0).sum()),
        'max_reprojection_error_pixels':float(error.max()),'mean_reprojection_error_pixels':float(error.mean()),
        'note':'Face/vertex labels vote among known pixels only; consult known/unknown coverage arrays before training. No unseen surfaces inferred.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));print('Outputs:',out.resolve())
    if a.open:webbrowser.open((out/'labels_3d.html').resolve().as_uri())

if __name__=='__main__':main()
