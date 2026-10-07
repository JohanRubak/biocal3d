"""Class-specific Grad-CAM over faces/points, feature ablation, and exact ROI overlays."""
from pathlib import Path
import html, json
import numpy as np
import torch
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data3d import (load_index,validate_geometry,geometry,inputs,to_device,digest,save_json,write_csv)
from models3d import build


def select_scans(rows,count):
    """Deterministic round robin across (class, scanner); no selection by heatmap quality."""
    buckets={}
    for r in sorted(rows,key=lambda x:x['scan_id']): buckets.setdefault((r['label'],r['scanner']),[]).append(r)
    selected=[]
    while buckets and len(selected)<count:
        for key in sorted(list(buckets)):
            if len(selected)>=count: break
            selected.append(buckets[key].pop(0))
            if not buckets[key]: del buckets[key]
    return selected


def interpolate(source,score,target):
    """Three-nearest inverse-distance interpolation, only for display."""
    d,idx=cKDTree(source).query(target,k=min(3,len(source)))
    if d.ndim==1: return score[idx].astype('float32')
    w=1/np.maximum(d,1e-8)**2
    return ((score[idx]*w).sum(1)/w.sum(1)).astype('float32')


def colour_values(values):
    colors=plt.get_cmap('inferno')(np.nan_to_num(values,nan=0))[:,:3]
    colors[~np.isfinite(values)]=[.65,.65,.65]
    return ['rgb(%d,%d,%d)'%tuple(np.rint(c*255).astype(int)) for c in colors]


def export_vtp(path,v,f,cam,raw_cam,ids):
    """Plain VTK XML, no VTK/PyVista dependency. NaN = unsampled face."""
    def arr(x): return ' '.join(map(str,np.asarray(x).ravel()))
    text='<?xml version="1.0"?><VTKFile type="PolyData" version="0.1" byte_order="LittleEndian"><PolyData><Piece NumberOfPoints="%d" NumberOfPolys="%d">'%(len(v),len(f))
    text+='<Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">'+arr(v)+'</DataArray></Points><Polys><DataArray type="Int64" Name="connectivity" format="ascii">'+arr(f)+'</DataArray><DataArray type="Int64" Name="offsets" format="ascii">'+arr(np.arange(1,len(f)+1)*3)+'</DataArray></Polys><CellData Scalars="CAM">'
    for name,x,dtype in [('CAM',cam,'Float32'),('signed_CAM',raw_cam,'Float32'),('original_face_id',ids,'Int64')]:
        text+=f'<DataArray type="{dtype}" Name="{name}" format="ascii">'+arr(x)+'</DataArray>'
    text+='</CellData></Piece></PolyData></VTKFile>'
    path.write_text(text,encoding='utf-8')


def ablation_test(net,batch,cam,target,baseline,repeats=10):
    n=len(cam); count=max(1,int(np.ceil(.1*n)))
    top=np.argsort(cam)[-count:]
    def confidence(indices):
        mask=torch.zeros(1,n,dtype=torch.bool,device=batch['xyz'].device); mask[0,indices]=True
        with torch.no_grad(): return float(net(batch,ablate=mask).softmax(-1)[0,target])
    rng=np.random.default_rng(20261008)
    top_conf=confidence(top)
    random_conf=[confidence(rng.choice(n,count,replace=False)) for _ in range(repeats)]
    return dict(method='zero local feature vectors before pooling/global abstraction; positions retained',
        fraction=count/n,baseline_probability=baseline,top_cam_probability=top_conf,
        top_cam_probability_drop=baseline-top_conf,
        random_probability_drop_mean=baseline-float(np.mean(random_conf)),random_probability_drop_std=float(np.std(random_conf)),
        random_repeats=repeats,positive_cam_available=bool(np.max(cam)>0))


def explain_checkpoint(checkpoint,index,output=None,scan_ids=None,count=6,target='predicted',device='auto'):
    checkpoint=Path(checkpoint).resolve(); index=Path(index).resolve()
    if device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'
    dev=torch.device(device)
    saved=torch.load(checkpoint,map_location=dev,weights_only=False); cfg=saved['config']; classes=cfg['classes']
    if digest(index)!=cfg['index_hash']: raise ValueError('Explanation index differs from training snapshot. Use the exact frozen index.')
    labels_csv=checkpoint.parent.parent/'labels.csv' if cfg.get('labels_csv') else None
    rows,_,_=load_index(index,labels_csv,classes)
    import hashlib
    targets_hash=hashlib.sha256(json.dumps([(r['scan_id'],r['label']) for r in rows],sort_keys=True).encode()).hexdigest()
    if targets_hash!=cfg['targets_hash']: raise ValueError('Classification labels changed since training.')
    test_ids=None
    test_path=checkpoint.parent/'test_predictions.csv'
    if test_path.exists():
        import csv
        test_ids={r['scan_id'] for r in csv.DictReader(open(test_path,encoding='utf-8-sig'))}
    if scan_ids:
        mapping={r['scan_id']:r for r in rows}
        missing=set(scan_ids)-set(mapping)
        if missing: raise ValueError('Unknown scan IDs: '+str(missing))
        chosen=[mapping[s] for s in scan_ids]
    else: chosen=select_scans([r for r in rows if test_ids is None or r['scan_id'] in test_ids],count)
    if not chosen: raise ValueError('No scans selected.')
    if target not in ['predicted','true'] + classes: raise ValueError('Unknown target class: '+target)
    validate_geometry(chosen,cfg['colour'])
    out=Path(output).resolve() if output else checkpoint.parent/'explanations'; out.mkdir(parents=True,exist_ok=True)
    net=build(cfg['model'],len(classes),cfg['colour'],radius=cfg['radius']).to(dev); net.load_state_dict(saved['state_dict']); net.eval()
    summaries=[]; pages=[]
    for row in chosen:
        folder=out/row['scan_id']; folder.mkdir(parents=True,exist_ok=True)
        batch,local=inputs(row,cfg['model'],cfg['elements'],cfg['colour'],cfg['scale_mm'],cfg['sampling_seed'])
        batch=to_device(batch,dev); net.zero_grad(set_to_none=True)
        logits,features,positions=net(batch,explain=True)
        if not torch.isfinite(logits).all() or not torch.isfinite(features).all(): raise ValueError('Nonfinite model output.')
        p=logits.softmax(-1)[0].detach().cpu().numpy(); predicted=int(p.argmax())
        ti=predicted if target=='predicted' else (row['target'] if target=='true' else classes.index(target))
        logits[0,ti].backward()
        grad=features.grad
        if grad is None or not torch.isfinite(grad).all(): raise ValueError('Missing/nonfinite attribution gradients.')
        alpha=grad.mean(-1,keepdim=True)
        signed=(alpha*features).sum(1)[0].detach().cpu().numpy()
        positive=np.maximum(signed,0); cam=positive/max(float(positive.max()),1e-12)
        pos=positions[0].detach().cpu().numpy()*cfg['scale_mm']
        v,f,_,rgb,area=geometry(row)
        if cfg['model']=='tsgcnet':
            face_cam=np.full(len(f),np.nan,'float32'); face_signed=face_cam.copy()
            face_cam[local['face_ids']]=cam; face_signed[local['face_ids']]=signed
            support='Direct Grad-CAM on sampled faces. Grey faces were not sampled.'
        else:
            centers=v[f].mean(1)
            face_cam=interpolate(pos,cam,centers); face_signed=interpolate(pos,signed,centers)
            support='CAM at the final local point stage, interpolated to faces for display. Interpolation is not measured face-level attribution.'
        np.savez_compressed(folder/'attribution.npz',positions_mm=pos,cam=cam,signed_cam=signed,
            face_cam=face_cam,face_signed_cam=face_signed,face_ids=np.arange(len(f)),sampled_face_ids=local['face_ids'],
            vertices=v,faces=f,target_class=classes[ti],probabilities=p,geometry_sha256=row['geometry_sha256'],
            checkpoint_sha256=digest(checkpoint),interpolated=cfg['model']=='pointnext_s')
        export_vtp(folder/'attribution.vtp',v,f,face_cam,face_signed,np.arange(len(f)))
        ablation=ablation_test(net,batch,cam,ti,float(p[ti])); save_json(folder/'feature_ablation.json',ablation)
        maps_path=Path(row['folder'])/'maps.npz'
        overlay_written=False
        if maps_path.is_file():
            if not row.get('maps_sha256') or digest(maps_path)!=row['maps_sha256']: raise ValueError('Mapping provenance mismatch: '+str(maps_path))
            with np.load(maps_path,allow_pickle=False) as maps:
                if not np.array_equal(maps['faces'],f) or not np.allclose(maps['vertices'],v,atol=1e-6,rtol=0):
                    raise ValueError('Mapping/geometry index mismatch.')
                tid=maps['triangle_id']; valid=maps['valid'].astype(bool)
                if np.any(valid & ((tid<0)|(tid>=len(f)))): raise ValueError('Invalid pixel triangle ID.')
                pixel_cam=np.full(tid.shape,np.nan,'float32'); pixel_cam[valid]=face_cam[tid[valid]]
                np.savez_compressed(folder/'pixel_cam.npz',cam=pixel_cam,valid=valid,triangle_id=tid,interpolated=cfg['model']=='pointnext_s')
            rgb_path=Path(row['folder'])/'rgb.png'
            if rgb_path.is_file():
                if not row.get('rgb_sha256') or digest(rgb_path)!=row['rgb_sha256']: raise ValueError('RGB provenance mismatch.')
                original=np.asarray(Image.open(rgb_path).convert('RGB'))/255.
                if original.shape[:2]!=pixel_cam.shape: raise ValueError('RGB/map dimensions mismatch.')
                colors=plt.get_cmap('inferno')(np.nan_to_num(pixel_cam,nan=0))[...,:3]
                overlay=original.copy(); mask=np.isfinite(pixel_cam)
                overlay[mask]=.45*original[mask]+.55*colors[mask]
                fig,axes=plt.subplots(1,3,figsize=(12,4))
                axes[0].imshow(original); axes[0].set_title('Original RGB')
                image=axes[1].imshow(pixel_cam,cmap='inferno',vmin=0,vmax=1); axes[1].set_title('Relative positive class evidence')
                axes[2].imshow(overlay); axes[2].set_title(f'{classes[ti]} evidence overlay')
                for a in axes: a.axis('off')
                fig.colorbar(image,ax=axes[1],fraction=.046); fig.suptitle(row['scan_id']+' · '+support,fontsize=8)
                fig.tight_layout(); fig.savefig(folder/'overlay.png',dpi=160); plt.close(fig); overlay_written=True
        fig=make_subplots(rows=1,cols=2,specs=[[{'type':'scene'},{'type':'scene'}]],subplot_titles=['Original surface colour','Class-specific evidence'])
        mesh=dict(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],flatshading=True,
                  lighting=dict(ambient=.8,diffuse=.2,specular=0),hoverinfo='skip')
        fig.add_trace(go.Mesh3d(**mesh,vertexcolor=colour_values_rgb(rgb)),row=1,col=1)
        fig.add_trace(go.Mesh3d(**mesh,facecolor=colour_values(face_cam)),row=1,col=2)
        if cfg['model']=='pointnext_s':
            fig.add_trace(go.Scatter3d(x=pos[:,0],y=pos[:,1],z=pos[:,2],mode='markers',
                marker=dict(size=3,color=cam,cmin=0,cmax=1,colorscale='Inferno',colorbar=dict(title='CAM')),
                name='Direct coarse point CAM'),row=1,col=2)
        settings=dict(aspectmode='data',xaxis_title='X (mm)',yaxis_title='Y (mm)',zaxis_title='Z (mm)')
        fig.update_layout(scene=settings,scene2=settings,height=650,title=f"{row['scan_id']} | true: {row['label']} | predicted: {classes[predicted]} ({p[predicted]:.3f}) | target: {classes[ti]}")
        fig.write_html(folder/'surface.html',include_plotlyjs=True,full_html=True)
        summary=dict(scan_id=row['scan_id'],physical_id=row['physical_id'],scanner=row['scanner'],true_label=row['label'],
            predicted_label=classes[predicted],target=classes[ti],target_probability=float(p[ti]),
            local_elements=len(cam),positive_cam_available=bool(positive.max()>0),support=support,
            explained_test_scan=test_ids is not None and row['scan_id'] in test_ids,
            top_feature_probability_drop=ablation['top_cam_probability_drop'],random_feature_probability_drop=ablation['random_probability_drop_mean'])
        summaries.append(summary); save_json(folder/'summary.json',summary)
        pages.append(f'<h2>{html.escape(row["scan_id"])}</h2><p>{html.escape(support)}</p><p>True: {html.escape(row["label"])}; predicted: {html.escape(classes[predicted])}; target: {html.escape(classes[ti])} ({p[ti]:.3f})</p><a href="{row["scan_id"]}/surface.html">Rotate the 3D surface</a> · <a href="{row["scan_id"]}/attribution.vtp">VTK surface</a> · <a href="{row["scan_id"]}/feature_ablation.json">Feature ablation check</a>'+ (f'<br><img width="1100" src="{row["scan_id"]}/overlay.png">' if overlay_written else ''))
        print('Explained:',row['scan_id'],cfg['model'],flush=True)
    write_csv(out/'explanation_summary.csv',summaries)
    page='<!doctype html><meta charset="utf-8"><title>BioCal3D 3D evidence</title><style>body{font:16px system-ui;margin:30px}img{max-width:100%}</style><h1>Class-specific 3D evidence</h1><p>Grad-CAM is computed from local feature gradients for the selected class logit. Warm colours indicate stronger positive evidence within this scan. Colour magnitudes are not comparable across scans/models. A zero map means no positive evidence at this layer. These maps are exploratory explanations, not plaque annotations or calibrated uncertainty. Feature ablation tests sensitivity of internal features, not causal effects of deleting physical surface regions.</p>'+''.join(pages)
    (out/'index.html').write_text(page,encoding='utf-8')
    return summaries


def colour_values_rgb(rgb):
    return ['rgb(%d,%d,%d)'%tuple(np.rint(np.clip(c,0,1)*255).astype(int)) for c in rgb]
