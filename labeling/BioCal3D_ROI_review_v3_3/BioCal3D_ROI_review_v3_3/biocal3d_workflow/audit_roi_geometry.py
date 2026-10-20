"""Audit current ROIs without modifying geometry or annotations."""
from pathlib import Path
import argparse,csv,html,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'annotator'))
from mesh_io import AnnotationStore,discover
from roi_review_store import load_review_context,read_reviewed
from roi_geometry import select_surface,face_geometry


def audit(root,output,annotations=None,limit=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    store=AnnotationStore(root,annotations or root/'annotations','geometry_audit')
    paths=discover(root)
    if limit:paths=paths[:limit]
    rows=[];cards=[]
    for i,path in enumerate(paths):
        try:
            mesh=read_reviewed(store,path);base,mask,s,message,_=load_review_context(store,mesh)
            proposed,metrics=select_surface(base.vertices,base.faces,s)
            # Only display the existing crop; proposed differences are suggestions, not ground truth.
            centers,normals,areas=face_geometry(base.vertices,base.faces)
            points=base.vertices[base.faces[mask]];selected=np.flatnonzero(mask)
            steep=np.abs(normals[mask]@s.frame()[3])<np.cos(np.deg2rad(s.max_angle))
            current_area=float(areas[mask].sum());suspect=float(areas[mask&~proposed].sum())
            row=dict(source=str(path.relative_to(root)),review_status=mesh.review_status,version=mesh.review_version,
                     current_faces=int(mask.sum()),proposed_faces=int(proposed.sum()),current_area_mm2=current_area,
                     suggested_excluded_area_mm2=suspect,steep_area_fraction=float(areas[mask][steep].sum()/current_area) if current_area else 0,
                     z_span_mm=float(np.ptp(points[:,:,2])),context_mode='full' if 'Full original' in message else 'local',
                     flags=';'.join(metrics['flags']),error='')
            fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
            color=np.where(proposed[selected,None],np.array([.2,.7,.5]),np.array([.9,.3,.3]))
            for ax,axes,title in [(axs[0],[0,1],'Top: X / Y'),(axs[1],[0,2],'Side: X / Z')]:
                ax.add_collection(PolyCollection(points[:,:,axes],facecolors=color,edgecolors='none',rasterized=True))
                ax.autoscale();ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('mm');ax.set_ylabel('mm')
            fig.suptitle(str(path.relative_to(root))+'\nRed: existing crop outside automatic proposal; inspect before accepting',fontsize=10)
            image=f'{i:05d}.png';fig.savefig(output/image,dpi=120);plt.close(fig)
            cards.append(f'<article><h3>{html.escape(row["source"])}</h3><p>{html.escape(row["review_status"])} • {html.escape(row["flags"])}</p><img src="{image}" loading="lazy"></article>')
        except Exception as e:row=dict(source=str(path.relative_to(root)),error=str(e));cards.append(f'<p>{html.escape(str(path))}: {html.escape(str(e))}</p>')
        rows.append(row);print(f'[{i+1}/{len(paths)}] {path.parent.name}',flush=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (output/'geometry_audit.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (output/'geometry_audit.html').write_text('<!doctype html><meta charset="utf-8"><title>BioCal3D geometry audit</title><style>body{font:15px system-ui;margin:30px;background:#f3f5f8}article{background:white;padding:18px;margin:20px 0;border-radius:12px}img{max-width:100%}</style><h1>BioCal3D geometry audit</h1><p>Automatic proposals require visual review. No inputs or labels were modified.</p>'+''.join(cards),encoding='utf-8')
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--annotations',type=Path);p.add_argument('--limit',type=int);a=p.parse_args();audit(a.root,a.output,a.annotations,a.limit)
