"""Export completed saved annotations as matched original-color PLY + label-color PLY + NPZ bundles."""
import argparse,csv,json,re
from pathlib import Path
import numpy as np
if __package__:
    from .mesh_io import AnnotationStore,load_mesh,export_ply,export_label_colors,atomic_npz
else:
    from mesh_io import AnnotationStore,load_mesh,export_ply,export_label_colors,atomic_npz


def export_annotations(root,annotations,destination,annotators=None,sources=None,include_drafts=False):
    root=Path(root).resolve();annotations=Path(annotations).resolve();destination=Path(destination).resolve()
    if destination==annotations or destination.is_relative_to(annotations):raise ValueError('Choose an export directory outside the working annotations directory.')
    names=annotators or sorted(p.name for p in annotations.iterdir() if p.is_dir() and p.name!='_sample_orientation' and re.fullmatch(r'[A-Za-z0-9_-]{1,64}',p.name))
    allowed={Path(s).resolve() for s in sources} if sources is not None else None
    rows=[];report=dict(exported=0,skipped=0,failed=0)
    destination.mkdir(parents=True,exist_ok=True)
    for name in names:
        store=AnnotationStore(root,annotations,name)
        for source in sorted((annotations/name).rglob('*.labels.npz')):
            try:
                with np.load(source,allow_pickle=False) as d:
                    data={k:d[k].copy() for k in d.files};relative=Path(str(d['source_relative'].item()));completed=bool(d['completed'])
                mesh_path=(root/relative).resolve()
                if not mesh_path.is_relative_to(root):raise ValueError('Source path escapes dataset.')
                if allowed is not None and mesh_path not in allowed:continue
                if not completed and not include_drafts:report['skipped']+=1;continue
                mesh=load_mesh(mesh_path);labels,_,_=store.load(mesh)
                texture=str(data.get('texture_path',''))
                if texture and Path(texture).is_file():mesh.texture=Path(texture)
                target=destination/name/relative.parent/(relative.name+'.labeled.ply')
                export_ply(mesh,labels,target)
                preview=target.with_name(relative.name+'.label_colors.ply')
                export_label_colors(mesh,labels,preview)
                label_target=target.with_name(relative.name+'.labels.npz');atomic_npz(label_target,**data)
                rows.append(dict(annotator=name,source_relative=str(relative),completed=completed,status='OK',ply_file=str(target.relative_to(destination)),label_colors_ply_file=str(preview.relative_to(destination)),npz_file=str(label_target.relative_to(destination)),error=''))
                report['exported']+=1
            except Exception as exc:
                report['failed']+=1;rows.append(dict(annotator=name,source_relative=str(source),completed=False,status='FAILED',ply_file='',label_colors_ply_file='',npz_file='',error=str(exc)))
    with (destination/'export_manifest.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['annotator','source_relative','completed','status','ply_file','label_colors_ply_file','npz_file','error']);writer.writeheader();writer.writerows(rows)
    (destination/'export_summary.json').write_text(json.dumps({**report,'source_root':str(root),'include_drafts':include_drafts},indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--annotations',type=Path)
    p.add_argument('--dest',type=Path,required=True);p.add_argument('--annotators',nargs='+');p.add_argument('--include-drafts',action='store_true')
    a=p.parse_args()
    try:
        report=export_annotations(a.root,a.annotations or a.root/'annotations',a.dest,a.annotators,include_drafts=a.include_drafts)
        print(json.dumps(report,indent=2));return 2 if report['failed'] else 0
    except Exception as exc:print('ERROR:',exc);return 2

if __name__=='__main__':raise SystemExit(main())
