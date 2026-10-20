"""Regenerate reviewed ROI geometry, 2D maps and manifests into a NEW snapshot.

Never reuse model feature/surface caches from the previous snapshot.
"""
from pathlib import Path
import argparse,csv,json,re,shutil,sys
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent/'annotator'))
from mesh_io import AnnotationStore,mesh_appearance,load_mesh,atomic_npz
from roi_review_store import read_reviewed,source_face_ids,read_state,current_version,source_domain,provenance,compact_mesh,manifest_reviews
from biocal3d_prepare_all import export_sample,create_qc,write_csv,write_json


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def normalize_export_row(row):
    """CSV fields must retain prepare-all's T<label>/numeric-index schema."""
    result=dict(row);identity=row.get('scan_id',row.get('sample_id','scan'))
    for key in ('group','time_idx','sample_number'):
        value=result.get(key)
        if value is not None and str(value).strip()!='':
            text=str(value).strip()
            if not re.fullmatch(r'\d+',text):raise ValueError(f'{identity}: invalid numeric {key}: {value!r}')
            result[key]=int(text)
    label=result.get('timepoint')
    if label is not None and str(label).strip()!='':
        match=re.fullmatch(r'T?(\d+)',str(label).strip(),flags=re.IGNORECASE)
        if match is None:raise ValueError(f'{identity}: invalid timepoint {label!r}; expected T0/T1/T2 or a numeric legacy index')
        index=int(match[1])
        if result.get('time_idx') not in (None,'') and result['time_idx']!=index:
            raise ValueError(f'{identity}: timepoint {label!r} conflicts with time_idx {result["time_idx"]!r}')
        result['timepoint']=f'T{index}';result['time_idx']=index
    elif isinstance(result.get('time_idx'),int):result['timepoint']=f'T{result["time_idx"]}'
    return result


def rebuild(root,dest,annotations=None,allow_pending=False):
    import pyvista as pv
    root=Path(root).resolve();dest=Path(dest).resolve()
    if dest==root or root in dest.parents or dest in root.parents:
        raise ValueError('Choose a separate sibling output snapshot, outside the original dataset.')
    if dest.exists() and any(dest.iterdir()):raise ValueError('Output must be empty/new.')
    store=AnnotationStore(root,annotations or root/'annotations','geometry_rebuild')
    checks=manifest_reviews(store)
    approved=sum(item['status']=='approved' for item in checks)
    print(f'ROI approval check: {approved}/{len(checks)} approved. Reviews: {store.output}',flush=True)
    blocked=[item for item in checks if item['status']=='flagged' or (item['status']!='approved' and not allow_pending)]
    if blocked:
        details='\n'.join(f"  {item['row'].get('scan_id',item['source'])}: {item['status']} ({item['source'].name})" for item in blocked)
        raise ValueError(f'{len(blocked)} exported ROI(s) still need review:\n{details}\nReview folder: {store.output}\nNo output was created. Check the full snapshot and the annotation output folder used in the app.')
    todo=[]
    for item in checks:
        row,source=normalize_export_row(item['row']),item['source'];sample=root/row['sample_dir']
        if source!=item['canonical']:
            print(f"Using existing {source.name} review for {row.get('scan_id',source.parent.name)} (manifest: {item['canonical'].name}).",flush=True)
        if not (sample/'geometry.npz').exists() or not (sample/'metadata.json').exists():
            raise ValueError(f'Missing geometry/metadata for {sample}')
        todo.append((row,source,sample,item['version']))
    dest.mkdir(parents=True,exist_ok=True);results=[];audit=[]
    try:
        for i,(row,source,sample,version) in enumerate(todo,1):
            print(f'[{i}/{len(todo)}] {row.get("scan_id",source)}',flush=True)
            reviewed_mesh=read_reviewed(store,source)
            mesh=reviewed_mesh
            if not np.all(mesh.active):mesh=compact_mesh(mesh,mesh.active,source)
            colors,uv=mesh_appearance(mesh)
            with np.load(sample/'geometry.npz',allow_pickle=False) as z:
                transform=np.array(z['original_to_canonical']);radius=float(z['roi_radius'])
                raw_source=str(z['source_ply'].item())
            previous=json.loads((sample/'metadata.json').read_text(encoding='utf-8'))
            source_state=read_state(store,source)
            if source_state and source_state.get('raw_path') and Path(source_state['raw_path']).is_file():raw_source=source_state['raw_path']
            if not Path(raw_source).is_file():raise ValueError(f'Original source needed for provenance hashing: {raw_source}')
            # Share vertices whenever UVs agree; duplicate only genuine texture seams.
            if uv is not None and mesh.texture:
                corner_ids=mesh.faces.ravel();corner_uv=uv.reshape(-1,2)
                keys=np.column_stack((corner_ids,corner_uv))
                _,first,inverse=np.unique(keys,axis=0,return_index=True,return_inverse=True)
                source_vertices=corner_ids[first]
                xyz=mesh.vertices[source_vertices];faces=inverse.reshape(-1,3)
                raw_vids=mesh.source_vertex_ids if mesh.source_vertex_ids is not None else np.full(len(mesh.vertices),-1)
                raw_vids=raw_vids[source_vertices]
                surface=pv.PolyData(xyz,np.c_[np.full(len(faces),3),faces])
                with Image.open(mesh.texture) as im:texture=np.asarray(im.convert('RGB'),np.float32)/255
                color_input=dict(rgb=None,uv=corner_uv[first],texture=texture,color_source='texture_uv',texture_path=str(mesh.texture))
            else:
                if colors is None:raise ValueError(f'No scanner colors/texture for {source}')
                surface=pv.PolyData(mesh.vertices,np.c_[np.full(len(mesh.faces),3),mesh.faces])
                color_input=dict(rgb=colors.astype(np.float32)/255,uv=None,texture=None,color_source='vertex_rgb',texture_path='')
                raw_vids=mesh.source_vertex_ids if mesh.source_vertex_ids is not None else np.full(len(mesh.vertices),-1)
            fids=source_face_ids(mesh)
            if fids is None:
                p=provenance(mesh);fids=p.get('raw_triangle_id',np.full(len(mesh.faces),-1))
            surface.point_data['RawVertexId']=raw_vids;surface.cell_data['RawTriangleId']=fids
            state=read_state(store,source)
            if state:
                radius=state['settings']['radius']
                surface.field_data['ROIExtractionJSON']=[json.dumps({'settings':state['settings'],**state['metrics']})]
            newrow=dict(row,source=raw_source)
            folder=dest/row['sample_dir']
            info=export_sample(newrow,surface,color_input,transform,radius,folder,int(previous['image_size']),float(previous['fov_mm']),True)
            info.update(geometry_review_status=mesh.review_status,geometry_review_version=version,
                        parent_snapshot=str(root),parent_sample_dir=row['sample_dir'])
            write_json(folder/'metadata.json',info);create_qc(folder)
            results.append({**row,**info,'sample_dir':row['sample_dir'],'annotation_mesh':row['sample_dir']+'/roi.ply'})
            # Transfer saved rater labels through the reviewed geometry; preserve their draft/completion status.
            for rater in sorted(store.output.iterdir()) if store.output.exists() else []:
                if not rater.is_dir() or rater.name.startswith(('_','.')):continue
                try:old_store=AnnotationStore(root,store.output,rater.name)
                except ValueError:continue
                if not old_store.path_for(source).exists():continue
                if old_store.status(source)=='geometry_changed':
                    raise ValueError(f'{rater.name}: reopen and save {source} on the reviewed surface before rebuilding.')
                labels,done,notes=old_store.load(reviewed_mesh)
                labels=labels[reviewed_mesh.active]
                new_store=AnnotationStore(dest,dest/'annotations',rater.name)
                new_mesh=load_mesh(folder/'roi.ply')
                # Same source-face order in export_sample; refuse any unexpected reorder.
                if len(new_mesh.faces)!=len(labels) or not np.array_equal(new_mesh.source_face_ids,fids):
                    raise ValueError('Export changed source-face ordering; labels were not copied.')
                new_store.save(new_mesh,labels,done,notes)
            # Copy the verified shared record to the new input identity (geometry may have seam expansion).
            if state:
                from roi_review_store import load_review_context,save_review
                new_source=folder/'roi.ply';new_store=AnnotationStore(dest,dest/'annotations','geometry_rebuild')
                new_mesh=load_mesh(new_source)
                new_mesh.source_domain=source_domain(mesh)
                new_mesh.source_face_ids=fids
                new_mesh.source_vertex_ids=raw_vids
                save_review(new_store,new_source,new_mesh,np.ones(len(new_mesh.faces),bool),
                            __import__('roi_geometry').Extraction(**state['settings']),state['status'],
                            'Rebuilt from '+str(root)+' / '+version,'',raw_source)
                # Annotation fingerprints refer to the same freshly exported topology. Update only review version.
                new_version=current_version(new_store,new_source)
                info['geometry_review_version']=new_version
                results[-1]['geometry_review_version']=new_version
                write_json(folder/'metadata.json',info)
                for rater in sorted((dest/'annotations').iterdir()):
                    if not rater.is_dir() or rater.name.startswith(('_','.')):continue
                    label_store=AnnotationStore(dest,dest/'annotations',rater.name);label_path=label_store.path_for(new_source)
                    if label_path.exists():
                        with np.load(label_path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
                        data['geometry_review_version']=np.array(new_version);data['geometry_review_status']=np.array(state['status'])
                        atomic_npz(label_path,**data)
            audit.append(dict(scan_id=row.get('scan_id',''),geometry_review_version=version,status=mesh.review_status,
                              review_source=str(source.relative_to(root)),manifest_source=row.get('annotation_mesh','')))
        # Ensure nobody changed a shared surface during this rebuild.
        for row,source,sample,version in todo:
            if current_version(store,source)!=version:raise RuntimeError('A shared surface changed while rebuilding. Discard this incomplete output and rerun.')
        write_csv(dest/'scan_manifest.csv',results)
        source_inventory=root/'annotation_inventory.csv'
        included={r['scan_id'] for r in rows(source_inventory)} if source_inventory.exists() else {r['scan_id'] for r in results}
        write_csv(dest/'annotation_inventory.csv',[r for r in results if r['scan_id'] in included])
        for name in ('splits.json','split_assignments.csv','specimen_splits.csv','splits.csv','classes.txt','class_map.json','annotation_classes.json'):
            if (root/name).is_file():shutil.copy2(root/name,dest/name)
        if (root/'references').is_dir():shutil.copytree(root/'references',dest/'references')
        config=json.loads((root/'run_config.json').read_text()) if (root/'run_config.json').exists() else {}
        config.update(parent_snapshot=str(root),output=str(dest),geometry_rebuilt=True,feature_caches_must_be_rebuilt=True)
        write_json(dest/'run_config.json',config);write_json(dest/'dataset_index.json',dict(schema_version=3,config=config,scans=results))
        write_json(dest/'geometry_rebuild.json',dict(status='complete',parent_snapshot=str(root),scans=audit,split_policy='copied unchanged',old_caches_compatible=False))
    except Exception as e:
        write_json(dest/'INCOMPLETE_REBUILD.json',dict(error=str(e),do_not_train=True));raise
    return dest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--annotations',type=Path);p.add_argument('--allow-pending',action='store_true',help='Exploratory rebuild; approved surfaces remain preferred. Flagged scans still block.')
    p.add_argument('--check-only',action='store_true',help='List approval status for the full scan manifest without writing output.')
    a=p.parse_args()
    if not a.check_only and a.output is None:p.error('--output is required unless --check-only is used')
    try:
        if a.check_only:
            store=AnnotationStore(a.root,a.annotations or a.root/'annotations','geometry_rebuild')
            checks=manifest_reviews(store);approved=sum(item['status']=='approved' for item in checks)
            print(f'ROI approval check: {approved}/{len(checks)} approved. Reviews: {store.output}')
            for item in checks:print(f"{item['row'].get('scan_id','')}: {item['status']} | {item['source']}")
            return 0 if approved==len(checks) else 2
        print(rebuild(a.root,a.output,a.annotations,a.allow_pending));return 0
    except (ValueError,RuntimeError,OSError) as e:
        print('Rebuild stopped: '+str(e),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
