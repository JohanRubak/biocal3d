"""Shared versioned geometry corrections, independent of any plaque rater."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib,json,os,sys,uuid
from datetime import datetime,timezone
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
if __package__:
    from .mesh_io import Mesh,load_mesh,mesh_appearance,polydata,atomic_npz
else:
    from mesh_io import Mesh,load_mesh,mesh_appearance,polydata,atomic_npz
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from roi_geometry import Extraction,select_surface,face_geometry,DEFAULT_HEIGHT_HALF_WIDTH
if __package__:
    from .source_correspondence import triangle_correspondence
else:
    from source_correspondence import triangle_correspondence


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def review_dir(store,path):
    rel=Path(path).resolve().relative_to(store.root)
    return store.output/'_roi_review'/rel.parent/rel.name


def read_state(store,path):
    p=review_dir(store,path)/'current.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def current_version(store,path):
    state=read_state(store,path)
    return state['version'] if state else ''


def read_reviewed(store,path):
    state=read_state(store,path)
    if state is None:return load_mesh(path)
    base=load_mesh(path)
    if base.fingerprint!=state['input_fingerprint']:
        raise ValueError('The original ROI changed since its geometry review. Use the original dataset snapshot.')
    payload=review_dir(store,path)/'versions'/state['version']/'surface.npz'
    m=load_mesh(payload)
    m.path=Path(path).resolve(); m.review_version=state['version']; m.review_status=state['status']
    m.review_directory=review_dir(store,path)
    if not m.texture and base.texture and m.display.GetPointData().GetTCoords() is not None:m.texture=base.texture
    return m


def provenance(mesh):
    """Require exact correspondence to the neighboring geometry file."""
    geo=mesh.path.with_name('geometry.npz')
    if geo.exists():
        with np.load(geo,allow_pickle=False) as z:
            if ('vertices' in z and 'faces' in z and z['vertices'].shape==mesh.vertices.shape
                    and np.array_equal(z['faces'],mesh.faces) and np.allclose(z['vertices'],mesh.vertices,atol=1e-6,rtol=0)):
                return {k:np.array(z[k]) for k in z.files if k in ('source_ply','source_texture','original_to_canonical','roi_radius',
                       'raw_triangle_id','raw_vertex_id','source_sha256','extraction_json')}
    return {}


def source_face_ids(mesh):
    ids=getattr(mesh,'source_face_ids',None)
    if ids is not None:return np.asarray(ids,np.int64)
    p=provenance(mesh)
    return np.asarray(p['raw_triangle_id'],np.int64) if 'raw_triangle_id' in p else None


def source_domain(mesh):
    if getattr(mesh,'source_domain',''):return mesh.source_domain
    p=provenance(mesh)
    ids=source_face_ids(mesh)
    if 'source_ply' in p and ids is not None and len(ids)==len(mesh.faces) and np.all(ids>=0) and len(np.unique(ids))==len(ids):
        checksum=str(p['source_sha256'].item()) if 'source_sha256' in p else ''
        metadata=mesh.path.with_name('metadata.json')
        if not checksum and metadata.exists():checksum=json.loads(metadata.read_text(encoding='utf-8')).get('source_sha256','')
        return 'raw-sha256:'+checksum if checksum else 'raw:'+str(p['source_ply'].item())
    return 'input:'+mesh.fingerprint


def compact_mesh(base,mask,path):
    ids=np.flatnonzero(mask)
    if not len(ids):raise ValueError('Cannot save an empty surface.')
    vids=np.unique(base.faces[ids]); remap=np.full(len(base.vertices),-1,np.int64); remap[vids]=np.arange(len(vids))
    faces=remap[base.faces[ids]]; colors,uv=mesh_appearance(base)
    if uv is not None:
        display=polydata(base.vertices[vids][faces.ravel()],np.arange(len(faces)*3).reshape(-1,3),
                         colors[vids][faces.ravel()] if colors is not None else None,uv[ids].reshape(-1,2))
    else:display=polydata(base.vertices[vids],faces,colors[vids] if colors is not None else None)
    m=Mesh(Path(path),base.vertices[vids],faces,display,np.ones(len(ids),bool),base.texture,base.warning)
    m.source_face_ids=np.asarray(base.source_face_ids)[ids]
    m.source_vertex_ids=np.asarray(base.source_vertex_ids)[vids]
    m.source_domain=base.source_domain
    return m


def load_review_context(store,mesh,raw_override=None):
    """Return an editable source domain; missing raw data still permits local cleanup."""
    original=load_mesh(mesh.path); state=read_state(store,mesh.path)
    p=provenance(original); raw_hint=str(p.get('source_ply',np.array('')).item())
    raw=Path(raw_override or raw_hint) if (raw_override or raw_hint) else None
    if state and not raw_override and state.get('raw_path'):raw=Path(state['raw_path'])
    settings=Extraction(radius=float(p.get('roi_radius',np.array(0.))))
    if settings.radius<=0:
        center=np.median(original.vertices,axis=0); settings.cx,settings.cy,settings.z=map(float,center)
        settings.radius=float(np.percentile(np.linalg.norm(original.vertices[:,:2]-center[:2],axis=1),99.9))
    if 'extraction_json' in p:
        extraction=json.loads(str(p['extraction_json'].item()))
        if extraction.get('settings'):settings=Extraction(**extraction['settings'])
    if state:settings=Extraction(**state['settings'])
    else:
        settings.below=DEFAULT_HEIGHT_HALF_WIDTH;settings.above=DEFAULT_HEIGHT_HALF_WIDTH
    transform=np.asarray(p.get('original_to_canonical',np.eye(4)),float)
    ids=source_face_ids(original)
    if raw and raw.is_file() and 'original_to_canonical' in p and ids is not None and np.all(ids>=0):
        metadata=mesh.path.with_name('metadata.json'); expected=''
        if metadata.exists():expected=json.loads(metadata.read_text(encoding='utf-8')).get('source_sha256','')
        if expected and digest(raw)!=expected:raise ValueError('Original-model checksum differs from preparation metadata.')
        if raw.suffix.lower()=='.ply':reader=vtk.vtkPLYReader()
        elif raw.suffix.lower()=='.vtp':reader=vtk.vtkXMLPolyDataReader()
        else:raise ValueError('Original context must be PLY or VTP.')
        reader.SetFileName(str(raw));reader.Update()
        triangle=vtk.vtkTriangleFilter();triangle.SetInputData(reader.GetOutput());triangle.Update()
        data=triangle.GetOutput()
        v=vtk_to_numpy(data.GetPoints().GetData()).astype(float)
        packed=vtk_to_numpy(data.GetPolys().GetData()).reshape(-1,4)
        f=packed[:,1:].astype(np.int64)
        xyz=(np.c_[v,np.ones(len(v))]@transform.T)[:,:3]
        if ids.max()>=len(f):raise ValueError('Source triangle IDs exceed original mesh.')
        # Verify every retained triangle, allowing cyclic/reversed vertex ordering.
        a=xyz[f[ids]];b=original.vertices[original.faces]
        distance=np.linalg.norm(a[:,:,None,:]-b[:,None,:,:],axis=-1)
        if not (np.all(distance.min(axis=1)<=2e-5) and np.all(distance.min(axis=2)<=2e-5)):
            raise ValueError('Original triangle mapping does not match this ROI. Full-model restoration is disabled.')
        data.GetPoints().SetData(__import__('vtk.util.numpy_support',fromlist=['numpy_to_vtk']).numpy_to_vtk(xyz,deep=True))
        # VTK defines the preparation source-face numbering. The appearance
        # loader may split/reindex texture vertices or reorder triangle corners.
        try:full=load_mesh(raw)
        except ValueError:full=None
        if full is not None:
            rows,corners=triangle_correspondence(full.vertices[full.faces],v[f])
            file_colors,file_uv=mesh_appearance(full)
            index=np.arange(len(f))[:,None]
            corner_colors=file_colors[full.faces[rows]][index,corners] if file_colors is not None else None
            corner_uv=file_uv.reshape(-1,3,2)[rows][index,corners] if file_uv is not None else None
            display=polydata(xyz[f.ravel()],np.arange(len(f)*3).reshape(-1,3),
                             corner_colors.reshape(-1,3) if corner_colors is not None else None,
                             corner_uv.reshape(-1,2) if corner_uv is not None else None)
        else:
            scalars=data.GetPointData().GetScalars();tcoords=data.GetPointData().GetTCoords()
            colors=vtk_to_numpy(scalars)[:,:3] if scalars is not None and scalars.GetNumberOfComponents() in (3,4) else None
            uv=vtk_to_numpy(tcoords) if tcoords is not None else None
            display=polydata(xyz,f,colors,uv)
        texture=full.texture if full else None
        texture_hint=str(p.get('source_texture',np.array('')).item())
        if not texture and texture_hint and Path(texture_hint).is_file():texture=Path(texture_hint)
        context=Mesh(raw,xyz,f,display,np.ones(len(f),bool),texture)
        context.source_face_ids=np.arange(len(f),dtype=np.int64);context.source_vertex_ids=np.arange(len(v),dtype=np.int64)
        context.source_domain=source_domain(original);context.context_kind='full'
        selected_ids=source_face_ids(mesh)
        mask=np.isin(context.source_face_ids,selected_ids[mesh.active])
        return context,mask,settings,'Full original model available; restore/exclude edits use verified source triangles.',str(raw)
    # Local mode uses immutable input topology and its own domain if no raw mapping exists.
    original.source_face_ids=ids if ids is not None and np.all(ids>=0) else np.arange(len(original.faces))
    p_ids=p.get('raw_vertex_id')
    original.source_vertex_ids=np.asarray(p_ids,np.int64) if p_ids is not None else (original.source_vertex_ids if original.source_vertex_ids is not None else np.arange(len(original.vertices)))
    original.source_domain=source_domain(original)
    if state:
        if mesh.source_domain!=original.source_domain:raise ValueError('Geometry review source domain differs.')
        mask=np.isin(original.source_face_ids,mesh.source_face_ids)
    else:mask=original.active.copy()
    original.active=np.ones(len(original.faces),bool);original.context_kind='local'
    return original,mask,settings,'Local ROI only. Locate the original model to restore surface outside this crop.',raw_hint


def save_review(store,input_path,base,mask,settings,status,notes,expected_version,raw_path=''):
    from PySide6.QtCore import QLockFile
    folder=review_dir(store,input_path);folder.mkdir(parents=True,exist_ok=True)
    lock=QLockFile(str(folder/'review.lock'));lock.setStaleLockTime(0)
    if not lock.tryLock(0):raise RuntimeError('Another reviewer is saving this surface.')
    try:
        if current_version(store,input_path)!=expected_version:raise RuntimeError('The shared ROI changed. Reopen review before saving.')
        original=load_mesh(input_path)
        if status not in ('pending','approved','flagged'):raise ValueError('Invalid review status.')
        version=uuid.uuid4().hex; dest=folder/'versions'/version;dest.mkdir(parents=True)
        m=compact_mesh(base,mask,input_path);colors,uv=mesh_appearance(m)
        data=dict(vertices=m.vertices,faces=m.faces,face_roi_mask=m.active,
                  source_face_ids=m.source_face_ids,source_vertex_ids=m.source_vertex_ids,source_domain=np.array(m.source_domain),
                  texture_path=np.array(str(m.texture) if m.texture else ''))
        if colors is not None:data['rgb']=colors
        if uv is not None:data['wedge_uv']=uv
        atomic_npz(dest/'surface.npz',**data)
        proposal,metrics=select_surface(base.vertices,base.faces,settings)
        metrics['automatic_retained_faces']=metrics['retained_faces']
        metrics['retained_faces']=int(mask.sum())
        metrics['retained_area_mm2']=float(face_geometry(base.vertices,base.faces)[2][mask].sum())
        metrics['manual_override_faces']=int(np.sum(mask!=proposal))
        metrics['review_status']=status
        record=dict(schema_version=1,version=version,parent_version=expected_version,status=status,
                    input_fingerprint=original.fingerprint,settings=asdict(settings),raw_path=raw_path,
                    notes=notes,reviewer=store.annotator,updated_utc=datetime.now(timezone.utc).isoformat(),
                    retained_faces=int(mask.sum()),metrics=metrics,source_domain=m.source_domain,
                    downstream_inputs_must_be_regenerated=True)
        (dest/'review.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        temp=folder/('current.'+version+'.tmp');temp.write_text(json.dumps(record,indent=2),encoding='utf-8');os.replace(temp,folder/'current.json')
        return record
    finally:lock.unlock()


def migrate_labels(store,mesh,data):
    """Transfer by verified identity only. Never nearest-neighbour label transfer."""
    old_labels=np.asarray(data['face_labels'],np.int8)
    if 'source_face_ids' in data and 'source_domain' in data:
        old_ids=np.asarray(data['source_face_ids']); domain=str(data['source_domain'].item())
    else:
        original=load_mesh(mesh.path)
        if str(data['fingerprint'].item())!=original.fingerprint:
            raise ValueError('Legacy annotation geometry cannot be verified; use its original snapshot.')
        old_ids=source_face_ids(original)
        if old_ids is None:old_ids=np.arange(len(original.faces))
        domain=source_domain(original)
    new_ids=source_face_ids(mesh)
    if len(old_ids)!=len(old_labels):raise ValueError('Source mapping and saved label lengths differ.')
    if domain!=source_domain(mesh) or new_ids is None:raise ValueError('Cannot safely transfer labels between source domains.')
    if len(np.unique(old_ids))!=len(old_ids) or np.any(old_ids<0):raise ValueError('Ambiguous source-face mapping.')
    order=np.argsort(old_ids); sorted_ids=old_ids[order];at=np.searchsorted(sorted_ids,new_ids)
    found=at<len(old_ids);found[found]&=sorted_ids[at[found]]==new_ids[found]
    result=np.full(len(mesh.faces),-2,np.int8);result[found]=old_labels[order[at[found]]]
    return result,False,str(data['notes'].item())+'\n[Geometry changed: matched source faces retained; completion reset.]'
