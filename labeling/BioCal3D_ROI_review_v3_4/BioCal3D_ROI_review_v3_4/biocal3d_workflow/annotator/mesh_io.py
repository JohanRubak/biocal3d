"""Mesh loading and index-preserving annotation storage; no Qt dependency."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib
import json
import os
import re
import tempfile
import shutil
import csv
from datetime import datetime, timezone
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

def _review_module():
    return importlib.import_module('.roi_review_store', __package__) if __package__ else importlib.import_module('roi_review_store')

LABELS = {-2: 'unreviewed', -1: 'uncertain', 0: 'clean', 1: 'plaque'}
PALETTE = {-2: (115, 128, 147), -1: (255, 190, 65), 0: (55, 205, 150), 1: (245, 75, 125)}

def rgb8(a):
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError('Colors must be N × 3 or N × 4.')
    a = a[:, :3]
    if np.issubdtype(a.dtype, np.floating) and a.size and np.nanmax(a) <= 1:
        a = a * 255
    return np.nan_to_num(a).clip(0, 255).astype(np.uint8)

def polydata(vertices, faces, colors=None, uv=None):
    p = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(vertices, dtype=np.float64), deep=True))
    p.SetPoints(points)
    cells = vtk.vtkCellArray()
    packed = np.column_stack((np.full(len(faces), 3), faces)).astype(np.int64).ravel()
    cells.SetCells(len(faces), numpy_to_vtkIdTypeArray(packed, deep=True))
    p.SetPolys(cells)
    if colors is not None:
        a = numpy_to_vtk(rgb8(colors), deep=True)
        a.SetName('RGB')
        p.GetPointData().SetScalars(a)
    if uv is not None:
        a = numpy_to_vtk(np.asarray(uv, dtype=np.float32), deep=True)
        a.SetName('TCoords')
        p.GetPointData().SetTCoords(a)
    return p

@dataclass
class Mesh:
    path: Path
    vertices: np.ndarray
    faces: np.ndarray
    display: vtk.vtkPolyData
    active: np.ndarray
    texture: Path | None = None
    warning: str = ''
    imported_labels: np.ndarray | None = None

    source_face_ids: np.ndarray | None = None
    source_vertex_ids: np.ndarray | None = None
    source_domain: str = ''
    review_version: str = ''
    review_status: str = 'pending'

    @property
    def fingerprint(self):
        h = hashlib.sha256()
        for a, dtype in [(self.vertices, '<f8'), (self.faces, '<i8')]:
            b = np.ascontiguousarray(a, dtype=dtype)
            h.update(str(b.shape).encode())
            h.update(b.tobytes())
        h.update(np.ascontiguousarray(self.active, dtype=np.uint8).tobytes())
        return h.hexdigest()

def find_texture(path, hint=None):
    if hint:
        candidate = path.parent / str(hint).strip().strip('"')
        if candidate.is_file():
            return candidate
    candidates = [p for p in path.parent.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    exact = [p for p in candidates if p.stem.lower() == path.stem.lower()]
    prefixed = [p for p in candidates if p.stem.lower().startswith(path.stem.lower())]
    if len(exact) == 1:
        return exact[0]
    if len(prefixed) == 1:
        return prefixed[0]
    # Do not silently use a QC image or unrelated photograph.
    return None

def _triangles(p):
    if p.GetNumberOfVerts() or p.GetNumberOfLines() or p.GetNumberOfStrips():
        raise ValueError('Only triangle surface meshes are supported; this file contains other cell types.')
    raw = vtk_to_numpy(p.GetPolys().GetData())
    if len(raw) % 4 or not np.all(raw.reshape(-1, 4)[:, 0] == 3):
        raise ValueError('Mesh contains non-triangular polygons. Export a triangulated copy first.')
    return raw.reshape(-1, 4)[:, 1:].copy()

def _array(p, names, cell=False):
    data = p.GetCellData() if cell else p.GetPointData()
    for name in names:
        a = data.GetArray(name)
        if a is not None:
            return vtk_to_numpy(a)
    return None

def load_mesh(path):
    path = Path(path).resolve()
    texture, warning, colors, uv, wedge_uv, active, imported_labels = None, '', None, None, None, None, None
    source_fids=source_vids=None; source_key=''
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as d:
            def get(names, required=False):
                for name in names:
                    if name in d:
                        return np.array(d[name])
                if required:
                    raise ValueError(f'Missing {names[0]}; NPZ keys are: {", ".join(d.files)}')
                return None
            v = get(['vertices', 'verts', 'points', 'xyz', 'V'], True)
            f = get(['faces', 'triangles', 'F'], True)
            colors = get(['rgb', 'colors', 'vertex_colors', 'vertex_rgb'])
            imported_labels = get(['face_labels', 'face_label'])
            source_fids=get(['source_face_ids','raw_triangle_id'])
            source_vids=get(['source_vertex_ids','raw_vertex_id'])
            wedge_uv=get(['wedge_uv'])
            key=get(['source_domain'])
            if key is not None:source_key=str(key.item())
            uv = get(['uv', 'tcoords', 'TCoords', 'texture_coordinates'])
            mask = get(['roi_mask', 'vertex_roi_mask', 'roi_vertex_mask'])
            face_mask = get(['face_roi_mask', 'roi_face_mask'])
            if face_mask is not None:
                active = np.asarray(face_mask, bool).reshape(-1)
            elif mask is not None and mask.size == len(v):
                active = np.asarray(mask, bool).reshape(-1)[np.asarray(f, int)].all(axis=1)
            hint = get(['texture_path', 'texture_file'])
            texture = find_texture(path, hint.item() if hint is not None and hint.size == 1 else None)
    elif path.suffix.lower() == '.ply':
        from plyfile import PlyData
        d = PlyData.read(str(path))
        if 'face' not in d:
            raise ValueError('PLY has no faces; point-cloud-only annotation is not supported.')
        vd, fd = d['vertex'].data, d['face'].data
        v = np.column_stack([vd[n] for n in ('x', 'y', 'z')])
        index_name = next((n for n in ('vertex_indices', 'vertex_index') if n in fd.dtype.names), None)
        if index_name is None or any(len(x) != 3 for x in fd[index_name]):
            raise ValueError('PLY must contain triangle faces with vertex_indices.')
        f = np.stack(fd[index_name]).astype(np.int64)
        if 'source_face_id' in fd.dtype.names:source_fids=np.asarray(fd['source_face_id'],np.int64)
        if 'source_vertex_id' in vd.dtype.names:source_vids=np.asarray(vd['source_vertex_id'],np.int64)
        domain=next((c.split(' ',1)[1] for c in d.comments if c.startswith('SourceDomain ')),None)
        if domain:source_key=domain
        if 'roi_active' in fd.dtype.names:active=fd['roi_active'].astype(bool)
        if 'face_label' in fd.dtype.names:imported_labels=np.array(fd['face_label'])
        if all(n in vd.dtype.names for n in ('red', 'green', 'blue')):
            colors = np.column_stack([vd[n] for n in ('red', 'green', 'blue')])
        for pair in [('texture_u', 'texture_v'), ('u', 'v'), ('s', 't')]:
            if all(n in vd.dtype.names for n in pair):
                uv = np.column_stack([vd[n] for n in pair])
                break
        if 'texcoord' in fd.dtype.names:
            if any(len(x) != 6 for x in fd['texcoord']):
                raise ValueError('PLY face texcoord must contain six UV numbers per triangle.')
            wedge_uv = np.stack(fd['texcoord']).reshape(-1, 2)
        hint = next((s.split(None, 1)[1] for s in d.comments
                     if s.lower().startswith('texturefile ')), None)
        texture = find_texture(path, hint)
    elif path.suffix.lower() == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(path)); reader.Update()
        p = reader.GetOutput()
        if p.GetNumberOfPoints() == 0:
            raise ValueError('VTP contains no points.')
        v = vtk_to_numpy(p.GetPoints().GetData()).copy()
        f = _triangles(p)
        colors = _array(p, ['RGB', 'rgb', 'colors', 'Colors', 'RGBA', 'vertex_colors'])
        if colors is None:
            scalars = p.GetPointData().GetScalars()
            if scalars and scalars.GetNumberOfComponents() in (3, 4):
                colors = vtk_to_numpy(scalars)
        t = p.GetPointData().GetTCoords()
        if t is not None:
            uv = vtk_to_numpy(t).copy()
        active = _array(p, ['roi_active', 'roi_mask', 'ROI'], cell=True)
        imported_labels = _array(p, ['face_label', 'face_labels'], cell=True)
        source_fids=_array(p,['RawTriangleId','source_face_id'],cell=True)
        source_vids=_array(p,['RawVertexId','source_vertex_id'])
        if active is None:
            mask = _array(p, ['roi_mask', 'ROI'])
            if mask is not None:
                active = np.asarray(mask, bool).reshape(-1)[f].all(axis=1)
        domain=p.GetFieldData().GetAbstractArray('SourceDomain')
        if domain is not None:source_key=domain.GetValue(0)
        field = p.GetFieldData().GetAbstractArray('TextureFile')
        texture = find_texture(path, field.GetValue(0) if field is not None else None)
    else:
        raise ValueError('Supported inputs: triangle PLY, VTP, and geometry NPZ.')
    v, f = np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError('Expected vertices N × 3 and faces M × 3.')
    if not len(v) or not len(f) or not np.isfinite(v).all() or f.min() < 0 or f.max() >= len(v):
        raise ValueError('Empty or invalid geometry.')
    if colors is not None and len(colors) != len(v):
        raise ValueError('Vertex color count differs from the vertex count.')
    if uv is not None and (uv.shape != (len(v), 2)):
        raise ValueError('UV array must have one pair per vertex.')
    active = np.ones(len(f), bool) if active is None else np.asarray(active, bool).reshape(-1)
    if len(active) != len(f) or not active.any():
        raise ValueError('ROI mask length is wrong or its ROI is empty.')
    if uv is not None or wedge_uv is not None:
        if texture is None and colors is None:
<<<<<<< HEAD
            warning = 'UVs found but no texture matched. Use Choose texture before labeling.'
=======
            warning = 'UVs found but no texture matched. Fix the prepared texture data before labeling.'
>>>>>>> 3dc377b2c13f84fb1a8141a5ae83a69198e2addb
    elif colors is None:
        warning = 'No vertex RGB or UV texture found. Do not judge plaque from the gray mesh.'
    # Expand only the display mesh for texture seams. Face order and source arrays remain unchanged.
    if wedge_uv is not None:
        p = polydata(v[f.ravel()], np.arange(len(f)*3).reshape(-1, 3),
                     colors[f.ravel()] if colors is not None else None, np.asarray(wedge_uv).reshape(-1,2))
    else:
        p = polydata(v, f, colors, uv)
    if imported_labels is not None:
        if imported_labels.shape!=(len(f),) or not np.isin(imported_labels,list(LABELS)).all():
            raise ValueError('Imported face labels are invalid.')
        imported_labels=imported_labels.astype(np.int8)
    result=Mesh(path, v, f, p, active, texture, warning, imported_labels)
    result.source_face_ids=None if source_fids is None else np.asarray(source_fids,np.int64)
    result.source_vertex_ids=None if source_vids is None else np.asarray(source_vids,np.int64)
    result.source_domain=source_key
    return result

def discover(root, contains=''):
    root = Path(root).resolve()
    candidates = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.ply', '.vtp', '.npz'}
                  and not any(s in {'annotations', 'annotation_output', 'labels', '_roi_review', '.venv', '__pycache__'}
                              for s in p.relative_to(root).parts[:-1])
                  and not p.name.endswith(('.labels.npz', '.labeled.vtp', '.labeled.ply', '.label_colors.ply'))]
    # Prepared sample directories: one representative input per sample.
    by_dir = {}
    for p in candidates:
        by_dir.setdefault(p.parent, []).append(p)
    selected = []
    for files in by_dir.values():
        known = {p.name.lower(): p for p in files}
<<<<<<< HEAD
        preferred = next((known[n] for n in ['roi.vtp', 'roi.ply', 'geometry.npz'] if n in known), None)
=======
        preferred = next((known[n] for n in ['roi.ply', 'roi.vtp', 'geometry.npz'] if n in known), None)
>>>>>>> 3dc377b2c13f84fb1a8141a5ae83a69198e2addb
        if preferred:
            selected.append(preferred)
        else:
            # Arbitrary NPZs could be model weights/features; only auto-discover geometry.npz.
            selected.extend(p for p in files if p.suffix.lower() in {'.ply', '.vtp'})
    selected = [p for p in selected if contains.lower() in str(p.relative_to(root)).lower()]
    return sorted(selected, key=lambda p: str(p.relative_to(root)).lower())

def manifest_queue(root, manifest, contains=''):
    """Only inventory-approved samples; do not discover CHECK/failed scans implicitly."""
    root=Path(root).resolve(); queue=[]
    with Path(manifest).open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            relative=row.get('annotation_mesh','')
            if relative:
                source=(root/relative).resolve()
            elif row.get('sample_dir'):
                folder=root/row['sample_dir']
                source=next((folder/name for name in ('roi.ply','roi.vtp') if (folder/name).is_file()),None)
                if source is None:raise ValueError(f'No ROI mesh for {row.get("scan_id",row["sample_dir"])}')
                source=source.resolve()
            else:raise ValueError('Annotation inventory needs annotation_mesh or sample_dir columns.')
            if not source.is_relative_to(root) or not source.is_file():
                raise ValueError(f'Invalid inventory source: {source}')
            if source.suffix.lower()=='.ply' and row.get('ply_sha256'):
                h=hashlib.sha256()
                with source.open('rb') as mesh_file:
                    for block in iter(lambda:mesh_file.read(1024*1024),b''):h.update(block)
                if h.hexdigest()!=row['ply_sha256']:
                    raise ValueError(f'ROI PLY changed since preparation: {source}. Restore the frozen mesh.')
            if contains.lower() in str(source.relative_to(root)).lower() and source not in queue:queue.append(source)
    return queue

def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

class AnnotationStore:
    def __init__(self, root, output, annotator):
        self.root, self.output = Path(root).resolve(), Path(output).resolve()
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', annotator):
            raise ValueError('Annotator name: use 1–64 letters, numbers, underscores or hyphens.')
        self.annotator = annotator

    def path_for(self, source):
        rel = Path(source).resolve().relative_to(self.root)
        return self.output / self.annotator / rel.parent / (rel.name + '.labels.npz')

    def status(self, source):
        path = self.path_for(source)
        if not path.exists():
            return 'new'
        try:
            with np.load(path, allow_pickle=False) as d:
                current_version=_review_module().current_version
                saved_version=str(d['geometry_review_version'].item()) if 'geometry_review_version' in d else ''
                if saved_version!=current_version(self,source):return 'geometry_changed'
                return 'done' if bool(d['completed']) else 'draft'
        except Exception:
            return 'error'

    def load(self, mesh):
        path = self.path_for(mesh.path)
        if not path.exists():
            labels = np.full(len(mesh.faces), -2, np.int8) if mesh.imported_labels is None else mesh.imported_labels.copy()
            labels[~mesh.active] = -1
            return labels, False, ''
        with np.load(path, allow_pickle=False) as d:
            if str(d['fingerprint'].item()) != mesh.fingerprint:
                if mesh.review_version:
                    migrate_labels=_review_module().migrate_labels
                    return migrate_labels(self,mesh,d)
                raise ValueError('Saved labels do not match this geometry/ROI. Restore the original input or use a new output folder.')
            saved_version=str(d['geometry_review_version'].item()) if 'geometry_review_version' in d else ''
            if saved_version!=mesh.review_version:
                return np.array(d['face_labels'],dtype=np.int8),False,str(d['notes'].item())
            labels = np.array(d['face_labels'], dtype=np.int8)
            if labels.shape != (len(mesh.faces),) or not np.isin(labels, list(LABELS)).all():
                raise ValueError('Invalid saved face labels.')
            return labels, bool(d['completed']), str(d['notes'].item())

    def save(self, mesh, labels, completed, notes, camera=None):
        review=_review_module()
        current_version,source_face_ids,source_domain=review.current_version,review.source_face_ids,review.source_domain
        if current_version(self,mesh.path)!=mesh.review_version:
            raise ValueError('Shared ROI changed since opening. Reopen the sample before saving labels.')
        labels = np.asarray(labels)
        if labels.shape != (len(mesh.faces),) or not np.isin(labels, list(LABELS)).all():
            raise ValueError('Labels must be one valid integer class per source face.')
        if completed and np.any(labels[mesh.active] == -2):
            raise ValueError('Unreviewed faces remain in the ROI.')
        previous=self.path_for(mesh.path)
        if previous.exists():
            with np.load(previous,allow_pickle=False) as old:
                old_version=str(old['geometry_review_version'].item()) if 'geometry_review_version' in old else ''
            if old_version!=mesh.review_version:
                history=previous.parent/'.geometry_history';history.mkdir(exist_ok=True)
                backup=history/(previous.name+'.'+(old_version or 'original')+'.npz')
                if not backup.exists():shutil.copy2(previous,backup)
        source_ids=source_face_ids(mesh)
        atomic_npz(self.path_for(mesh.path), schema_version=np.array(2), face_labels=labels,
                   source_face_ids=source_ids if source_ids is not None else np.arange(len(mesh.faces)),
                   source_domain=np.array(source_domain(mesh)),
                   geometry_review_version=np.array(mesh.review_version), geometry_review_status=np.array(mesh.review_status),
                   active_faces=mesh.active, face_indices=np.arange(len(mesh.faces)),
                   fingerprint=np.array(mesh.fingerprint), source_relative=np.array(str(mesh.path.relative_to(self.root))),
                   annotator=np.array(self.annotator), completed=np.array(completed), notes=np.array(notes),
                   updated_utc=np.array(datetime.now(timezone.utc).isoformat()),
                   label_names=np.array(json.dumps(LABELS)), camera=np.array(json.dumps(camera or {})),
                   texture_path=np.array(str(mesh.texture) if mesh.texture else ''))

def area_summary(mesh, labels):
    tri = mesh.vertices[mesh.faces]
    areas = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)/2
    return {name: float(areas[(labels == value) & mesh.active].sum()) for value, name in LABELS.items()}

def surface_normals(vertices, faces):
    """Area-weighted vertex normals without changing topology or triangle winding."""
    tri = vertices[faces]
    raw = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    face = raw / np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-30)
    vertex = np.zeros_like(vertices,dtype=float)
    for corner in range(3):
        np.add.at(vertex, faces[:,corner], raw)
    vertex /= np.maximum(np.linalg.norm(vertex,axis=1,keepdims=True),1e-30)
    return vertex, face

def shaded_display(mesh):
    """Normals use original shared vertices, including across texture seams."""
    vertex,face=surface_normals(mesh.vertices,mesh.faces)
    p=vtk.vtkPolyData(); p.ShallowCopy(mesh.display)
    if p.GetNumberOfPoints()!=len(mesh.vertices) or not np.array_equal(_triangles(p),mesh.faces):
        vertex=vertex[mesh.faces.ravel()]
    a=numpy_to_vtk(vertex.astype(np.float32),deep=True); a.SetName('Normals'); p.GetPointData().SetNormals(a)
    a=numpy_to_vtk(face.astype(np.float32),deep=True); a.SetName('Normals'); p.GetCellData().SetNormals(a)
    return p

def mesh_appearance(mesh):
    """Original RGB plus per-face-corner UVs, also for seam-expanded display data."""
    p=mesh.display
    data=p.GetPointData().GetScalars()
    colors=rgb8(vtk_to_numpy(data)) if data is not None and data.GetNumberOfComponents()>=3 else None
    display_faces=_triangles(p)
    if colors is not None and (len(colors)!=len(mesh.vertices) or not np.array_equal(display_faces,mesh.faces)):
        original=np.zeros((len(mesh.vertices),3),np.uint8)
        original[mesh.faces.ravel()]=colors[display_faces.ravel()]
        colors=original
    data=p.GetPointData().GetTCoords()
    uv=vtk_to_numpy(data)[display_faces].reshape(-1,6).copy() if data is not None else None
    return colors,uv

def export_ply(mesh, labels, target, binary=True, face_ids=None):
    """Original vertices/faces, labels as separate properties, and portable texture copy."""
    from plyfile import PlyData, PlyElement
    target=Path(target).resolve(); target.parent.mkdir(parents=True,exist_ok=True)
    if target==mesh.path:
        raise ValueError('Choose a new filename; exports must not overwrite the input mesh.')
    colors,uv=mesh_appearance(mesh)
    normals,_=surface_normals(mesh.vertices,mesh.faces)
    dtype=[(n,'f8') for n in ('x','y','z')]+[(n,'f4') for n in ('nx','ny','nz')]
    if colors is not None:
        dtype += [(n,'u1') for n in ('red','green','blue')]
    vd=np.empty(len(mesh.vertices),dtype=dtype)
    for i,n in enumerate(('x','y','z')):vd[n]=mesh.vertices[:,i]
    for i,n in enumerate(('nx','ny','nz')):vd[n]=normals[:,i]
    if colors is not None:
        for i,n in enumerate(('red','green','blue')):vd[n]=colors[:,i]
    dtype=[('vertex_indices','i4',(3,)),('source_face_id','i4'),('roi_active','u1')]
    if labels is not None:
        labels=np.asarray(labels)
        if labels.shape!=(len(mesh.faces),) or not np.isin(labels,list(LABELS)).all():
            raise ValueError('Invalid face labels for PLY export.')
        dtype += [('face_label','i4'),('quality','f4')]
    if uv is not None:dtype += [('texcoord','f4',(6,))]
    fd=np.empty(len(mesh.faces),dtype=dtype)
    fd['vertex_indices']=mesh.faces
    fd['source_face_id']=(mesh.source_face_ids if mesh.source_face_ids is not None else np.arange(len(mesh.faces))) if face_ids is None else face_ids
    fd['roi_active']=mesh.active
    if labels is not None:
        fd['face_label']=labels; fd['quality']=labels
    if uv is not None:fd['texcoord']=uv
    comments=['BioCal3D triangle mesh; RGB preserved; source_face_id is a source mapping']
    if mesh.source_domain:comments.append('SourceDomain '+mesh.source_domain)
    if labels is not None:
        comments.append('face_label and quality: -2 unreviewed; -1 uncertain; 0 clean; 1 plaque')
    if uv is not None and mesh.texture:
        # PLY texture convention is an external image; it is not embedded in PLY bytes.
        texture_target=target.with_name(target.stem+'.texture'+mesh.texture.suffix.lower())
        if texture_target.resolve()!=mesh.texture.resolve():
            if texture_target.exists() and texture_target.read_bytes()!=mesh.texture.read_bytes():
                raise FileExistsError(f'Texture destination already contains another image: {texture_target}')
            if not texture_target.exists():shutil.copy2(mesh.texture,texture_target)
        comments.append('TextureFile '+texture_target.name)
    fd_handle,tmp=tempfile.mkstemp(prefix=target.name+'.',suffix='.tmp',dir=target.parent)
    os.close(fd_handle)
    try:
        PlyData([PlyElement.describe(vd,'vertex'),PlyElement.describe(fd,'face')],
                text=not binary,byte_order='<',comments=comments).write(tmp)
        os.replace(tmp,target)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def export_label_colors(mesh, labels, target, binary=True):
    """Viewing PLY: crisp per-face class colors with original-face/vertex mappings.

    Vertices are duplicated per triangle to avoid interpolating colors across
    class boundaries. Original-color exports and annotation NPZs are unchanged.
    """
    from plyfile import PlyData, PlyElement
    target = Path(target).resolve()
    if target == mesh.path.resolve():
        raise ValueError('Choose a new filename; exports must not overwrite the input mesh.')
    labels = np.asarray(labels)
    if labels.shape != (len(mesh.faces),) or not np.isin(labels, list(LABELS)).all():
        raise ValueError('Invalid face labels for colored PLY export.')
    target.parent.mkdir(parents=True, exist_ok=True)
    source_ids = mesh.faces.reshape(-1)
    coords = mesh.vertices[source_ids]
    colors = np.repeat(np.array([PALETTE[int(value)] for value in labels], dtype=np.uint8), 3, axis=0)
    vd = np.empty(len(coords), dtype=[('x','f8'),('y','f8'),('z','f8'),
                                     ('red','u1'),('green','u1'),('blue','u1'),('source_vertex_id','i4')])
    for i, name in enumerate(('x','y','z')):vd[name] = coords[:,i]
    for i, name in enumerate(('red','green','blue')):vd[name] = colors[:,i]
    vd['source_vertex_id'] = source_ids
    fd = np.empty(len(mesh.faces), dtype=[('vertex_indices','i4',(3,)),('source_face_id','i4'),
                                         ('roi_active','u1'),('face_label','i4'),('quality','f4')])
    fd['vertex_indices'] = np.arange(len(coords), dtype=np.int32).reshape(-1,3)
    fd['source_face_id'] = np.arange(len(mesh.faces), dtype=np.int32)
    fd['roi_active'] = mesh.active
    fd['face_label'] = labels; fd['quality'] = labels
    comments = ['BioCal3D LABEL COLORS VIEWING COPY; use original-color PLY and labels NPZ for training',
                'Vertices duplicated per face; source_vertex_id and source_face_id map to annotation input',
                'face_label: -2 gray unreviewed; -1 orange uncertain; 0 green clean; 1 pink plaque']
    descriptor, temporary = tempfile.mkstemp(prefix=target.name+'.', suffix='.tmp', dir=target.parent)
    os.close(descriptor)
    try:
        PlyData([PlyElement.describe(vd,'vertex'), PlyElement.describe(fd,'face')],
                text=not binary, byte_order='<', comments=comments).write(temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def prepare_roi_ply(mesh, target):
    """Crop only by the existing ROI mask, compact vertices, and save an explicit mapping."""
    face_ids=np.flatnonzero(mesh.active)
    vertex_ids=np.unique(mesh.faces[face_ids])
    reverse=np.full(len(mesh.vertices),-1,np.int64); reverse[vertex_ids]=np.arange(len(vertex_ids))
    faces=reverse[mesh.faces[face_ids]]
    colors,uv=mesh_appearance(mesh)
    cropped=Mesh(Path(mesh.path),mesh.vertices[vertex_ids],faces,
                 polydata(mesh.vertices[vertex_ids],faces,colors[vertex_ids] if colors is not None else None),
                 np.ones(len(faces),bool),mesh.texture,mesh.warning)
    if uv is not None:
        cropped.display=polydata(cropped.vertices[faces.ravel()],np.arange(len(faces)*3).reshape(-1,3),
                                colors[vertex_ids][faces.ravel()] if colors is not None else None,
                                uv[face_ids].reshape(-1,2))
    target=Path(target)
    mapping=target.with_suffix('.mapping.npz')
    if target.exists() or mapping.exists():
        raise FileExistsError(f'Prepared ROI already exists: {target}. Use a new output folder.')
    export_ply(cropped,None,target,face_ids=face_ids)
    atomic_npz(mapping,source_path=np.array(str(mesh.path)),source_fingerprint=np.array(mesh.fingerprint),
               source_vertex_id=vertex_ids,source_face_id=face_ids)
    return target

def export_vtp(mesh, labels, target):
    if Path(target).resolve()==mesh.path:
        raise ValueError('Choose a new filename; exports must not overwrite the input mesh.')
    p = vtk.vtkPolyData(); p.DeepCopy(mesh.display)
    for name, values in [('face_label', labels.astype(np.int32)),
                         ('source_face_id', mesh.source_face_ids if mesh.source_face_ids is not None else np.arange(len(labels), dtype=np.int64)),
                         ('roi_active', mesh.active.astype(np.uint8))]:
        a = numpy_to_vtk(values, deep=True); a.SetName(name); p.GetCellData().AddArray(a)
    if mesh.source_domain:
        domain=vtk.vtkStringArray();domain.SetName('SourceDomain');domain.InsertNextValue(mesh.source_domain);p.GetFieldData().AddArray(domain)
    if mesh.texture:
        a = vtk.vtkStringArray(); a.SetName('TextureFile'); a.InsertNextValue(str(mesh.texture))
        p.GetFieldData().AddArray(a)
    writer = vtk.vtkXMLPolyDataWriter(); writer.SetFileName(str(target)); writer.SetInputData(p)
    if not writer.Write():
        raise OSError('VTK export failed.')

def create_demo(directory):
    """Synthetic curved disks, not experimental samples or ground truth."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    n = 65
    x, y = np.meshgrid(np.linspace(-7, 7, n), np.linspace(-7, 7, n))
    vertices = np.column_stack((x.ravel(), y.ravel(), (0.015*(x*x+y*y)).ravel()))
    faces = []
    for j in range(n-1):
        for i in range(n-1):
            a = j*n+i
            for f in [(a, a+1, a+n), (a+1, a+n+1, a+n)]:
                if np.all(np.linalg.norm(vertices[list(f), :2], axis=1) < 6.8):
                    faces.append(f)
    faces = np.asarray(faces)
    for k in range(3):
        plaque = ((x+1.8-k*.4)**2+(y-.8)**2 < (2.6-k*.3)**2).ravel()
        colors = np.tile([225, 216, 186], (len(vertices), 1)).astype(np.uint8)
        colors[plaque] = [156, 45, 111]
        p = polydata(vertices, faces, colors)
        folder = directory / f'SYNTHETIC_DEMO_{k+1}'
        folder.mkdir(exist_ok=True)
        w = vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(folder/'roi.vtp')); w.SetInputData(p); w.Write()
    return directory
