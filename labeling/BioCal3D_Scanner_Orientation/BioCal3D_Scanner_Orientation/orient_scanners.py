"""BioCal3D: review one in-plane rotation per scanner, then rotate safe dataset copies."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'vendor'))
from mesh_io import load_mesh, AnnotationStore, atomic_npz
from roi_review_store import read_reviewed, read_state, resolve_review_source
from original_models import OriginalModelIndex

SCANNERS = ('TRIOS5', 'TRIOS3', 'iTERO', 'LABscanner', 'PRIMESCAN')
EXCLUDED = {'__pycache__', '.geometry_history', 'surface_cache', 'feature_cache',
            'code_backups', '.venv', '.labeling-env', '.git'}
STAMP = 'scanner_orientation.json'

def now():
    return datetime.now(timezone.utc).isoformat()

def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_rows(path, data):
    fields = list(dict.fromkeys(k for r in data for k in r))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(data)

def rotation(degrees):
    a = math.radians(float(degrees)); c, s = math.cos(a), math.sin(a)
    t = np.eye(4); t[:2, :2] = [[c, -s], [s, c]]
    return t

def points(p, t):
    p = np.asarray(p, float)
    if p.shape[-1:] != (3,) or not np.isfinite(p).all():
        raise ValueError('Expected finite XYZ coordinates.')
    return p @ t[:3, :3].T + t[:3, 3]

def directions(v, t):
    return np.asarray(v) @ t[:3, :3].T

def extraction(d, t):
    """Rotate the circular crop center and its tilted normal, without recropping."""
    d = dict(d)
    if 'settings' in d: d['settings'] = extraction(d['settings'], t)
    if all(k in d for k in ('cx', 'cy', 'z')):
        p = points(np.array([d['cx'], d['cy'], d['z']]), t)
        d.update(zip(('cx', 'cy', 'z'), map(float, p)))
    if 'tilt_x' in d and 'tilt_y' in d:
        tx, ty = np.deg2rad([d['tilt_x'], d['tilt_y']])
        n = directions([np.tan(ty), -np.tan(tx), 1.], t)
        d['tilt_x'] = float(np.rad2deg(np.arctan2(-n[1], n[2])))
        d['tilt_y'] = float(np.rad2deg(np.arctan2(n[0], n[2])))
    return d

def transform_npz(path, t):
    d = arrays(path)
    for k in ('vertices', 'verts', 'points', 'xyz', 'V', 'xyz_canonical'):
        if k in d: d[k] = points(d[k], t)
    for k in ('normals', 'vertex_normals', 'face_normals'):
        if k in d: d[k] = directions(d[k], t)
    if 'origin' in d: d['origin'] = points(d['origin'], t)
    if 'basis' in d: d['basis'] = t[:3, :3] @ d['basis']
    if 'original_to_canonical' in d:
        d['original_to_canonical'] = t @ d['original_to_canonical']
        d['canonical_to_original'] = np.linalg.inv(d['original_to_canonical'])
    elif 'canonical_to_original' in d:
        d['canonical_to_original'] = d['canonical_to_original'] @ np.linalg.inv(t)
    if 'extraction_json' in d:
        d['extraction_json'] = np.array(json.dumps(extraction(json.loads(str(d['extraction_json'].item())), t)))
    atomic_npz(path, **d)

def transform_mesh(path, t):
    """Edit coordinates/normals only: preserve all PLY properties and VTP arrays."""
    if path.suffix.lower() == '.ply':
        from plyfile import PlyData
        d = PlyData.read(str(path), mmap=False)
        v = d['vertex'].data
        for names, fn in [(('x','y','z'), points), (('nx','ny','nz'), directions)]:
            if all(n in v.dtype.names for n in names):
                q = fn(np.column_stack([v[n] for n in names]), t)
                for i, n in enumerate(names): v[n] = q[:, i]
        d.write(str(path))
    else:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
        reader = vtk.vtkXMLPolyDataReader(); reader.SetFileName(str(path)); reader.Update()
        p = vtk.vtkPolyData(); p.DeepCopy(reader.GetOutput())
        p.GetPoints().SetData(numpy_to_vtk(points(vtk_to_numpy(p.GetPoints().GetData()), t), deep=True))
        for data in (p.GetPointData(), p.GetCellData()):
            for i in range(data.GetNumberOfArrays()):
                a = data.GetArray(i)
                if a is not None and a.GetNumberOfComponents() == 3 and a.GetName() in ('Normals','normals','vertex_normals','face_normals'):
                    vtk_to_numpy(a)[:] = directions(vtk_to_numpy(a), t)
        a = p.GetFieldData().GetAbstractArray('ROIExtractionJSON')
        if a is not None:
            for i in range(a.GetNumberOfValues()): a.SetValue(i, json.dumps(extraction(json.loads(a.GetValue(i)), t)))
        writer = vtk.vtkXMLPolyDataWriter(); writer.SetFileName(str(path)); writer.SetInputData(p)
        if writer.Write() != 1: raise IOError('Could not write '+str(path))

def scanner_name(value):
    key = ''.join(c for c in str(value).lower() if c.isalnum())
    return next((s for s in SCANNERS if s.lower() == key), None)

def inventory(root):
    root = Path(root).resolve()
    manifest = root/'scan_manifest.csv'
    if not manifest.exists(): manifest = root/'annotation_inventory.csv'
    if not manifest.exists(): raise ValueError('Select a prepared/approved dataset folder containing scan_manifest.csv or annotation_inventory.csv.')
    result = rows(manifest)
    if not result: raise ValueError('The dataset inventory is empty.')
    seen = set()
    for row in result:
        s = scanner_name(row.get('scanner', ''))
        if not s: raise ValueError('Unknown/missing scanner in inventory: '+str(row))
        row['scanner'] = s
        folder = (root/row['sample_dir']).resolve()
        if not folder.is_relative_to(root) or not folder.is_dir() or folder in seen:
            raise ValueError('Invalid/duplicate sample directory: '+str(folder))
        seen.add(folder)
        if not (folder/'geometry.npz').is_file():
            raise ValueError('Missing geometry.npz: '+str(folder))
        mesh = (root/row.get('annotation_mesh', '')).resolve() if row.get('annotation_mesh') else next((folder/n for n in ('roi.ply','roi.vtp') if (folder/n).exists()), None)
        if mesh is None or not mesh.is_relative_to(folder) or not mesh.is_file():
            raise ValueError('Missing/invalid annotation mesh: '+str(folder))
        row['annotation_mesh'] = mesh.relative_to(root).as_posix()
        if not row.get('scan_id'): row['scan_id'] = s+'__'+folder.name
    if len({r['scan_id'] for r in result}) != len(result): raise ValueError('Duplicate scan identities.')
    return result

def existing_stamp(root):
    return (Path(root)/STAMP).exists()

def mesh_equal(a, b, t=None, strict_attributes=True):
    expected = a.vertices if t is None else points(a.vertices, t)
    if not np.array_equal(a.faces, b.faces) or not np.array_equal(a.active, b.active):
        raise ValueError('Mesh connectivity or active-face mask changed.')
    if expected.shape != b.vertices.shape or not np.allclose(expected, b.vertices, atol=2e-6, rtol=1e-7):
        raise ValueError('Unexpected mesh coordinate difference.')
    for name in ('source_face_ids', 'source_vertex_ids', 'imported_labels'):
        x, y = getattr(a, name), getattr(b, name)
        if not strict_attributes and (x is None or y is None): continue
        if (x is None) != (y is None) or (x is not None and not np.array_equal(x,y)):
            raise ValueError('Source IDs or embedded labels changed: '+name)

def validated_mesh(root, row):
    store = AnnotationStore(root, Path(root)/'annotations', 'orientation_check')
    path = Path(root)/row['annotation_mesh']
    m = read_reviewed(store, path)
    base = load_mesh(path)
    mesh_equal(base, m, strict_attributes=False)  # reviews may carry extra provenance arrays
    state = read_state(store, m.path)
    if state and state['status'] != 'approved': raise ValueError('ROI is not approved: '+str(m.path))
    return m

def frame_evidence(root, rr):
    out = {}
    for r in rr:
        g = arrays(Path(root)/r['sample_dir']/'geometry.npz')
        if 'original_to_canonical' not in g: raise ValueError('Missing original_to_canonical for '+r['scan_id'])
        t = np.asarray(g['original_to_canonical'], float)
        if t.shape != (4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]):
            raise ValueError('Invalid coordinate mapping for '+r['scan_id'])
        out[r['scan_id']] = {'scanner':r['scanner'], 'original_to_canonical':t.tolist(),
                             'source_sha256':str(g['source_sha256'].item()) if 'source_sha256' in g else ''}
    return out

def load_config(path):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    if c.get('schema') != 'biocal3d_scanner_orientation_v1' or c.get('reference') != 'TRIOS5':
        raise ValueError('Not a supported TRIOS5 orientation configuration.')
    if not c.get('corrections'): raise ValueError('No saved corrections.')
    for s, r in c['corrections'].items():
        if not r.get('reviewed'): raise ValueError('Unreviewed correction: '+s)
        a = float(r['angle_deg'])
        if not math.isfinite(a) or not np.allclose(rotation(a), r['matrix'], atol=1e-10):
            raise ValueError('Invalid rotation matrix: '+s)
    if 'TRIOS5' not in c['corrections'] or not np.allclose(c['corrections']['TRIOS5']['matrix'],np.eye(4)):
        raise ValueError('TRIOS5 must remain the unchanged orientation reference.')
    return c

def preflight(root, config):
    root = Path(root).resolve()
    if existing_stamp(root): raise ValueError('This dataset has already been oriented; use the original source folder.')
    if (root/'INCOMPLETE_REBUILD.json').exists(): raise ValueError('An incomplete rebuilt snapshot cannot be used.')
    rr = inventory(root); ev = frame_evidence(root,rr)
    for r in rr:
        s, key = r['scanner'], r['scan_id']
        if s not in config['corrections']: raise ValueError('No reviewed correction for '+s)
        old = config.get('frame_evidence',{}).get(key)
        if old is None: raise ValueError('Scan not in calibration frame inventory: '+key+'. Establish orientations using the full approved dataset first.')
        if old['scanner'] != s or not np.allclose(old['original_to_canonical'],ev[key]['original_to_canonical'],atol=1e-8,rtol=0):
            raise ValueError('Coordinate frame differs from the calibration dataset: '+key)
        if old.get('source_sha256') and old['source_sha256'] != ev[key]['source_sha256']:
            raise ValueError('Raw scan identity differs: '+key)
        validated_mesh(root,r)
    return rr

def rerender_maps(folder, t):
    """Rebuild pixel mappings in fixed canonical X/Y, not a co-rotated image basis."""
    from biocal3d_raster import rasterize
    from PIL import Image
    path = folder/'maps.npz'
    if not path.exists(): return
    old = arrays(path); g = arrays(folder/'geometry.npz')
    v, f = g['vertices'], g['faces']
    required = {'origin','basis','fov_mm','rgb','vertices','faces'}
    if not required.issubset(old): raise ValueError('Unsupported maps.npz schema: '+str(path))
    if not np.array_equal(old['faces'], f): raise ValueError('maps.npz face order differs from geometry.')
    origin = points(old['origin'],t)
    normal = old['basis'][:,2] @ t[:3,:3].T
    normal /= np.linalg.norm(normal)
    u = np.array([1.,0.,0.])-normal[0]*normal; u /= np.linalg.norm(u)
    basis = np.column_stack((u,np.cross(normal,u),normal))
    xyz = (v-origin) @ basis
    size = old['rgb'].shape[0]; fov = float(old['fov_mm'])
    if old['rgb'].shape[:2] != (size,size): raise ValueError('Expected square raster maps.')
    if np.abs(xyz[:,:2]).max() > fov/2+1e-5: raise ValueError('Rotated ROI exceeds the existing image FOV; no clipping is permitted.')
    color = g.get('vertex_rgb')
    if color is None: raise ValueError('Missing vertex_rgb needed to regenerate maps.')
    uv = g.get('uv'); texture = None; flip_v = True
    meta_path = folder/'metadata.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    if meta.get('color_source') == 'texture_uv':
        candidate = next((folder/n for n in ('roi.ply','roi.vtp') if (folder/n).exists()),None)
        texture_path = load_mesh(candidate).texture if candidate else None
        if uv is None or texture_path is None or not texture_path.is_file(): raise ValueError('Missing portable UV/texture for '+str(folder))
        texture = np.asarray(Image.open(texture_path).convert('RGB'),dtype=np.float32)/255
        # Recover the actual historical V convention, including --no-flip-v runs.
        from biocal3d_raster import sample_texture
        valid = np.flatnonzero(old['valid'].ravel())
        pick = valid[np.linspace(0,len(valid)-1,min(len(valid),2048),dtype=int)]
        tids = old['triangle_id'].ravel()[pick]
        bary = old['barycentric'].reshape(-1,3)[pick]
        tex_uv = np.einsum('ni,nij->nj',bary,uv[f[tids]])
        expected = old['rgb'].reshape(-1,3)[pick]
        errors = [np.mean(np.abs(sample_texture(texture,tex_uv,b)-expected)) for b in (True,False)]
        if min(errors)>.01: raise ValueError('Portable texture does not reproduce the saved RGB maps; texture convention/identity must be resolved first.')
        flip_v = errors[0]<=errors[1]
    maps = rasterize({'vertices':v,'faces':f,'rgb':color,'uv':uv,'texture':texture},xyz,size,fov,flip_v)
    old.update(maps,vertices=v,faces=f,origin=origin,basis=basis,
               original_to_canonical=g['original_to_canonical'], canonical_to_original=g['canonical_to_original'])
    # Raw-space arrays are intentionally unchanged.
    atomic_npz(path,**old)
    Image.fromarray(np.uint8(np.rint(maps['rgb']*255))).save(folder/'rgb.png')
    Image.fromarray(maps['valid'].astype(np.uint8)*255).save(folder/'valid_mask.png')
    meta.update(valid_pixels=int(maps['valid'].sum()),overlap_fraction=float(maps['overlap_fraction']),projection_extent_mm=float(2*np.abs(xyz[:,:2]).max()),
                raster_surface_area_mm2=float(maps['pixel_surface_area_mm2'].sum()))
    dump(meta_path,meta)

def rebase(value, old_root, new_root):
    if isinstance(value,dict): return {k:rebase(v,old_root,new_root) for k,v in value.items()}
    if isinstance(value,list): return [rebase(v,old_root,new_root) for v in value]
    if isinstance(value,str):
        a = str(old_root); b = value.replace('\\','/'); n = a.replace('\\','/')
        if b == n or b.startswith(n+'/'): return str(new_root)+b[len(n):]
    return value

def included_files(root):
    result = []
    for directory,dirs,names in os.walk(root,followlinks=False):
        base = Path(directory)
        if any((base/d).is_symlink() for d in dirs): raise ValueError('Dataset contains symlink directories.')
        dirs[:] = [d for d in dirs if d not in EXCLUDED]
        for n in names:
            p = base/n
            if p.is_symlink(): raise ValueError('Dataset contains symlink files.')
            if n.endswith(('.lock','.tmp')): raise ValueError('Dataset appears open or has a pending save: '+str(p))
            result.append(p)
    return result

def apply_dataset(root, output, config, log=print):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Output must be a separate sibling/new folder outside the input tree.')
    if output.exists(): raise ValueError('Output already exists; choose a new folder.')
    rr = preflight(root,config)
    files = included_files(root)
    signatures = {p:(p.stat().st_size,p.stat().st_mtime_ns) for p in files}
    output.parent.mkdir(parents=True,exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'_',dir=output.parent))
    log(f'Copying {len(files)} files; originals remain unchanged.')
    try:
        for p in files:
            q = staging/p.relative_to(root); q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
        # Scanner reference definitions and previous model outputs belong to the old frame.
        # Keep them clearly archived, not as active preprocessing/model inputs.
        for name in ('references','labels','predictions','x_anylabeling'):
            if (staging/name).exists():
                shutil.move(staging/name,staging/('orientation_previous_'+name))
        label_paths = [p for p in staging.rglob('*.labels.npz')
                       if not any(part.startswith('orientation_previous_') for part in p.relative_to(staging).parts)]
        labels_by_folder = {}
        for p in label_paths:
            d = arrays(p)
            rel = str(d.get('source_relative','')).replace('\\','/')
            if not rel or Path(rel).is_absolute() or '..' in Path(rel).parts:
                raise ValueError('Invalid/missing label source_relative: '+str(p))
            labels_by_folder.setdefault(Path(rel).parent.as_posix(),[]).append(p)
        known_folders = {Path(r['sample_dir']).as_posix() for r in rr}
        if set(labels_by_folder)-known_folders: raise ValueError('Annotation sources are missing from the scan inventory.')
        audit = []
        for index,r in enumerate(rr,1):
            log(f'[{index}/{len(rr)}] {r["scan_id"]}: {config["corrections"][r["scanner"]]["angle_deg"]:.3f} degrees')
            t = np.asarray(config['corrections'][r['scanner']]['matrix'])
            folder = staging/r['sample_dir']; old_folder = root/r['sample_dir']
            # Only existing ROI coordinate assets are transformed. No recropping/reindexing.
            for p in folder.iterdir():
                if p.suffix.lower() in ('.ply','.vtp'): transform_mesh(p,t)
                elif p.suffix.lower() == '.npz' and p.name != 'maps.npz':
                    d = arrays(p)
                    if any(k in d for k in ('vertices','xyz_canonical')): transform_npz(p,t)
            # Review version IDs/status are preserved; every payload and input fingerprint is updated.
            review_root = staging/'annotations'/'_roi_review'/r['sample_dir']
            if review_root.exists():
                for history in review_root.iterdir():
                    if not history.is_dir(): continue
                    mesh_path = folder/history.name
                    if not mesh_path.is_file(): raise ValueError('Review input missing: '+str(mesh_path))
                    fingerprint = load_mesh(mesh_path).fingerprint
                    for p in history.rglob('surface.npz'): transform_npz(p,t)
                    for p in history.rglob('*.json'):
                        d = json.loads(p.read_text(encoding='utf-8'))
                        if 'input_fingerprint' in d: d['input_fingerprint'] = fingerprint
                        if 'settings' in d: d['settings'] = extraction(d['settings'],t)
                        d['orientation_config_id'] = config['config_id']; dump(p,d)
            # Review aliases and application hash checks must see the newly written mesh files.
            meta_path = folder/'metadata.json'
            meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
            if 'extraction' in meta: meta['extraction'] = extraction(meta['extraction'],t)
            meta.update(orientation='Scanner-specific in-plane rotation to TRIOS5; no pointwise cross-scanner correspondence',
                        orientation_config_id=config['config_id'],orientation_angle_deg=config['corrections'][r['scanner']]['angle_deg'])
            dump(meta_path,meta)
            rerender_maps(folder,t)
            # Saved rater files refer to the exact effective mesh, including shared approved reviews.
            for p in labels_by_folder.get(Path(r['sample_dir']).as_posix(),[]):
                d = arrays(p)
                relative = str(d.get('source_relative','')).replace('\\','/')
                if not relative: raise ValueError('Label file has no source_relative: '+str(p))
                if Path(relative).parent.as_posix() != Path(r['sample_dir']).as_posix(): continue
                rel = Path(relative)
                if rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe annotation source path.')
                a_store = AnnotationStore(root,root/'annotations','orientation_check')
                b_store = AnnotationStore(staging,staging/'annotations','orientation_check')
                a, b = read_reviewed(a_store,root/rel),read_reviewed(b_store,staging/rel)
                if str(d.get('fingerprint','')) != a.fingerprint: raise ValueError('Existing label/mesh fingerprint mismatch: '+str(p))
                mesh_equal(a,b,t)
                if len(d['face_labels']) != len(b.faces): raise ValueError('Label count mismatch.')
                d['fingerprint'] = np.array(b.fingerprint)
                if 'camera' in d:
                    d['orientation_previous_camera'] = d['camera']
                    d['camera'] = np.array('{}')  # reopen in the common canonical top view
                if 'texture_path' in d and b.texture: d['texture_path'] = np.array(str(output/b.texture.relative_to(staging)))
                d['orientation_config_id'] = np.array(config['config_id'])
                atomic_npz(p,**d)
            # Exported labeled meshes may live outside samples; handle them using their scanner path.
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            hashes = {}
            for name,key in [('roi.ply','ply_sha256'),('geometry.npz','geometry_sha256'),('maps.npz','maps_sha256'),('rgb.png','rgb_sha256')]:
                if (folder/name).exists(): hashes[key] = sha(folder/name)
            meta.update(hashes); dump(meta_path,meta)
            audit.append({'scan_id':r['scan_id'],'scanner':r['scanner'],'angle_deg':config['corrections'][r['scanner']]['angle_deg'],
                          'vertices':len(load_mesh(folder/'geometry.npz').vertices),'labels_preserved':True,'hashes':hashes})
            # Retain inventory fields such as study phase/splits; change hashes/orientation only.
            r.update(hashes,orientation_config_id=config['config_id'],orientation_angle_deg=config['corrections'][r['scanner']]['angle_deg'])
        # Transform standalone exported meshes, keeping every PLY face property.
        for p in staging.rglob('*'):
            if p.suffix.lower() not in ('.ply','.vtp') or any(p.is_relative_to(staging/r['sample_dir']) for r in rr): continue
            if '_roi_review' in p.parts: continue
            if any(part.startswith('orientation_previous_') for part in p.relative_to(staging).parts): continue
            matches = {scanner_name(part) for part in p.relative_to(staging).parts if scanner_name(part)}
            if len(matches) != 1: raise ValueError('Cannot identify scanner for additional mesh: '+str(p))
            transform_mesh(p,np.asarray(config['corrections'][matches.pop()]['matrix']))
        by_id = {r['scan_id']:r for r in rr}
        def update_row(r):
            if r.get('scan_id') in by_id:
                current = by_id[r['scan_id']]
                r.update({k:v for k,v in current.items() if k.endswith('_sha256') or k.startswith('orientation_')})
            return r
        for p in staging.rglob('*.csv'):
            data = rows(p)
            if data and 'scan_id' in data[0]: write_rows(p,[update_row(r) for r in data])
        index = staging/'dataset_index.json'
        if index.exists():
            d = json.loads(index.read_text(encoding='utf-8')); d['scans'] = [update_row(r) for r in d['scans']]
            d.setdefault('config',{}).update(orientation_config_id=config['config_id'],output=str(output),feature_caches_must_be_rebuilt=True)
            dump(index,d)
        # Existing QC and 2D polygons are historical; they must not masquerade as rotated outputs.
        stale = []
        for p in list(staging.rglob('*')):
            if not p.is_file(): continue
            if p.name in ('qc.png','context_qc.png') or p.name.startswith('contact_sheet'):
                dest = staging/'orientation_previous_previews'/p.relative_to(staging)
                dest.parent.mkdir(parents=True,exist_ok=True); shutil.move(p,dest); stale.append(str(p.relative_to(staging)))
        # Rebase internal absolute dataset paths after all geometry work has finished.
        for p in staging.rglob('*.json'):
            if any(part.startswith('orientation_previous_') for part in p.relative_to(staging).parts): continue
            d = json.loads(p.read_text(encoding='utf-8-sig'))
            d = rebase(rebase(d,root,output),staging,output)
            if p.parent.name == 'labeling_plan' and p.name == 'plan.json':
                d['root'] = str(output)
                d['inventory_sha256'] = sha(staging/'annotation_inventory.csv')
            dump(p,d)
        # Verify preserved current annotations through the exact annotator loader.
        label_count = 0
        for p in label_paths:
            d = arrays(p); rel = Path(str(d['source_relative']).replace('\\','/'))
            b = read_reviewed(AnnotationStore(staging,staging/'annotations','orientation_check'),staging/rel)
            if str(d['fingerprint'].item()) != b.fingerprint: raise ValueError('Final label fingerprint mismatch.')
            source = arrays(root/p.relative_to(staging))
            for k in ('face_labels','active_faces','face_indices','source_face_ids','source_domain','completed','notes','annotator','geometry_review_version','geometry_review_status'):
                if k in source and not np.array_equal(source[k],d[k]): raise ValueError('Annotation value changed: '+k)
            label_count += 1
        for r in rr:
            a = validated_mesh(root,r); b = validated_mesh(staging,r)
            mesh_equal(a,b,np.asarray(config['corrections'][r['scanner']]['matrix']))
        if set(included_files(root)) != set(files) or any((p.stat().st_size,p.stat().st_mtime_ns)!=sig for p,sig in signatures.items()):
            raise ValueError('The input dataset changed while copying. Close all annotators and retry.')
        dump(staging/STAMP,{'status':'complete','created_utc':now(),'config_id':config['config_id'],'reference':'TRIOS5',
                           'source_root':str(root),'scans':len(rr),'annotation_files_verified':label_count,
                           'old_features_and_predictions_compatible':False,'historical_previews':stale,'audit':audit})
        dump(staging/'orientation_corrections_used.json',config)
        staging.rename(output)
        log(f'Complete: {output} ({len(rr)} scans; {label_count} label files verified).')
        return output
    except Exception:
        shutil.rmtree(staging,ignore_errors=True)
        raise

def dataset_roots(root):
    root = Path(root).resolve()
    if (root/'scan_manifest.csv').exists() or (root/'annotation_inventory.csv').exists(): return [root]
    found = sorted({p.parent for p in root.rglob('annotation_inventory.csv')})
    if not found: raise ValueError('No prepared datasets found. Select the Approved ROI snapshot or a calibration/agreement folder.')
    return found

def apply_tree(root,output,config,log=print):
    root,output = Path(root).resolve(),Path(output).resolve()
    roots = dataset_roots(root)
    if len(roots)==1 and roots[0]==root: return apply_dataset(root,output,config,log)
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Choose a new output folder outside the source tree.')
    for r in roots: preflight(r,config)
    output.mkdir(parents=True)
    try:
        for r in roots: apply_dataset(r,output/r.relative_to(root),config,log)
        dump(output/'orientation_collection.json',{'status':'complete','datasets':[str(r.relative_to(root)) for r in roots],
                                                 'source_root':str(root),'config_id':config['config_id']})
    except Exception:
        dump(output/'INCOMPLETE_ORIENTATION.json',{'status':'incomplete','do_not_use':True})
        raise
    return output

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command')
    for command in ('apply','check'):
        s = sub.add_parser(command); s.add_argument('--input',type=Path,required=True); s.add_argument('--config',type=Path,required=True)
        if command=='apply': s.add_argument('--output',type=Path,required=True)
    a = p.parse_args(argv)
    try:
        if a.command:
            c = load_config(a.config)
            if a.command=='check':
                for r in dataset_roots(a.input): print(f'{r}: {len(preflight(r,c))} scans passed frame/ROI checks.')
            else: apply_tree(a.input,a.output,c)
        else:
            from orientation_gui import launch
            launch()
        return 0
    except Exception as e:
        print('ERROR:',e,file=sys.stderr); return 2

if __name__=='__main__': raise SystemExit(main())
