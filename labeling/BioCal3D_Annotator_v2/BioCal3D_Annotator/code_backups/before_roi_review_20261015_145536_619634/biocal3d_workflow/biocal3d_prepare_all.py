"""BioCal3D: raw PLY -> registered ROI + PLY/RGB/maps -> direct 3D labeling.

Run `python biocal3d_prepare_all.py prepare --data-root "C:\\path\\to\\data"`.
Run `python biocal3d_prepare_all.py label --output "C:\\path\\to\\snapshot"`.
Run `python biocal3d_prepare_all.py project-labels --output "C:\\path\\to\\snapshot"`.
Keep the two companion Python modules beside this script. See README.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import sys
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from biocal3d_raster import rasterize, sample_texture, triangles
from annotator.mesh_io import Mesh, polydata, export_ply, LABELS

def default_data_root():
    if os.environ.get('BIOCAL3D_DATA_ROOT'):
        return Path(os.environ['BIOCAL3D_DATA_ROOT']).expanduser()
    return None



DEFAULT_ROOT = default_data_root()
SAMPLE = re.compile(r"(?:BCG|G)?(\d+)\s*T(\d+)\s*[-_]\s*(\d+)", re.I)
SCANNERS = {"trios3": "TRIOS3", "trios5": "TRIOS5", "itero": "iTERO", "labscanner": "LABscanner", "primescan": "PRIMESCAN"}
RGB_KEYS = ('RGB', 'rgb', 'colors', 'Colors', 'Color', 'color', 'vertex_colors', 'VertexColors', 'texture_rgb', 'RGBA', 'rgba', 'RGB (uint8)', 'RGB Colors')
UV_KEYS = ('TCoords', 'Texture Coordinates', 'TextureCoordinates', 'tcoords', 'uv', 'UV')
CLASSES = {"clean": 0, "plaque": 1, "ignore": 255}
# Protocol metadata is separate from folder discovery. Unknown positive groups
# are still exported, with their timepoint preserved rather than a guessed stage.
GROUP_PROTOCOLS = {
    1: ('fresh_12h', {0: 'baseline', 1: 'clean'}),
    2: ('fresh_12h', {0: 'baseline', 1: 'clean'}),
    3: ('mature_24h', {0: 'baseline', 1: 'partial', 2: 'clean'}),
    4: ('mature_24h', {0: 'baseline', 1: 'partial', 2: 'clean'}),
    5: ('mature_24h', {0: 'baseline', 1: 'partial', 2: 'clean'}),
    6: ('mature_24h', {0: 'baseline', 1: 'partial', 2: 'clean'}),
}


def dependency_versions():
    versions = {}
    for name in ('pyvista', 'vtk', 'numpy', 'open3d'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not installed'
    return versions


def load_pyvista():
    try:
        import pyvista as pv
        # Force lazy imports now so missing VTK symbols fail before output creation.
        pv.PolyData()
        return pv
    except ImportError as e:
        raise RuntimeError(
            f'PyVista/VTK import failed: {e}\n'
            f'Interpreter: {sys.executable}\nInstalled packages: {dependency_versions()}\n'
            'Use this interpreter to install the supported pair:\n'
            f'& "{sys.executable}" -m pip install --upgrade "pyvista==0.44.2"\n'
            f'& "{sys.executable}" -m pip install --force-reinstall --no-deps "vtk==9.3.1"'
        ) from e


def check_environment():
    print(f'Interpreter: {sys.executable}')
    print(f'Packages: {dependency_versions()}')
    print(f'Default data root: {DEFAULT_ROOT}')
    pv = load_pyvista()
    import vtk
    import vtkmodules.vtkCommonDataModel as model
    print(f'VTK runtime: {vtk.vtkVersion.GetVTKVersion()}')
    print(f'VTK module: {model.__file__}')
    print(f'PyVista module: {pv.__file__}')
    if not hasattr(model, 'VTK_TRIQUADRATIC_PYRAMID'):
        raise RuntimeError('VTK module lacks VTK_TRIQUADRATIC_PYRAMID; reinstall the supported VTK wheel.')
    print('PyVista/VTK mesh import check: OK')
    return 0


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(dict.fromkeys(k for r in rows for k in r)) or ['error']
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def identify(path, root):
    parts = path.relative_to(root).parts
    scanners = {v for part in parts[:-1] for k, v in SCANNERS.items()
                if re.sub(r'[ _-]', '', part.lower()) == k}
    if len(scanners) != 1:
        raise ValueError('Expected exactly one known scanner folder (LABscanner/iTERO/TRIOS3/TRIOS5/PRIMESCAN).')
    match = next((m for p in reversed(parts) if (m := SAMPLE.search(p))), None)
    if match is None:
        raise ValueError('No BCG<group>T<time>-<sample> identity in the path.')
    g, t, n = map(int, match.groups())
    if g < 1 or t < 0 or n < 1:
        raise ValueError(f'Unexpected group/time/sample: G{g}, T{t}, S{n}.')
    growth, stages = GROUP_PROTOCOLS.get(g, ('unspecified', {}))
    if stages and t not in stages:
        raise ValueError(f'G{g} T{t} is outside its configured protocol; update GROUP_PROTOCOLS if intentional.')
    scanner = scanners.pop()
    sample = f'BCG{g}T{t}-{n}'
    return dict(scanner=scanner, group=g, time_idx=t, timepoint=f'T{t}',
                sample_number=n, sample_id=sample, sample_name=sample,
                physical_id=f'G{g}_S{n}', scan_id=f'{scanner}__{sample}',
                stage=stages.get(t, f'timepoint_T{t}'), growth_condition=growth)


def discover(root, pattern, excluded):
    rows, failures, keys = [], [], {}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or (p.suffix.lower() != '.ply' if pattern == 'raw' else p.name != 'roi.vtp'):
            continue
        if any(x == p or x in p.parents for x in excluded):
            continue
        if pattern == 'raw' and any(x.startswith('_') for x in p.relative_to(root).parts[:-1]):
            continue
        try:
            r = identify(p, root)
        except ValueError as e:
            failures.append(dict(source=str(p), status='FAILED_DISCOVERY', error=str(e)))
            continue
        if r['scan_id'] in keys:
            raise ValueError(f"Duplicate {r['scan_id']}: {keys[r['scan_id']]} and {p}. "
                             'Keep one authoritative raw scan per identity; separate experiments into separate roots.')
        keys[r['scan_id']] = p
        rows.append({**r, 'source': str(p)})
    if not rows:
        raise RuntimeError(f'No identifiable {pattern} meshes under {root}. Discovery errors: {failures[:3]}')
    return rows, failures


def prepare_reference(core, scanner, rows, reference_root):
    import open3d as o3d
    core.REFERENCE_ROOT = reference_root
    template = core.get_template_file(scanner)
    if template.exists():
        reference = core.load_reference(scanner)
        reference_id = reference['reference_sample_id']
    else:
        reference = None
        reference_id = core.REFERENCE_SAMPLE_BY_SCANNER[scanner]
    matches = [r for r in rows if r['scanner'] == scanner and r['sample_id'] == reference_id]
    if not matches and reference is None and scanner == 'PRIMESCAN':
        baseline = sorted([r for r in rows if r['scanner'] == scanner and r['time_idx'] == 0],
                          key=lambda r: (r['group'], r['sample_number']))
        if baseline:
            matches = [baseline[0]]
            reference_id = matches[0]['sample_id']
            print(f'PRIMESCAN: BCG1T0-1 absent; using baseline {reference_id} as reference.', flush=True)
    if len(matches) != 1:
        raise ValueError(f'{scanner}: reference {reference_id} must be present exactly once under the data root.')
    record = {**matches[0], 'ply_path': Path(matches[0]['source'])}
    record['texture_path'] = core.find_texture(record['ply_path'], reference_id)
    mesh, _, _ = core.load_mesh(record['ply_path'])
    if reference is None:
        print(f'{scanner}: define the reference circle once in the interactive window.', flush=True)
        reference = core.define_reference_roi(mesh, core.load_texture(record['texture_path']), scanner, reference_id)
        core.save_reference(scanner, reference, record)
    points = np.asarray(mesh.points)
    down, feature = core.prepare_cloud(points, core.VOXEL_SIZE)
    holder_points = points[core.holder_mask(points, reference, core.HOLDER_HALF_WIDTH_FACTOR)]
    if len(holder_points) < 20:
        raise ValueError(f'{scanner}: saved reference has too few holder points; inspect the reference.')
    holder = core.points_to_o3d(holder_points).voxel_down_sample(.25)
    core.estimate_normals(holder, 1.)
    return dict(record=record, reference=reference, down=down, feature=feature,
                holder_cloud=holder, holder_points=holder_points)


def register_roi(core, mesh, system, source):
    import open3d as o3d
    o3d.utility.random.seed(42)
    is_reference = source.resolve() == system['record']['ply_path'].resolve()
    candidates, evaluated, errors = [], [], []
    if is_reference:
        candidates.append(dict(transform=np.eye(4), method='REFERENCE', global_fitness=1.,
                               global_rmse=0., full_fitness=1., full_rmse=0., holder_fitness=1.,
                               holder_rmse=0., holder_points=len(system['holder_points'])))
    else:
        try:
            candidate = core.primary_registration(mesh, system)
            if candidate is not None:
                candidates.append(candidate)
        except Exception as e:
            errors.append(f'Primary: {e}')
    def evaluate(candidate):
        try:
            evaluated.append(core.evaluate_registration(mesh, system['reference'], candidate))
        except Exception as e:
            errors.append(f"{candidate['method']}: {e}")
    for candidate in candidates:
        evaluate(candidate)
    weak = not candidates or not core.registration_is_strong(candidates[0])
    valid = [x for x in evaluated if x['quality']['ok']]
    if not is_reference and system['record']['scanner'] in core.TRIOS_SCANNERS and (weak or not valid):
        try:
            fallback = core.trios_rgb_fallback_registration(mesh, system)
            if fallback is not None:
                evaluate(fallback)
        except Exception as e:
            errors.append(f'RGB fallback: {e}')
        valid = [x for x in evaluated if x['quality']['ok']]
    if not valid:
        raise ValueError(f'No candidate passed ROI quality: {[x["quality"] for x in evaluated]}; {errors}')
    return max(valid, key=lambda x: core.registration_score(x['registration']))


def get_array(data, keys):
    return next((np.asarray(data[k]) for k in keys if k in data), None)


def mesh_colors(mesh, texture_path, flip_v, prefer_texture=False):
    # Embedded vertex RGB takes precedence; this avoids the old scrambled iTERO UV mapping.
    rgb = None if prefer_texture else get_array(mesh.point_data, RGB_KEYS)
    if rgb is not None:
        if rgb.ndim != 2 or rgb.shape[1] < 3 or len(rgb) != mesh.n_points:
            raise ValueError('Invalid vertex RGB shape.')
        rgb = rgb[:, :3]
        scale = 255. if np.issubdtype(rgb.dtype, np.integer) or np.nanmax(rgb) > 1.5 else 1.
        rgb = np.asarray(rgb, np.float32) / scale
        if not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 1.0001):
            raise ValueError('Non-finite/out-of-range vertex RGB.')
        return dict(rgb=rgb, uv=None, texture=None, color_source='vertex_rgb', texture_path='')
    uv = get_array(mesh.point_data, UV_KEYS)
    if uv is None and mesh.active_texture_coordinates is not None:
        uv = np.asarray(mesh.active_texture_coordinates)
    if uv is None or uv.ndim != 2 or uv.shape[1] < 2 or len(uv) != mesh.n_points:
        raise ValueError('No usable RGB or UV coordinates; refusing to substitute gray geometry for color.')
    if not texture_path or not Path(texture_path).is_file():
        raise ValueError('UV mesh needs its original external JPG/PNG texture.')
    uv = np.asarray(uv[:, :2], float)
    if not np.isfinite(uv).all():
        raise ValueError('Non-finite texture coordinates.')
    with Image.open(texture_path) as im:
        texture = np.asarray(im.convert('RGB'), dtype=np.float32) / 255.
    sample_texture(texture, uv, flip_v)  # validate UV range/orientation convention
    return dict(rgb=None, uv=uv, texture=texture, color_source='texture_uv', texture_path=str(texture_path))


def project_plane(vertices, faces):
    tri = vertices[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)/2
    weights = np.bincount(faces.ravel(), weights=np.repeat(areas/3, 3), minlength=len(vertices))
    if weights.sum() <= 0:
        raise ValueError('Degenerate ROI mesh.')
    origin = np.average(vertices, axis=0, weights=weights)
    c = vertices-origin
    _, axes = np.linalg.eigh((c*weights[:, None]).T@c/weights.sum())
    normal = axes[:, 0]
    if normal[2] < 0:
        normal = -normal
    if abs(normal[2]) < .5:
        raise ValueError('ROI plane not approximately canonical XY; inspect reference/crop.')
    # Prefer canonical X to avoid arbitrary PCA rotations between repeated runs.
    u = np.array([1., 0., 0.])-normal[0]*normal
    u /= np.linalg.norm(u)
    basis = np.column_stack((u, np.cross(normal, u), normal))
    xyz = c@basis
    xycenter = (xyz[:, :2].min(0)+xyz[:, :2].max(0))/2
    origin += basis[:, :2]@xycenter
    xyz[:, :2] -= xycenter
    return xyz, origin, basis, float(2*np.abs(xyz[:, :2]).max()), areas


def export_sample(row, mesh, colors, transform, roi_radius, folder, size, fov, flip_v):
    mesh = mesh.triangulate()
    vertices = np.asarray(mesh.points, float)
    faces = triangles(mesh.faces)
    if not np.isfinite(vertices).all() or len(faces) == 0:
        raise ValueError('Invalid ROI vertices/triangles.')
    xyz, origin, basis, extent, areas = project_plane(vertices, faces)
    if extent > fov + 1e-6:
        raise ValueError(f'ROI span {extent:.3f} mm exceeds FOV {fov:.3f}; use a new run with --fov-mm.')
    maps = rasterize(dict(vertices=vertices, faces=faces, **colors), xyz, size, fov, flip_v)
    folder.mkdir(parents=True, exist_ok=True)
    # Exact row order of roi.vtp is the index convention for every saved map.
    vertex_rgb = colors['rgb'] if colors['rgb'] is not None else sample_texture(colors['texture'], colors['uv'], flip_v)
    mesh.point_data['RGB'] = np.uint8(np.rint(vertex_rgb*255))
    mesh = mesh.compute_normals(cell_normals=True, point_normals=True, split_vertices=False,
                               consistent_normals=False, auto_orient_normals=False)
    mesh.save(folder/'roi.vtp')
    raw_vid = np.asarray(mesh.point_data.get('RawVertexId', np.full(len(vertices), -1)), np.int64)
    raw_fid = np.asarray(mesh.cell_data.get('RawTriangleId', np.full(len(faces), -1)), np.int64)
    display_uv=colors['uv'].copy() if colors['uv'] is not None else None
    if display_uv is not None and not flip_v:
        display_uv[:,1]=1-display_uv[:,1]
    display=polydata(vertices,faces,np.uint8(np.rint(vertex_rgb*255)),display_uv)
    annotation_mesh=Mesh(Path(row['source']).resolve(),vertices,faces,display,np.ones(len(faces),bool),
                         Path(colors['texture_path']).resolve() if display_uv is not None else None)
    export_ply(annotation_mesh,None,folder/'roi.ply',face_ids=raw_fid)
    raw_xyz = (np.column_stack((vertices, np.ones(len(vertices))))@np.linalg.inv(transform).T)[:, :3]
    geom = dict(vertices=vertices, faces=faces, normals=np.asarray(mesh.point_data['Normals']),
                vertex_rgb=vertex_rgb.astype(np.float32), vertices_raw=raw_xyz,
                raw_vertex_id=raw_vid, raw_triangle_id=raw_fid, roi_mask=np.ones(len(vertices), bool),
                original_to_canonical=transform, canonical_to_original=np.linalg.inv(transform),
                roi_radius=float(roi_radius), source_ply=row['source'], source_texture=colors['texture_path'],
                scanner=row['scanner'], group=row['group'], timepoint=row['timepoint'], sample_id=row['sample_id'],
                registration_method=row['registration_method'], holder_fitness=row['holder_fitness'],
                holder_rmse=row['holder_rmse'])
    for key in ('roi_points', 'roi_area_coverage', 'roi_edge_reach'):
        if key in row:
            geom[key] = row[key]
    if colors['uv'] is not None:
        geom['uv'] = colors['uv']
    np.savez_compressed(folder/'geometry.npz', **geom)
    Image.fromarray(np.uint8(np.rint(maps['rgb']*255))).save(folder/'rgb.png')
    Image.fromarray(maps['valid'].astype(np.uint8)*255).save(folder/'valid_mask.png')
    np.savez_compressed(folder/'maps.npz', **maps, vertices=vertices, faces=faces,
                        vertices_raw=raw_xyz, raw_vertex_id=raw_vid, raw_triangle_id=raw_fid,
                        origin=origin, basis=basis, fov_mm=fov, mm_per_pixel=fov/size,
                        original_to_canonical=transform, canonical_to_original=np.linalg.inv(transform))
    metadata = {**row, 'schema_version': 2, 'color_source': colors['color_source'],
                'texture_path': colors['texture_path'], 'image_size': size, 'fov_mm': fov,
                'mm_per_pixel': fov/size, 'roi_radius': float(roi_radius),
                'valid_pixels': int(maps['valid'].sum()), 'projection_extent_mm': extent,
                'mesh_surface_area_mm2': float(areas.sum()),
                'raster_surface_area_mm2': float(maps['pixel_surface_area_mm2'].sum()),
                'overlap_fraction': maps['overlap_fraction'], 'rgb_sha256': sha256(folder/'rgb.png'),
                'maps_sha256': sha256(folder/'maps.npz'), 'geometry_sha256': sha256(folder/'geometry.npz'),
                'orientation': 'Per-scan plane alignment; no pointwise cross-scan registration',
                'mapping': 'row,column -> triangle_id -> faces -> barycentric -> canonical/raw XYZ',
                'units': 'mm (assumed from scanner export; verify before interpreting dimensions)',
                'source_sha256': sha256(row['source'])}
    metadata['ply_sha256']=sha256(folder/'roi.ply')
    metadata['ply_texture']=str(annotation_mesh.texture) if display_uv is not None else ''
    metadata['ply_uv_convention']='standard VTK UV; texture V adjusted when --no-flip-v is used'
    if colors['texture_path']:
        metadata['texture_sha256'] = sha256(colors['texture_path'])
    write_json(folder/'metadata.json', metadata)
    return metadata


def create_qc(folder):
    with np.load(folder/'maps.npz', allow_pickle=False) as z:
        rgb = np.uint8(np.rint(z['rgb']*255)); valid = z['valid']; height = z['height_mm']
    # RGB left; surface coverage middle; grayscale height right. These are QA, never inputs.
    h, w = valid.shape
    cov = np.zeros((h, w, 3), np.uint8); cov[valid] = [40, 190, 100]
    gray = np.zeros((h, w), np.uint8)
    lo, hi = np.percentile(height[valid], [2, 98])
    gray[valid] = np.uint8(np.clip((height[valid]-lo)/max(hi-lo, 1e-8), 0, 1)*255)
    sheet = Image.new('RGB', (w*3, h+28), 'white')
    for i, (title, arr) in enumerate([('Unlit RGB', rgb), ('Valid surface', cov), ('Height (per-scan plane)', np.repeat(gray[..., None], 3, axis=2))]):
        sheet.paste(Image.fromarray(arr), (i*w, 28))
        ImageDraw.Draw(sheet).text((i*w+8, 8), title, fill='black')
    sheet.thumbnail((1500, 530)); sheet.save(folder/'qc.png')


def contact_sheets(rows, out):
    for scanner in sorted({r['scanner'] for r in rows}):
        scans = [r for r in rows if r['scanner'] == scanner]
        for start in range(0, len(scans), 12):
            sheet = Image.new('RGB', (1440, 660), 'white')
            draw = ImageDraw.Draw(sheet)
            for i, row in enumerate(scans[start:start+12]):
                x, y = (i % 3)*480, (i//3)*165
                folder = out/row['sample_dir']
                source = folder/'context_qc.png'
                if not source.exists():
                    source = folder/'qc.png'
                with Image.open(source) as im:
                    tile = im.convert('RGB'); tile.thumbnail((472, 138))
                    sheet.paste(tile, (x+4, y+2))
                draw.text((x+6, y+144), f"{row['sample_id']} | {row['status']} | {row['registration_method']}", fill='black')
            path = out/'qc_overviews'/f'{scanner}_{start//12+1:03d}.jpg'
            path.parent.mkdir(parents=True, exist_ok=True); sheet.save(path, quality=92)


def make_splits(rows, output, import_path):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    specimens = pd.DataFrame(rows)[['physical_id', 'group', 'sample_number', 'growth_condition']].drop_duplicates().sort_values(['group', 'sample_number'])
    if import_path:
        splits = pd.read_csv(import_path)
        if not {'physical_id', 'split'} <= set(splits) or splits.physical_id.duplicated().any() or not set(splits.split) <= {'train', 'val', 'test'}:
            raise ValueError('Invalid imported specimen split file.')
        missing = set(specimens.physical_id)-set(splits.physical_id)
        if missing:
            raise ValueError(f'Imported split lacks new specimens {sorted(missing)}. Define a new experiment split explicitly.')
        splits = specimens.merge(splits[['physical_id', 'split']], on='physical_id', validate='one_to_one')
    else:
        if (specimens.growth_condition == 'unspecified').any():
            raise ValueError('New groups have unspecified growth protocols. Configure GROUP_PROTOCOLS before '
                             '--make-splits, supply an existing --splits file, or prepare without splits.')
        try:
            dev, test = train_test_split(specimens, test_size=.1, stratify=specimens.growth_condition, random_state=42)
            train, val = train_test_split(dev, test_size=1/9, stratify=dev.growth_condition, random_state=43)
        except ValueError as e:
            raise ValueError('Too few specimens for growth-stratified 80/10/10. Prepare without --make-splits, or supply --splits.') from e
        splits = pd.concat([x.assign(split=s) for x, s in [(train, 'train'), (val, 'val'), (test, 'test')]])
    splits.sort_values(['group', 'sample_number']).to_csv(output/'specimen_splits.csv', index=False)
    return splits.set_index('physical_id')['split'].to_dict()


def prepare(args):
    pv = load_pyvista()
    root = args.data_root.resolve()
    if not root.is_dir():
        raise ValueError(f'Data root does not exist: {root}')
    if args.processed_root and not args.processed_root.is_dir():
        raise ValueError('--processed-root must exist.')
    out = (args.output or root/('_development_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if out == root or out in root.parents or (args.processed_root and out == args.processed_root.resolve()):
        raise ValueError('Output must be a separate new directory.')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is not empty. Use a new output snapshot to protect images, labels, and maps.')
    out.mkdir(parents=True, exist_ok=True)
    config = dict(schema_version=2, data_root=str(root), output=str(out), image_size=args.size,
                  flip_v=not args.no_flip_v, coordinate_units='mm', group_protocols=GROUP_PROTOCOLS,
                  created_at=datetime.now().astimezone().isoformat(),
                  script_sha256=sha256(__file__), dependency_versions=dependency_versions(), companion_sha256={n: sha256(Path(__file__).with_name(n))
                      for n in ['biocal3d_roi_core.py', 'biocal3d_raster.py']})
    config['annotation_mode']='direct_3d_faces'
    config['annotation_mesh_format']='binary_little_endian_ply'
    config['export_x_anylabeling']=args.export_x_anylabeling
    config['annotator_mesh_io_sha256']=sha256(Path(__file__).parent/'annotator'/'mesh_io.py')
    rows, failures = discover(args.processed_root.resolve() if args.processed_root else root,
                              'roi' if args.processed_root else 'raw', [out])
    for group in sorted({r['group'] for r in rows if r['growth_condition'] == 'unspecified'}):
        print(f'G{group}: included with original timepoint names; growth/stage protocol unspecified.', flush=True)
    # Failures and split assignments are visible even if a later step stops the run.
    write_csv(out/'discovery_inventory.csv', rows)
    write_csv(out/'failures.csv', failures)
    assignments = make_splits(rows, out, args.splits) if args.make_splits or args.splits else {}
    for row in rows:
        row['split'] = assignments.get(row['physical_id'], 'unassigned')
    systems, core = {}, None
    if not args.processed_root:
        import biocal3d_roi_core as core
        core.DATA_ROOT = root
        core.OUTPUT_ROOT = out
        core.REFERENCE_ROOT = out/'references'
        core.CONTACT_SHEET_ROOT = out/'contact_sheets'
        core.MANIFEST_FILE = out/'manifest.csv'
        if args.references:
            if not args.references.is_dir():
                raise ValueError(f'Reference directory does not exist: {args.references}')
            previous_roots = [args.references.resolve()]
        else:
            previous_roots = [root/'_processed_ROI'/'_references']
            snapshots = sorted([p for p in root.glob('_development_*') if p.is_dir() and p != out],
                               key=lambda p: p.name)
            previous_roots.extend(p/'references' for p in snapshots)
        config['reference_sources'] = [str(p) for p in previous_roots if p.is_dir()]
        for previous in previous_roots:
            if not previous.is_dir():
                continue
            # Only copy per-scanner reference templates, never old processed meshes.
            for p in previous.rglob('reference_roi.npz'):
                dest = core.REFERENCE_ROOT/p.relative_to(previous)
                dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
        for scanner in sorted({r['scanner'] for r in rows}):
            try:
                print(f'Preparing {scanner} reference...', flush=True)
                systems[scanner] = prepare_reference(core, scanner, rows, core.REFERENCE_ROOT)
            except Exception as e:
                failures.append(dict(scanner=scanner, status='FAILED_REFERENCE', error=str(e)))
                print(f'{scanner} reference FAILED: {e}', flush=True)
        if not systems:
            write_csv(out/'failures.csv', failures)
            raise ValueError('No usable references. See failures.csv.')
    fov = args.fov_mm
    if fov is None and systems:
        fov = math.ceil(max(s['reference']['roi_radius']*2 for s in systems.values())*1.10*10)/10
    if fov is None:
        # Maps-only mode: conservative physical extent from all ROI geometry, never labels.
        extents = []
        for r in rows:
            try:
                m = pv.read(r['source']).triangulate()
                extents.append(project_plane(np.asarray(m.points), triangles(m.faces))[3])
            except Exception as e:
                failures.append(dict(scan_id=r['scan_id'], status='FAILED_EXTENT', error=str(e)))
        if not extents:
            write_csv(out/'failures.csv', failures)
            raise ValueError('No readable ROI geometry.')
        fov = math.ceil(max(extents)*1.10*10)/10
    config['fov_mm'] = fov
    config['fov_source'] = 'explicit' if args.fov_mm else ('reference circle' if systems else 'all ROI extents')
    write_json(out/'run_config.json', config)
    results, queue = [], []
    for i, row in enumerate(rows, 1):
        print(f'[{i}/{len(rows)}] {row["scan_id"]}', flush=True)
        folder = out/'samples'/row['scanner']/f'G{row["group"]}'/row['sample_id']
        try:
            texture_path = None
            prefer_texture = False
            if core:
                if row['scanner'] not in systems:
                    raise ValueError('Scanner reference failed.')
                raw = Path(row['source'])
                mesh, _, _ = core.load_mesh(raw)
                mesh = mesh.triangulate()
                texture_path = core.find_texture(raw, row['sample_id'])
                chosen = register_roi(core, mesh, systems[row['scanner']], raw)
                mesh = chosen['specimen'].triangulate()
                transform = chosen['canonical']@chosen['registration']['transform']
                radius = chosen['roi_radius']
                reg, q = chosen['registration'], chosen['quality']
                status = 'CHECK' if reg['holder_fitness'] < core.MIN_HOLDER_FITNESS or reg['holder_rmse'] > core.MAX_HOLDER_RMSE else 'OK'
                row.update(status=status, registration_method=reg['method'], holder_fitness=float(reg['holder_fitness']),
                           holder_rmse=float(reg['holder_rmse']), roi_points=q['n_points'],
                           roi_area_coverage=q['area_coverage'], roi_edge_reach=q['edge_reach'])
            else:
                mesh = pv.read(row['source']).triangulate()
                previous_metadata = Path(row['source']).with_name('metadata.json')
                if previous_metadata.exists():
                    previous_info = json.loads(previous_metadata.read_text(encoding='utf-8'))
                    prefer_texture = previous_info.get('color_source') == 'texture_uv'
                geo = Path(row['source']).with_name('geometry.npz')
                # Require transform: silently inventing raw coordinates would invalidate back-projection.
                if not geo.exists():
                    raise ValueError('Maps-only requires geometry.npz with original_to_canonical beside roi.vtp.')
                with np.load(geo, allow_pickle=False) as z:
                    transform = np.asarray(z['original_to_canonical'], float)
                    radius = float(z['roi_radius'])
                    texture_path = str(z['source_texture']) if 'source_texture' in z else None
                    if 'source_ply' in z and Path(str(z['source_ply'])).is_file():
                        row['source'] = str(z['source_ply'])
                    row['registration_method'] = str(z['registration_method']) if 'registration_method' in z else 'EXISTING_ROI'
                    row['holder_fitness'] = float(z['holder_fitness']) if 'holder_fitness' in z else 0.
                    row['holder_rmse'] = float(z['holder_rmse']) if 'holder_rmse' in z else 1e9
                row['status'] = 'OK' if row['holder_fitness'] >= .25 and row['holder_rmse'] <= .70 else 'CHECK'
                if texture_path and not Path(texture_path).is_file():
                    # Recover relocated external textures using sample identity, avoiding other experiments.
                    candidates = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
                                  and not any(x.startswith('_') for x in p.relative_to(root).parts[:-1])]
                    matches = []
                    for p in candidates:
                        try:
                            identity = identify(p, root)
                            if identity['scan_id'] == row['scan_id']: matches.append(p)
                        except ValueError:
                            pass
                    if len(matches) == 1: texture_path = str(matches[0])
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError('Invalid original_to_canonical transform.')
            colors = mesh_colors(mesh, texture_path, not args.no_flip_v, prefer_texture)
            info = export_sample(row, mesh, colors, transform, radius, folder, args.size, fov, not args.no_flip_v)
            create_qc(folder)
            if core and not args.no_context_qc:
                try:
                    context = core.create_context(chosen['aligned_mesh'], systems[row['scanner']]['reference'],
                                                  chosen['z_offset'], chosen['canonical'])
                    core.save_qc_preview(context, chosen['specimen'], core.load_texture(texture_path), radius, folder/'context_qc.png')
                except Exception as e:
                    failures.append(dict(scan_id=row['scan_id'], status='QC_WARNING', error=str(e)))
            rel = folder.relative_to(out).as_posix()
            entry = {**info, 'sample_dir': rel, 'annotation_mesh': rel+'/roi.ply', 'annotation_image': '', 'annotation_json': ''}
            if row['status'] == 'OK' or args.include_check:
                if args.export_x_anylabeling:
                    image = out/'x_anylabeling'/(row['scan_id']+'.png')
                    image.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(folder/'rgb.png', image)
                    entry['annotation_image'] = image.relative_to(out).as_posix()
                    entry['annotation_json'] = image.with_suffix('.json').relative_to(out).as_posix()
                queue.append({**entry, 'label_status': 'unlabelled', 'priority': {'partial': 1, 'baseline': 2, 'clean': 3}.get(row['stage'], 4)})
            results.append(entry)
        except Exception as e:
            failures.append({**row, 'status': 'FAILED_PREPARATION', 'error': str(e)})
            print(f'FAILED: {e}', flush=True)
        write_csv(out/'scan_manifest.csv', results)
        write_csv(out/'failures.csv', failures)
    queue.sort(key=lambda r: (r['split'], r['priority'], r['physical_id'], r['scanner']))
    write_csv(out/'annotation_inventory.csv', queue)
    (out/'classes.txt').write_text('clean\nplaque\nignore\n', encoding='utf-8')
    write_json(out/'class_map.json', CLASSES)
    write_json(out/'annotation_classes.json', LABELS)
    write_json(out/'dataset_index.json', {'schema_version': 2, 'config': config, 'scans': results})
    if results:
        try:
            write_json(Path(__file__).parent/'last_prepared.json',{'output':str(out),'prepared_scans':len(results),'queued_scans':len(queue)})
        except OSError as e:
            print(f'Could not save the last-run shortcut ({e}); use label --output "{out}".',flush=True)
    contact_sheets(results, out)
    print(f'\nPrepared {len(results)}/{len(rows)} scans; {len(queue)} 3D annotation meshes. Output: {out}', flush=True)
    print('Review failures.csv, CHECK scans, context QC, texture orientation and valid coverage before labeling.')
    print(f'Launch 3D labeling: python biocal3d_prepare_all.py label --output "{out}"')
    return 2 if any(r.get('status', '').startswith('FAILED') for r in failures) else 0


def shape_mask(shape, size):
    points = np.asarray(shape.get('points', []), float)
    kind = shape.get('shape_type', 'polygon')
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Invalid shape points.')
    mask = Image.new('L', size, 0); draw = ImageDraw.Draw(mask)
    pts = [tuple(p) for p in points]
    if kind in {'polygon', 'quadrilateral'} and len(pts) >= 3:
        draw.polygon(pts, fill=1)
    elif kind == 'rectangle' and len(pts) == 2:
        a, b = np.minimum(points[0], points[1]), np.maximum(points[0], points[1])
        draw.rectangle([tuple(a), tuple(b)], fill=1)
    elif kind == 'circle' and len(pts) == 2:
        x, y = points[0]; r = float(np.linalg.norm(points[1]-points[0]))
        draw.ellipse((x-r, y-r, x+r, y+r), fill=1)
    else:
        raise ValueError(f'Unsupported segmentation shape: {kind}. Use polygons/rectangles/circles.')
    return np.asarray(mask, bool)


def annotations_to_mask(data, valid, background):
    h, w = valid.shape
    if data.get('imageHeight') != h or data.get('imageWidth') != w:
        raise ValueError('Annotation size differs from saved image/maps. Do not resize labeling images.')
    if background == 'clean' and not (data.get('checked') is True or (data.get('flags') or {}).get('complete') is True):
        raise ValueError('--background clean requires checked=true or flags.complete=true for every image. '
                         'Review the entire surface before declaring unmarked pixels clean.')
    mask = np.full((h, w), 255, np.uint8)
    if background == 'clean': mask[valid] = 0
    grouped = {label: [] for label in CLASSES}
    for shape in data.get('shapes', []):
        label = shape.get('label', '').strip().lower()
        if label not in CLASSES:
            raise ValueError(f'Unknown label {label!r}; expected clean/plaque/ignore.')
        grouped[label].append(shape)
    # Explicit precedence is independent of drawing order: clean < plaque < ignore.
    for label in ('clean', 'plaque', 'ignore'):
        for shape in grouped[label]:
            mask[shape_mask(shape, (w, h)) & valid] = CLASSES[label]
    mask[~valid] = 255
    return mask


def label_votes(mask, maps, minimum_fraction=.5):
    valid = maps['valid']; tid = maps['triangle_id'][valid]; bary = maps['barycentric'][valid]
    faces = maps['faces']; vertices = maps['vertices']; lab = mask[valid]
    area = maps['pixel_surface_area_mm2'][valid]
    nv, nf = len(vertices), len(faces)
    all_v = np.zeros(nv); all_f = np.bincount(tid, weights=area, minlength=nf)
    votes_v = np.zeros((nv, 2)); votes_f = np.zeros((nf, 2))
    for k in range(3):
        idx = faces[tid, k]; support = area*np.clip(bary[:, k], 0, 1)
        all_v += np.bincount(idx, weights=support, minlength=nv)
        for label in (0, 1):
            take = lab == label
            votes_v[:, label] += np.bincount(idx[take], weights=support[take], minlength=nv)
    for label in (0, 1):
        take = lab == label
        votes_f[:, label] = np.bincount(tid[take], weights=area[take], minlength=nf)
    def classify(votes, support):
        fraction = np.divide(votes.sum(1), support, out=np.zeros_like(support), where=support > 0)
        labels = np.full(len(support), 255, np.uint8)
        take = (support > 0) & (fraction >= minimum_fraction) & ~np.isclose(votes[:, 0], votes[:, 1], rtol=1e-6, atol=1e-12)
        labels[take] = np.argmax(votes[take], axis=1).astype(np.uint8)
        return labels, fraction
    vl, vf = classify(votes_v, all_v); fl, ff = classify(votes_f, all_f)
    return dict(vertex_labels=vl, face_labels=fl, vertex_labelled_fraction=vf, face_labelled_fraction=ff,
                vertex_visible_support_mm2=all_v, face_visible_area_mm2=all_f,
                vertex_class_votes_mm2=votes_v, face_class_area_mm2=votes_f)


def project_labels(args):
    pv = load_pyvista()
    out = args.output.resolve()
    index = json.loads((out/'dataset_index.json').read_text(encoding='utf-8'))
    annotation_root = args.annotations.resolve() if args.annotations else out/'x_anylabeling'
    if not annotation_root.is_dir():
        raise ValueError(f'Annotation directory missing: {annotation_root}')
    results, errors = [], []
    for row in index['scans']:
        annotation = annotation_root/(row['scan_id']+'.json')
        if not annotation.exists():
            continue  # Unlabelled scans do not acquire inferred clean labels.
        print(f'Projecting {row["scan_id"]}', flush=True)
        try:
            folder = out/row['sample_dir']
            for name, key in [('rgb.png', 'rgb_sha256'), ('maps.npz', 'maps_sha256'), ('geometry.npz', 'geometry_sha256')]:
                if sha256(folder/name) != row[key]:
                    raise ValueError(f'{name} changed since preparation; mapping identity is no longer valid.')
            image = out/row['annotation_image'] if row.get('annotation_image') else None
            if image is None or sha256(image) != row['rgb_sha256']:
                raise ValueError('Annotation image missing or changed since preparation.')
            data = json.loads(annotation.read_text(encoding='utf-8-sig'))
            # X-AnyLabeling may use a relative imagePath when labels are saved elsewhere.
            image_name = str(data.get('imagePath', '')).replace('\\', '/').rsplit('/', 1)[-1]
            if image_name != row['scan_id']+'.png':
                raise ValueError('JSON imagePath does not match the expected unique scan image.')
            with np.load(folder/'maps.npz', allow_pickle=False) as z:
                maps = {k: z[k] for k in z.files}
            mask = annotations_to_mask(data, maps['valid'], args.background)
            votes = label_votes(mask, maps, args.min_label_fraction)
            dest = folder/'labels'; dest.mkdir(exist_ok=True)
            rr, cc = np.nonzero(maps['valid'] & (mask != 255))
            tid = maps['triangle_id'][rr, cc]; bary = maps['barycentric'][rr, cc]
            xyz = np.einsum('ni,nij->nj', bary, maps['vertices'][maps['faces'][tid]])
            raw_xyz = np.einsum('ni,nij->nj', bary, maps['vertices_raw'][maps['faces'][tid]])
            Image.fromarray(mask).save(dest/'mask.png')
            np.savez_compressed(dest/'pixel_labels_3d.npz', pixel_row=rr.astype(np.int32),
                                pixel_col=cc.astype(np.int32), labels=mask[rr, cc], triangle_id=tid,
                                barycentric=bary, xyz_canonical=xyz, xyz_raw=raw_xyz,
                                pixel_surface_area_mm2=maps['pixel_surface_area_mm2'][rr, cc],
                                raw_triangle_id=maps['raw_triangle_id'][tid])
            np.savez_compressed(dest/'mesh_labels.npz', **votes, vertices=maps['vertices'],
                                faces=maps['faces'], vertices_raw=maps['vertices_raw'],
                                raw_vertex_id=maps['raw_vertex_id'], raw_triangle_id=maps['raw_triangle_id'])
            # Build from the map's exact vertex/face arrays rather than trusting an edited VTP.
            mesh = pv.PolyData(maps['vertices'], np.column_stack((np.full(len(maps['faces']), 3), maps['faces'])).ravel())
            with np.load(folder/'geometry.npz', allow_pickle=False) as z:
                mesh.point_data['RGB'] = np.uint8(np.rint(z['vertex_rgb']*255))
            mesh.point_data['plaque_label'] = votes['vertex_labels']
            mesh.point_data['labelled_fraction'] = votes['vertex_labelled_fraction']
            mesh.point_data['RawVertexId'] = maps['raw_vertex_id']
            mesh.cell_data['plaque_label'] = votes['face_labels']
            mesh.cell_data['labelled_fraction'] = votes['face_labelled_fraction']
            mesh.cell_data['RawTriangleId'] = maps['raw_triangle_id']
            mesh.save(dest/'roi_labelled.vtp')
            mesh.points = maps['vertices_raw']; mesh.save(dest/'roi_labelled_raw.vtp')
            rgb = np.uint8(np.rint(maps['rgb']*255)).copy()
            for label, color in [(0, [30, 210, 90]), (1, [255, 30, 80])]:
                take = mask == label
                rgb[take] = np.uint8(.55*rgb[take]+.45*np.asarray(color))
            Image.fromarray(rgb).save(dest/'overlay.png')
            info = dict(scan_id=row['scan_id'], status='OK', labelled_pixels=len(rr),
                        clean_pixels=int((mask == 0).sum()), plaque_pixels=int((mask == 1).sum()),
                        ignored_valid_pixels=int(((mask == 255)&maps['valid']).sum()),
                        background=args.background, min_label_fraction=args.min_label_fraction,
                        annotation_sha256=sha256(annotation), maps_sha256=row['maps_sha256'],
                        annotation_file=str(annotation), pixel_mapping='Exact interpolated surface points for labelled visible pixels',
                        mesh_mapping='Approximate area/barycentric-weighted majority; unseen/unsupported/tied labels are 255')
            write_json(dest/'label_metadata.json', info); results.append(info)
        except Exception as e:
            errors.append(dict(scan_id=row['scan_id'], error=str(e)))
            print(f'FAILED: {e}', flush=True)
    write_csv(out/'label_projection_manifest.csv', results)
    write_csv(out/'label_projection_failures.csv', errors)
    print(f'Projected {len(results)} annotations; failures {len(errors)}. Masks use 0=clean, 1=plaque, 255=ignore.')
    return 2 if errors else (0 if results else 1)


def launch_labeler(args):
    if args.output:
        out=args.output.resolve()
    else:
        saved=Path(__file__).parent/'last_prepared.json'
        if saved.exists():
            out=Path(json.loads(saved.read_text(encoding='utf-8'))['output']).resolve()
        else:
            candidates=sorted([p for p in DEFAULT_ROOT.glob('_development_*') if (p/'annotation_inventory.csv').is_file()]) if DEFAULT_ROOT is not None else []
            if not candidates:raise ValueError('No prepared snapshot found. Run prepare first or pass label --output SNAPSHOT.')
            out=candidates[-1].resolve()
    manifest=out/'annotation_inventory.csv'
    if not manifest.is_file():raise ValueError(f'Annotation inventory missing: {manifest}')
    entry=Path(__file__).parent/'annotator'/'biocal3d_annotator.py'
    command=[sys.executable,str(entry),'--root',str(out),'--manifest',str(manifest),'--annotator',args.annotator]
    if args.labels_dir:command+=['--output',str(args.labels_dir.resolve())]
    print(f'Opening 3D labeling queue: {out}',flush=True)
    return subprocess.call(command)

def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('check-env', help='Check interpreter, PyVista/VTK versions and mesh imports without touching data.')
    prep = sub.add_parser('prepare', help='Rerun all raw preprocessing and export geometry/RGB/maps.')
    prep.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    prep.add_argument('--output', type=Path, help='NEW empty output directory; default timestamped snapshot under data root.')
    prep.add_argument('--references', type=Path, help='Existing scanner templates; default legacy references plus newest per-scanner templates from previous snapshots.')
    prep.add_argument('--processed-root', type=Path, help='Optional maps-only mode using existing roi.vtp + geometry.npz.')
    prep.add_argument('--size', type=int, default=1024, help='Square annotation image size; default 1024.')
    prep.add_argument('--fov-mm', type=float, help='Fixed physical field of view; default reference diameter + 10%%.')
    prep.add_argument('--no-flip-v', action='store_true', help='Change only after checking texture V orientation.')
    prep.add_argument('--no-context-qc', action='store_true', help='Skip VTK screenshots; RGB/coverage/height QC is still exported.')
    prep.add_argument('--include-check', action='store_true', help='Also queue weak-registration CHECK scans; inspect before labeling.')
    prep.add_argument('--export-x-anylabeling',action='store_true',help='Also create the legacy 2D annotation image folder.')
    prep.add_argument('--make-splits', action='store_true', help='Create growth-stratified 80/10/10 physical-specimen splits.')
    prep.add_argument('--splits', type=Path, help='Reuse existing physical-specimen split assignments (must cover every specimen).')
    label=sub.add_parser('label',help='Open the direct 3D annotator using the approved ROI inventory.')
    label.add_argument('--output',type=Path,help='Prepared snapshot; default last prepared snapshot.')
    label.add_argument('--annotator',default='rater1',help='Independent rater ID; default rater1.')
    label.add_argument('--labels-dir',type=Path,help='Optional annotation destination; default SNAPSHOT/annotations.')
    proj = sub.add_parser('project-labels', help='Read X-AnyLabeling JSON and map labels to 3D.')
    proj.add_argument('--output', type=Path, required=True, help='Existing preparation snapshot.')
    proj.add_argument('--annotations', type=Path, help='JSON directory; default OUTPUT/x_anylabeling.')
    proj.add_argument('--background', choices=['ignore', 'clean'], default='ignore',
                      help='Unmarked valid pixels: ignore by default; clean requires reviewed complete images.')
    proj.add_argument('--min-label-fraction', type=float, default=.5, help='Minimum labelled fraction of visible support for vertex/face majority.')
    return p


def main():
    p = parser(); args = p.parse_args()
    if args.command == 'prepare' and args.data_root is None:
        p.error('Pass --data-root or set BIOCAL3D_DATA_ROOT to your raw dataset folder.')
    if args.command == 'prepare' and (args.size < 32 or (args.fov_mm is not None and (not np.isfinite(args.fov_mm) or args.fov_mm <= 0))):
        p.error('--size must be >=32 and --fov-mm finite/positive.')
    if args.command == 'project-labels' and not 0 < args.min_label_fraction <= 1:
        p.error('--min-label-fraction must be in (0,1].')
    try:
        if args.command == 'check-env':
            return check_environment()
        if args.command == 'label':
            return launch_labeler(args)
        return prepare(args) if args.command == 'prepare' else project_labels(args)
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
