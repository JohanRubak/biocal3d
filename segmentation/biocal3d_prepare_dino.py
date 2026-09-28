"""BioCal3D: specimen splits, triangle-rasterized ROI maps and frozen DINOv2 inspection.

Python 3.11+. Install maps dependencies:
  python -m pip install numpy pandas scipy scikit-learn pillow pyvista matplotlib
Optional features: install torch and torchvision for your CPU/CUDA installation.

Run (edit DATA_ROOT below, or pass --data-root):
  python biocal3d_prepare_dino.py
  python biocal3d_prepare_dino.py --features

No plaque labels are generated and no encoder is trained by this script.
DINO clusters are appearance clusters, NOT plaque predictions.
Source model: https://github.com/facebookresearch/dinov2

Inputs: recursively discovered roi.vtp or geometry.npz, plus original textures.
Use --processed-root to select ONE authoritative preprocessing tree if duplicates
are reported. --data-root must still include the raw textures.
Output: _dino_preparation_80_10_10 (override with --output).
Live: python biocal3d_prepare_dino.py --feature-only --live
Step through: add --step; press N/right-arrow to advance, Q to close preview.
Split: 80/10/10 by specimen, stratified by fresh/mature growth condition.
With 40 specimens: 32 train, 4 validation, 4 test. Four held-out specimens
cannot cover all five groups; growth-condition stratification is used instead.
The old 60/20/20 output directory remains unchanged.

Specimen identity assumes Gx sample n is the SAME physical specimen across
all stages and scanners, and different groups contain different specimens.
If specimens were reused across groups, correct the IDs BEFORE using this script.

Saved splits are immutable on reruns. New specimens require an explicit new
experiment/output directory. Missing scans do not change existing assignments.
RGB maps have no rendering lights, histogram normalization or automatic contrast.
Projection is per-scan plane alignment, NOT pixel registration across scans.
Mesh coordinates are assumed millimetres; verify your scanner export units.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

DATA_ROOT = Path(r"C:\Users\johan\repos\biocal3d\data\02-09-2026-Data collected")
SEED = 42
SAMPLE = re.compile(r"(?:BCG)?G?(\d+)\s*T(\d+)\s*[-_]\s*(\d+)", re.I)
RGB_KEYS = ('RGB', 'rgb', 'colors', 'Colors', 'vertex_colors', 'VertexColors', 'texture_rgb', 'RGBA', 'rgba', 'RGB (uint8)', 'RGB Colors')
UV_KEYS = ('TCoords', 'Texture Coordinates', 'TextureCoordinates', 'tcoords', 'uv', 'UV')
SCANNERS = {'trios3': 'TRIOS3', 'trios5': 'TRIOS5', 'itero': 'iTERO', 'labscanner': 'LABscanner'}


def identify(path):
    scanner = None
    match = None
    for part in reversed(path.parts):
        compact = re.sub(r'[ _-]', '', part.lower())
        if scanner is None:
            scanner = next((v for k, v in SCANNERS.items() if k in compact), None)
        if match is None:
            match = SAMPLE.search(part)
    if not scanner or not match:
        return None
    g, t, n = map(int, match.groups())
    if g not in range(1, 6) or t not in (range(2) if g <= 2 else range(3)):
        raise ValueError(f'Unexpected group/time: {path}')
    stage = 'baseline' if t == 0 else ('partial' if g >= 3 and t == 1 else 'clean')
    return dict(scanner=scanner, group=g, time_idx=t, specimen_num=n,
                physical_id=f'G{g}_S{n}', sample_name=f'BCG{g}T{t}-{n}', stage=stage)


def discover(root, output):
    found = {}
    for pattern in ('roi.vtp', 'geometry.npz'):
        for p in sorted(root.rglob(pattern)):
            if output == p.parent or output in p.parents:
                continue
            meta = identify(p.parent)
            if meta is None:
                continue
            key = (meta['scanner'], meta['sample_name'])
            item = found.setdefault(key, {**meta, 'assets': {}})
            if p.name in item['assets']:
                raise ValueError(f'Duplicate {key}: {p} and {item["assets"][p.name]}. '
                                 'Use --processed-root for one authoritative ROI tree.')
            item['assets'][p.name] = p
    if not found:
        raise RuntimeError(f'No identifiable roi.vtp/geometry.npz under {root}')
    return [found[k] for k in sorted(found)]


def lock_splits(scans, output):
    path = output / 'specimen_splits.csv'
    specimens = pd.DataFrame(scans)[['physical_id', 'group', 'specimen_num']].drop_duplicates()
    specimens = specimens.sort_values(['group', 'specimen_num']).reset_index(drop=True)
    specimens['growth_condition'] = np.where(specimens.group <= 2, 'fresh_12h', 'mature_24h')
    if path.exists():
        split = pd.read_csv(path)
        if split.physical_id.duplicated().any() or not set(split.split) <= {'train', 'val', 'test'}:
            raise ValueError('Invalid saved specimen_splits.csv')
        if not set(specimens.physical_id) <= set(split.physical_id):
            raise ValueError('New specimens detected; use a new output directory to define a new experiment.')
        expected_test = int(np.ceil(len(split)*.1))
        expected_val = int(np.ceil((len(split)-expected_test)/9))
        counts = split.groupby('split').size().to_dict()
        if counts != {'train': len(split)-expected_test-expected_val, 'val': expected_val, 'test': expected_test}:
            raise ValueError('Saved split is not 80/10/10. Use the new default output folder or another new directory.')
        return split
    try:
        dev, test = train_test_split(specimens, test_size=.1, stratify=specimens.growth_condition, random_state=SEED)
        train, val = train_test_split(dev, test_size=1/9, stratify=dev.growth_condition, random_state=SEED + 1)
    except ValueError as exc:
        raise ValueError('Too few specimens per growth condition for stratified 80/10/10 split. '
                         'Run on your full processed dataset, not a small subset.') from exc
    split = pd.concat([x.assign(split=s) for x, s in [(train, 'train'), (val, 'val'), (test, 'test')]])
    split = split.sort_values(['group', 'specimen_num'])
    split.to_csv(path, index=False)
    return split


def texture_index(root, output):
    index = {}
    for p in root.rglob('*'):
        if p.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'} or not p.is_file():
            continue
        if output in p.parents or any(x.startswith('_') for x in p.relative_to(root).parts[:-1]):
            continue
        meta = identify(p)
        if meta and not any(x in p.stem.lower() for x in ['qc', 'overview', 'figure']):
            index.setdefault((meta['scanner'], meta['sample_name']), []).append(p)
    return index


def choose_texture(paths):
    if not paths:
        raise ValueError('UV coordinates found but no matching raw texture. Keep raw scanner/sample folders under --data-root.')
    def score(p):
        s = p.stem.lower()
        return 100 * ('texture' in s) + 30 * ('shell' in s) + 20 * ('occlusion' in s)
    ordered = sorted(paths, key=lambda p: (-score(p), str(p)))
    if len(ordered) > 1 and score(ordered[0]) == score(ordered[1]):
        raise ValueError(f'Ambiguous texture candidates: {ordered}. Retain one authoritative raw texture.')
    return ordered[0]


def get_array(mapping, names):
    return next((np.asarray(mapping[k]) for k in names if k in mapping), None)


def triangles(raw):
    f = np.asarray(raw, dtype=np.int64)
    if f.ndim == 2 and f.shape[1] == 3:
        return f
    if f.ndim == 2 and f.shape[1] == 4 and np.all(f[:, 0] == 3):
        return f[:, 1:]
    if f.ndim == 1 and f.size % 4 == 0 and np.all(f.reshape(-1, 4)[:, 0] == 3):
        return f.reshape(-1, 4)[:, 1:]
    raise ValueError('Missing or unsupported triangle connectivity; use roi.vtp.')


def load_mesh(scan, texindex):
    errors = []
    for name in ('roi.vtp', 'geometry.npz'):
        path = scan['assets'].get(name)
        if path is None:
            continue
        try:
            if name == 'roi.vtp':
                import pyvista as pv
                mesh = pv.read(path).triangulate()
                v = np.asarray(mesh.points, dtype=float)
                f = triangles(mesh.faces)
                arrays = mesh.point_data
                keep = np.ones(len(v), bool)
                rgb = get_array(arrays, RGB_KEYS)
                uv = get_array(arrays, UV_KEYS)
                if uv is None and mesh.active_texture_coordinates is not None:
                    uv = np.asarray(mesh.active_texture_coordinates)
            else:
                with np.load(path, allow_pickle=False) as z:
                    arrays = {k: z[k] for k in z.files}
                v = get_array(arrays, ('vertices', 'verts', 'points', 'xyz'))
                v = np.asarray(v, dtype=float)
                f = triangles(get_array(arrays, ('faces', 'triangles', 'cells')))
                rgb = get_array(arrays, RGB_KEYS)
                uv = get_array(arrays, UV_KEYS)
                keep = get_array(arrays, ('roi_mask', 'valid_mask', 'mask'))
                keep = np.ones(len(v), bool) if keep is None else np.asarray(keep, bool).reshape(-1)
            if v.ndim != 2 or v.shape[1] != 3 or len(keep) != len(v):
                raise ValueError('Invalid vertices/ROI mask')
            if len(f) == 0 or f.min() < 0 or f.max() >= len(v):
                raise ValueError('Invalid faces')
            texture = None
            texture_path = ''
            # iTERO: require vertex RGB; try the next geometry source if absent.
            if scan['scanner'] == 'iTERO':
                if rgb is None:
                    raise ValueError('iTERO requires vertex RGB. Trying another geometry source; texture is not substituted.')
                uv = None
            # Other scanners retain texture support.
            if uv is not None:
                uv = np.asarray(uv, float)[:, :2]
                if len(uv) != len(v):
                    raise ValueError('UV length mismatch')
                tp = choose_texture(texindex.get((scan['scanner'], scan['sample_name']), []))
                texture = np.asarray(Image.open(tp).convert('RGB'), dtype=np.float32) / 255
                texture_path = str(tp)
                keep &= np.isfinite(uv).all(axis=1)
            elif rgb is not None:
                rgb = np.asarray(rgb)[:, :3]
                if len(rgb) != len(v):
                    raise ValueError('RGB length mismatch')
                scale = 255 if np.issubdtype(rgb.dtype, np.integer) or np.nanmax(rgb) > 1.5 else 1
                rgb = rgb.astype(np.float32) / scale
                keep &= np.isfinite(rgb).all(axis=1)
            else:
                raise ValueError('No explicitly named RGB array or UV coordinates. Lab-only data are not used as source RGB.')
            keep &= np.isfinite(v).all(axis=1)
            f = f[np.all(keep[f], axis=1)]
            if len(f) == 0:
                raise ValueError('No triangles inside valid ROI')
            used = np.unique(f)
            remap = np.full(len(v), -1, dtype=np.int64)
            remap[used] = np.arange(len(used))
            return dict(vertices=v[used], faces=remap[f],
                        rgb=None if rgb is None else rgb[used],
                        uv=None if uv is None else uv[used], texture=texture,
                        source=str(path), texture_path=texture_path, color_source='texture_uv' if texture is not None else 'vertex_rgb')
        except Exception as exc:
            errors.append(f'{path}: {exc}')
    raise ValueError('\n'.join(errors))


def plane(mesh):
    v, f = mesh['vertices'], mesh['faces']
    tri = v[f]
    areas = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)/2
    weights = np.bincount(f.ravel(), weights=np.repeat(areas/3, 3), minlength=len(v))
    if weights.sum() <= 0:
        raise ValueError('Degenerate mesh')
    origin = np.average(v, axis=0, weights=weights)
    c = v-origin
    _, axes = np.linalg.eigh((c * weights[:, None]).T @ c / weights.sum())
    normal = axes[:, 0]
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    ref = np.eye(3)[np.argmin(np.abs(normal))]
    u = ref - np.dot(ref, normal)*normal
    u /= np.linalg.norm(u)
    w = np.cross(normal, u)
    basis = np.column_stack([u, w, normal])
    xyz = c @ basis
    # Fit circle to boundary vertices to avoid a mesh-density-biased crop center.
    edges = np.sort(np.concatenate([f[:, [0,1]], f[:, [1,2]], f[:, [2,0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = np.unique(edges[counts == 1])
    # Bounding-box center is stable for a complete circular ROI, not for missing large sectors.
    center2 = (xyz[:, :2].min(0) + xyz[:, :2].max(0))/2
    origin = origin + basis[:, :2] @ center2
    xyz[:, :2] -= center2
    extent = np.max(np.abs(xyz[:, :2])) * 2
    return xyz, origin, basis, extent, areas.sum(), len(boundary)


def sample_texture(tex, uv, flip_v=True):
    if np.any(uv < -1e-4) or np.any(uv > 1.0001):
        raise ValueError('UV outside [0,1]; repeat/wrap textures require an explicit mapping rule.')
    uv = np.clip(uv, 0, 1)
    x = uv[:, 0]*(tex.shape[1]-1)
    y = (1-uv[:, 1] if flip_v else uv[:, 1])*(tex.shape[0]-1)
    x0, y0 = x.astype(int), y.astype(int)
    x1, y1 = np.minimum(x0+1, tex.shape[1]-1), np.minimum(y0+1, tex.shape[0]-1)
    a, b = (x-x0)[:, None], (y-y0)[:, None]
    return (1-b)*((1-a)*tex[y0,x0]+a*tex[y0,x1])+b*((1-a)*tex[y1,x0]+a*tex[y1,x1])


def rasterize(mesh, xyz, size, fov, flip_v=True):
    # Pixel row 0 is +v; centers spaced fov/size mm. No hole filling.
    px = np.column_stack([(xyz[:,0]/fov+.5)*size-.5, (.5-xyz[:,1]/fov)*size-.5])
    tid = np.full((size,size), -1, np.int32)
    bary = np.zeros((size,size,3), np.float32)
    zbuf = np.full((size,size), -np.inf)
    overlap = np.zeros((size,size), np.uint16)
    for i, face in enumerate(mesh['faces']):
        a,b,c = px[face]
        lo = np.maximum(np.ceil(np.min([a,b,c],axis=0)).astype(int),0)
        hi = np.minimum(np.floor(np.max([a,b,c],axis=0)).astype(int),size-1)
        if np.any(lo>hi):
            continue
        det=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(det)<1e-12:
            continue
        xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1))
        wa=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/det
        wb=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/det
        wc=1-wa-wb
        inside=(wa>=-1e-7)&(wb>=-1e-7)&(wc>=-1e-7)
        weights=np.stack([wa,wb,wc],axis=-1)
        depth=weights@xyz[face,2]
        overlap[yy[inside],xx[inside]]+=1
        take=inside&(depth>zbuf[yy,xx])
        rr,cc=yy[take],xx[take]
        zbuf[rr,cc]=depth[take]
        tid[rr,cc]=i
        bary[rr,cc]=weights[take]
    valid=tid>=0
    if valid.sum()==0:
        raise ValueError('Empty projected map')
    faces=mesh['faces'][tid[valid]]
    w=bary[valid]
    rgb=np.full((size,size,3), [.485,.456,.406],np.float32)
    if mesh['texture'] is not None:
        uv=np.einsum('ni,nij->nj',w,mesh['uv'][faces])
        rgb[valid]=sample_texture(mesh['texture'],uv,flip_v)
    else:
        rgb[valid]=np.einsum('ni,nij->nj',w,mesh['rgb'][faces])
    height=np.where(valid,zbuf,np.nan).astype(np.float32)
    # Surface/projected-area Jacobian allows approximate surface-weighted coverage.
    tr=xyz[mesh['faces']]
    cross=np.cross(tr[:,1]-tr[:,0],tr[:,2]-tr[:,0])
    ratio=np.linalg.norm(cross,axis=1)/np.maximum(np.abs(cross[:,2]),1e-12)
    area=np.zeros((size,size),np.float32)
    area[valid]=(fov/size)**2*ratio[tid[valid]]
    return dict(rgb=np.clip(rgb,0,1),valid=valid,height_mm=height,
                triangle_id=tid,barycentric=bary,pixel_surface_area_mm2=area,
                overlap_fraction=float(np.mean(overlap[valid]>1)))


def save_maps(scan, mesh, proj, fov, size, out, flip_v):
    xyz, origin, basis, extent, surface_area, _ = proj
    if extent > fov:
        raise ValueError(f'ROI span {extent:.3f} mm exceeds locked FOV {fov:.3f}; no silent cropping. '
                         'Check ROI or create a new experiment with a physically justified --fov-mm.')
    maps=rasterize(mesh,xyz,size,fov,flip_v)
    folder=out/'maps'/scan['split']/scan['scanner']/scan['sample_name']
    folder.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.uint8(np.rint(maps['rgb']*255))).save(folder/'rgb.png')
    Image.fromarray(maps['valid'].astype(np.uint8)*255).save(folder/'valid_mask.png')
    # Labels are intentionally absent. Later annotation convention: 0 clean,
    # 1 plaque, 255 ignored/uncertain/outside valid surface.
    np.savez_compressed(folder/'maps.npz', **maps, vertices=mesh['vertices'], faces=mesh['faces'],
                        origin=origin,basis=basis, fov_mm=fov, mm_per_pixel=fov/size)
    info={k:scan[k] for k in ['scanner','sample_name','physical_id','group','stage','split']}
    info.update(source=mesh['source'], texture_path=mesh['texture_path'], color_source=mesh['color_source'], fov_mm=fov,
                image_size=size, mesh_surface_area_mm2=float(surface_area),
                raster_surface_area_mm2=float(maps['pixel_surface_area_mm2'].sum()),
                valid_pixels=int(maps['valid'].sum()),overlap_fraction=maps['overlap_fraction'],
                orientation='Per-scan plane; NOT cross-scan registered',
                projection_extent_mm=float(extent))
    (folder/'metadata.json').write_text(json.dumps(info,indent=2))
    return {**info,'map_dir':str(folder.resolve())}


class LivePreview:
    """Optional GUI; closing it disables display and processing continues."""
    def __init__(self, enabled=False, step=False, seconds=.3):
        self.enabled = enabled
        self.step = step
        self.seconds = max(.01, seconds)
        self.advance = False
        if enabled:
            import matplotlib.pyplot as plt
            self.plt = plt
            backend = plt.get_backend().lower()
            if backend in {'agg', 'pdf', 'svg', 'ps', 'template', 'cairo'} or 'inline' in backend:
                print('No interactive Matplotlib backend. Saved previews and terminal progress remain available.', flush=True)
                self.enabled = False
                return
            plt.ion()
            self.fig, self.axes = plt.subplots(1, 3, figsize=(13, 5))
            self.fig.canvas.mpl_connect('key_press_event', self.on_key)
            plt.show(block=False)

    def on_key(self, event):
        if event.key in ('n', 'right', 'enter'):
            self.advance = True
        elif event.key == ' ':
            self.step = not self.step
            self.advance = True
        elif event.key in ('q', 'escape'):
            self.plt.close(self.fig)
            self.enabled = False

    def show(self, panels, titles, caption, wait=True):
        if not self.enabled:
            return
        if not self.plt.fignum_exists(self.fig.number):
            self.enabled = False
            return
        for ax, panel, title in zip(self.axes, panels, titles):
            ax.clear()
            ax.imshow(panel, interpolation='nearest')
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        self.fig.suptitle(caption + '\nN/right: next | Space: toggle stepping | Q: close preview', fontsize=10)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.advance = False
        self.plt.pause(.01)
        if not wait:
            return
        if self.step:
            while self.enabled and self.plt.fignum_exists(self.fig.number) and not self.advance:
                self.plt.pause(.05)
        else:
            self.plt.pause(self.seconds)

    def close(self):
        if self.enabled:
            self.plt.close(self.fig)


def pca_preview(features, valid):
    """Per-image PCA for immediate preview; colours are NOT comparable across scans."""
    from sklearn.decomposition import PCA
    result = np.zeros((*valid.shape, 3), np.float32)
    x = features[valid.ravel()]
    if len(x) >= 3:
        y = PCA(n_components=3, svd_solver='randomized', random_state=SEED).fit_transform(x)
        lo, hi = np.percentile(y, [1, 99], axis=0)
        result[valid] = np.clip((y-lo)/np.maximum(hi-lo, 1e-9), 0, 1)
    return result


def save_preview(folder, rgb, panel2, panel3, caption, titles, name):
    """Save a labelled contact sheet without requiring an interactive backend."""
    from PIL import ImageDraw
    width = 360
    canvas = Image.new('RGB', (width*3, width+90), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), caption, fill='black')
    for i, (panel, title) in enumerate(zip([rgb, panel2, panel3], titles)):
        a = np.asarray(panel)
        if a.dtype != np.uint8:
            a = np.uint8(np.clip(a, 0, 1)*255)
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        im = Image.fromarray(a).resize((width, width), Image.Resampling.NEAREST)
        canvas.paste(im, (i*width, 65))
        draw.text((i*width+8, 40), title, fill='black')
    canvas.save(folder/name)


def inspect_features(manifest, output, size, repo, device, clusters,
                     live=False, step=False, preview_seconds=.3):
    import torch
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans
    if size % 14:
        raise ValueError('DINOv2 map size must be divisible by 14')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    print(f'Loading pretrained DINOv2 ViT-S/14 with registers on {device}. First run may download weights.', flush=True)
    model=torch.hub.load(repo,'dinov2_vits14_reg',pretrained=True,trust_repo=True).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    rng=np.random.default_rng(SEED)
    samples=[]
    rows=[]
    selected=manifest[manifest.split.isin(['train','val'])].to_dict('records')
    viewer=LivePreview(live or step, step, preview_seconds)
    start=time.perf_counter()
    try:
        for i,row in enumerate(selected,1):
            folder=Path(row['map_dir'])
            caption=f"{row['scanner']} | {row['sample_name']} | {row['split']} | {row['stage']}"
            print(f"[DINO {i}/{len(selected)}] Feeding {caption}",flush=True)
            with np.load(folder/'maps.npz') as z:
                rgb=z['rgb']; valid=z['valid']
            viewer.show([rgb,valid,np.zeros_like(rgb)],['RGB input','Valid surface','Running encoder...'],caption,wait=False)
            tensor=torch.from_numpy(rgb.transpose(2,0,1).copy()).unsqueeze(0).to(device)
            infer_start=time.perf_counter()
            with torch.inference_mode():
                f=model.forward_features((tensor-mean)/std)['x_norm_patchtokens'][0].float().cpu().numpy()
            infer_seconds=time.perf_counter()-infer_start
            grid=size//14
            patch_valid=valid.reshape(grid,14,grid,14).mean((1,3))>=.9
            np.savez_compressed(folder/'dino_features.npz',features=f.reshape(grid,grid,-1).astype(np.float16),
                                patch_valid=patch_valid, model='dinov2_vits14_reg',
                                map_sha256=hashlib.sha256((folder/'maps.npz').read_bytes()).hexdigest())
            pc=pca_preview(f,patch_valid)
            Image.fromarray(np.uint8(pc*255)).resize((size,size),Image.Resampling.NEAREST).save(folder/'dino_pca_per_image.png')
            titles=['RGB input','Valid surface','Per-image PCA (NOT plaque)']
            save_preview(folder,rgb,valid,pc,caption,titles,'live_feature_preview.png')
            elapsed=time.perf_counter()-start
            eta=elapsed/i*(len(selected)-i)
            print(f'  Done: {infer_seconds:.2f}s inference | {int(patch_valid.sum())} valid patches | '
                  f'elapsed {elapsed/60:.1f} min | ETA ~{eta/60:.1f} min (includes preview time)',flush=True)
            viewer.show([rgb,valid,pc],titles,caption)
            if row['split']=='train':
                usable=f[patch_valid.ravel()]
                if len(usable):
                    samples.append(usable[rng.choice(len(usable),min(256,len(usable)),replace=False)])
            rows.append((folder,caption))
        if not samples:
            raise RuntimeError('No valid training feature patches')
        X=np.concatenate(samples)
        if len(X)<max(3,clusters):
            raise RuntimeError('Too few training patches')
        print(f'Fitting shared PCA and {clusters} appearance clusters on {len(X)} TRAIN patches only...',flush=True)
        pca=PCA(n_components=3,random_state=SEED).fit(X)
        norm=lambda x:x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
        km=MiniBatchKMeans(n_clusters=clusters,random_state=SEED,n_init=10,batch_size=2048).fit(norm(X))
        scores=pca.transform(X)
        low,high=np.percentile(scores,[1,99],axis=0)
        palette=np.random.default_rng(SEED+2).integers(35,240,(clusters,3),dtype=np.uint8)
        for i,(folder,caption) in enumerate(rows,1):
            with np.load(folder/'dino_features.npz') as z:
                f=z['features'].astype(np.float32).reshape(grid*grid,-1);valid=z['patch_valid']
            with np.load(folder/'maps.npz') as z:
                rgb=z['rgb']
            pc=np.clip((pca.transform(f)-low)/np.maximum(high-low,1e-9),0,1).reshape(*valid.shape,3)
            ids=km.predict(norm(f)).reshape(valid.shape)
            view=palette[ids];view[~valid]=0;pc[~valid]=0
            Image.fromarray(np.uint8(pc*255)).resize((size,size),Image.Resampling.NEAREST).save(folder/'dino_pca.png')
            Image.fromarray(view).resize((size,size),Image.Resampling.NEAREST).save(folder/'appearance_clusters.png')
            np.save(folder/'appearance_cluster_ids.npy',np.where(valid,ids,-1).astype(np.int16))
            titles=['RGB input','Shared PCA (train fitted)','Appearance clusters (NOT plaque)']
            save_preview(folder,rgb,pc,view,caption,titles,'shared_feature_preview.png')
            print(f'[Shared preview {i}/{len(rows)}] {caption}',flush=True)
            viewer.show([rgb,pc,view],titles,caption)
        np.savez_compressed(output/'feature_inspection_fit.npz',pca_mean=pca.mean_,pca_components=pca.components_,
                            pca_low=low,pca_high=high,cluster_centers=km.cluster_centers_,palette=palette)
        (output/'feature_run.json').write_text(json.dumps(dict(model='dinov2_vits14_reg',repository=repo,
            torch_version=torch.__version__,device=device,fit_split='train',inference_splits=['train','val'],
            warning='Unsupervised appearance clusters; NOT plaque masks',
            live_pca='Per-image colours not comparable; shared PCA is fitted on train only'),indent=2))
        print('Feature inspection finished. No encoder weights updated; no segmentation head fitted.',flush=True)
    finally:
        viewer.close()


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root',type=Path,default=DATA_ROOT)
    p.add_argument('--processed-root',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--size',type=int,default=518,help='518 = 37 x 14, compatible with DINOv2')
    p.add_argument('--fov-mm',type=float,help='Fixed physical field of view; default train maximum span + 10%%')
    p.add_argument('--features',action='store_true')
    p.add_argument('--feature-only',action='store_true',help='Use existing maps without rebuilding')
    p.add_argument('--live',action='store_true',help='Show current image and spatial features in a live window')
    p.add_argument('--step',action='store_true',help='Live viewer waits for N/right/Enter after each result')
    p.add_argument('--preview-seconds',type=float,default=.3,help='Display time per result in automatic mode')
    p.add_argument('--device',default='auto')
    p.add_argument('--repo',default='facebookresearch/dinov2',help='PyTorch Hub repo; may append :commit for reproducibility')
    p.add_argument('--clusters',type=int,default=4)
    p.add_argument('--no-flip-v',action='store_true',help='Only change after verifying texture orientation')
    args=p.parse_args()
    root=args.data_root.resolve();out=(args.output or root/'_dino_preparation_80_10_10').resolve()
    out.mkdir(parents=True,exist_ok=True)
    if args.clusters < 2:
        p.error('--clusters must be at least 2')
    if args.size<=0 or args.size%14:
        p.error('--size must be a positive multiple of 14')
    if args.fov_mm is not None and args.fov_mm<=0:
        p.error('--fov-mm must be positive')
    if args.feature_only:
        config=json.loads((out/'map_config.json').read_text())
        inspect_features(pd.read_csv(out/'scan_manifest.csv'),out,config['size'],args.repo,args.device,args.clusters,args.live,args.step,args.preview_seconds)
        return
    scans=discover((args.processed_root or root).resolve(),out)
    splits=lock_splits(scans,out)
    assignments=splits.set_index('physical_id').split.to_dict()
    for s in scans:
        s['split']=assignments[s['physical_id']]
    textures=texture_index(root,out)
    configpath=out/'map_config.json'
    failures=[];ready=[]
    for i,s in enumerate(scans):
        print(f'[{i+1}/{len(scans)}] {s["scanner"]} {s["sample_name"]}',flush=True)
        try:
            mesh=load_mesh(s,textures)
            proj=plane(mesh)
            # Keep only metadata for next pass, not every mesh in RAM.
            ready.append((s,float(proj[3])))
        except Exception as exc:
            failures.append(dict(scanner=s['scanner'],sample_name=s['sample_name'],split=s['split'],error=str(exc)))
    if not ready:
        pd.DataFrame(failures).to_csv(out/'loading_failures.csv',index=False)
        raise RuntimeError('No usable meshes; inspect loading_failures.csv')
    if configpath.exists():
        config=json.loads(configpath.read_text())
        if config['size']!=args.size or config['flip_v']!=(not args.no_flip_v) or (args.fov_mm is not None and not np.isclose(config['fov_mm'],args.fov_mm)):
            raise ValueError('Map settings differ from locked configuration; use a new output directory.')
        fov=config['fov_mm']
    else:
        extents=[extent for s,extent in ready if s['split']=='train']
        if not extents:
            raise RuntimeError('No valid training meshes for field-of-view definition')
        fov=args.fov_mm or float(np.ceil(max(extents)*1.10*10)/10)
        config=dict(size=args.size,fov_mm=fov,flip_v=not args.no_flip_v,seed=SEED,
                    fov_source='explicit' if args.fov_mm else 'training maximum span + 10 percent',
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        configpath.write_text(json.dumps(config,indent=2))
    rows=[]
    for i,(s,_) in enumerate(ready,1):
        print(f'[Maps {i}/{len(ready)}] {s["scanner"]} {s["sample_name"]}',flush=True)
        try:
            mesh=load_mesh(s,textures)
            rows.append(save_maps(s,mesh,plane(mesh),fov,args.size,out,not args.no_flip_v))
        except Exception as exc:
            failures.append(dict(scanner=s['scanner'],sample_name=s['sample_name'],split=s['split'],error=str(exc)))
    pd.DataFrame(failures,columns=['scanner','sample_name','split','error']).to_csv(out/'loading_failures.csv',index=False)
    manifest=pd.DataFrame(rows)
    manifest.to_csv(out/'scan_manifest.csv',index=False)
    if not rows:
        raise RuntimeError('No maps exported; inspect loading_failures.csv')
    manifest.groupby(['split','scanner','stage']).size().rename('n_scans').to_csv(out/'split_scan_counts.csv')
    # Select labels later from train and val; reserve test annotations for final evaluation.
    queue=manifest.copy()
    queue['priority']=queue.stage.map({'partial':1,'baseline':2,'clean':3})
    queue['label_status']='unlabelled'
    queue.sort_values(['split','priority','physical_id','scanner']).to_csv(out/'annotation_inventory.csv',index=False)
    print('\nSpecimens:',splits.groupby('split').size().to_dict())
    print(f'Exported {len(rows)} maps; failures {len(failures)}. FOV={fov:.2f} mm.')
    print('Check source colours, mask coverage, and overlap/area QC before any training.')
    if failures:
        print('Resolve loading_failures.csv before treating this as the complete dataset.')
    if args.features:
        inspect_features(manifest,out,args.size,args.repo,args.device,args.clusters,args.live,args.step,args.preview_seconds)


if __name__=='__main__':
    main()
