"""Resolve each original model inside the selected data root; never edit it."""
from __future__ import annotations
from pathlib import Path, PureWindowsPath
import json
import os
import re
import numpy as np
import roi_browser_core as core
from original_models import OriginalModelIndex, identity

MESH_EXTENSIONS = {'.ply', '.vtp', '.obj', '.stl'}
SOURCE_KEYS = ('source_ply', 'raw_path', 'original_path', 'original_mesh', 'source_path', 'source_file', 'source_mesh', 'input_path')
HARMONIZED_KEYS = ('harmonized_source_ply', 'harmonized_source_sha256')
TEXTURE_KEYS = ('source_texture', 'texture_path', 'texture_file')


def parts_of(value):
    return PureWindowsPath(str(value)).parts if '\\' in str(value) else Path(str(value)).parts


def sample_metadata(scan):
    """Read small source/transform fields, not mesh arrays, from the approved sample."""
    result = {}
    path = scan.path if scan.path.suffix.lower() == '.npz' else scan.path.with_name('geometry.npz')
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            for key in (*SOURCE_KEYS, *HARMONIZED_KEYS, *TEXTURE_KEYS, 'original_to_canonical'):
                if key in data:
                    value = data[key]
                    result[key] = value.item() if value.size == 1 else value.tolist()
    for name in ('meta.json', 'metadata.json', 'source.json', 'review.json', 'sample.json'):
        path = scan.path.parent / name
        if not path.is_file(): continue
        try: data = json.loads(path.read_text(encoding='utf-8-sig'))
        except (ValueError, OSError): continue
        def fields(item):
            if not isinstance(item, dict): return
            for key, value in item.items():
                if key in (*SOURCE_KEYS, *HARMONIZED_KEYS, *TEXTURE_KEYS, 'original_to_canonical') and key not in result: result[key] = value
                if isinstance(value, dict): fields(value)
        fields(data)
    return result


def normal_name(value):
    return re.sub("[^a-z0-9]", "", str(value).lower())


def model_frame(scan, points, transform):
    """Use recorded vertex correspondence to detect already aligned full exports."""
    path = scan.path.with_name('geometry.npz')
    if not path.is_file(): return 'native'
    with np.load(path, allow_pickle=False) as data:
        if not {'raw_vertex_id', 'vertices'}.issubset(data.files): return 'native'
        ids = data['raw_vertex_id']
        vertices = data['vertices']
    if ids.shape != (len(vertices),) or not np.issubdtype(ids.dtype, np.integer):
        return 'unverified'
    valid = np.flatnonzero(ids >= 0)
    if len(valid) < 3: return 'unverified'
    if len(valid) > 512:
        valid = valid[np.linspace(0, len(valid) - 1, 512, dtype=int)]
    target = vertices[valid]
    if ids[valid].max() < len(points):
        source = points[ids[valid]]
        if np.allclose(source, target, atol=2e-5, rtol=0): return 'canonical'
        native_to_roi = source @ transform[:3, :3].T + transform[:3, 3]
        if np.allclose(native_to_roi, target, atol=2e-5, rtol=0): return 'native'
    # Re-exporters can reorder vertices or split texture seams. Compare the
    # sampled ROI points spatially before aligning a reordered full model.
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    native_target = np.linalg.solve(transform[:3, :3], (target - transform[:3, 3]).T).T
    canonical_match = np.all(tree.query(target)[0] <= 2e-5)
    native_match = np.all(tree.query(native_target)[0] <= 2e-5)
    if canonical_match and (not native_match or np.allclose(target, native_target, atol=2e-5, rtol=0)):
        return 'canonical'
    if native_match and not canonical_match: return 'native'
    return 'unverified'



class OriginalModels:
    def __init__(self, dataset_root, original_root=None):
        self.dataset_root = Path(dataset_root).resolve()
        self.original_root = Path(original_root).expanduser().resolve() if original_root else None
        if self.original_root and not self.original_root.is_dir(): raise ValueError(f'Original data folder unavailable: {self.original_root}')
        self.index = {}; self.resolved = {}
        self.checksums = OriginalModelIndex(self.original_root) if self.original_root else None

    def inside_root(self, path):
        path = Path(path).resolve()
        return self.original_root is None or path == self.original_root or self.original_root in path.parents

    def rebase(self, recorded):
        if not recorded or not isinstance(recorded, str): return []
        parts = parts_of(recorded); candidates = [Path(recorded)]
        if self.original_root:
            candidates.append(self.original_root.joinpath(*parts)) if not PureWindowsPath(recorded).is_absolute() and not Path(recorded).is_absolute() else None
            # Match either the original-data folder's name or a scanner folder.
            for i, part in enumerate(parts):
                if part.casefold() == self.original_root.name.casefold(): candidates.append(self.original_root.joinpath(*parts[i + 1:]))
                if core.scanner_name([part]):
                    candidates.append(self.original_root.joinpath(*parts[i:]))
                    if normal_name(self.original_root.name) == normal_name(part): candidates.append(self.original_root.joinpath(*parts[i + 1:]))
        return list(dict.fromkeys(p.resolve() for p in candidates if p.is_file() and self.inside_root(p)))

    def scan_index(self, scanner):
        if self.index: return self.index.get(scanner, {})
        root = self.original_root
        if root is None: return {}
        index = {name: {} for name in core.SCANNERS}
        root_scanners, root_samples = identity(root.name)
        for folder, directories, files in os.walk(root):
            directories[:] = sorted(d for d in directories if not d.startswith(('.', '_')) and
                d.casefold() not in {'approved rois', 'annotations', 'versions', 'exports', 'runs', 'feature_cache', 'surface_cache'})
            folder = Path(folder)
            if folder != root and ('geometry_rebuild.json' in files or
                    ('scan_manifest.csv' in files and (folder / 'samples').is_dir())):
                directories[:] = []
                continue
            for filename in files:
                path = folder / filename
                if path.suffix.lower() not in MESH_EXTENSIONS or not self.inside_root(path): continue
                if filename.casefold() in {'roi.ply', 'roi.vtp'}: continue
                scanners, samples = identity(path.relative_to(root))
                scanners |= root_scanners
                samples |= root_samples
                if len(scanners) != 1 or len(samples) != 1: continue
                name = next((name for name in core.SCANNERS if name.casefold() in scanners), None)
                if name is None: continue
                group, stage, specimen = next(iter(samples))
                index[name].setdefault(f'BCG{group}T{stage}-{specimen}', []).append(path)
        self.index = index
        return index[scanner]

    def locate(self, scan, metadata):
        if scan.key in self.resolved: return self.resolved[scan.key]
        harmonized_path = metadata.get('harmonized_source_ply')
        checksum = metadata.get('harmonized_source_sha256')
        if checksum:
            if self.checksums is None:
                raise FileNotFoundError('Select the harmonized full-model folder for this ROI.')
            try:
                return self.checksums.resolve([harmonized_path], scanner=scan.scanner,
                                              sample=scan.sample_name, checksum=checksum)
            except ValueError as error:
                raise ValueError(f'{scan.scanner}/{scan.sample_name}: could not verify the harmonized full model. '
                                 'Select the folder containing the harmonized exports used to create these ROIs. '
                                 f'{error}') from error
        recorded = [str(harmonized_path)] if harmonized_path else [
            str(metadata[k]) for k in SOURCE_KEYS if metadata.get(k)]
        for value in recorded:
            for candidate in self.rebase(value):
                scanners, samples = identity(candidate.relative_to(self.original_root) if self.original_root else candidate)
                if (candidate.suffix.lower() in MESH_EXTENSIONS and
                        (not scanners or scanners == {scan.scanner.casefold()}) and
                        (not samples or samples == {(scan.group, scan.timepoint, scan.specimen)})):
                    self.resolved[scan.key] = candidate; return candidate
        candidates = [p for p in dict.fromkeys(self.scan_index(scan.scanner).get(scan.sample_name.upper(), [])) if self.inside_root(p)]
        basenames = {parts_of(value)[-1].casefold() for value in recorded}
        matches = [p for p in candidates if p.name.casefold() in basenames]
        if len(matches) == 1: selected = matches[0]
        else:
            # A colour PLY is preferred over the scanner's accompanying STL.
            preferred = [p for p in candidates if p.suffix.lower() == '.ply']
            if not preferred: preferred = [p for p in candidates if p.suffix.lower() in ('.vtp', '.obj')]
            if not preferred: preferred = candidates
            if len(preferred) == 1: selected = preferred[0]
            elif not preferred: raise FileNotFoundError(f'Original missing: {scan.scanner} / {scan.sample_name}. Choose the original data folder.')
            else: raise ValueError(f'Ambiguous original: {scan.scanner} / {scan.sample_name}; {len(preferred)} possible models. Saved source filename is needed to distinguish them.')
        self.resolved[scan.key] = selected; return selected

    def texture_for(self, path, metadata):
        recorded = [str(metadata[k]) for k in TEXTURE_KEYS if metadata.get(k)]
        # PLY headers commonly name their own texture file.
        if path.suffix.lower() == '.ply':
            with path.open('rb') as stream:
                for _ in range(500):
                    line = stream.readline(4096).decode('ascii', errors='replace').strip()
                    if line == 'end_header' or not line: break
                    match = re.match(r'comment\s+TextureFile\s+(.+)', line, re.I)
                    if match: recorded.insert(0, match.group(1).strip().strip('"'))
        for value in recorded:
            candidates = [path.parent / parts_of(value)[-1], *self.rebase(value)]
            for candidate in candidates:
                if candidate.is_file() and self.inside_root(candidate): return candidate.resolve()
        images = [p for p in path.parent.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'} and p.is_file()]
        exact = [p for p in images if p.stem.casefold() in {path.stem.casefold(), path.stem.casefold() + '_texture'}]
        return exact[0] if len(exact) == 1 else (images[0] if len(images) == 1 else None)

    def load(self, scan, use_texture=False):
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
        metadata = sample_metadata(scan); source = self.locate(scan, metadata)
        source_scan = core.Scan(scan.scanner, scan.group, scan.timepoint, scan.specimen, source)
        mesh = core.load_mesh(source_scan, self.dataset_root, False)
        mesh['coordinate_frame'] = 'original scanner coordinates (not aligned to ROI)'
        transform = metadata.get('original_to_canonical')
        if transform is not None:
            matrix = np.asarray(transform, dtype=float)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]) or abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
                raise ValueError('Invalid saved original_to_canonical transform')
            roi = core.load_mesh(scan, self.dataset_root, False)
            original = vtk_to_numpy(mesh['poly'].GetPoints().GetData()) + np.asarray(mesh['center_translation'])
            frame = model_frame(scan, original, matrix)
            if frame != 'unverified':
                canonical = original if frame == 'canonical' else original @ matrix[:3, :3].T + matrix[:3, 3]
                anchor = np.asarray(roi['center_translation'])
                vertices = np.ascontiguousarray(canonical - anchor)
                mesh['poly'].GetPoints().SetData(numpy_to_vtk(vertices, deep=True))
                mesh['poly'].GetPointData().SetNormals(None); mesh['poly'].GetCellData().SetNormals(None)
                mesh['bounds'] = list(mesh['poly'].GetBounds()); mesh['center_translation'] = anchor.tolist()
                mesh['coordinate_frame'] = ('already in approved ROI coordinates' if frame == 'canonical'
                                            else 'aligned to approved ROI using saved transform')
                mesh['original_to_canonical'] = (np.eye(4) if frame == 'canonical' else matrix).tolist()
            else:
                mesh['coordinate_frame'] = 'full-model frame unverified; displayed independently'
        texture_path = self.texture_for(source, metadata)
        if texture_path and mesh['poly'].GetPointData().GetTCoords() is not None and (use_texture or mesh['association'] == 'none'):
            factory = vtk.vtkImageReader2Factory(); reader = factory.CreateImageReader2(str(texture_path))
            if reader:
                reader.SetFileName(str(texture_path)); reader.Update()
                texture = vtk.vtkTexture(); texture.SetInputConnection(reader.GetOutputPort()); texture.InterpolateOn()
                mesh['texture'] = texture; mesh['texture_path'] = str(texture_path)
                mesh['colour_source'] = 'Original texture: ' + texture_path.name
        elif use_texture: mesh['colour_source'] += ' | original texture unavailable'
        mesh['model_representation'] = 'original'
        return mesh
