"""Resolve each original model inside the selected data root; never edit it."""
from __future__ import annotations
from pathlib import Path, PureWindowsPath
import json
import os
import re
import numpy as np
import roi_browser_core as core

MESH_EXTENSIONS = {'.ply', '.vtp', '.obj', '.stl'}
SOURCE_KEYS = ('source_ply', 'raw_path', 'original_path', 'original_mesh', 'source_path', 'source_file', 'source_mesh', 'input_path')
TEXTURE_KEYS = ('source_texture', 'texture_path', 'texture_file')


def parts_of(value):
    return PureWindowsPath(str(value)).parts if '\\' in str(value) else Path(str(value)).parts


def sample_metadata(scan):
    """Read small source/transform fields, not mesh arrays, from the approved sample."""
    result = {}
    path = scan.path if scan.path.suffix.lower() == '.npz' else scan.path.with_name('geometry.npz')
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            for key in (*SOURCE_KEYS, *TEXTURE_KEYS, 'original_to_canonical'):
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
                if key in (*SOURCE_KEYS, *TEXTURE_KEYS, 'original_to_canonical') and key not in result: result[key] = value
                if isinstance(value, dict): fields(value)
        fields(data)
    return result


def normal_name(value): return re.sub('[^a-z0-9]', '', str(value).lower())


class OriginalModels:
    def __init__(self, dataset_root, original_root=None):
        self.dataset_root = Path(dataset_root).resolve()
        self.original_root = Path(original_root).expanduser().resolve() if original_root else None
        if self.original_root and not self.original_root.is_dir(): raise ValueError(f'Original data folder unavailable: {self.original_root}')
        self.index = {}; self.resolved = {}

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
        if scanner in self.index: return self.index[scanner]
        root = self.original_root
        if root is None: return {}
        direct = [p for p in root.iterdir() if p.is_dir() and normal_name(p.name) == normal_name(scanner)]
        starts = [root] if normal_name(root.name) == normal_name(scanner) else direct
        if not starts: starts = [root]
        index = {}
        for start in starts:
            for folder, directories, files in os.walk(start):
                directories[:] = sorted(d for d in directories if not d.startswith(('.', '_')) and
                    d.casefold() not in {'approved rois', 'annotations', 'versions', 'exports', 'runs', 'feature_cache', 'surface_cache'})
                folder = Path(folder)
                if not core.SAMPLE.fullmatch(folder.name): continue
                if core.scanner_name(folder.parts) != scanner: directories[:] = []; continue
                found = []
                for subfolder, children, names in os.walk(folder):
                    if len(Path(subfolder).relative_to(folder).parts) >= 2: children[:] = []
                    children[:] = [d for d in children if not d.startswith(('.', '_'))]
                    found.extend(Path(subfolder) / name for name in names if Path(name).suffix.lower() in MESH_EXTENSIONS)
                index.setdefault(folder.name.upper(), []).extend(found)
                directories[:] = []
        self.index[scanner] = index
        return index

    def locate(self, scan, metadata):
        if scan.key in self.resolved: return self.resolved[scan.key]
        recorded = [str(metadata[k]) for k in SOURCE_KEYS if metadata.get(k)]
        for value in recorded:
            for candidate in self.rebase(value):
                if candidate.suffix.lower() in MESH_EXTENSIONS:
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
            canonical = np.column_stack((original, np.ones(len(original)))) @ matrix.T
            anchor = np.asarray(roi['center_translation'])
            vertices = np.ascontiguousarray(canonical[:, :3] - anchor)
            mesh['poly'].GetPoints().SetData(numpy_to_vtk(vertices, deep=True))
            mesh['poly'].GetPointData().SetNormals(None); mesh['poly'].GetCellData().SetNormals(None)
            mesh['bounds'] = list(mesh['poly'].GetBounds()); mesh['center_translation'] = anchor.tolist()
            mesh['coordinate_frame'] = 'aligned to approved ROI using saved transform'
            mesh['original_to_canonical'] = matrix.tolist()
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
