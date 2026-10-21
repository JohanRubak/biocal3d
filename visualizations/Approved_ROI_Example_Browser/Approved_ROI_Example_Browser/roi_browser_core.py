"""Read-only Approved ROIs discovery, mesh loading and reproducible image export."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import re
import sys
import numpy as np

VERSION = '1.2.0'
SCANNERS = ('iTERO', 'TRIOS3', 'TRIOS5', 'LABscanner')
STAGES = ('baseline', 'partial', 'clean')
STAGE_LABELS = {'baseline': 'Baseline', 'partial': 'Partial cleaning', 'clean': 'Final cleaning'}
SAMPLE = re.compile(r'^BCG([1-6])T(\d+)-(\d+)$', re.I)


@dataclass(frozen=True)
class Scan:
    scanner: str
    group: int
    timepoint: int
    specimen: int
    path: Path

    @property
    def physical_id(self): return f'BCG{self.group}-{self.specimen}'
    @property
    def sample_name(self): return f'BCG{self.group}T{self.timepoint}-{self.specimen}'
    @property
    def stage(self):
        return {0: 'baseline', 1: 'clean' if self.group <= 2 else 'partial', 2: 'clean'}.get(self.timepoint, 'unknown')
    @property
    def key(self): return self.physical_id, self.stage, self.scanner


def scanner_name(parts):
    for part in reversed(parts):
        key = re.sub('[^a-z0-9]', '', part.lower())
        for name in SCANNERS:
            if key == name.lower(): return name
    return None


def discover(root):
    """Inspect sample folders only. Never traverse review versions or annotations."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir(): raise ValueError(f'Folder does not exist: {root}')
    start = root / 'samples' if (root / 'samples').is_dir() else root
    scans, warnings, seen = [], [], {}
    for folder, dirs, files in os.walk(start):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d.lower() not in
                         {'annotations', 'versions', 'exports', '__pycache__', 'feature_cache'})
        folder = Path(folder)
        match = SAMPLE.fullmatch(folder.name)
        if not match: continue
        # A sample is terminal: roi.vtp can also be a review directory, never a mesh.
        dirs[:] = []
        scanner = scanner_name(folder.parts)
        if scanner is None:
            warnings.append(f'Cannot identify scanner; skipped {folder}'); continue
        group, timepoint, specimen = map(int, match.groups())
        if specimen < 1 or timepoint not in ((0, 1) if group <= 2 else (0, 1, 2)):
            warnings.append(f'Unexpected sample ID; skipped {folder}'); continue
        path = next((folder / name for name in ('geometry.npz', 'roi.vtp') if name in files), None)
        if path is None:
            warnings.append(f'No geometry.npz or roi.vtp file in {folder}'); continue
        scan = Scan(scanner, group, timepoint, specimen, path)
        if scan.key in seen:
            raise ValueError(f'Duplicate scan identity {scan.key}:\n{seen[scan.key]}\n{path}\nChoose one Approved ROIs dataset root.')
        seen[scan.key] = path; scans.append(scan)
    scans.sort(key=lambda s: (s.group, s.specimen, s.timepoint, SCANNERS.index(s.scanner)))
    if not scans:
        raise ValueError('No prepared scans found. Choose Approved ROIs (containing samples/SCANNER/G#/BCG#T#-#/geometry.npz or roi.vtp).')
    return root, scans, warnings


def write_inventory(scans, path):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['physical_id', 'sample_name', 'group', 'specimen', 'scanner', 'stage', 'timepoint', 'mesh_path'])
        for s in scans: writer.writerow([s.physical_id, s.sample_name, s.group, s.specimen, s.scanner, s.stage, s.timepoint, s.path])


def safe_output(root, output):
    output = Path(output).expanduser().resolve()
    if output == root or root in output.parents:
        raise ValueError('Choose an export folder outside Approved ROIs so the dataset remains read-only.')
    output.mkdir(parents=True, exist_ok=True)
    return output


def find_texture(recorded, scan_path, root):
    if not recorded: return None
    parts = PureWindowsPath(recorded).parts if '\\' in recorded else Path(recorded).parts
    candidates = [Path(recorded), scan_path.parent / parts[-1]]
    data_root = next((p for p in (root, *root.parents) if p.name.lower() == 'data'), root.parent)
    # Rebase original .../data/... paths onto the current project's data folder.
    for i, part in enumerate(parts):
        if part.lower() == 'data': candidates.append(data_root.joinpath(*parts[i + 1:]))
    return next((p for p in candidates if p.is_file()), None)


def rgb_array(value, count):
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[0] != count or value.shape[1] not in (3, 4): return None
    value = value[:, :3]
    if not np.isfinite(value).all(): raise ValueError('RGB contains non-finite values')
    if np.issubdtype(value.dtype, np.floating) and value.size and value.max() <= 1.00001:
        value = value * 255
    return np.clip(np.rint(value), 0, 255).astype(np.uint8)


def load_mesh(scan, root, use_texture=False):
    """Load approved triangles unchanged; center translation is display-only."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    path = scan.path
    texture = None; note = ''; recorded = ''; uv = wedge = None
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            def first(names):
                return next((np.asarray(data[k]) for k in names if k in data), None)
            vertices = first(('vertices', 'points', 'verts'))
            faces = first(('faces', 'triangles'))
            rgb = first(('rgb', 'vertex_colors', 'colors', 'colours', 'vertex_rgb'))
            face_rgb = first(('face_rgb', 'face_colors'))
            uv = first(('uv', 'texcoords', 'texture_coordinates', 'tcoords'))
            wedge = first(('wedge_uv',))
            recorded_array = first(('texture_path',))
            if recorded_array is not None and recorded_array.size == 1: recorded = str(recorded_array.item())
            mask = first(('face_roi_mask',))
        if vertices is None or faces is None:
            companion = path.with_name('roi.vtp')
            if companion.is_file():
                return load_mesh(Scan(scan.scanner, scan.group, scan.timepoint, scan.specimen, companion), root, use_texture)
            raise ValueError(f'Unsupported NPZ keys in {path}; expected vertices/points and faces/triangles.')
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=np.int64)
        if faces.ndim == 2 and faces.shape[1] == 4 and np.all(faces[:, 0] == 3): faces = faces[:, 1:]
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f'Expected Nx3 vertices and Mx3 triangular faces in {path}')
        if not len(faces) or not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError(f'Invalid or empty mesh in {path}')
        if mask is not None:
            mask = np.asarray(mask, bool)
            if mask.shape != (len(faces),): raise ValueError('Invalid face_roi_mask shape')
            faces = faces[mask]
            if face_rgb is not None: face_rgb = face_rgb[mask]
            if wedge is not None: wedge = wedge[mask]
        if not len(faces): raise ValueError('Approved ROI mask contains no faces')
        rgb = rgb_array(rgb, len(vertices)) if rgb is not None else None
        face_rgb = rgb_array(face_rgb, len(faces)) if face_rgb is not None else None
        if rgb is None and face_rgb is None and path.with_name('roi.vtp').is_file():
            companion = Scan(scan.scanner, scan.group, scan.timepoint, scan.specimen, path.with_name('roi.vtp'))
            return load_mesh(companion, root, use_texture)
        used = vertices[np.unique(faces)]
        center = (used.min(0) + used.max(0)) / 2
        vertices = vertices - center
        poly = vtk.vtkPolyData(); points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.ascontiguousarray(vertices), deep=True)); poly.SetPoints(points)
        cells = vtk.vtkCellArray()
        cells.SetCells(len(faces), numpy_to_vtkIdTypeArray(np.column_stack((np.full(len(faces), 3), faces)).ravel(), deep=True))
        poly.SetPolys(cells)
        association = 'none'
        if rgb is not None:
            array = numpy_to_vtk(rgb, deep=True); array.SetName('ScannerRGB'); poly.GetPointData().SetScalars(array); association = 'point'
        elif face_rgb is not None:
            array = numpy_to_vtk(face_rgb, deep=True); array.SetName('ScannerRGB'); poly.GetCellData().SetScalars(array); association = 'cell'
        if uv is not None and uv.shape == (len(vertices), 2):
            array = numpy_to_vtk(np.asarray(uv, np.float32), deep=True); poly.GetPointData().SetTCoords(array)
        if use_texture and wedge is not None and wedge.size == len(faces) * 6:
            # Separate corners to preserve texture seams, without smoothing or decimation.
            corner_vertices = vertices[faces.ravel()]
            points = vtk.vtkPoints(); points.SetData(numpy_to_vtk(corner_vertices, deep=True))
            poly.SetPoints(points)
            cells = vtk.vtkCellArray(); cells.SetCells(len(faces), numpy_to_vtkIdTypeArray(
                np.column_stack((np.full(len(faces), 3), np.arange(len(faces) * 3).reshape(-1, 3))).ravel(), deep=True))
            poly.SetPolys(cells); poly.GetPointData().Initialize()
            array = numpy_to_vtk(np.asarray(wedge.reshape(-1, 2), np.float32), deep=True); poly.GetPointData().SetTCoords(array)
            if rgb is not None:
                array = numpy_to_vtk(rgb[faces.ravel()], deep=True); array.SetName('ScannerRGB'); poly.GetPointData().SetScalars(array)
        note = 'Saved vertex RGB' if association == 'point' else ('Saved face RGB' if association == 'cell' else 'NO COLOUR DATA — geometry only')
    else:
        reader_types = {'.vtp': vtk.vtkXMLPolyDataReader, '.ply': vtk.vtkPLYReader,
                        '.stl': vtk.vtkSTLReader, '.obj': vtk.vtkOBJReader}
        if path.suffix.lower() not in reader_types: raise ValueError(f'Unsupported mesh format: {path.suffix}')
        reader = reader_types[path.suffix.lower()](); reader.SetFileName(str(path)); reader.Update()
        source = reader.GetOutput()
        if not source.GetNumberOfPoints(): raise ValueError(f'Empty mesh: {path}')
        triangles = vtk.vtkTriangleFilter(); triangles.SetInputData(source); triangles.PassLinesOff(); triangles.PassVertsOff(); triangles.Update()
        poly = vtk.vtkPolyData(); poly.DeepCopy(triangles.GetOutput())
        if not poly.GetNumberOfPolys(): raise ValueError(f'No surface triangles in {path}')
        # Explicitly choose RGB only; labels and numeric scalars must never be colour-mapped.
        association = 'none'
        for data, count, mode in ((poly.GetPointData(), poly.GetNumberOfPoints(), 'point'),
                                  (poly.GetCellData(), poly.GetNumberOfCells(), 'cell')):
            candidates = sorted([data.GetArray(i) for i in range(data.GetNumberOfArrays())],
                                key=lambda a: 0 if (a.GetName() or '').lower() in ('rgb', 'rgba', 'colors', 'scannerrgb') else 1)
            selected = next((a for a in candidates if a.GetNumberOfComponents() in (3, 4) and
                             ((a.GetName() or '').lower() in ('rgb', 'rgba', 'colors', 'colour', 'color', 'scannerrgb') or a.GetDataType() == vtk.VTK_UNSIGNED_CHAR)), None)
            if selected is not None:
                colors = rgb_array(vtk_to_numpy(selected), count)
                if colors is not None:
                    array = numpy_to_vtk(colors, deep=True); array.SetName('ScannerRGB'); data.SetScalars(array); association = mode; break
        vertices = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
        center = (vertices.min(0) + vertices.max(0)) / 2; vertices -= center
        poly.GetPoints().SetData(numpy_to_vtk(vertices, deep=True))
        companion = path.with_name('geometry.npz')
        if companion.is_file():
            with np.load(companion, allow_pickle=False) as data:
                if 'texture_path' in data: recorded = str(data['texture_path'].item())
        note = f'{path.suffix[1:].upper()} {association} RGB' if association != 'none' else 'NO COLOUR DATA — geometry only'
    texture_path = find_texture(recorded, path, root)
    if use_texture:
        if texture_path and poly.GetPointData().GetTCoords() is not None:
            factory = vtk.vtkImageReader2Factory(); reader = factory.CreateImageReader2(str(texture_path))
            if reader:
                reader.SetFileName(str(texture_path)); reader.Update()
                texture = vtk.vtkTexture(); texture.SetInputConnection(reader.GetOutputPort()); texture.InterpolateOn()
                note = f'Original texture: {texture_path.name}'
        if texture is None: note += ' | texture unavailable; RGB fallback'
    bounds = poly.GetBounds()
    return {'poly': poly, 'association': association, 'texture': texture, 'colour_source': note,
            'texture_path': str(texture_path) if texture is not None else None,
            'center_translation': center.tolist(), 'bounds': list(bounds),
            'vertices': poly.GetNumberOfPoints(), 'faces': poly.GetNumberOfCells(), 'loaded_path': str(path)}


def actor_for(mesh, mode='colour', angle=0):
    import vtk
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(mesh['poly'])
    mapper.SetColorModeToDirectScalars()
    if mesh['association'] == 'point': mapper.SetScalarModeToUsePointData()
    else: mapper.SetScalarModeToUseCellData()
    mapper.SetScalarVisibility(mode == 'colour' and mesh['texture'] is None and mesh['association'] != 'none')
    actor = vtk.vtkActor(); actor.SetMapper(mapper); actor.RotateZ(angle)
    prop = actor.GetProperty(); prop.SetColor(.69, .72, .75); prop.SetSpecular(0)
    if mode == 'colour':
        prop.LightingOff()
        if mesh['texture'] is not None:
            prop.SetColor(1, 1, 1); actor.SetTexture(mesh['texture'])
    else:
        prop.LightingOn(); prop.SetAmbient(.25); prop.SetDiffuse(.75); prop.SetInterpolationToFlat()
        if mode == 'wireframe': prop.EdgeVisibilityOn(); prop.SetEdgeColor(.14, .18, .22)
    return actor


def default_camera(meshes):
    # One view and one scale for the whole paired set. No per-image stretching.
    radius = max((np.linalg.norm([max(abs(m['bounds'][0]), abs(m['bounds'][1])),
                                 max(abs(m['bounds'][2]), abs(m['bounds'][3])),
                                 max(abs(m['bounds'][4]), abs(m['bounds'][5]))]) for m in meshes), default=7)
    radius = max(radius * 1.08, .001)
    return {'position': [0, 0, radius * 6], 'focal_point': [0, 0, 0], 'view_up': [0, 1, 0], 'parallel_scale': radius}


def set_camera(renderer, state):
    c = renderer.GetActiveCamera(); c.ParallelProjectionOn()
    c.SetPosition(*state['position']); c.SetFocalPoint(*state['focal_point']); c.SetViewUp(*state['view_up'])
    c.SetParallelScale(state['parallel_scale']); renderer.ResetCameraClippingRange()


def camera_state(renderer):
    c = renderer.GetActiveCamera()
    return {'position': list(c.GetPosition()), 'focal_point': list(c.GetFocalPoint()),
            'view_up': list(c.GetViewUp()), 'parallel_scale': c.GetParallelScale()}


def render_png(mesh, path, camera, pixels=1600, mode='colour', angle=0):
    import vtk
    renderer = vtk.vtkRenderer(); renderer.SetBackground(1, 1, 1)
    renderer.AddActor(actor_for(mesh, mode, angle)); set_camera(renderer, camera)
    window = vtk.vtkRenderWindow(); window.SetOffScreenRendering(1); window.SetMultiSamples(0)
    window.SetSize(pixels, pixels); window.AddRenderer(renderer)
    render_errors = []
    window.AddObserver('ErrorEvent', lambda *_: render_errors.append('VTK rendering error'))
    try:
        window.Render()
        if render_errors: raise RuntimeError('VTK could not render the image. Check the desktop OpenGL context and terminal messages.')
        capture = vtk.vtkWindowToImageFilter(); capture.SetInput(window); capture.ReadFrontBufferOff(); capture.Update()
        writer = vtk.vtkPNGWriter(); writer.SetFileName(str(path)); writer.SetInputConnection(capture.GetOutputPort()); writer.Write()
        if writer.GetErrorCode() or not Path(path).is_file(): raise OSError(f'PNG export failed: {path}')
        from PIL import Image
        with Image.open(path) as image: image.save(path, dpi=(300, 300))
    finally: window.Finalize()


def compose_montage(folder, rows, scanners, cells, tile=620):
    from PIL import Image, ImageDraw, ImageFont
    def font(size):
        for name in ('arial.ttf', 'DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf'):
            try: return ImageFont.truetype(name, size)
            except OSError: pass
        return ImageFont.load_default()
    pad, gap, title_height = 18, 24, 105
    panel_width = len(rows) * tile + 2 * pad
    panel_height = tile + title_height + pad
    image = Image.new('RGB', (2 * panel_width + 3 * gap, 2 * panel_height + 3 * gap + 55), 'white')
    draw = ImageDraw.Draw(image)
    panel_files = {}
    for index, scanner in enumerate(scanners):
        panel = Image.new('RGB', (panel_width, panel_height), 'white')
        labels = ImageDraw.Draw(panel)
        labels.rectangle((0, 0, panel_width - 1, panel_height - 1), outline='#cbd6df', width=2)
        labels.text((pad, 16), scanner, fill='#172b39', font=font(32))
        for col, stage in enumerate(rows):
            x = pad + col * tile
            labels.text((x + tile / 2, 66), STAGE_LABELS[stage], fill='#172b39', font=font(25), anchor='mt')
            path = cells.get((stage, scanner))
            if path:
                with Image.open(path) as scan:
                    scan = scan.convert('RGB'); scan.thumbnail((tile - pad, tile - pad), Image.Resampling.LANCZOS)
                    panel.paste(scan, (x + (tile - scan.width) // 2, title_height + (tile - scan.height) // 2))
            else:
                labels.text((x + tile / 2, title_height + tile / 2), 'Missing / failed scan', fill='#8a4550', font=font(23), anchor='mm')
        filename = scanner + '_stages.png'
        panel.save(folder / filename, dpi=(300, 300)); panel_files[scanner] = filename
        image.paste(panel, (gap + (index % 2) * (panel_width + gap), gap + (index // 2) * (panel_height + gap)))
    draw.text((gap, image.height - 40), 'Matched physical specimen; four scanner panels with cleaning stages from left to right.', fill='#4c5860', font=font(22))
    image.save(folder / 'labelled_overview.png', dpi=(300, 300))
    return panel_files


def export_set(root, output, scans, physical_id, stages=None, camera=None, mode='colour',
               angles=None, use_texture=False, pixels=1600, note='', progress=None, fit_full=True,
               show_original=False, original_root=None):
    output = safe_output(root, output); angles = angles or {}
    chosen = [s for s in scans if s.physical_id == physical_id and (stages is None or s.stage in stages)]
    if not chosen: raise ValueError('No scans match the selected specimen and stage.')
    group = chosen[0].group
    rows = [s for s in STAGES if s in (stages or (('baseline', 'clean') if group <= 2 else STAGES))]
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder = output / f'{physical_id}_{stamp}'; folder.mkdir()
    resolver = None
    if show_original:
        from original_models import OriginalModels
        resolver = OriginalModels(root, original_root)
    meshes, errors = {}, []
    for scan in chosen:
        if progress: progress(f'Loading {scan.scanner} / {scan.sample_name}')
        try: meshes[scan.key] = resolver.load(scan, use_texture) if resolver else load_mesh(scan, root, use_texture)
        except Exception as exc: errors.append({'scanner': scan.scanner, 'stage': scan.stage, 'source': str(scan.path), 'error': str(exc)})
    if not meshes: raise ValueError('No meshes could be loaded: ' + '; '.join(e['error'] for e in errors))
    camera = camera or default_camera(list(meshes.values()))
    camera = dict(camera)
    if fit_full:
        # A bounding sphere keeps every ROI in frame for any shared camera angle.
        radius = max(np.linalg.norm([max(abs(m['bounds'][0]), abs(m['bounds'][1])),
                                     max(abs(m['bounds'][2]), abs(m['bounds'][3])),
                                     max(abs(m['bounds'][4]), abs(m['bounds'][5]))]) for m in meshes.values())
        direction = np.asarray(camera['position']) - np.asarray(camera['focal_point'])
        direction /= max(np.linalg.norm(direction), 1e-12)
        camera['focal_point'] = [0, 0, 0]; camera['position'] = (direction * max(radius * 6, .01)).tolist()
        camera['parallel_scale'] = max(radius * 1.08, .001)
    cells, records = {}, []
    for scan in chosen:
        mesh = meshes.get(scan.key)
        if mesh is None: continue
        if progress: progress(f'Exporting {scan.scanner} / {scan.sample_name}')
        filename = f'{scan.sample_name}_{scan.scanner}_{scan.stage}.png'
        path = folder / filename
        angle = float(angles.get(scan.scanner + '|' + scan.stage, angles.get(scan.scanner, 0)))
        render_png(mesh, path, camera, pixels, mode, angle)
        cells[scan.stage, scan.scanner] = path
        source = Path(mesh['loaded_path'])
        records.append({'sample_name': scan.sample_name, 'scanner': scan.scanner, 'stage': scan.stage,
                        'source_path': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                        'image': filename, 'colour_source': mesh['colour_source'], 'texture_path': mesh['texture_path'],
                        'approved_roi_path': str(scan.path), 'coordinate_frame': mesh.get('coordinate_frame', 'centred approved ROI'),
                        'original_to_canonical': mesh.get('original_to_canonical'),
                        'texture_sha256': hashlib.sha256(Path(mesh['texture_path']).read_bytes()).hexdigest() if mesh['texture_path'] else None,
                        'display_center_translation': mesh['center_translation'], 'rotation_z_degrees': angle,
                        'vertices': mesh['vertices'], 'faces': mesh['faces']})
    panel_files = compose_montage(folder, rows, SCANNERS, cells)
    missing = [{'stage': stage, 'scanner': scanner} for stage in rows for scanner in SCANNERS if (stage, scanner) not in cells]
    metadata = {'browser_version': VERSION, 'exported_utc': datetime.now(timezone.utc).isoformat(),
                'dataset_root': str(root), 'physical_id': physical_id, 'group': group,
                'cultivation_hours': 12 if group <= 2 else 24, 'stages': rows, 'render_mode': mode,
                'original_texture_requested': use_texture, 'individual_image_pixels': [pixels, pixels], 'dpi_metadata': 300,
                'model_representation': 'original' if show_original else 'approved_roi',
                'original_data_root': str(original_root) if original_root else None,
                'camera': camera, 'projection': 'parallel', 'note': note,
                'overview_layout': 'Four scanner panels (2 by 2), with stage columns inside each panel',
                'scanner_panel_images': panel_files,
                'fit_full_roi_on_export': fit_full,
                'display_policy': 'No colour correction, smoothing or decimation. Approved ROI coordinates centred for display. Original models use the saved original_to_canonical transform and ROI centre where available; otherwise they use their own centred scanner coordinates. No new registration is performed. Source coordinate units retained.',
                'missing_or_failed_panels': missing, 'errors': errors, 'scans': records,
                'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__}}
    import vtk
    metadata['runtime']['vtk'] = vtk.vtkVersion.GetVTKVersion()
    (folder / 'export_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    write_inventory(chosen, folder / 'selected_scans.csv')
    return folder, metadata
