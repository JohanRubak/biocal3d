"""Render BCG3T1-1 full models and approved ROIs before and after harmonization."""
import argparse
from uuid import uuid4
from pathlib import Path, PureWindowsPath
import re

import numpy as np
import pandas as pd

from analyse_harmonization import rgb_values
from investigate_approved_rois import SCANNERS, Figures, discover, plt, read_json

RAW_MODELS = r"C:\Users\au662213\repos\biocal3d\data\02-09-2026-Data collected"
CORRECTED_MODELS = r"C:\Users\au662213\repos\biocal3d\data\Color-harmonized Data"


def find_model(root, scanner, sample, source=None):
    if not root.is_dir():
        raise FileNotFoundError(f"Full-model folder is unavailable: {root}")
    pattern = re.compile(re.escape(sample).replace(r"\-", r"[-_]")+r"(?!\d)", re.I)
    extensions = {".ply", ".obj", ".vtp"}
    if source:
        recorded = Path(source)
        parts = PureWindowsPath(source).parts
        for i, part in enumerate(parts):
            if part.casefold() == root.name.casefold():
                recorded = root.joinpath(*parts[i+1:])
                break
        if (recorded.resolve().is_relative_to(root.resolve()) and
                recorded.suffix.lower() in extensions and recorded.is_file()):
            return recorded

    candidates = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in extensions or not path.is_file():
            continue
        relative = path.relative_to(root)
        if not pattern.search(str(relative)):
            continue
        tokens = re.sub(r"[^a-z0-9]", "", str(relative).lower())
        if scanner.lower() in tokens:
            candidates.append(path)
    named = [path for path in candidates if pattern.search(path.stem)]
    if len(named) == 1:
        return named[0]
    if source:
        basename = PureWindowsPath(source).name.casefold()
        recorded = [path for path in candidates if path.name.casefold() == basename]
        if len(recorded) == 1:
            return recorded[0]
    if len(candidates) != 1:
        raise ValueError(f"{scanner}/{sample}: found {len(candidates)} full models in {root}. "
                         "Use --models-manifest to specify exact files. Candidates: "+
                         ", ".join(str(p) for p in candidates))
    return candidates[0]


def load_mesh(path):
    import vtk
    readers = {".ply": vtk.vtkPLYReader, ".obj": vtk.vtkOBJReader, ".vtp": vtk.vtkXMLPolyDataReader}
    reader = readers[path.suffix.lower()]()
    reader.SetFileName(str(path))
    reader.Update()
    mesh = reader.GetOutput()
    if mesh.GetNumberOfPoints() == 0:
        raise ValueError(f"Empty mesh: {path}")
    texture_path = None
    if path.suffix.lower() == ".ply":
        with path.open("rb") as f:
            for _ in range(1000):
                line = f.readline().decode("utf-8", errors="replace").strip()
                if line.lower().startswith("comment texturefile "):
                    texture_path = path.parent / line.split(None, 2)[2]
                if line == "end_header":
                    break
    elif path.suffix.lower() == ".obj":
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"^mtllib\s+(.+)$", text, re.M)
        if matches:
            materials = (path.parent / matches[0].strip()).read_text(encoding="utf-8", errors="replace")
            images = re.findall(r"^map_Kd\s+(.+)$", materials, re.M)
            if len(set(images)) > 1:
                raise ValueError(f"Multiple OBJ textures in {path}. Export a single textured mesh or colored PLY.")
            if images:
                texture_path = path.parent / images[0].strip()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().LightingOff()
    if texture_path:
        if not texture_path.is_file():
            raise FileNotFoundError(f"Texture missing: {texture_path}")
        if mesh.GetPointData().GetTCoords() is None:
            raise ValueError(f"Texture coordinates missing: {path}")
        image_reader = vtk.vtkImageReader2Factory().CreateImageReader2(str(texture_path))
        if image_reader is None:
            raise ValueError(f"Unsupported texture format: {texture_path}")
        image_reader.SetFileName(str(texture_path))
        image_reader.Update()
        texture = vtk.vtkTexture()
        texture.SetInputConnection(image_reader.GetOutputPort())
        texture.InterpolateOn()
        actor.SetTexture(texture)
        mapper.ScalarVisibilityOff()
    else:
        colors = None
        for data, mode in [(mesh.GetPointData(), "point"), (mesh.GetCellData(), "cell")]:
            names = [data.GetArrayName(i) for i in range(data.GetNumberOfArrays())]
            for name in ["RGB", "RGBA", "Colors", "colors", "Color", "color"]:
                array = data.GetArray(name)
                if array is not None and array.GetNumberOfComponents() in (3, 4):
                    colors = array
                    break
            active = data.GetScalars()
            if colors is None and active is not None and active.GetNumberOfComponents() in (3, 4):
                colors = active
            if colors is not None:
                data.SetScalars(colors)
                if mode == "point":
                    mapper.SetScalarModeToUsePointData()
                else:
                    mapper.SetScalarModeToUseCellData()
                break
        if colors is None:
            raise ValueError(f"No RGB colors or usable texture in {path}")
        mapper.SetColorModeToDirectScalars()
        mapper.ScalarVisibilityOn()
    return mesh, actor


def align_full_models(raw, corrected):
    from scipy.spatial import cKDTree

    # Compare surfaces independently of vertex order and duplicated vertices.
    raw_tree = cKDTree(raw)
    span = max(np.linalg.norm(np.ptp(raw, axis=0)), 1.)
    tolerance = span * .002

    def matches(points):
        forward = raw_tree.query(points)[0]
        reverse = cKDTree(points).query(raw)[0]
        return max(np.quantile(forward, .95), np.quantile(reverse, .95)) <= tolerance

    if matches(corrected):
        return corrected, "Shared view"
    raw_center, corrected_center = raw.mean(axis=0), corrected.mean(axis=0)
    raw_axes = np.linalg.eigh(np.cov(raw.T))[1]
    corrected_axes = np.linalg.eigh(np.cov(corrected.T))[1]
    rotations = []
    if raw.shape == corrected.shape:
        u, _, vt = np.linalg.svd((corrected-corrected_center).T @ (raw-raw_center))
        signs = np.diag([1., 1., np.linalg.det(u @ vt)])
        rotations.append(u @ signs @ vt)
    for signs in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]):
        rotation = corrected_axes @ np.diag(signs) @ raw_axes.T
        if np.linalg.det(rotation) < 0:
            rotation = corrected_axes @ np.diag(-np.asarray(signs)) @ raw_axes.T
        rotations.append(rotation)
    for rotation in rotations:
        aligned = (corrected-corrected_center) @ rotation + raw_center
        if matches(aligned):
            return aligned, "Shared view after rigid alignment"
    return corrected, "Independent views (geometry differs)"


def roi_frame(points, geometry_path):
    from scipy.spatial import cKDTree

    with np.load(geometry_path, allow_pickle=False) as geometry:
        target = np.asarray(geometry["vertices"], float)
        source = geometry.get("vertices_raw")
        ids = geometry.get("raw_vertex_id")
    candidates = []
    if source is not None and source.shape == target.shape:
        candidates.append(source)
    if (ids is not None and ids.shape == (len(target),) and
            np.issubdtype(ids.dtype, np.integer) and np.all((ids >= 0) & (ids < len(points)))):
        candidates.append(points[ids])
    tolerance = max(np.linalg.norm(np.ptp(target, axis=0)), 1.) * .002
    for source in candidates:
        if not np.isfinite(source).all():
            continue
        origin, center = source.mean(axis=0), target.mean(axis=0)
        u, _, vt = np.linalg.svd((source-origin).T @ (target-center))
        rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
        errors = np.linalg.norm((source-origin) @ rotation + center-target, axis=1)
        if (np.quantile(errors, .95) <= tolerance and
                np.quantile(cKDTree(points).query(source)[0], .95) <= tolerance):
            return rotation, center-origin @ rotation
    return None


def full_model_camera(points, radius, top_down=False):
    import vtk

    center = (points.min(0)+points.max(0))/2
    _, axes = np.linalg.eigh(np.cov(points.T))
    normal, up = axes[:, 0], axes[:, 2]
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    if up[np.argmax(np.abs(up))] < 0:
        up = -up
    if top_down:
        normal, up = np.array([0., 0., 1.]), np.array([0., 1., 0.])
    camera = vtk.vtkCamera()
    camera.SetPosition(*(center+normal*radius*4))
    camera.SetFocalPoint(*center)
    camera.SetViewUp(*up)
    camera.ParallelProjectionOn()
    camera.SetParallelScale(radius*1.04)
    return camera


def render_pair(raw_path, corrected_path, raw_geometry, corrected_geometry):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    raw_mesh, raw_actor = load_mesh(raw_path)
    corrected_mesh, corrected_actor = load_mesh(corrected_path)
    raw_points = vtk_to_numpy(raw_mesh.GetPoints().GetData()).astype(float)
    corrected_points = vtk_to_numpy(corrected_mesh.GetPoints().GetData()).astype(float)
    aligned, status = align_full_models(raw_points, corrected_points)
    independent = status.startswith("Independent")
    raw_frame = roi_frame(raw_points, raw_geometry)
    corrected_frame = roi_frame(aligned, corrected_geometry) if independent else raw_frame
    if raw_frame is not None:
        raw_points = raw_points @ raw_frame[0] + raw_frame[1]
        raw_mesh.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(raw_points), deep=True))
        raw_mesh.Modified()
    if corrected_frame is not None:
        aligned = aligned @ corrected_frame[0] + corrected_frame[1]
    corrected_mesh.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(aligned), deep=True))
    corrected_mesh.Modified()
    radius = max(*(np.linalg.norm(points-(points.min(0)+points.max(0))/2, axis=1).max()
                   for points in (raw_points, aligned)), 1.)
    raw_camera = full_model_camera(raw_points, radius, top_down=raw_frame is not None)
    corrected_camera = (full_model_camera(aligned, radius, top_down=corrected_frame is not None)
                        if independent else raw_camera)
    if raw_frame is None or corrected_frame is None:
        status += "; ROI orientation could not be verified for one or both models"
    else:
        status += "; top-down in approved ROI coordinates"
    images = []
    for actor, camera in [(raw_actor, raw_camera), (corrected_actor, corrected_camera)]:
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(1, 1, 1)
        renderer.AddActor(actor)
        renderer.SetActiveCamera(camera)
        renderer.ResetCameraClippingRange()
        window = vtk.vtkRenderWindow()
        window.SetOffScreenRendering(1)
        window.SetSize(700, 700)
        window.SetMultiSamples(0)
        window.AddRenderer(renderer)
        window.Render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(window)
        capture.SetInputBufferTypeToRGB()
        capture.ReadFrontBufferOff()
        capture.Update()
        image = capture.GetOutput()
        width, height, _ = image.GetDimensions()
        images.append(np.flipud(vtk_to_numpy(image.GetPointData().GetScalars()).reshape(height, width, 3)).copy())
        window.Finalize()
    return images, status


def make_figure(images, view_labels, sample, filename, figs, independent=(), scanners_as_rows=False):
    shape = (len(SCANNERS), len(view_labels)) if scanners_as_rows else (len(view_labels), len(SCANNERS))
    fig, axes = plt.subplots(*shape, figsize=(12, 2.8*shape[0]),
                             squeeze=False, layout="constrained")
    for i, view_label in enumerate(view_labels):
        for j, scanner in enumerate(SCANNERS):
            ax = axes[j, i] if scanners_as_rows else axes[i, j]
            ax.imshow(images[scanner][i])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            scanner_label = scanner + ("\nFull-model view differs" if scanner in independent else "")
            if scanners_as_rows:
                if j == 0:
                    ax.set_title(view_label)
                if i == 0:
                    ax.set_ylabel(scanner_label)
            else:
                if i == 0:
                    ax.set_title(scanner_label)
                if j == 0:
                    ax.set_ylabel(view_label)
    fig.suptitle(sample+" · partial cleaning")
    figs.save(fig, filename, "", "")


def write_model_status(output, statuses):
    table = pd.DataFrame(statuses)
    path = output / "full_model_status.csv"
    try:
        table.to_csv(path, index=False)
    except PermissionError:
        path = output / f"full_model_status_{uuid4().hex[:8]}.csv"
        try:
            table.to_csv(path, index=False)
        except OSError as error:
            print(f"Could not save the model status CSV: {error}", flush=True)
            print(table.to_string(index=False), flush=True)
            return
        print(f"Status CSV was locked or read-only. Saved status to {path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--corrected-root", type=Path, required=True)
    p.add_argument("--raw-model-root", type=Path, default=Path(RAW_MODELS))
    p.add_argument("--corrected-model-root", type=Path, default=Path(CORRECTED_MODELS))
    p.add_argument("--models-manifest", type=Path, help="Optional CSV with scanner,raw_model,corrected_model.")
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent/"results"/"specimen_comparison")
    p.add_argument("--sample", default="BCG3T1-1")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    args.raw_root = args.raw_root.resolve()
    inventory = discover(args.raw_root)
    selected = inventory[inventory.sample_id.str.casefold() == args.sample.casefold()]
    if set(selected.scanner) != set(SCANNERS):
        p.error(f"Expected {args.sample} in all four scanners.")
    args.output.mkdir(parents=True, exist_ok=True)
    figs = Figures(args.output, args)
    roi_images, model_images, statuses = {}, {}, []
    manifest = None
    if args.models_manifest:
        manifest = pd.read_csv(args.models_manifest).set_index("scanner")
        if manifest.index.duplicated().any():
            p.error("Duplicate scanner in full-model manifest.")
    index = read_json(args.raw_root / "dataset_index.json")
    sources = {item.get("scan_id"): item.get("source") for item in index.get("scans", [])}
    for row in selected.itertuples():
        relative = Path(row.sample_dir).relative_to(args.raw_root)
        images = []
        valid_reference = None
        for root in [args.raw_root, args.corrected_root]:
            with np.load(root/relative/"maps.npz", allow_pickle=False) as data:
                valid = data["valid"].astype(bool)
                if valid_reference is not None and not np.array_equal(valid, valid_reference):
                    raise ValueError(f"Raw/harmonized ROI masks differ for {row.scanner}")
                valid_reference = valid
                image = np.ones((*valid.shape, 3), dtype=np.float32)
                image[valid] = rgb_values(data["rgb"][valid])
                images.append(image)
        roi_images[row.scanner] = images
        raw_path = corrected_path = None
        try:
            if manifest is not None:
                raw_path = args.models_manifest.parent / manifest.loc[row.scanner, "raw_model"]
                corrected_path = args.models_manifest.parent / manifest.loc[row.scanner, "corrected_model"]
            else:
                metadata = read_json(Path(row.sample_dir) / "metadata.json")
                source = metadata.get("source") or sources.get(row.scan_id)
                raw_path = find_model(args.raw_model_root, row.scanner, args.sample, source)
                corrected_path = find_model(args.corrected_model_root, row.scanner, args.sample)
            if raw_path.resolve() == corrected_path.resolve():
                raise ValueError("Raw and harmonized full-model paths coincide.")
            model_images[row.scanner], message = render_pair(
                raw_path, corrected_path, args.raw_root/relative/"geometry.npz",
                args.corrected_root/relative/"geometry.npz")
        except (OSError, ValueError, KeyError, ImportError) as error:
            message = str(error)
            print(f"Full-model example needs attention for {row.scanner}: {message}", flush=True)
        statuses.append(dict(scanner=row.scanner, raw_model=raw_path, corrected_model=corrected_path, status=message))
    make_figure(roi_images, ["ROI before", "ROI after"], args.sample, "04_BCG3T1_1_ROI_before_after", figs)
    if len(model_images) == len(SCANNERS):
        combined = {
            scanner: model_images[scanner] + roi_images[scanner]
            for scanner in SCANNERS
        }
        make_figure(combined, ["Full model · Before", "Full model · After", "ROI · Before", "ROI · After"],
                    args.sample, "04_BCG3T1_1_full_model_and_ROI", figs, scanners_as_rows=True,
                    independent={row["scanner"] for row in statuses if row["status"].startswith("Independent") or "could not be verified" in row["status"]})
    write_model_status(args.output, statuses)
    if len(model_images) != len(SCANNERS):
        print("ROI figure saved. Full-model figure is pending. See the model status output.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
