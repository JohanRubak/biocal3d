"""Transfer harmonized vertex RGB to the exact surfaces in an approved ROI export."""
import argparse
import csv
from itertools import permutations
from pathlib import Path
import sys
import re
import shutil
from tempfile import mkdtemp
from time import sleep

sys.path.append(str(Path(__file__).resolve().parents[2] / "studio" / "app"))
from dataset_paths import APPROVED_REORIENTED, FULL_HARMONIZED, data_root

import numpy as np
from numpy.lib.recfunctions import drop_fields
from PIL import Image
from plyfile import PlyData, PlyElement, PlyParseError
from scipy.spatial import cKDTree

from study_data import canonical_scanner, digest, identity, write_csv, write_json
from study_io import read_json, save_npz

DATA = data_root()
XYZ_TOL = 2e-5  # mm, to allow float32 PLY coordinate roundoff
CORNER_ORDERS = tuple(permutations(range(3)))
RGB = ("red", "green", "blue")
HASHED_FILES = {"geometry_sha256": "geometry.npz", "maps_sha256": "maps.npz",
                "ply_sha256": "roi.ply", "rgb_sha256": "rgb.png"}


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def harmonized_paths(root):
    result = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            scanner = canonical_scanner(folder.name)
        except ValueError:
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() != ".ply":
                continue
            try:
                group, _, _, sample, _, _ = identity(path.stem)
            except ValueError as exc:
                raise ValueError(f"Cannot identify harmonized scan {path}") from exc
            for part in path.relative_to(folder).parts[:-1]:
                group_folder = re.match(r"^G(\d+)\b", part, re.I)
                if group_folder and int(group_folder[1]) != group:
                    raise ValueError(f"Scan and group folder disagree: {path}")
            key = scanner, sample
            if key in result:
                raise ValueError(f"Duplicate harmonized scan {key}: {result[key]} and {path}")
            result[key] = path
    if not result:
        raise ValueError(f"No scan PLYs found under scanner folders in {root}")
    return result


def approved_rows(root):
    index = read_json(root / "dataset_index.json")
    rows = index.get("scans")
    if not isinstance(rows, list) or not rows:
        raise ValueError("The approved dataset index must contain a nonempty scans list.")
    seen, folders, scan_ids = set(), set(), set()
    for row in rows:
        rel = Path(str(row["sample_dir"]).replace("\\", "/"))
        folder = (root / rel).resolve()
        if not folder.is_relative_to(root) or folder == root:
            raise ValueError(f"Unsafe approved sample directory: {rel}")
        names = [row.get("scan_id", ""), row.get("sample_id", ""), row.get("sample_name", ""), str(rel)]
        samples = {identity(name)[3] for name in names if re.search(r"BCG\d+T\d+[_-]\d+", str(name), re.I)}
        if len(samples) != 1:
            raise ValueError(f"Missing or conflicting approved scan identity: {rel}")
        key = canonical_scanner(row["scanner"]), samples.pop()
        scan_id = row.get("scan_id") or "__".join(key)
        if key in seen or folder in folders or scan_id in scan_ids:
            raise ValueError(f"Duplicate approved scan or directory: {key}, {rel}")
        seen.add(key)
        folders.add(folder)
        scan_ids.add(scan_id)
        yield dict(row, scan_id=scan_id), folder.relative_to(root), key


def ply_geometry(ply):
    vertex = ply["vertex"].data
    xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float64)
    polygons = ply["face"].data
    field = "vertex_indices" if "vertex_indices" in polygons.dtype.names else "vertex_index"
    faces = np.stack(polygons[field])
    if (not len(xyz) or not np.isfinite(xyz).all() or faces.ndim != 2 or faces.shape[1] != 3
            or not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= len(xyz)):
        raise ValueError("Expected finite PLY coordinates and nonempty triangular connectivity.")
    return xyz, faces


def read_harmonized(path):
    ply = PlyData.read(path)
    xyz, faces = ply_geometry(ply)
    vertex = ply["vertex"].data
    if not all(name in vertex.dtype.names for name in RGB):
        raise ValueError("Harmonized PLY needs red, green and blue vertex properties.")
    rgb = np.column_stack([vertex[name] for name in RGB])
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
        raise ValueError("Harmonized vertex RGB values are out of range.")
    if np.issubdtype(rgb.dtype, np.integer) or rgb.max() > 1:
        rgb = rgb.astype(np.float32) / 255
    return xyz, faces, rgb.astype(np.float32)


def valid_ids(ids, length, limit):
    return (ids.shape == (length,) and np.issubdtype(ids.dtype, np.integer)
            and np.all((ids >= 0) & (ids < limit)))


def colours_on_roi(geo, source_xyz, source_faces, source_rgb):
    raw_xyz = geo["vertices_raw"]
    faces = geo["faces"]
    if (raw_xyz.ndim != 2 or raw_xyz.shape[1] != 3 or raw_xyz.shape != geo["vertices"].shape
            or not np.isfinite(raw_xyz).all()):
        raise ValueError("Missing usable raw-coordinate geometry for the approved ROI.")
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(raw_xyz)):
        raise ValueError("Invalid approved ROI triangles.")
    vertex_ids = geo.get("raw_vertex_id", np.empty(0))
    face_ids = geo.get("raw_triangle_id", np.empty(0))
    if (valid_ids(vertex_ids, len(raw_xyz), len(source_xyz))
            and valid_ids(face_ids, len(faces), len(source_faces))
            and np.all(np.linalg.norm(source_xyz[vertex_ids] - raw_xyz, axis=1) <= XYZ_TOL)
            and np.array_equal(np.sort(source_faces[face_ids], axis=1), np.sort(vertex_ids[faces], axis=1))):
        return source_rgb[vertex_ids], "verified source indices"

    # Match the same triangles by XYZ, allowing vertex/face reordering and
    # texture-seam splits that do not change the approved surface.
    target = raw_xyz[faces]
    source = source_xyz[source_faces]
    neighbours = cKDTree(source.mean(axis=1)).query_ball_point(target.mean(axis=1), XYZ_TOL)
    counts = np.fromiter(map(len, neighbours), dtype=np.int64, count=len(faces))
    if np.any(counts == 0):
        raise ValueError(f"{np.count_nonzero(counts == 0)} approved triangles have no matching harmonized XYZ.")
    target_indices = np.repeat(np.arange(len(faces)), counts)
    source_indices = np.concatenate(neighbours).astype(np.int64)
    offsets = target[target_indices, :, None, :] - source[source_indices, None, :, :]
    close = np.einsum("...i,...i->...", offsets, offsets) <= XYZ_TOL ** 2
    matched_faces, matched_corners = [], []
    for order in CORNER_ORDERS:
        valid = close[:, np.arange(3), order].all(axis=1)
        matched_faces.append(target_indices[valid])
        matched_corners.append(source_faces[source_indices[valid]][:, order])
    face_indices = np.concatenate(matched_faces)
    counts = np.bincount(face_indices, minlength=len(faces))
    if np.any(counts != 1):
        raise ValueError(f"Approved triangles without a unique geometry match: "
                         f"{np.count_nonzero(counts == 0)} missing, {np.count_nonzero(counts > 1)} ambiguous.")
    ordered_corners = np.empty_like(faces, dtype=np.int64)
    ordered_corners[face_indices] = np.concatenate(matched_corners)
    corner_rgb = source_rgb[ordered_corners]
    rgb = np.full((len(raw_xyz), 3), np.nan, dtype=np.float32)
    rgb[faces.ravel()] = corner_rgb.reshape(-1, 3)
    if not np.isfinite(rgb).all() or not np.array_equal(rgb[faces], corner_rgb):
        raise ValueError("Harmonized triangle colours disagree at shared approved vertices, "
                         "or the approved mesh has unreferenced vertices.")
    return rgb, "verified triangle coordinates"


def check_mesh(xyz, faces, geometry):
    if (xyz.shape != geometry["vertices"].shape or not np.array_equal(faces, geometry["faces"])
            or not np.allclose(xyz, geometry["vertices"], atol=XYZ_TOL, rtol=0)):
        raise ValueError("Approved mesh geometry or ordering differs from geometry.npz.")


def recolor_ply(source, dest, geometry, rgb8):
    ply = PlyData.read(source)
    check_mesh(*ply_geometry(ply), geometry)
    elements = []
    for element in ply.elements:
        data = element.data
        names = list(data.dtype.names)
        if element.name == "vertex":
            names = [name for name in names if name not in RGB]
            dtype = [(name, data.dtype[name]) for name in names] + [(name, "u1") for name in RGB]
            recolored = np.empty(len(data), dtype=dtype)
            for name in names:
                recolored[name] = data[name]
            for i, name in enumerate(RGB):
                recolored[name] = rgb8[:, i]
            data = recolored
        elif element.name == "face" and "texcoord" in names:
            data = drop_fields(data, "texcoord")
        elements.append(PlyElement.describe(data, element.name, comments=element.comments))
    comments = [line for line in ply.comments if not line.casefold().startswith("texturefile ")]
    comments.append("BioCal3D harmonized vertex RGB on approved geometry")
    PlyData(elements, text=ply.text, byte_order=ply.byte_order, comments=comments, obj_info=ply.obj_info).write(dest)


def recolor_vtp(source, dest, geometry, rgb8):
    from vtk import vtkXMLPolyDataReader, vtkXMLPolyDataWriter
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    reader = vtkXMLPolyDataReader()
    reader.SetFileName(str(source))
    reader.Update()
    mesh = reader.GetOutput()
    cells = mesh.GetPolys()
    if mesh.GetPoints() is None or not np.all(np.diff(vtk_to_numpy(cells.GetOffsetsArray())) == 3):
        raise ValueError("Expected a triangular approved VTP mesh.")
    check_mesh(vtk_to_numpy(mesh.GetPoints().GetData()),
               vtk_to_numpy(cells.GetConnectivityArray()).reshape(-1, 3), geometry)
    colors = numpy_to_vtk(rgb8, deep=True)
    colors.SetName("RGB")
    point_data = mesh.GetPointData()
    point_data.AddArray(colors)
    point_data.SetActiveScalars("RGB")
    point_data.SetTCoords(None)
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(str(dest))
    writer.SetInputData(mesh)
    if writer.Write() != 1:
        raise OSError(f"Failed to write {dest}")


def export_scan(source, dest, geometry, rgb):
    dest.mkdir(parents=True)
    rgb8 = np.rint(rgb * 255).astype(np.uint8)
    for name, writer in (("roi.ply", recolor_ply), ("roi.vtp", recolor_vtp)):
        if (source / name).is_file():
            writer(source / name, dest / name, geometry, rgb8)
    geometry["vertex_rgb"] = rgb
    geometry.pop("uv", None)
    save_npz(dest / "geometry.npz", **geometry)
    if (source / "maps.npz").is_file():
        maps = load_arrays(source / "maps.npz")
        check_mesh(maps["vertices"], maps["faces"], geometry)
        valid = maps["valid"].astype(bool)
        triangles = maps["triangle_id"][valid]
        if np.any((triangles < 0) | (triangles >= len(geometry["faces"]))):
            raise ValueError("Approved raster references a triangle outside its ROI.")
        weights = maps["barycentric"][valid]
        maps["rgb"][valid] = np.clip(np.einsum("ni,nij->nj", weights, rgb[geometry["faces"][triangles]]), 0, 1)
        save_npz(dest / "maps.npz", **maps)
        Image.fromarray(np.rint(maps["rgb"] * 255).astype(np.uint8)).save(dest / "rgb.png")
        Image.fromarray(valid.astype(np.uint8) * 255).save(dest / "valid_mask.png")
    elif any((source / name).exists() for name in ("rgb.png", "valid_mask.png")):
        raise ValueError("Cannot regenerate approved RGB preview without maps.npz.")
    return {key: digest(dest / name) for key, name in HASHED_FILES.items()
            if (dest / name).is_file()}


def write_manifests(approved, staging, output, rows, changes, audit):
    index = read_json(approved / "dataset_index.json")
    config = dict(index.get("config", {}), output=str(output), harmonized_roi_export=True,
                  approved_roi_template=str(approved), feature_caches_must_be_rebuilt=True)
    index.update(scans=rows, config=config)
    write_json(staging / "dataset_index.json", index)
    for name in ("scan_manifest.csv", "annotation_inventory.csv"):
        old = approved / name
        if old.is_file():
            with old.open(encoding="utf-8-sig", newline="") as stream:
                records = list(csv.DictReader(stream))
            write_csv(staging / name, [dict(row, **changes.get(row.get("scan_id"), {})) for row in records])
        elif name == "scan_manifest.csv":
            write_csv(staging / name, rows)
    stamp = approved / "scanner_orientation.json"
    if stamp.is_file():
        orientation = read_json(stamp)
        for scan in orientation.get("audit", []):
            if scan.get("scan_id") in changes:
                scan["hashes"] = {k: v for k, v in changes[scan["scan_id"]].items()
                                  if k in HASHED_FILES}
        orientation.update(annotation_files_verified=0, historical_previews=[])
        write_json(staging / stamp.name, orientation)
    for name in ("geometry_rebuild.json", "orientation_corrections_used.json",
                 "splits.json", "split_assignments.csv", "specimen_splits.csv", "splits.csv",
                 "classes.txt", "class_map.json", "annotation_classes.json"):
        old = approved / name
        if old.is_file():
            shutil.copy2(old, staging / name)
    write_json(staging / "run_config.json", config)
    write_csv(staging / "harmonized_roi_audit.csv", audit)


def publish(staging, output):
    for delay in (0, .5, 1, 2, 4):
        if delay:
            sleep(delay)
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}. Completed export kept at {staging}")
        try:
            staging.rename(output)
            return
        except PermissionError as exc:
            blocked = exc
        except OSError as exc:
            raise OSError(f"Could not publish {output}: {exc}. Completed export kept at {staging}") from exc
    raise PermissionError(f"Could not publish {output}: {blocked}. Completed export kept at {staging}. "
                          f'Close applications using the folder and retry with --publish-staged "{staging}".') from blocked


def publish_staged(staging):
    staging = staging.resolve()
    if staging.name != "export" or not staging.parent.name.startswith("biocal3d_harmonized_"):
        raise ValueError("Choose the staging export folder shown in the rename error.")
    index = read_json(staging / "dataset_index.json")
    config = index["config"]
    output = Path(config["output"]).resolve()
    if config.get("harmonized_roi_export") is not True or output.parent != staging.parent.parent:
        raise ValueError("Staged index does not identify a harmonized export beside its destination.")
    rows = list(approved_rows(staging))
    with (staging / "harmonized_roi_audit.csv").open(encoding="utf-8-sig", newline="") as stream:
        audit_ids = [row["scan_id"] for row in csv.DictReader(stream)]
    if audit_ids != [row["scan_id"] for row, _, _ in rows]:
        raise ValueError("Staged export audit is incomplete or disagrees with the dataset index.")
    template = Path(config["approved_roi_template"])
    for row, rel, _ in rows:
        folder = staging / rel
        if not row.get("geometry_sha256"):
            raise ValueError(f"Staged index is missing the geometry hash for {rel}")
        for key, name in HASHED_FILES.items():
            if row.get(key) and digest(folder / name) != row[key]:
                raise ValueError(f"Staged file differs from its export hash: {folder / name}")
        required = ["metadata.json"]
        if row.get("maps_sha256"):
            required.append("valid_mask.png")
        if (template / rel / "roi.vtp").is_file():
            required.append("roi.vtp")
        for name in required:
            if not (folder / name).is_file():
                raise FileNotFoundError(f"Staged export is missing {folder / name}")
    publish(staging, output)
    try:
        staging.parent.rmdir()
    except OSError:
        pass
    print(f"Created {output}", flush=True)


def export_dataset(harmonized, approved, output, check_only=False):
    harmonized, approved, output = (path.resolve() for path in (harmonized, approved, output))
    stamp = approved / "scanner_orientation.json"
    if (approved / "INCOMPLETE_ORIENTATION.json").exists() or (
            stamp.is_file() and read_json(stamp).get("status") != "complete"):
        raise ValueError("The input ROI orientation export is incomplete.")
    if not check_only:
        if output.exists():
            raise FileExistsError(f"Choose a new output folder: {output}")
        if any(output.is_relative_to(root) or root.is_relative_to(output) for root in (approved, harmonized)):
            raise ValueError("Choose an output folder separate from both input datasets.")
    pairs = list(approved_rows(approved))
    inputs = harmonized_paths(harmonized)
    expected = {key for _, _, key in pairs}
    missing = sorted(expected - inputs.keys())
    if missing:
        examples = ", ".join(f"{scanner}/{scan}" for scanner, scan in missing[:12])
        raise ValueError(f"{len(missing)} of {len(pairs)} approved scans are missing harmonized PLYs. First: {examples}")
    print(f"Matched {len(pairs)} filenames. {len(inputs.keys() - expected)} extra PLYs outside the approved index.",
          flush=True)
    parent = None if check_only else output.parent
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    temp = Path(mkdtemp(prefix="biocal3d_harmonized_", dir=parent))
    staging = temp / "export"
    ready = False
    try:
        rows, changes, audit = [], {}, []
        for i, (row, rel, key) in enumerate(pairs, 1):
            source = inputs[key]
            original = approved / rel
            geometry_path = original / "geometry.npz"
            try:
                if row.get("geometry_sha256") and digest(geometry_path) != row["geometry_sha256"]:
                    raise ValueError("Approved geometry does not match the dataset index hash.")
                geometry = load_arrays(geometry_path)
                rgb, method = colours_on_roi(geometry, *read_harmonized(source))
                if not check_only:
                    color_meta = dict(color_source="vertex_rgb", texture_path="",
                                      harmonized_source_ply=str(source), harmonized_source_sha256=digest(source))
                    geometry.update(harmonized_source_ply=np.array(str(source)),
                                    harmonized_source_sha256=np.array(color_meta["harmonized_source_sha256"]))
                    hashes = export_scan(original, staging / rel, geometry, rgb)
                    update = dict(color_meta, **hashes)
                    metadata = read_json(original / "metadata.json")
                    write_json(staging / rel / "metadata.json", dict(metadata, **update))
                    changes[row["scan_id"]] = update
                    rows.append(dict(row, **update))
                    audit.append(dict(scan_id=row["scan_id"], scanner=key[0], sample_name=key[1],
                                      sample_dir=row["sample_dir"], match_method=method,
                                      vertices=len(rgb), faces=len(geometry["faces"]),
                                      harmonized_source_ply=str(source)))
            except (ValueError, OSError, KeyError, ImportError, PlyParseError) as exc:
                raise ValueError(f"{key[0]}/{key[1]} ({source}): {exc}") from exc
            print(f"[{i}/{len(pairs)}] {key[0]} / {key[1]} · {method}", flush=True)
        if not check_only:
            write_manifests(approved, staging, output, rows, changes, audit)
            ready = True
            publish(staging, output)
    finally:
        if not ready or not staging.exists():
            shutil.rmtree(temp, ignore_errors=True)
    print(f"Geometry verified for {len(pairs)} scans." if check_only else f"Created {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmonized", type=Path, default=DATA / FULL_HARMONIZED)
    parser.add_argument("--approved", type=Path, default=DATA / APPROVED_REORIENTED)
    parser.add_argument("--output", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check-only", action="store_true", help="Validate scan and geometry correspondence.")
    action.add_argument("--publish-staged", type=Path, help="Retry publishing an already completed staged export.")
    args = parser.parse_args()
    output = args.output or args.approved.with_name(args.approved.name + "_harmonized")
    try:
        if args.publish_staged:
            publish_staged(args.publish_staged)
        else:
            export_dataset(args.harmonized, args.approved, output, args.check_only)
    except (ValueError, OSError, KeyError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
