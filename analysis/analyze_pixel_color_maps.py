"""BioCal3D Step 1-3 pixel-level colour analysis.

This script consumes the ROI outputs made by ``preprocessing.py``::

    DATA_ROOT/_processed_ROI/manifest.csv
    DATA_ROOT/_processed_ROI/<scanner>/G<group>/<sample_id>/roi.vtp
    DATA_ROOT/_processed_ROI/<scanner>/G<group>/<sample_id>/geometry.npz

It performs three registration-independent steps:

1. Project each circular ROI onto the saved canonical specimen plane and
   interpolate its vertex colour onto an equal-sized square grid.
2. Calculate one set of colour-distribution and spatial features per ROI.
   Pixels are never treated as independent biological observations.
3. Fit an exploratory CIELAB colour-cluster model separately for each scanner,
   using a balanced pixel sample from every ROI. Cluster 1 is the cluster with
   the largest a* centre (most red/magenta), but it is NOT automatically plaque.

Main outputs
------------
tables/pixel_color_features.csv
    One row per ROI, suitable for later scanner/stage robustness analysis.
tables/radial_color_features.csv
    Central, middle and outer-ring summaries for each ROI.
tables/color_cluster_centers.csv
    Scanner-specific Lab cluster centres and their redness ranks.
tables/pixel_color_qc.csv
    Missing-colour, interpolation-coverage and processing diagnostics.
maps/<scanner>/G<group>/<sample_id>/color_maps.npz
    Standardized RGB, Lab, chroma, height, mask and cluster-label rasters.
figures/roi_qc/
    A six-panel colour/map/cluster quality-control figure per accepted ROI.
models/
    Reusable fitted scanner-specific clustering models.

The maps are orientation-standardized in the canonical coordinates saved by
preprocessing. They are useful for spatial summaries and future image models,
but are not interpreted as pointwise longitudinal change maps.

Examples
--------
    python analyze_pixel_color_maps.py
    python analyze_pixel_color_maps.py --data-root "D:\\BioCal3D\\Data collected"
    python analyze_pixel_color_maps.py --clusters 5 --grid-size 192

Dependencies
------------
numpy, pandas, scipy, matplotlib, scikit-learn, Pillow, joblib and vtk.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError as exc:  # pragma: no cover - depends on user environment
    raise ImportError(
        "This script requires VTK. Use the same Python environment as the "
        "BioCal3D preprocessing pipeline, or install it with `pip install vtk`."
    ) from exc

try:
    import pyvista as pv
except ImportError:  # VTK fallback keeps the script usable without PyVista.
    pv = None


DEFAULT_DATA_ROOT = Path(
    r"C:\Users\johan\repos\biocal3d\data\02-09-2026-Data collected"
)
SCANNER_ORDER = ["LABscanner", "iTERO", "TRIOS3", "TRIOS5"]
RINGS = (("central", 0.00, 0.33), ("middle", 0.33, 0.66), ("outer", 0.66, 1.00))
MIN_POINTS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Default: DATA_ROOT/_pixel_color_analysis",
    )
    parser.add_argument(
        "--grid-size", type=int, default=256,
        help="Pixels on each side of every standardized square map (default: 256).",
    )
    parser.add_argument(
        "--clusters", type=int, default=4,
        help="Number of scanner-specific exploratory Lab clusters (default: 4).",
    )
    parser.add_argument(
        "--cluster-method", choices=("gmm", "kmeans"), default="gmm",
        help="Gaussian mixture model (default) or k-means.",
    )
    parser.add_argument(
        "--training-pixels-per-roi", type=int, default=1500,
        help="Balanced maximum pixel contribution from each ROI (default: 1500).",
    )
    parser.add_argument(
        "--min-coverage", type=float, default=0.75,
        help="Minimum fraction of disk pixels with interpolated colour (default: 0.75).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--skip-roi-figures", action="store_true",
        help="Create maps/tables but omit the per-ROI six-panel QC PNGs.",
    )
    return parser.parse_args()


def finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def safe_filename(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def parse_timepoint_number(value: object, sample_id: str) -> float:
    if value is not None and not pd.isna(value):
        match = re.search(r"(\d+)", str(value))
        if match:
            return float(match.group(1))
    match = re.search(r"T(\d+)", sample_id, flags=re.IGNORECASE)
    return float(match.group(1)) if match else np.nan


def biological_stage(group: object, timepoint: object) -> str:
    group_value = finite_float(group)
    time_value = finite_float(timepoint)
    if not np.isfinite(group_value) or not np.isfinite(time_value):
        return "Unknown"
    group_int, time_int = int(group_value), int(time_value)
    if time_int == 0:
        return "Baseline"
    if group_int in (1, 2) and time_int == 1:
        return "Clean"
    if group_int in (3, 4, 5) and time_int == 1:
        return "Partial"
    if group_int in (3, 4, 5) and time_int == 2:
        return "Clean"
    return f"T{time_int}"


def physical_sample_id(group: object, sample_number: object) -> str:
    group_value, sample_value = finite_float(group), finite_float(sample_number)
    if np.isfinite(group_value) and np.isfinite(sample_value):
        return f"BCG{int(group_value)}-{int(sample_value)}"
    return ""


def resolve_output_file(
    recorded_path: object, processed_root: Path, row: pd.Series, filename: str
) -> Path:
    if recorded_path is not None and not pd.isna(recorded_path):
        candidate = Path(str(recorded_path))
        if candidate.is_file():
            return candidate
    group = finite_float(row.get("group"))
    group_folder = f"G{int(group)}" if np.isfinite(group) else "G_UNKNOWN"
    return (
        processed_root / str(row.get("scanner", "")) / group_folder
        / str(row.get("sample_id", "")) / filename
    )


def recorded_file(value: object) -> Path | None:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    candidate = Path(str(value))
    return candidate if candidate.is_file() else None


def path_stored_in_geometry(geometry_path: Path, key: str) -> Path | None:
    if not geometry_path.is_file():
        return None
    try:
        with np.load(geometry_path, allow_pickle=False) as data:
            if key not in data:
                return None
            value = str(np.asarray(data[key]).item()).strip()
    except (OSError, TypeError, ValueError):
        return None
    candidate = Path(value)
    return candidate if value and candidate.is_file() else None


def normalized_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def build_source_ply_index(data_root: Path, scanners: list[str]) -> dict[tuple[str, str], Path]:
    """Rediscover raw PLYs if absolute paths stored during preprocessing moved."""
    scanner_tokens = {normalized_token(scanner): scanner for scanner in scanners}
    sample_pattern = re.compile(r"BCG\d+T\d+-\d+", flags=re.IGNORECASE)
    excluded = {"_processed_roi", "_pixel_color_analysis", "_geometry_analysis"}
    candidates: dict[tuple[str, str], list[Path]] = {}
    for path in data_root.rglob("*.ply"):
        try:
            relative = path.relative_to(data_root)
        except ValueError:
            continue
        if any(part.casefold() in excluded for part in relative.parts):
            continue
        match = sample_pattern.search(" ".join((path.stem, *relative.parts)))
        if match is None:
            continue
        parts = [normalized_token(part) for part in relative.parts]
        scanner = next(
            (name for token, name in scanner_tokens.items()
             if any(token == part or token in part for part in parts)),
            None,
        )
        if scanner is None:
            continue
        key = (scanner.casefold(), match.group(0).upper())
        candidates.setdefault(key, []).append(path)
    result = {}
    for key, paths in candidates.items():
        sample = key[1].casefold()
        paths.sort(key=lambda p: (p.stem.casefold() != sample, len(p.parts), str(p)))
        result[key] = paths[0]
    return result


def find_texture_for_source(source_ply: Path | None, sample_id: str) -> Path | None:
    if source_ply is None or not source_ply.is_file():
        return None
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in source_ply.parent.iterdir()
                    if p.is_file() and p.suffix.casefold() in extensions)
    if not images:
        return None
    stem = source_ply.stem.casefold()
    preferred = {stem, stem + "_texture", stem + "-texture", stem + "texture"}
    for image in images:
        if image.stem.casefold() in preferred:
            return image
    matches = [p for p in images if sample_id.casefold() in p.stem.casefold()]
    return matches[0] if len(matches) == 1 else (images[0] if len(images) == 1 else None)


def extract_point_colour_and_uv(polydata: vtk.vtkPolyData) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    point_data = polydata.GetPointData()
    n_points = polydata.GetNumberOfPoints()
    rgb = None
    description = "Colour unavailable"
    for name in ("RGB", "rgb", "RGBA", "rgba", "Colors", "colors", "Color", "color"):
        vtk_array = point_data.GetArray(name)
        if vtk_array is None:
            continue
        candidate = np.asarray(vtk_to_numpy(vtk_array))
        if candidate.ndim == 2 and candidate.shape[0] == n_points and candidate.shape[1] in (3, 4):
            rgb = candidate[:, :3].astype(float)
            if np.any(np.isfinite(rgb)) and np.nanmax(rgb) > 1.0:
                rgb /= 255.0
            rgb = np.clip(rgb, 0.0, 1.0)
            description = f"Embedded {name}"
            break
    if rgb is None:
        for names in (("red", "green", "blue"), ("Red", "Green", "Blue"),
                      ("RED", "GREEN", "BLUE"),
                      ("diffuse_red", "diffuse_green", "diffuse_blue")):
            arrays = [point_data.GetArray(name) for name in names]
            if any(array is None for array in arrays):
                continue
            channels = [np.asarray(vtk_to_numpy(array)).reshape(-1) for array in arrays]
            if all(len(channel) == n_points for channel in channels):
                rgb = np.column_stack(channels).astype(float)
                if np.any(np.isfinite(rgb)) and np.nanmax(rgb) > 1.0:
                    rgb /= 255.0
                rgb = np.clip(rgb, 0.0, 1.0)
                description = "Embedded separate RGB channels"
                break
    uv = None
    active = point_data.GetTCoords()
    if active is not None:
        candidate = np.asarray(vtk_to_numpy(active))
        if candidate.shape == (n_points, 2):
            uv = candidate.astype(float)
    if uv is None:
        for name in ("TCoords", "tcoords", "Texture Coordinates", "TextureCoordinates",
                     "texture_coordinates", "UV", "UVs", "uv", "uvs", "TexCoord", "TexCoords"):
            array = point_data.GetArray(name)
            if array is None:
                continue
            candidate = np.asarray(vtk_to_numpy(array))
            if candidate.shape == (n_points, 2):
                uv = candidate.astype(float)
                break
    return rgb, uv, description


def sample_texture(uv: np.ndarray | None, texture_path: Path | None) -> np.ndarray | None:
    """Bilinearly sample an sRGB texture at VTK UV coordinates."""
    if uv is None or texture_path is None or not texture_path.is_file():
        return None
    with Image.open(texture_path) as opened:
        image = np.asarray(opened.convert("RGB"), dtype=float) / 255.0
    x = np.clip(uv[:, 0], 0.0, 1.0) * (image.shape[1] - 1)
    y = (1.0 - np.clip(uv[:, 1], 0.0, 1.0)) * (image.shape[0] - 1)
    return np.column_stack([
        map_coordinates(image[:, :, channel], [y, x], order=1, mode="nearest")
        for channel in range(3)
    ])


def read_vtp(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, str]:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise ValueError(f"No points found in {path}")
    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(float)
    rgb, uv, description = extract_point_colour_and_uv(polydata)
    return points, rgb, uv, description


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, str]:
    if pv is not None:
        mesh = pv.read(path).extract_surface().triangulate().clean()
        points = np.asarray(mesh.points, dtype=float)
        rgb, uv, description = extract_point_colour_and_uv(mesh)
        return points, rgb, uv, description
    reader = vtk.vtkPLYReader()
    reader.SetFileName(str(path))
    reader.Update()
    mesh = reader.GetOutput()
    if mesh is None or mesh.GetNumberOfPoints() == 0:
        raise ValueError(f"No points found in {path}")
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
    rgb, uv, description = extract_point_colour_and_uv(mesh)
    return points, rgb, uv, description


def recover_roi_colours_from_source(
    roi_points: np.ndarray, source_ply: Path | None,
    texture_path: Path | None, geometry_path: Path,
) -> tuple[np.ndarray | None, str]:
    if source_ply is None or not source_ply.is_file() or not geometry_path.is_file():
        return None, "Raw source/geometry unavailable"
    try:
        source_points, source_rgb, source_uv, description = read_ply(source_ply)
        if source_rgb is None:
            source_rgb = sample_texture(source_uv, texture_path)
            if source_rgb is not None:
                description = f"Source PLY + texture: {texture_path.name}"
        if source_rgb is None:
            return None, "Raw source has no usable RGB or UV texture"
        with np.load(geometry_path, allow_pickle=False) as data:
            if "original_to_canonical" not in data:
                return None, "original_to_canonical transform unavailable"
            transform = np.asarray(data["original_to_canonical"], dtype=float)
        if transform.shape != (4, 4):
            return None, "original_to_canonical transform invalid"
        homogeneous = np.column_stack([source_points, np.ones(len(source_points))])
        canonical = (transform @ homogeneous.T).T[:, :3]
        distances, nearest = cKDTree(canonical).query(roi_points, k=1)
        spacing = cKDTree(roi_points).query(roi_points, k=2)[0][:, 1]
        tolerance = max(0.001, 0.20 * float(np.nanmedian(spacing)))
        matched = np.isfinite(distances) & (distances <= tolerance)
        if np.mean(matched) < 0.95:
            return None, f"Only {np.mean(matched):.1%} of ROI vertices matched raw colour"
        mapped = np.asarray(source_rgb[nearest], dtype=float)
        mapped[~matched] = np.nan
        return mapped, description
    except Exception as exc:
        return None, f"Raw-colour recovery failed: {type(exc).__name__}: {exc}"


def load_geometry_metadata(path: Path, points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    radius = np.nan
    saved_center = None
    saved_normal = None
    with np.load(path, allow_pickle=False) as data:
        if "roi_radius" in data:
            radius = finite_float(data["roi_radius"])
        for key in ("roi_center", "circle_center", "center"):
            if key in data and np.asarray(data[key]).size == 3:
                saved_center = np.asarray(data[key], dtype=float).reshape(3)
                break
        for key in ("roi_normal", "circle_normal", "normal"):
            if key in data and np.asarray(data[key]).size == 3:
                saved_normal = np.asarray(data[key], dtype=float).reshape(3)
                break
    finite_points = points[np.isfinite(points).all(axis=1)]
    if saved_center is None or not np.isfinite(saved_center).all():
        saved_center = np.mean(finite_points, axis=0)
    if saved_normal is None or not np.isfinite(saved_normal).all() or np.linalg.norm(saved_normal) == 0:
        _, _, vh = np.linalg.svd(finite_points - np.mean(finite_points, axis=0), full_matrices=False)
        saved_normal = vh[-1]
    saved_normal = saved_normal / np.linalg.norm(saved_normal)
    if saved_normal[2] < 0:
        saved_normal = -saved_normal
    if not np.isfinite(radius) or radius <= 0:
        distances = np.linalg.norm(finite_points - saved_center, axis=1)
        radius = float(np.nanpercentile(distances, 99))
    return float(radius), saved_center, saved_normal


def plane_coordinates(points: np.ndarray, center: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic in-plane axes in preprocessing's canonical frame."""
    reference = np.array([1.0, 0.0, 0.0])
    u = reference - np.dot(reference, normal) * normal
    if np.linalg.norm(u) < 0.2:
        reference = np.array([0.0, 1.0, 0.0])
        u = reference - np.dot(reference, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    centred = points - center
    xy = np.column_stack((centred @ u, centred @ v))
    height = centred @ normal
    return xy, height


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    result = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055)
    return np.clip(result, 0.0, 1.0)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB D65 [0,1] to CIELAB without requiring scikit-image."""
    shape = rgb.shape
    linear = srgb_to_linear(np.asarray(rgb, dtype=float).reshape(-1, 3))
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ matrix.T
    xyz /= np.array([0.95047, 1.00000, 1.08883])
    delta = 6 / 29
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)
    lab = np.column_stack((116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])))
    return lab.reshape(shape)


def create_standardized_map(
    points: np.ndarray, rgb: np.ndarray, radius: float,
    center: np.ndarray, normal: np.ndarray, grid_size: int,
) -> dict[str, np.ndarray | float]:
    finite = np.isfinite(points).all(axis=1) & np.isfinite(rgb).all(axis=1)
    if np.sum(finite) < MIN_POINTS:
        raise ValueError(f"Only {np.sum(finite)} finite coloured vertices")
    xy, height = plane_coordinates(points[finite], center, normal)
    rgb = np.clip(rgb[finite], 0.0, 1.0)
    axis = np.linspace(-radius, radius, grid_size)
    xx, yy = np.meshgrid(axis, axis)
    disk = xx ** 2 + yy ** 2 <= radius ** 2
    try:
        linear_rgb = griddata(xy, srgb_to_linear(rgb), (xx, yy), method="linear")
        height_grid = griddata(xy, height, (xx, yy), method="linear")
    except Exception as exc:
        raise ValueError(f"2D colour interpolation failed: {exc}") from exc
    valid = disk & np.isfinite(linear_rgb).all(axis=2)
    rgb_grid = linear_to_srgb(linear_rgb)
    rgb_grid[~valid] = np.nan
    lab = rgb_to_lab(rgb_grid)
    lab[~valid] = np.nan
    chroma = np.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2)
    hue = np.arctan2(lab[:, :, 2], lab[:, :, 1])
    height_grid[~(disk & np.isfinite(height_grid))] = np.nan
    return {
        "x_mm": xx, "y_mm": yy, "disk_mask": disk, "valid_mask": valid,
        "rgb": rgb_grid, "lab": lab, "chroma": chroma, "hue_rad": hue,
        "height_mm": height_grid,
        "coverage": float(np.sum(valid) / max(np.sum(disk), 1)),
        "pixel_spacing_mm": float(2 * radius / max(grid_size - 1, 1)),
    }


def robust_mad(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def histogram_entropy(values: np.ndarray, value_range: tuple[float, float], bins: int = 32) -> float:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    counts, _ = np.histogram(values, bins=bins, range=value_range)
    probabilities = counts[counts > 0] / np.sum(counts)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def describe_channel(values: np.ndarray, prefix: str, value_range: tuple[float, float]) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {f"{prefix}_{name}": np.nan for name in
                ("mean", "median", "std", "mad", "iqr", "p05", "p10", "p90", "p95", "entropy")}
    p05, p10, p25, p75, p90, p95 = np.percentile(values, [5, 10, 25, 75, 90, 95])
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        f"{prefix}_mad": robust_mad(values),
        f"{prefix}_iqr": float(p75 - p25),
        f"{prefix}_p05": float(p05), f"{prefix}_p10": float(p10),
        f"{prefix}_p90": float(p90), f"{prefix}_p95": float(p95),
        f"{prefix}_entropy": histogram_entropy(values, value_range),
    }


def local_standard_deviation(values: np.ndarray, valid: np.ndarray, sigma_pixels: float) -> np.ndarray:
    sigma_pixels = max(float(sigma_pixels), 0.5)
    weights = gaussian_filter(valid.astype(float), sigma_pixels, mode="constant")
    filled = np.where(valid, values, 0.0)
    mean = gaussian_filter(filled, sigma_pixels, mode="constant") / np.maximum(weights, 1e-12)
    second = gaussian_filter(filled ** 2, sigma_pixels, mode="constant") / np.maximum(weights, 1e-12)
    result = np.sqrt(np.maximum(second - mean ** 2, 0.0))
    result[(~valid) | (weights < 0.5)] = np.nan
    return result


def extract_global_features(data: dict[str, np.ndarray | float]) -> dict[str, float]:
    valid = np.asarray(data["valid_mask"], dtype=bool)
    rgb, lab = np.asarray(data["rgb"]), np.asarray(data["lab"])
    chroma, hue = np.asarray(data["chroma"]), np.asarray(data["hue_rad"])
    features: dict[str, float] = {
        "map_coverage": float(data["coverage"]),
        "map_valid_pixels": int(np.sum(valid)),
        "map_pixel_spacing_mm": float(data["pixel_spacing_mm"]),
    }
    channels = (
        (rgb[:, :, 0], "rgb_r", (0, 1)), (rgb[:, :, 1], "rgb_g", (0, 1)),
        (rgb[:, :, 2], "rgb_b", (0, 1)), (lab[:, :, 0], "lab_L", (0, 100)),
        (lab[:, :, 1], "lab_a", (-128, 127)), (lab[:, :, 2], "lab_b", (-128, 127)),
        (chroma, "lab_chroma", (0, 181)),
    )
    for values, prefix, bounds in channels:
        features.update(describe_channel(values[valid], prefix, bounds))
    valid_hue = hue[valid & np.isfinite(hue)]
    valid_chroma = chroma[valid & np.isfinite(hue)]
    if len(valid_hue):
        weights = np.maximum(valid_chroma, 1e-6)
        vector = np.sum(weights * np.exp(1j * valid_hue)) / np.sum(weights)
        features["lab_hue_circular_mean_deg"] = float(np.degrees(np.angle(vector)) % 360)
        features["lab_hue_resultant_length"] = float(np.abs(vector))
    else:
        features["lab_hue_circular_mean_deg"] = np.nan
        features["lab_hue_resultant_length"] = np.nan
    a_values, b_values = lab[:, :, 1][valid], lab[:, :, 2][valid]
    counts, _, _ = np.histogram2d(a_values, b_values, bins=24, range=[[-128, 127], [-128, 127]])
    probabilities = counts[counts > 0] / max(np.sum(counts), 1)
    features["lab_ab_joint_entropy"] = float(-np.sum(probabilities * np.log2(probabilities)))
    features["fraction_a_positive"] = float(np.mean(a_values > 0))
    features["fraction_a_gt_20"] = float(np.mean(a_values > 20))
    features["fraction_chroma_gt_30"] = float(np.mean(chroma[valid] > 30))
    features["fraction_red_high_chroma"] = float(np.mean((a_values > 15) & (chroma[valid] > 20)))
    spacing = float(data["pixel_spacing_mm"])
    for scale_mm in (0.5, 1.0):
        sigma = scale_mm / max(spacing, 1e-9)
        for values, name in ((lab[:, :, 1], "lab_a"), (chroma, "lab_chroma")):
            local_sd = local_standard_deviation(values, valid, sigma)
            features[f"{name}_local_sd_{scale_mm:g}mm_mean"] = float(np.nanmean(local_sd))
            features[f"{name}_local_sd_{scale_mm:g}mm_p90"] = float(np.nanpercentile(local_sd, 90))
    return features


def extract_radial_features(
    data: dict[str, np.ndarray | float], metadata: dict[str, object]
) -> list[dict[str, object]]:
    x, y = np.asarray(data["x_mm"]), np.asarray(data["y_mm"])
    valid = np.asarray(data["valid_mask"], dtype=bool)
    radius = float(max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y))))
    normalized_radius = np.sqrt(x ** 2 + y ** 2) / radius
    lab, chroma = np.asarray(data["lab"]), np.asarray(data["chroma"])
    rows = []
    for ring, lower, upper in RINGS:
        mask = valid & (normalized_radius >= lower) & (normalized_radius < upper + 1e-12)
        row: dict[str, object] = {**metadata, "ring": ring, "radius_min": lower, "radius_max": upper,
                                  "valid_pixels": int(np.sum(mask))}
        for channel, name in ((lab[:, :, 0], "lab_L"), (lab[:, :, 1], "lab_a"),
                              (lab[:, :, 2], "lab_b"), (chroma, "lab_chroma")):
            values = channel[mask]
            row[f"{name}_mean"] = float(np.nanmean(values)) if len(values) else np.nan
            row[f"{name}_median"] = float(np.nanmedian(values)) if len(values) else np.nan
            row[f"{name}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    return rows


def save_map(path: Path, data: dict[str, np.ndarray | float], cluster_labels: np.ndarray | None = None,
             cluster_confidence: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in data.items()}
    if cluster_labels is not None:
        payload["cluster_label"] = cluster_labels.astype(np.int16)
    if cluster_confidence is not None:
        payload["cluster_confidence"] = cluster_confidence.astype(np.float32)
    np.savez_compressed(path, **payload)


def load_map(path: Path) -> dict[str, np.ndarray | float]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def fit_scanner_model(samples: np.ndarray, method: str, clusters: int, random_state: int):
    if len(samples) < max(50, clusters * 10):
        raise ValueError(f"Only {len(samples)} training pixels")
    if method == "gmm":
        model = GaussianMixture(
            n_components=clusters, covariance_type="diag", n_init=5,
            reg_covar=1e-4, random_state=random_state,
        ).fit(samples)
        centers = model.means_
        weights = model.weights_
    else:
        model = KMeans(n_clusters=clusters, n_init=20, random_state=random_state).fit(samples)
        centers = model.cluster_centers_
        raw_labels = model.labels_
        weights = np.bincount(raw_labels, minlength=clusters) / len(raw_labels)
    order = np.argsort(-centers[:, 1])  # descending CIELAB a*: red/magenta first
    old_to_rank = np.empty(clusters, dtype=int)
    old_to_rank[order] = np.arange(1, clusters + 1)
    return model, centers, weights, old_to_rank


def apply_scanner_model(data: dict[str, np.ndarray | float], model, old_to_rank: np.ndarray,
                        method: str) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(data["valid_mask"], dtype=bool)
    lab = np.asarray(data["lab"])
    labels = np.zeros(valid.shape, dtype=np.int16)
    confidence = np.full(valid.shape, np.nan, dtype=float)
    raw = model.predict(lab[valid])
    labels[valid] = old_to_rank[raw]
    if method == "gmm":
        confidence[valid] = np.max(model.predict_proba(lab[valid]), axis=1)
    return labels, confidence


def cluster_features(labels: np.ndarray, confidence: np.ndarray, valid: np.ndarray,
                     clusters: int) -> dict[str, float]:
    features = {
        f"color_cluster_{rank}_fraction": float(np.mean(labels[valid] == rank))
        for rank in range(1, clusters + 1)
    }
    features["color_cluster_mean_confidence"] = (
        float(np.nanmean(confidence[valid])) if np.any(np.isfinite(confidence[valid])) else np.nan
    )
    return features


def add_radial_clusters(rows: list[dict[str, object]], data: dict[str, np.ndarray | float],
                        labels: np.ndarray, clusters: int) -> None:
    x, y = np.asarray(data["x_mm"]), np.asarray(data["y_mm"])
    valid = np.asarray(data["valid_mask"], dtype=bool)
    radius = float(max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y))))
    normalized = np.sqrt(x ** 2 + y ** 2) / radius
    by_ring = {str(row["ring"]): row for row in rows}
    for name, lower, upper in RINGS:
        mask = valid & (normalized >= lower) & (normalized < upper + 1e-12)
        for rank in range(1, clusters + 1):
            by_ring[name][f"color_cluster_{rank}_fraction"] = (
                float(np.mean(labels[mask] == rank)) if np.any(mask) else np.nan
            )


def save_roi_figure(path: Path, data: dict[str, np.ndarray | float], labels: np.ndarray,
                    scanner: str, sample_id: str, coverage: float, clusters: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb, lab = np.asarray(data["rgb"]), np.asarray(data["lab"])
    chroma = np.asarray(data["chroma"])
    extent = [float(np.nanmin(data["x_mm"])), float(np.nanmax(data["x_mm"])),
              float(np.nanmin(data["y_mm"])), float(np.nanmax(data["y_mm"]))]
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    panels = [
        (rgb, "Standardized RGB", None, None, None),
        (lab[:, :, 0], "CIELAB L*", "gray", 0, 100),
        (lab[:, :, 1], "CIELAB a*", "coolwarm", -60, 60),
        (lab[:, :, 2], "CIELAB b*", "coolwarm", -60, 60),
        (chroma, "Chroma C*", "magma", 0, None),
        (np.ma.masked_where(labels == 0, labels), "Scanner-specific colour cluster", ListedColormap(plt.cm.tab10.colors[:clusters]), 0.5, clusters + 0.5),
    ]
    for axis, (image, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        shown = axis.imshow(image, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("canonical x (mm)")
        axis.set_ylabel("canonical y (mm)")
        axis.set_aspect("equal")
        if np.asarray(image).ndim == 2:
            figure.colorbar(shown, ax=axis, shrink=0.72)
    figure.suptitle(f"{scanner} | {sample_id} | map coverage {coverage:.1%}\nCluster 1 = highest scanner-specific a* centre; exploratory, not a plaque label")
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def save_cluster_overview(path: Path, centers: pd.DataFrame) -> None:
    if centers.empty:
        return
    scanners = list(dict.fromkeys(centers["scanner"].astype(str)))
    figure, axes = plt.subplots(1, len(scanners), figsize=(4.2 * len(scanners), 4), squeeze=False, constrained_layout=True)
    for axis, scanner in zip(axes.ravel(), scanners):
        subset = centers[centers["scanner"] == scanner]
        sizes = 400 * np.maximum(subset["training_weight"].to_numpy(float), 0.03)
        scatter = axis.scatter(subset["center_a"], subset["center_b"], c=subset["center_L"],
                               s=sizes, cmap="viridis", vmin=0, vmax=100, edgecolor="black")
        for _, row in subset.iterrows():
            axis.annotate(f"C{int(row['redness_rank'])}", (row["center_a"], row["center_b"]),
                          xytext=(4, 4), textcoords="offset points")
        axis.axvline(0, color="0.8", linewidth=0.8)
        axis.axhline(0, color="0.8", linewidth=0.8)
        axis.set_title(scanner)
        axis.set_xlabel("a* (red +)")
        axis.set_ylabel("b* (yellow +)")
        figure.colorbar(scatter, ax=axis, label="L*")
    figure.suptitle("Scanner-specific exploratory Lab cluster centres\nPoint size = training fraction; C1 has highest a*")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_readme(output_root: Path, args: argparse.Namespace) -> None:
    text = f"""BioCal3D pixel-level colour analysis (Steps 1-3)

SETTINGS
grid size: {args.grid_size} x {args.grid_size}
cluster method: {args.cluster_method}
clusters per scanner: {args.clusters}
maximum training pixels contributed by each ROI: {args.training_pixels_per_roi}
minimum accepted colour-map coverage: {args.min_coverage:.0%}
random state: {args.random_state}

INTERPRETATION
- Each ROI is the biological/statistical unit. Raster pixels are not replicates.
- RGB is interpolated in linear-light space, then converted to CIELAB (D65).
- Models are fitted separately by scanner because raw scanner colour is not yet calibrated.
- Cluster 1 is the scanner-specific cluster with the highest a* centre. It can be
  described as the reddest/magenta-most colour phenotype, but not as plaque unless
  validated against labelled regions or a physical colour calibration target.
- Map orientation follows the canonical coordinates saved by preprocessing.
- The standardized maps support spatial summaries; they do not establish reliable
  pointwise before/after correspondence between separately scanned surfaces.

NEXT USE
Use pixel_color_features.csv as one-row-per-ROI input to the later mixed-model /
robustness analysis. Retain scanner as a factor and physical_sample_id as the
repeated-sample identifier. Re-run this complete pipeline after the fifth scanner
and remaining groups have been collected; do not append clusters from a separately
fitted run if you need directly comparable cluster fractions.
"""
    (output_root / "README_results.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.grid_size < 32:
        raise ValueError("--grid-size must be at least 32")
    if args.clusters < 2:
        raise ValueError("--clusters must be at least 2")
    if not 0 < args.min_coverage <= 1:
        raise ValueError("--min-coverage must lie in (0, 1]")

    data_root = args.data_root
    processed_root = data_root / "_processed_ROI"
    manifest_path = args.manifest or processed_root / "manifest.csv"
    output_root = args.output or data_root / "_pixel_color_analysis"
    table_root, map_root = output_root / "tables", output_root / "maps"
    figure_root, model_root = output_root / "figures", output_root / "models"
    for directory in (table_root, map_root, figure_root, model_root):
        directory.mkdir(parents=True, exist_ok=True)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = ["scanner", "group", "timepoint", "sample_number", "sample_id"]
    missing = [column for column in required if column not in manifest]
    if missing:
        raise ValueError("Manifest is missing columns: " + ", ".join(missing))

    scanners = list(dict.fromkeys([*SCANNER_ORDER, *manifest["scanner"].dropna().astype(str)]))
    source_index = build_source_ply_index(data_root, scanners)
    rng = np.random.default_rng(args.random_state)
    training: dict[str, list[np.ndarray]] = {}
    records: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    radial_by_key: dict[str, list[dict[str, object]]] = {}

    print("=" * 76)
    print("BIOCAL3D PIXEL-LEVEL COLOUR ANALYSIS: STEPS 1-3")
    print("=" * 76)
    print(f"Manifest: {manifest_path}")
    print(f"Rows: {len(manifest)} | standardized grid: {args.grid_size} x {args.grid_size}")

    for index, (_, row) in enumerate(manifest.iterrows(), 1):
        scanner, sample_id = str(row.get("scanner", "")), str(row.get("sample_id", ""))
        group = finite_float(row.get("group"))
        group_folder = f"G{int(group)}" if np.isfinite(group) else "G_UNKNOWN"
        timepoint_number = parse_timepoint_number(row.get("timepoint"), sample_id)
        roi_path = resolve_output_file(row.get("roi_vtp"), processed_root, row, "roi.vtp")
        geometry_path = resolve_output_file(row.get("geometry_npz"), processed_root, row, "geometry.npz")
        map_path = map_root / safe_filename(scanner) / group_folder / safe_filename(sample_id) / "color_maps.npz"
        metadata: dict[str, object] = {
            "scanner": scanner, "group": group, "timepoint": row.get("timepoint"),
            "timepoint_number": timepoint_number,
            "stage": biological_stage(group, timepoint_number),
            "sample_number": finite_float(row.get("sample_number")), "sample_id": sample_id,
            "physical_sample_id": physical_sample_id(group, row.get("sample_number")),
            "roi_vtp": str(roi_path), "geometry_npz": str(geometry_path),
            "map_npz": str(map_path),
        }
        qc: dict[str, object] = {**metadata, "analysis_ok": False, "qc_reasons": "", "analysis_error": ""}
        print(f"[{index:03d}/{len(manifest):03d}] {scanner} | {sample_id}")
        reasons = []
        if str(row.get("status", "")).upper() == "FAILED":
            reasons.append("preprocessing_failed")
        if not roi_path.is_file():
            reasons.append("roi_file_missing")
        if not geometry_path.is_file():
            reasons.append("geometry_file_missing")
        if reasons:
            qc["qc_reasons"] = ";".join(reasons)
            qc_rows.append(qc)
            continue
        try:
            points, rgb, uv, colour_source = read_vtp(roi_path)
            source_ply = recorded_file(row.get("source_ply")) or path_stored_in_geometry(geometry_path, "source_ply")
            if source_ply is None:
                source_ply = source_index.get((scanner.casefold(), sample_id.upper()))
            texture = recorded_file(row.get("source_texture")) or path_stored_in_geometry(geometry_path, "source_texture")
            if texture is None:
                texture = find_texture_for_source(source_ply, sample_id)
            if rgb is None:
                rgb = sample_texture(uv, texture)
                if rgb is not None:
                    colour_source = f"ROI UV + texture: {texture.name}"
            if rgb is None:
                rgb, colour_source = recover_roi_colours_from_source(points, source_ply, texture, geometry_path)
            if rgb is None:
                raise ValueError(colour_source)
            radius, center, normal = load_geometry_metadata(geometry_path, points)
            data = create_standardized_map(points, rgb, radius, center, normal, args.grid_size)
            features = extract_global_features(data)
            if float(data["coverage"]) < args.min_coverage:
                reasons.append("low_map_coverage")
            if str(row.get("status", "")).upper() == "CHECK":
                reasons.append("preprocessing_status_check")
            analysis_ok = len(reasons) == 0
            qc.update({
                "analysis_ok": analysis_ok, "qc_reasons": ";".join(reasons),
                "colour_source": colour_source, "input_vertices": len(points),
                "coloured_vertices": int(np.sum(np.isfinite(rgb).all(axis=1))),
                "saved_roi_radius_mm": radius, "map_coverage": float(data["coverage"]),
                "source_ply": str(source_ply) if source_ply else "",
                "source_texture": str(texture) if texture else "",
            })
            save_map(map_path, data)
            record = {**metadata, "colour_source": colour_source, "saved_roi_radius_mm": radius,
                      **features, "analysis_ok": analysis_ok}
            records.append(record)
            key = str(map_path)
            radial_by_key[key] = extract_radial_features(data, metadata)
            if analysis_ok:
                valid_lab = np.asarray(data["lab"])[np.asarray(data["valid_mask"], dtype=bool)]
                take = min(args.training_pixels_per_roi, len(valid_lab))
                chosen = rng.choice(len(valid_lab), size=take, replace=False)
                training.setdefault(scanner, []).append(valid_lab[chosen])
        except Exception as exc:
            qc["analysis_error"] = f"{type(exc).__name__}: {exc}"
            qc["qc_reasons"] = "analysis_exception"
            print(f"  FAILED: {exc}")
        qc_rows.append(qc)

    if not records:
        pd.DataFrame(qc_rows).to_csv(table_root / "pixel_color_qc.csv", index=False)
        raise RuntimeError("No ROI colour map could be created. Inspect pixel_color_qc.csv.")

    center_rows: list[dict[str, object]] = []
    fitted: dict[str, tuple[object, np.ndarray]] = {}
    for scanner, parts in training.items():
        samples = np.vstack(parts)
        try:
            model, centers, weights, old_to_rank = fit_scanner_model(
                samples, args.cluster_method, args.clusters, args.random_state
            )
        except Exception as exc:
            warnings.warn(f"No cluster model for {scanner}: {exc}")
            continue
        fitted[scanner] = (model, old_to_rank)
        joblib.dump(
            {"model": model, "old_label_to_redness_rank": old_to_rank,
             "method": args.cluster_method, "lab_standard": "CIELAB D65",
             "cluster_1_definition": "largest centre a*"},
            model_root / f"{safe_filename(scanner)}_{args.cluster_method}_lab_clusters.joblib",
        )
        for old_label in range(args.clusters):
            rank = int(old_to_rank[old_label])
            L, a, b = centers[old_label]
            center_rows.append({
                "scanner": scanner, "cluster_method": args.cluster_method,
                "old_model_label": old_label, "redness_rank": rank,
                "center_L": L, "center_a": a, "center_b": b,
                "center_chroma": math.hypot(a, b), "training_weight": weights[old_label],
                "training_pixels": len(samples), "training_rois": len(parts),
            })

    record_by_map = {str(record["map_npz"]): record for record in records}
    for map_key, record in record_by_map.items():
        scanner = str(record["scanner"])
        if scanner not in fitted:
            record["cluster_model_available"] = False
            continue
        data = load_map(Path(map_key))
        model, old_to_rank = fitted[scanner]
        labels, confidence = apply_scanner_model(data, model, old_to_rank, args.cluster_method)
        save_map(Path(map_key), data, labels, confidence)
        record.update(cluster_features(labels, confidence, np.asarray(data["valid_mask"], dtype=bool), args.clusters))
        record["cluster_model_available"] = True
        add_radial_clusters(radial_by_key[map_key], data, labels, args.clusters)
        if not args.skip_roi_figures:
            figure_path = (figure_root / "roi_qc" / safe_filename(scanner)
                           / f"{safe_filename(record['sample_id'])}_pixel_colour_qc.png")
            save_roi_figure(figure_path, data, labels, scanner, str(record["sample_id"]),
                            float(record["map_coverage"]), args.clusters)

    features_df = pd.DataFrame(records)
    qc_df = pd.DataFrame(qc_rows)
    radial_df = pd.DataFrame([row for rows in radial_by_key.values() for row in rows])
    centers_df = pd.DataFrame(center_rows)
    features_df.to_csv(table_root / "pixel_color_features.csv", index=False)
    qc_df.to_csv(table_root / "pixel_color_qc.csv", index=False)
    radial_df.to_csv(table_root / "radial_color_features.csv", index=False)
    centers_df.to_csv(table_root / "color_cluster_centers.csv", index=False)
    save_cluster_overview(figure_root / "scanner_color_cluster_centers.png", centers_df)
    write_readme(output_root, args)
    settings = vars(args).copy()
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in settings.items()}
    (output_root / "analysis_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    accepted = int(features_df["analysis_ok"].sum())
    print("\nFinished")
    print(f"Created maps for {len(features_df)} ROIs; {accepted} passed all QC thresholds.")
    print(f"Scanner-specific cluster models: {len(fitted)}")
    print(f"Results: {output_root}")


if __name__ == "__main__":
    main()
