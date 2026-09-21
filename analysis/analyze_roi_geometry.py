"""Registration-independent geometric analysis of extracted BioCal3D ROIs.

This script reads the outputs produced by ``preprocessing.py``:

    _processed_ROI/manifest.csv
    _processed_ROI/<scanner>/G<group>/<sample_id>/roi.vtp
    _processed_ROI/<scanner>/G<group>/<sample_id>/geometry.npz

It deliberately does NOT calculate pointwise before/after surface differences.
Every ROI is analysed independently.  Scanner and cleaning-stage comparisons
are subsequently performed on one set of summary features per scan.

The script analyses every triangle already present in each extracted
``roi.vtp``. It does not perform a second radial clipping step. The saved ROI
radius is used only as the nominal-area denominator, grid extent and radial-zone
definition. The jagged ROI boundary is never interpreted as specimen geometry.

Main outputs
------------
tables/roi_geometry_features.csv
    One row per successfully analysed ROI, including native-mesh and
    standardized-grid features.

tables/roi_geometry_qc.csv
    Registration, ROI-extraction and full-ROI quality information.

tables/roi_radial_geometry_features.csv
    Rotation-independent summaries for central, middle and outer rings.

tables/scanner_geometry_summary.csv
tables/scanner_pairwise_geometry_tests.csv
tables/scanner_friedman_geometry_tests.csv
    Descriptive and matched scanner comparisons.

tables/individual_stage_geometry_changes.csv
tables/stage_geometry_tests.csv
    Paired global feature changes between biological stages.  These are
    stage-associated changes, not localized surface loss or plaque volume.

figures/
    Geometry overview, QC, correlation, PCA and stage-change plots.

figures/processing_examples/
    Detailed nine-panel examples for one matched Group 3 specimen at T0, T1
    and T2 for every scanner. Each example places the actual ROI colour beside
    the surface-height maps. Use --save-all-diagnostics for every ROI.

figures/G3_examples/
    One compact T0/T1/T2 colour-versus-height comparison per scanner for the
    automatically selected matched Group 3 specimen.

GEOMETRY_FEATURE_GUIDE.md and tables/geometry_feature_dictionary.csv
    Plain-language calculation, interpretation, limitation and recommended use
    for every output feature family.

Usage
-----
Run with the default project path configured below:

    python analyze_roi_geometry.py

Or supply another data root:

    python analyze_roi_geometry.py --data-root "D:\\path\\to\\Data collected"

Dependencies
------------
numpy, pandas, scipy, matplotlib, scikit-learn and vtk.  VTK is already a
dependency of PyVista, which is used by the preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.stats import (
    friedmanchisquare,
    kurtosis,
    rankdata,
    skew,
    wilcoxon,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise ImportError(
        "This script requires VTK. Install it with `pip install vtk`, or use "
        "the same environment as preprocessing.py/PyVista."
    ) from exc

try:
    import pyvista as pv
except ImportError:  # VTK fallback below keeps geometry analysis usable.
    pv = None


# ============================================================================
# DEFAULT PATHS AND SETTINGS
# ============================================================================

DEFAULT_DATA_ROOT = Path(
    r"C:\Users\johan\repos\biocal3d\data\02-09-2026-Data collected"
)

SCANNER_ORDER = [
    "LABscanner",
    "iTERO",
    "TRIOS3",
    "TRIOS5",
]

SCANNER_FILE_CODES = {
    "LABscanner": "LAB",
    "iTERO": "iTERO",
    "TRIOS3": "T3",
    "TRIOS5": "T5",
}

# Standardized raster resolution used for density-independent roughness.
GRID_SPACING_MM = 0.20

# Spatial scales for high-pass roughness and approximate curvature.
ROUGHNESS_SCALES_MM = (0.50, 1.00)

# Conservative QC thresholds.  All computed rows are retained in the output;
# these thresholds only determine analysis_ok and which rows enter statistics.
MIN_ROI_POINTS = 50
MIN_ROI_TRIANGLES = 50
MIN_PROJECTED_AREA_COVERAGE = 0.75
MIN_GRID_COVERAGE = 0.75

RING_DEFINITIONS = [
    ("central", 0.00, 0.33),
    ("middle", 0.33, 0.66),
    ("outer_roi", 0.66, 1.00),
]

EXAMPLE_GROUP = 3
EXAMPLE_TIMEPOINTS = (0, 1, 2)


# Features selected for formal summaries and exploratory statistical tests.
# Vertices are never treated as independent statistical observations.
ANALYSIS_METRICS = [
    "roi_vertices_per_nominal_mm2",
    "roi_faces_per_nominal_mm2",
    "median_nearest_neighbor_mm",
    "median_unique_edge_length_mm",
    "median_triangle_area_mm2",
    "median_triangle_edge_ratio",
    "median_triangle_min_angle_deg",
    "plane_rmse_mm",
    "native_Sa_mm",
    "native_Sq_mm",
    "native_height_p95_minus_p05_mm",
    "surface_to_projected_area_ratio",
    "area_weighted_normal_angle_median_deg",
    "grid_Sa_mm",
    "grid_Sq_mm",
    "grid_height_p95_minus_p05_mm",
    "grid_rms_slope",
]

PLOT_METRICS = [
    "roi_vertices_per_nominal_mm2",
    "median_nearest_neighbor_mm",
    "median_unique_edge_length_mm",
    "median_triangle_area_mm2",
    "plane_rmse_mm",
    "native_Sq_mm",
    "grid_Sq_mm",
    "surface_to_projected_area_ratio",
    "area_weighted_normal_angle_median_deg",
]

FEATURE_LABELS = {
    "roi_vertices_per_nominal_mm2": "Vertices / nominal mm²",
    "roi_faces_per_nominal_mm2": "Faces / nominal mm²",
    "median_nearest_neighbor_mm": "Median nearest-neighbour distance (mm)",
    "median_unique_edge_length_mm": "Median unique edge length (mm)",
    "median_triangle_area_mm2": "Median triangle area (mm²)",
    "median_triangle_edge_ratio": "Median longest/shortest edge ratio",
    "median_triangle_min_angle_deg": "Median minimum triangle angle (°)",
    "plane_rmse_mm": "Robust plane RMSE (mm)",
    "native_Sa_mm": "Native area-weighted Sa (mm)",
    "native_Sq_mm": "Native area-weighted Sq (mm)",
    "native_height_p95_minus_p05_mm": "Native P95–P05 height (mm)",
    "surface_to_projected_area_ratio": "3D / projected surface area",
    "area_weighted_normal_angle_median_deg": "Median normal deviation (°)",
    "grid_Sa_mm": "Grid Sa (mm)",
    "grid_Sq_mm": "Grid Sq (mm)",
    "grid_height_p95_minus_p05_mm": "Grid P95–P05 height (mm)",
    "grid_rms_slope": "Grid RMS slope",
}


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root containing _processed_ROI (default: configured Windows path).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional explicit manifest.csv path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output directory (default: DATA_ROOT/_geometry_analysis).",
    )
    parser.add_argument(
        "--grid-spacing-mm",
        type=float,
        default=GRID_SPACING_MM,
        help=f"Grid spacing for standardized features (default: {GRID_SPACING_MM}).",
    )
    parser.add_argument(
        "--save-all-diagnostics",
        action="store_true",
        help=(
            "Save the nine-panel processing example for every accepted ROI. "
            "By default, matched Group 3 T0/T1/T2 examples are saved."
        ),
    )
    return parser.parse_args()


def finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def parse_timepoint_number(value: object, sample_id: str) -> float:
    if value is not None and not pd.isna(value):
        match = re.search(r"(\d+)", str(value))
        if match:
            return float(match.group(1))
    match = re.search(r"T(\d+)", str(sample_id), flags=re.IGNORECASE)
    return float(match.group(1)) if match else np.nan


def physical_sample_id(group: object, sample_number: object) -> str:
    group_value = finite_float(group)
    sample_value = finite_float(sample_number)
    if np.isfinite(group_value) and np.isfinite(sample_value):
        return f"BCG{int(group_value)}-{int(sample_value)}"
    return ""


def biological_stage(group: object, timepoint_number: object) -> str:
    group_value = finite_float(group)
    time_value = finite_float(timepoint_number)
    if not np.isfinite(group_value) or not np.isfinite(time_value):
        return "Unknown"
    group_int = int(group_value)
    time_int = int(time_value)
    if time_int == 0:
        return "Baseline"
    if group_int in (1, 2) and time_int == 1:
        return "Clean"
    if group_int in (3, 4, 5) and time_int == 1:
        return "Partial"
    if group_int in (3, 4, 5) and time_int == 2:
        return "Clean"
    return f"T{time_int}"


def choose_group3_example_sample(manifest: pd.DataFrame) -> int | None:
    """Choose the smallest G3 specimen with T0/T1/T2 for every scanner.

    If no fully complete specimen exists, use the specimen with the greatest
    scanner-by-stage coverage so the script can still produce useful examples.
    """
    candidates = manifest.copy()
    candidates["_group"] = candidates["group"].map(finite_float)
    candidates["_sample"] = candidates["sample_number"].map(finite_float)
    candidates["_time"] = [
        parse_timepoint_number(timepoint, sample_id)
        for timepoint, sample_id in zip(
            candidates["timepoint"], candidates["sample_id"]
        )
    ]
    candidates = candidates[
        (candidates["_group"] == EXAMPLE_GROUP)
        & candidates["_sample"].notna()
        & candidates["_time"].isin(EXAMPLE_TIMEPOINTS)
        & candidates["scanner"].isin(SCANNER_ORDER)
    ].copy()
    if "status" in candidates:
        candidates = candidates[
            candidates["status"].fillna("").astype(str).str.upper() != "FAILED"
        ]
    if candidates.empty:
        return None

    target_pairs = {
        (scanner, timepoint)
        for scanner in SCANNER_ORDER
        for timepoint in EXAMPLE_TIMEPOINTS
    }
    coverage: list[tuple[int, int, bool]] = []
    for sample_value, rows in candidates.groupby("_sample"):
        pairs = {
            (str(scanner), int(timepoint))
            for scanner, timepoint in zip(rows["scanner"], rows["_time"])
            if np.isfinite(timepoint)
        }
        coverage.append(
            (int(sample_value), len(pairs & target_pairs), target_pairs <= pairs)
        )

    complete = sorted(sample for sample, _, is_complete in coverage if is_complete)
    if complete:
        return complete[0]
    coverage.sort(key=lambda item: (-item[1], item[0]))
    return coverage[0][0]


def resolve_output_file(
    recorded_path: object,
    processed_root: Path,
    row: pd.Series,
    filename: str,
) -> Path:
    """Resolve both original Windows manifest paths and reconstructed paths."""
    if recorded_path is not None and not pd.isna(recorded_path):
        recorded = Path(str(recorded_path))
        if recorded.is_file():
            return recorded

    group_value = finite_float(row.get("group"))
    group_folder = f"G{int(group_value)}" if np.isfinite(group_value) else "G_UNKNOWN"
    reconstructed = (
        processed_root
        / str(row.get("scanner", ""))
        / group_folder
        / str(row.get("sample_id", ""))
        / filename
    )
    return reconstructed


def resolve_recorded_file(recorded_path: object) -> Path | None:
    """Return a usable file path recorded by preprocessing."""
    if recorded_path is None or pd.isna(recorded_path) or not str(recorded_path).strip():
        return None
    candidate = Path(str(recorded_path))
    return candidate if candidate.is_file() else None


def resolve_texture_file(recorded_path: object) -> Path | None:
    """Return a usable external texture path recorded by preprocessing."""
    return resolve_recorded_file(recorded_path)


def path_stored_in_geometry(geometry_path: Path, key: str) -> Path | None:
    """Recover an original data path stored inside preprocessing geometry.npz."""
    if not geometry_path.is_file():
        return None
    try:
        with np.load(geometry_path, allow_pickle=False) as data:
            if key not in data:
                return None
            value = str(np.asarray(data[key]).item()).strip()
    except (OSError, ValueError, TypeError):
        return None
    if not value:
        return None
    candidate = Path(value)
    return candidate if candidate.is_file() else None


def find_texture_for_source(
    source_ply: Path | None,
    sample_id: str,
) -> Path | None:
    """Apply the texture-name rules used by load_data_get_overview.py."""
    if source_ply is None or not source_ply.is_file():
        return None
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(
        path
        for path in source_ply.parent.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not images:
        return None

    ply_stem = source_ply.stem.lower()
    for image in images:
        if image.stem.lower() == ply_stem:
            return image

    expected = {
        ply_stem + "_texture",
        ply_stem + "-texture",
        ply_stem + "texture",
    }
    for image in images:
        if image.stem.lower() in expected:
            return image

    sample_matches = [
        image for image in images if sample_id.lower() in image.stem.lower()
    ]
    if len(sample_matches) == 1:
        return sample_matches[0]
    texture_matches = [
        image for image in sample_matches if "texture" in image.stem.lower()
    ]
    if len(texture_matches) == 1:
        return texture_matches[0]
    return images[0] if len(images) == 1 else None


def normalized_path_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def build_source_ply_index(data_root: Path) -> dict[tuple[str, str], Path]:
    """Rediscover raw PLYs when absolute paths saved earlier have moved.

    The key is ``(scanner, sample_id)``. Processed/output folders are excluded,
    and exact sample-ID filenames are preferred when duplicate candidates exist.
    """
    scanner_tokens = {
        normalized_path_token(scanner): scanner for scanner in SCANNER_ORDER
    }
    candidates: dict[tuple[str, str], list[Path]] = {}
    sample_pattern = re.compile(r"BCG\d+T\d+-\d+", flags=re.IGNORECASE)
    excluded = {"_processed_roi", "_geometry_analysis", "_loading_qc", "_loading_qc_v2"}

    for path in data_root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() != ".ply":
            continue
        try:
            relative = path.relative_to(data_root)
        except ValueError:
            continue
        if any(part.casefold() in excluded for part in relative.parts):
            continue

        searchable = " ".join([path.stem, *relative.parts])
        sample_match = sample_pattern.search(searchable)
        if sample_match is None:
            continue
        sample_id = sample_match.group(0).upper()

        normalized_parts = [normalized_path_token(part) for part in relative.parts]
        scanner = next(
            (
                scanner_name
                for token, scanner_name in scanner_tokens.items()
                if any(token == part or token in part for part in normalized_parts)
            ),
            None,
        )
        if scanner is None:
            continue
        key = (scanner.casefold(), sample_id)
        candidates.setdefault(key, []).append(path)

    index: dict[tuple[str, str], Path] = {}
    for key, paths in candidates.items():
        sample_id = key[1].casefold()
        paths.sort(
            key=lambda path: (
                path.stem.casefold() != sample_id,
                len(path.parts),
                str(path).casefold(),
            )
        )
        index[key] = paths[0]
    return index


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def weighted_quantile(
    values: np.ndarray,
    quantiles: float | list[float] | np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles_array = np.atleast_1d(quantiles).astype(float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.full(len(quantiles_array), np.nan)
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= np.sum(weights)
    return np.interp(quantiles_array, cumulative, values)


def rank_biserial_from_differences(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if len(differences) == 0:
        return np.nan
    ranks = rankdata(np.abs(differences))
    positive = float(np.sum(ranks[differences > 0]))
    negative = float(np.sum(ranks[differences < 0]))
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else np.nan


def holm_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Holm step-down multiplicity adjustment without extra dependencies."""
    p_values = np.asarray(p_values, dtype=float)
    if len(p_values) == 0:
        return np.array([], dtype=float)
    order = np.argsort(p_values)
    ordered = p_values[order]
    multipliers = len(ordered) - np.arange(len(ordered))
    adjusted_ordered = np.maximum.accumulate(ordered * multipliers)
    adjusted_ordered = np.clip(adjusted_ordered, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = adjusted_ordered
    return adjusted


# ============================================================================
# VTP LOADING
# ============================================================================

def extract_point_colour_and_uv(
    polydata: vtk.vtkPolyData,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Read embedded vertex RGB and UV coordinates without altering geometry."""
    point_data = polydata.GetPointData()
    n_points = polydata.GetNumberOfPoints()

    rgb: np.ndarray | None = None
    colour_source = "Colour unavailable"
    for name in ("RGB", "rgb", "RGBA", "rgba", "Colors", "colors", "Color", "color"):
        vtk_array = point_data.GetArray(name)
        if vtk_array is None:
            continue
        array = np.asarray(vtk_to_numpy(vtk_array))
        if array.ndim == 2 and array.shape[0] == n_points and array.shape[1] in (3, 4):
            rgb = array[:, :3].astype(float)
            finite_max = np.nanmax(rgb) if np.any(np.isfinite(rgb)) else np.nan
            if np.isfinite(finite_max) and finite_max > 1.0:
                rgb /= 255.0
            rgb = np.clip(rgb, 0.0, 1.0)
            colour_source = f"Embedded {name}"
            break

    if rgb is None:
        for names in (
            ("red", "green", "blue"),
            ("Red", "Green", "Blue"),
            ("RED", "GREEN", "BLUE"),
            ("diffuse_red", "diffuse_green", "diffuse_blue"),
        ):
            arrays = [point_data.GetArray(name) for name in names]
            if any(array is None for array in arrays):
                continue
            channels = [np.asarray(vtk_to_numpy(array)).reshape(-1) for array in arrays]
            if all(len(channel) == n_points for channel in channels):
                rgb = np.column_stack(channels).astype(float)
                finite_max = np.nanmax(rgb) if np.any(np.isfinite(rgb)) else np.nan
                if np.isfinite(finite_max) and finite_max > 1.0:
                    rgb /= 255.0
                rgb = np.clip(rgb, 0.0, 1.0)
                colour_source = "Embedded separate RGB channels"
                break

    uv: np.ndarray | None = None
    active_tcoords = point_data.GetTCoords()
    if active_tcoords is not None:
        candidate = np.asarray(vtk_to_numpy(active_tcoords))
        if candidate.ndim == 2 and candidate.shape == (n_points, 2):
            uv = candidate.astype(float)
    if uv is None:
        for name in (
            "TCoords",
            "tcoords",
            "Texture Coordinates",
            "TextureCoordinates",
            "texture_coordinates",
            "UV",
            "UVs",
            "uv",
            "uvs",
            "TexCoord",
            "TexCoords",
        ):
            vtk_array = point_data.GetArray(name)
            if vtk_array is None:
                continue
            candidate = np.asarray(vtk_to_numpy(vtk_array))
            if candidate.ndim == 2 and candidate.shape == (n_points, 2):
                uv = candidate.astype(float)
                break
    return rgb, uv, colour_source


def texture_vertex_colours(
    uv: np.ndarray | None,
    texture_path: Path | None,
) -> np.ndarray | None:
    """Sample an external texture at vertex UV coordinates for visualization."""
    if uv is None or texture_path is None or not texture_path.is_file():
        return None
    try:
        loaded_image = np.asarray(mpimg.imread(texture_path))
    except (OSError, ValueError):
        return None
    image = loaded_image
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    image = image[:, :, :3].astype(float)
    if np.issubdtype(loaded_image.dtype, np.integer):
        image /= np.iinfo(loaded_image.dtype).max
    elif np.nanmax(image) > 1.0:
        image /= 255.0
    image = np.clip(image, 0.0, 1.0)
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)
    x = np.rint(u * (image.shape[1] - 1)).astype(int)
    y = np.rint((1.0 - v) * (image.shape[0] - 1)).astype(int)
    return image[y, x]


def recover_roi_colours_from_source(
    roi_points: np.ndarray,
    source_ply: Path | None,
    texture_path: Path | None,
    geometry_path: Path,
) -> tuple[np.ndarray | None, str]:
    """Recover colour from the original textured PLY and map it onto the ROI.

    LABscanner stores its appearance as an external image plus UV coordinates.
    If those UV coordinates did not survive the VTP export, the original PLY is
    loaded exactly as in ``load_data_get_overview.py``. Its vertices are moved
    into the saved canonical coordinate system and matched to the existing ROI.
    Geometry is never replaced or re-extracted by this fallback.
    """
    if source_ply is None or not source_ply.is_file():
        return None, "Source PLY unavailable for colour recovery"
    if not geometry_path.is_file():
        return None, "geometry.npz unavailable for colour recovery"

    try:
        if pv is not None:
            # This is the same loading path used by load_data_get_overview.py.
            source_mesh = pv.read(source_ply)
            source_mesh = source_mesh.extract_surface().triangulate().clean()
            source_points = np.asarray(source_mesh.points, dtype=float)
        else:
            reader = vtk.vtkPLYReader()
            reader.SetFileName(str(source_ply))
            reader.Update()
            triangle_filter = vtk.vtkTriangleFilter()
            triangle_filter.SetInputData(reader.GetOutput())
            triangle_filter.Update()
            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputData(triangle_filter.GetOutput())
            cleaner.Update()
            source_mesh = cleaner.GetOutput()
            source_points = vtk_to_numpy(
                source_mesh.GetPoints().GetData()
            ).astype(float)
        source_rgb, source_uv, source_description = extract_point_colour_and_uv(
            source_mesh
        )
        if source_rgb is None:
            source_rgb = texture_vertex_colours(source_uv, texture_path)
            if source_rgb is not None and texture_path is not None:
                source_description = f"Source PLY + texture: {texture_path.name}"
        if source_rgb is None:
            return None, "Source PLY has no usable RGB/UV texture colour"

        with np.load(geometry_path, allow_pickle=False) as data:
            if "original_to_canonical" not in data:
                return None, "Saved original-to-canonical transform unavailable"
            transform = np.asarray(data["original_to_canonical"], dtype=float)
        if transform.shape != (4, 4):
            return None, "Saved original-to-canonical transform is invalid"

        homogeneous = np.column_stack([source_points, np.ones(len(source_points))])
        canonical_points = (transform @ homogeneous.T).T[:, :3]

        tree = cKDTree(canonical_points)
        distances, nearest = tree.query(roi_points, k=1)
        if len(roi_points) >= 2:
            roi_spacing = cKDTree(roi_points).query(roi_points, k=2)[0][:, 1]
            typical_spacing = float(np.nanmedian(roi_spacing))
        else:
            typical_spacing = 0.0
        tolerance = max(0.001, 0.15 * typical_spacing)
        matched = np.isfinite(distances) & (distances <= tolerance)
        if np.mean(matched) < 0.95:
            return (
                None,
                "Source-colour transfer rejected: fewer than 95% of ROI vertices matched",
            )

        mapped_rgb = np.asarray(source_rgb[nearest], dtype=float)
        mapped_rgb[~matched] = np.nan
        return mapped_rgb, source_description
    except Exception as exc:
        return None, f"Source-colour recovery failed: {type(exc).__name__}"


def read_vtp_triangles(
    path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, int],
    np.ndarray | None,
    np.ndarray | None,
    str,
]:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise ValueError(f"No points found in {path}")

    raw_counts = {
        "roi_total_vertices": int(polydata.GetNumberOfPoints()),
        "roi_total_cells": int(polydata.GetNumberOfCells()),
    }

    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(polydata)
    triangle_filter.PassVertsOff()
    triangle_filter.PassLinesOff()
    triangle_filter.Update()
    triangulated = triangle_filter.GetOutput()

    points = vtk_to_numpy(triangulated.GetPoints().GetData()).astype(float)
    point_rgb, point_uv, colour_source = extract_point_colour_and_uv(triangulated)
    cells = triangulated.GetPolys()

    if hasattr(cells, "GetConnectivityArray"):
        connectivity = vtk_to_numpy(cells.GetConnectivityArray()).astype(np.int64)
        offsets = vtk_to_numpy(cells.GetOffsetsArray()).astype(np.int64)
        faces_list = []
        for start, stop in zip(offsets[:-1], offsets[1:]):
            cell = connectivity[start:stop]
            if len(cell) == 3:
                faces_list.append(cell)
        faces = np.asarray(faces_list, dtype=np.int64)
    else:  # Compatibility with older VTK cell-array representation.
        legacy = vtk_to_numpy(cells.GetData()).astype(np.int64)
        faces_list = []
        cursor = 0
        while cursor < len(legacy):
            size = int(legacy[cursor])
            cell = legacy[cursor + 1 : cursor + 1 + size]
            if size == 3:
                faces_list.append(cell)
            cursor += size + 1
        faces = np.asarray(faces_list, dtype=np.int64)

    if faces.size == 0:
        raise ValueError(f"No triangular faces found in {path}")
    return (
        points,
        faces.reshape(-1, 3),
        raw_counts,
        point_rgb,
        point_uv,
        colour_source,
    )


def load_roi_radius(path: Path) -> float:
    if not path.is_file():
        return np.nan
    with np.load(path, allow_pickle=False) as data:
        return finite_float(data["roi_radius"]) if "roi_radius" in data else np.nan


# ============================================================================
# GEOMETRY CALCULATIONS
# ============================================================================

def full_roi_mesh(
    points: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return every triangle and every vertex referenced by the saved ROI."""
    referenced = np.unique(faces.ravel()) if len(faces) else np.array([], dtype=int)
    return faces, referenced


def robust_plane_fit(
    points: np.ndarray,
    maximum_iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit z = ax + by + c using iterative MAD-based trimming."""
    design = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    z = points[:, 2]
    keep = np.isfinite(design).all(axis=1) & np.isfinite(z)
    if np.sum(keep) < 3:
        raise ValueError("Too few finite points for plane fitting.")

    coefficients = np.linalg.lstsq(design[keep], z[keep], rcond=None)[0]
    for _ in range(maximum_iterations):
        vertical = z - design @ coefficients
        centre = np.median(vertical[keep])
        mad = np.median(np.abs(vertical[keep] - centre))
        if not np.isfinite(mad) or mad <= 1e-12:
            break
        new_keep = keep & (np.abs(vertical - centre) <= 4.0 * 1.4826 * mad)
        if np.sum(new_keep) < max(3, int(0.50 * np.sum(keep))):
            break
        new_coefficients = np.linalg.lstsq(
            design[new_keep], z[new_keep], rcond=None
        )[0]
        if np.allclose(new_coefficients, coefficients, rtol=1e-8, atol=1e-10):
            keep = new_keep
            coefficients = new_coefficients
            break
        keep = new_keep
        coefficients = new_coefficients

    vertical = z - design @ coefficients
    normalizer = math.sqrt(coefficients[0] ** 2 + coefficients[1] ** 2 + 1.0)
    perpendicular_residual = vertical / normalizer
    plane_normal = np.array(
        [-coefficients[0], -coefficients[1], 1.0], dtype=float
    )
    plane_normal /= np.linalg.norm(plane_normal)
    return coefficients, perpendicular_residual, plane_normal


def triangle_geometry(
    points: np.ndarray,
    faces: np.ndarray,
) -> dict[str, np.ndarray]:
    triangles = points[faces]
    edge01 = triangles[:, 1] - triangles[:, 0]
    edge12 = triangles[:, 2] - triangles[:, 1]
    edge20 = triangles[:, 0] - triangles[:, 2]
    lengths = np.column_stack(
        [
            np.linalg.norm(edge01, axis=1),
            np.linalg.norm(edge12, axis=1),
            np.linalg.norm(edge20, axis=1),
        ]
    )
    cross = np.cross(edge01, triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    area_3d = 0.5 * double_area

    xy = triangles[:, :, :2]
    projected_double_area = np.abs(
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 1, 1] - xy[:, 0, 1]) * (xy[:, 2, 0] - xy[:, 0, 0])
    )
    projected_area = 0.5 * projected_double_area

    safe_double_area = np.where(double_area > 1e-15, double_area, np.nan)
    normals = cross / safe_double_area[:, None]

    shortest = np.min(lengths, axis=1)
    longest = np.max(lengths, axis=1)
    edge_ratio = np.divide(
        longest,
        shortest,
        out=np.full_like(longest, np.nan),
        where=shortest > 1e-15,
    )

    # Interior angles from the law of cosines.
    a, b, c = lengths[:, 1], lengths[:, 2], lengths[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        angle_a = np.degrees(
            np.arccos(np.clip((b * b + c * c - a * a) / (2 * b * c), -1, 1))
        )
        angle_b = np.degrees(
            np.arccos(np.clip((a * a + c * c - b * b) / (2 * a * c), -1, 1))
        )
        angle_c = 180.0 - angle_a - angle_b
    minimum_angle = np.nanmin(np.column_stack([angle_a, angle_b, angle_c]), axis=1)

    return {
        "area_3d": area_3d,
        "projected_area": projected_area,
        "normals": normals,
        "edge_lengths_per_face": lengths,
        "edge_ratio": edge_ratio,
        "minimum_angle": minimum_angle,
    }


def unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def vertex_area_weights(
    number_of_points: int,
    faces: np.ndarray,
    triangle_areas: np.ndarray,
) -> np.ndarray:
    weights = np.zeros(number_of_points, dtype=float)
    contribution = np.repeat(triangle_areas / 3.0, 3)
    np.add.at(weights, faces.ravel(), contribution)
    return weights


def normalized_gaussian(
    values: np.ndarray,
    valid: np.ndarray,
    sigma_pixels: float,
) -> np.ndarray:
    numerator = gaussian_filter(
        np.where(valid, values, 0.0), sigma=sigma_pixels, mode="nearest"
    )
    denominator = gaussian_filter(valid.astype(float), sigma=sigma_pixels, mode="nearest")
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 1e-8,
    )


def standardized_grid_features(
    xy: np.ndarray,
    heights: np.ndarray,
    radius_mm: float,
    spacing_mm: float,
) -> tuple[dict[str, float], list[dict[str, float]], dict[str, np.ndarray]]:
    axis = np.arange(-radius_mm, radius_mm + 0.5 * spacing_mm, spacing_mm)
    grid_x, grid_y = np.meshgrid(axis, axis)
    disk = (grid_x**2 + grid_y**2) <= radius_mm**2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_z = griddata(xy, heights, (grid_x, grid_y), method="linear")

    valid = disk & np.isfinite(grid_z)
    coverage = float(np.sum(valid) / np.sum(disk)) if np.any(disk) else np.nan
    values = grid_z[valid]

    features: dict[str, float] = {
        "grid_spacing_mm": spacing_mm,
        "grid_n_valid_pixels": int(np.sum(valid)),
        "grid_disk_pixels": int(np.sum(disk)),
        "grid_coverage": coverage,
        "grid_Sa_mm": float(np.mean(np.abs(values))) if len(values) else np.nan,
        "grid_Sq_mm": float(np.sqrt(np.mean(values**2))) if len(values) else np.nan,
        "grid_height_p05_mm": float(np.percentile(values, 5)) if len(values) else np.nan,
        "grid_height_p50_mm": float(np.percentile(values, 50)) if len(values) else np.nan,
        "grid_height_p95_mm": float(np.percentile(values, 95)) if len(values) else np.nan,
    }
    features["grid_height_p95_minus_p05_mm"] = (
        features["grid_height_p95_mm"] - features["grid_height_p05_mm"]
        if len(values)
        else np.nan
    )

    # Nearest filling is used only to calculate derivatives.  Results are then
    # evaluated inside an eroded version of the original valid-data mask.
    if np.any(valid):
        nearest = griddata(xy, heights, (grid_x, grid_y), method="nearest")
        derivative_surface = np.where(valid, grid_z, nearest)
        derivative_valid = binary_erosion(valid, structure=np.ones((3, 3)))
        dz_dy, dz_dx = np.gradient(derivative_surface, spacing_mm, spacing_mm)
        squared_slope = dz_dx**2 + dz_dy**2
        features["grid_rms_slope"] = (
            float(np.sqrt(np.mean(squared_slope[derivative_valid])))
            if np.any(derivative_valid)
            else np.nan
        )

        for scale_mm in ROUGHNESS_SCALES_MM:
            sigma_pixels = scale_mm / spacing_mm
            smooth = normalized_gaussian(
                derivative_surface, valid, sigma_pixels=sigma_pixels
            )
            scale_valid = valid & np.isfinite(smooth)
            high_pass = derivative_surface - smooth
            suffix = str(scale_mm).replace(".", "p")
            features[f"grid_highpass_Sq_below_{suffix}mm"] = (
                float(np.sqrt(np.mean(high_pass[scale_valid] ** 2)))
                if np.any(scale_valid)
                else np.nan
            )

            smooth_filled = np.where(np.isfinite(smooth), smooth, derivative_surface)
            inner = binary_erosion(valid, structure=np.ones((5, 5)))
            first_y, first_x = np.gradient(smooth_filled, spacing_mm, spacing_mm)
            _, second_x = np.gradient(first_x, spacing_mm, spacing_mm)
            second_y, _ = np.gradient(first_y, spacing_mm, spacing_mm)
            approximate_mean_curvature = 0.5 * (second_x + second_y)
            curvature_values = np.abs(approximate_mean_curvature[inner])
            features[f"grid_abs_curvature_median_at_{suffix}mm"] = (
                float(np.median(curvature_values)) if len(curvature_values) else np.nan
            )
            features[f"grid_abs_curvature_p95_at_{suffix}mm"] = (
                float(np.percentile(curvature_values, 95))
                if len(curvature_values)
                else np.nan
            )
    else:
        features["grid_rms_slope"] = np.nan
        for scale_mm in ROUGHNESS_SCALES_MM:
            suffix = str(scale_mm).replace(".", "p")
            features[f"grid_highpass_Sq_below_{suffix}mm"] = np.nan
            features[f"grid_abs_curvature_median_at_{suffix}mm"] = np.nan
            features[f"grid_abs_curvature_p95_at_{suffix}mm"] = np.nan

    radial_fraction = np.sqrt(grid_x**2 + grid_y**2) / radius_mm
    ring_rows: list[dict[str, float]] = []
    for ring_name, lower, upper in RING_DEFINITIONS:
        ring_mask = valid & (radial_fraction >= lower) & (radial_fraction < upper)
        ring_values = grid_z[ring_mask]
        ring_rows.append(
            {
                "ring": ring_name,
                "radius_fraction_lower": lower,
                "radius_fraction_upper": upper,
                "n_valid_pixels": int(len(ring_values)),
                "mean_height_mm": (
                    float(np.mean(ring_values)) if len(ring_values) else np.nan
                ),
                "Sa_mm": (
                    float(np.mean(np.abs(ring_values))) if len(ring_values) else np.nan
                ),
                "Sq_mm": (
                    float(np.sqrt(np.mean(ring_values**2)))
                    if len(ring_values)
                    else np.nan
                ),
                "height_p05_mm": (
                    float(np.percentile(ring_values, 5)) if len(ring_values) else np.nan
                ),
                "height_p95_mm": (
                    float(np.percentile(ring_values, 95)) if len(ring_values) else np.nan
                ),
            }
        )
    grid_diagnostic = {
        "x": grid_x,
        "y": grid_y,
        "height": grid_z,
        "valid": valid,
        "disk": disk,
    }
    return features, ring_rows, grid_diagnostic


def analyse_one_roi(
    path: Path,
    roi_radius_mm: float,
    grid_spacing_mm: float,
    texture_path: Path | None = None,
    source_ply: Path | None = None,
    geometry_path: Path | None = None,
    return_diagnostic: bool = False,
) -> tuple[
    dict[str, float],
    list[dict[str, float]],
    dict[str, object] | None,
]:
    (
        points,
        faces,
        raw_counts,
        point_rgb,
        point_uv,
        colour_source,
    ) = read_vtp_triangles(path)
    faces_roi, referenced = full_roi_mesh(points, faces)
    roi_points = points[referenced]

    roi_rgb: np.ndarray | None = None
    if point_rgb is None:
        point_rgb = texture_vertex_colours(point_uv, texture_path)
        if point_rgb is not None:
            colour_source = f"External texture: {texture_path.name}"
    if point_rgb is None and geometry_path is not None:
        roi_rgb, colour_source = recover_roi_colours_from_source(
            roi_points,
            source_ply,
            texture_path,
            geometry_path,
        )
    elif point_rgb is not None:
        roi_rgb = point_rgb[referenced]

    if len(roi_points) < MIN_ROI_POINTS:
        raise ValueError(
            f"Only {len(roi_points)} points are referenced by the extracted ROI mesh."
        )
    if len(faces_roi) < MIN_ROI_TRIANGLES:
        raise ValueError(
            f"Only {len(faces_roi)} triangles are present in the extracted ROI mesh."
        )

    geometry = triangle_geometry(points, faces_roi)
    area_3d = geometry["area_3d"]
    projected_area = geometry["projected_area"]
    nondegenerate = np.isfinite(area_3d) & (area_3d > 1e-12)

    # The same robust METHOD is used for all scans, but a separate plane is fitted
    # to every extracted ROI. This removes scan-specific pose/tilt without aligning
    # the surface to any other stage or scanner.
    coefficients, residuals, plane_normal = robust_plane_fit(roi_points)

    vertex_weights_full = vertex_area_weights(len(points), faces_roi, area_3d)
    vertex_weights = vertex_weights_full[referenced]

    q05, q50, q95 = weighted_quantile(
        residuals, [0.05, 0.50, 0.95], vertex_weights
    )
    weighted_centre = weighted_mean(residuals, vertex_weights)
    centred = residuals - weighted_centre
    native_sa = weighted_mean(np.abs(centred), vertex_weights)
    native_sq = math.sqrt(weighted_mean(centred**2, vertex_weights))

    edges = unique_edges(faces_roi)
    edge_lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    valence = np.bincount(edges.ravel(), minlength=len(points))[referenced]

    if len(roi_points) >= 2:
        tree = cKDTree(roi_points)
        nearest_distance = tree.query(roi_points, k=2)[0][:, 1]
    else:
        nearest_distance = np.array([], dtype=float)

    try:
        hull_area = float(ConvexHull(roi_points[:, :2]).volume)
    except (QhullError, ValueError):
        hull_area = np.nan

    nominal_area = math.pi * roi_radius_mm**2
    summed_projected_area = float(np.nansum(projected_area))
    summed_surface_area = float(np.nansum(area_3d))

    normal_dots = np.abs(geometry["normals"] @ plane_normal)
    normal_angles = np.degrees(np.arccos(np.clip(normal_dots, 0.0, 1.0)))
    normal_median = weighted_quantile(normal_angles, 0.50, area_3d)[0]
    normal_p95 = weighted_quantile(normal_angles, 0.95, area_3d)[0]
    normal_rms = math.sqrt(weighted_mean(normal_angles**2, area_3d))

    # Exact duplicates are a mesh-export property.  Six decimals correspond to
    # sub-micrometre tolerance when coordinates are in millimetres.
    rounded = np.round(roi_points, decimals=6)
    duplicate_points = len(rounded) - len(np.unique(rounded, axis=0))

    grid_features, ring_rows, grid_diagnostic = standardized_grid_features(
        roi_points[:, :2],
        residuals,
        roi_radius_mm,
        grid_spacing_mm,
    )

    features: dict[str, float] = {
        **raw_counts,
        "analysis_roi_radius_mm": roi_radius_mm,
        "roi_nominal_area_mm2": nominal_area,
        "roi_vertices_referenced_by_faces": int(len(referenced)),
        "roi_faces": int(len(faces_roi)),
        "roi_unique_edges": int(len(edges)),
        "roi_duplicate_vertices_1e_6mm": int(duplicate_points),
        "roi_vertices_per_nominal_mm2": len(roi_points) / nominal_area,
        "roi_faces_per_nominal_mm2": len(faces_roi) / nominal_area,
        "roi_vertices_per_covered_mm2": (
            len(roi_points) / summed_projected_area
            if summed_projected_area > 0
            else np.nan
        ),
        "projected_triangle_area_mm2": summed_projected_area,
        "projected_area_coverage": summed_projected_area / nominal_area,
        "convex_hull_projected_area_mm2": hull_area,
        "convex_hull_area_coverage": hull_area / nominal_area,
        "surface_area_mm2": summed_surface_area,
        "surface_to_projected_area_ratio": (
            summed_surface_area / summed_projected_area
            if summed_projected_area > 0
            else np.nan
        ),
        "median_nearest_neighbor_mm": (
            float(np.median(nearest_distance)) if len(nearest_distance) else np.nan
        ),
        "p05_nearest_neighbor_mm": (
            float(np.percentile(nearest_distance, 5)) if len(nearest_distance) else np.nan
        ),
        "p95_nearest_neighbor_mm": (
            float(np.percentile(nearest_distance, 95)) if len(nearest_distance) else np.nan
        ),
        "median_unique_edge_length_mm": float(np.median(edge_lengths)),
        "p05_unique_edge_length_mm": float(np.percentile(edge_lengths, 5)),
        "p95_unique_edge_length_mm": float(np.percentile(edge_lengths, 95)),
        "median_triangle_area_mm2": float(np.nanmedian(area_3d)),
        "p05_triangle_area_mm2": float(np.nanpercentile(area_3d, 5)),
        "p95_triangle_area_mm2": float(np.nanpercentile(area_3d, 95)),
        "degenerate_triangle_percent": 100.0 * (1.0 - np.mean(nondegenerate)),
        "median_triangle_edge_ratio": float(np.nanmedian(geometry["edge_ratio"])),
        "p95_triangle_edge_ratio": float(np.nanpercentile(geometry["edge_ratio"], 95)),
        "median_triangle_min_angle_deg": float(
            np.nanmedian(geometry["minimum_angle"])
        ),
        "p05_triangle_min_angle_deg": float(
            np.nanpercentile(geometry["minimum_angle"], 5)
        ),
        "median_vertex_valence": float(np.median(valence)),
        "p95_vertex_valence": float(np.percentile(valence, 95)),
        "plane_a": float(coefficients[0]),
        "plane_b": float(coefficients[1]),
        "plane_c": float(coefficients[2]),
        "plane_tilt_deg": float(
            np.degrees(np.arctan(np.hypot(coefficients[0], coefficients[1])))
        ),
        "plane_rmse_mm": float(np.sqrt(np.mean(residuals**2))),
        "plane_residual_mad_mm": float(
            np.median(np.abs(residuals - np.median(residuals)))
        ),
        "native_Sa_mm": native_sa,
        "native_Sq_mm": native_sq,
        "native_height_p05_mm": float(q05),
        "native_height_p50_mm": float(q50),
        "native_height_p95_mm": float(q95),
        "native_height_p95_minus_p05_mm": float(q95 - q05),
        "native_height_skewness": (
            float(skew(residuals, bias=False)) if len(residuals) >= 3 else np.nan
        ),
        "native_height_excess_kurtosis": (
            float(kurtosis(residuals, bias=False)) if len(residuals) >= 4 else np.nan
        ),
        "area_weighted_normal_angle_median_deg": float(normal_median),
        "area_weighted_normal_angle_p95_deg": float(normal_p95),
        "area_weighted_normal_angle_rms_deg": float(normal_rms),
        **grid_features,
    }
    diagnostic = None
    if return_diagnostic:
        remap = np.full(len(points), -1, dtype=np.int64)
        remap[referenced] = np.arange(len(referenced))
        diagnostic = {
            "all_roi_points": points,
            "roi_points": roi_points,
            "roi_faces": remap[faces_roi],
            "roi_rgb": roi_rgb,
            "colour_source": colour_source,
            "roi_heights": residuals,
            "plane_coefficients": coefficients,
            "grid_x": grid_diagnostic["x"],
            "grid_y": grid_diagnostic["y"],
            "grid_height": grid_diagnostic["height"],
            "grid_valid": grid_diagnostic["valid"],
            "grid_disk": grid_diagnostic["disk"],
        }
    return features, ring_rows, diagnostic


# ============================================================================
# PROCESSING EXAMPLES AND FEATURE DICTIONARY
# ============================================================================

def safe_filename(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def plot_true_colour_roi(
    axis: plt.Axes,
    roi_points: np.ndarray,
    roi_faces: np.ndarray,
    roi_rgb: np.ndarray | None,
    title: str,
    colour_source: str,
) -> None:
    """Render the native ROI triangles using vertex colour when available."""
    polygons = roi_points[roi_faces, :2]
    if roi_rgb is not None and len(roi_rgb) == len(roi_points):
        face_colours = np.nanmean(roi_rgb[roi_faces], axis=1)
        face_colours[~np.isfinite(face_colours)] = 0.70
        collection = PolyCollection(
            polygons,
            facecolors=np.clip(face_colours, 0.0, 1.0),
            edgecolors="none",
            rasterized=True,
        )
        axis.add_collection(collection)
        axis.autoscale_view()
        source_line = colour_source
    else:
        collection = PolyCollection(
            polygons,
            facecolors="#BDBDBD",
            edgecolors="#777777",
            linewidths=0.12,
            rasterized=True,
        )
        axis.add_collection(collection)
        axis.autoscale_view()
        source_line = (
            colour_source
            if colour_source and colour_source != "Colour unavailable"
            else "No embedded RGB or usable texture"
        )
    axis.set_title(title)
    axis.set_xlabel("Canonical x (mm)")
    axis.set_ylabel("Canonical y (mm)")
    axis.set_aspect("equal")
    axis.text(
        0.02,
        0.02,
        source_line,
        transform=axis.transAxes,
        fontsize=7.5,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )


def save_processing_example(
    diagnostic: dict[str, object],
    features: dict[str, float],
    scanner: str,
    sample_id: str,
    saved_roi_radius_mm: float,
    output: Path,
) -> None:
    """Show exactly which surface is analysed and how it is transformed."""
    output.parent.mkdir(parents=True, exist_ok=True)
    all_points = diagnostic["all_roi_points"]
    roi_points = diagnostic["roi_points"]
    roi_faces = diagnostic["roi_faces"]
    roi_rgb = diagnostic["roi_rgb"]
    colour_source = str(diagnostic["colour_source"])
    roi_heights = diagnostic["roi_heights"]
    plane_coefficients = diagnostic["plane_coefficients"]
    grid_x = diagnostic["grid_x"]
    grid_y = diagnostic["grid_y"]
    grid_height = diagnostic["grid_height"]
    grid_valid = diagnostic["grid_valid"]
    roi_radius = float(features["analysis_roi_radius_mm"])

    if not np.isfinite(saved_roi_radius_mm):
        saved_roi_radius_mm = roi_radius

    height_pool = [roi_heights[np.isfinite(roi_heights)]]
    if np.any(grid_valid):
        height_pool.append(grid_height[grid_valid])
    height_pool_array = np.concatenate(height_pool)
    color_limit = float(np.percentile(np.abs(height_pool_array), 98))
    color_limit = max(color_limit, 1e-6)

    figure = plt.figure(figsize=(18, 16))
    axes = [
        figure.add_subplot(3, 3, 1),
        figure.add_subplot(3, 3, 2),
        figure.add_subplot(3, 3, 3, projection="3d"),
        figure.add_subplot(3, 3, 4),
        figure.add_subplot(3, 3, 5),
        figure.add_subplot(3, 3, 6),
        figure.add_subplot(3, 3, 7),
        figure.add_subplot(3, 3, 8),
        figure.add_subplot(3, 3, 9),
    ]

    # A: the saved ROI is used directly; there is no second radial extraction.
    axis = axes[0]
    if len(all_points) > 20000:
        display_indices = np.linspace(0, len(all_points) - 1, 20000).astype(int)
        displayed = all_points[display_indices]
    else:
        displayed = all_points
    axis.scatter(displayed[:, 0], displayed[:, 1], s=1.8, color="#D62728", alpha=0.65)
    axis.add_patch(
        Circle(
            (0, 0),
            saved_roi_radius_mm,
            fill=False,
            linestyle="--",
            linewidth=1.5,
            color="#555555",
            label="Saved reference radius",
        )
    )
    axis.set_title("1. Full extracted ROI used for analysis")
    axis.legend(loc="upper right", fontsize=8, frameon=True)
    axis.set_xlabel("Canonical x (mm)")
    axis.set_ylabel("Canonical y (mm)")
    axis.set_aspect("equal")

    # B: native triangles used for scanner sampling and mesh-quality features.
    triangulation = Triangulation(
        roi_points[:, 0], roi_points[:, 1], triangles=roi_faces
    )
    axis = axes[1]
    axis.triplot(triangulation, color="#264653", linewidth=0.25, alpha=0.75)
    axis.set_title("2. Every native ROI triangle analysed")
    axis.set_xlabel("Canonical x (mm)")
    axis.set_ylabel("Canonical y (mm)")
    axis.set_aspect("equal")

    # C: explicit 3D visualization of the independently fitted robust plane.
    axis = axes[2]
    if len(roi_points) > 6000:
        plane_display_indices = np.linspace(0, len(roi_points) - 1, 6000).astype(int)
        plane_display = roi_points[plane_display_indices]
        plane_display_heights = roi_heights[plane_display_indices]
    else:
        plane_display = roi_points
        plane_display_heights = roi_heights
    plane_color_limit = max(float(np.percentile(np.abs(plane_display_heights), 98)), 1e-6)
    axis.scatter(
        plane_display[:, 0],
        plane_display[:, 1],
        plane_display[:, 2],
        c=plane_display_heights,
        cmap="coolwarm",
        vmin=-plane_color_limit,
        vmax=plane_color_limit,
        s=2.0,
        alpha=0.60,
    )
    plane_axis = np.linspace(-roi_radius, roi_radius, 31)
    plane_x, plane_y = np.meshgrid(plane_axis, plane_axis)
    plane_z = (
        plane_coefficients[0] * plane_x
        + plane_coefficients[1] * plane_y
        + plane_coefficients[2]
    )
    plane_z[(plane_x**2 + plane_y**2) > roi_radius**2] = np.nan
    axis.plot_surface(
        plane_x,
        plane_y,
        plane_z,
        color="#2A9D8F",
        alpha=0.38,
        linewidth=0,
        antialiased=True,
    )
    axis.set_title("3. ROI surface and fitted plane\n(vertical scale enlarged)")
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_zlabel("z (mm)")
    axis.view_init(elev=24, azim=-55)
    try:
        axis.set_box_aspect((1, 1, 0.35))
    except AttributeError:
        pass

    # D: the actual scan colour, placed directly beside the two height maps.
    axis = axes[3]
    plot_true_colour_roi(
        axis,
        roi_points,
        roi_faces,
        roi_rgb,
        "4. Actual ROI colour",
        colour_source,
    )

    # E: topography after removing residual tilt with that fitted plane.
    axis = axes[4]
    native_map = axis.tripcolor(
        triangulation,
        roi_heights * 1000.0,
        shading="gouraud",
        cmap="coolwarm",
        vmin=-1000.0 * color_limit,
        vmax=1000.0 * color_limit,
    )
    axis.set_title("5. Plane-detrended native height")
    axis.set_xlabel("Canonical x (mm)")
    axis.set_ylabel("Canonical y (mm)")
    axis.set_aspect("equal")
    figure.colorbar(native_map, ax=axis, label="Residual height (µm)", shrink=0.82)

    # F: identical physical grid for density-independent comparisons.
    axis = axes[5]
    masked_grid = np.ma.masked_where(~grid_valid, grid_height * 1000.0)
    grid_map = axis.pcolormesh(
        grid_x,
        grid_y,
        masked_grid,
        shading="auto",
        cmap="coolwarm",
        vmin=-1000.0 * color_limit,
        vmax=1000.0 * color_limit,
    )
    axis.add_patch(Circle((0, 0), roi_radius, fill=False, color="black", linewidth=1.0))
    axis.set_title("6. Standardized-grid height map")
    axis.set_xlabel("Canonical x (mm)")
    axis.set_ylabel("Canonical y (mm)")
    axis.set_aspect("equal")
    figure.colorbar(grid_map, ax=axis, label="Residual height (µm)", shrink=0.82)

    # G: rotation-independent radial regions used for spatial summaries.
    axis = axes[6]
    radial_fraction = np.linalg.norm(roi_points[:, :2], axis=1) / roi_radius
    ring_index = np.select(
        [radial_fraction < 0.33, radial_fraction < 0.66], [0, 1], default=2
    )
    ring_colors = np.array(["#2A9D8F", "#E9C46A", "#E76F51"])
    axis.scatter(
        roi_points[:, 0],
        roi_points[:, 1],
        c=ring_colors[ring_index],
        s=2.0,
        alpha=0.70,
    )
    for fraction, color in [(0.33, "#2A9D8F"), (0.66, "#C48B19"), (1.0, "#C4422B")]:
        axis.add_patch(
            Circle((0, 0), fraction * roi_radius, fill=False, color=color, linewidth=1.5)
        )
    axis.set_title("7. Rotation-independent radial zones")
    axis.set_xlabel("Canonical x (mm)")
    axis.set_ylabel("Canonical y (mm)")
    axis.set_aspect("equal")
    axis.text(
        0.02,
        0.02,
        "Central: 0–0.33r\nMiddle: 0.33–0.66r\nOuter ROI: 0.66–1.00r",
        transform=axis.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "none"},
    )

    # H: distributions make outliers and grid standardization visible.
    axis = axes[7]
    native_micrometres = roi_heights[np.isfinite(roi_heights)] * 1000.0
    grid_micrometres = grid_height[grid_valid] * 1000.0
    bins = np.linspace(-1000.0 * color_limit, 1000.0 * color_limit, 41)
    axis.hist(
        native_micrometres,
        bins=bins,
        density=True,
        alpha=0.45,
        color="#264653",
        label="Native vertices",
    )
    axis.hist(
        grid_micrometres,
        bins=bins,
        density=True,
        alpha=0.45,
        color="#E76F51",
        label="Standardized grid",
    )
    axis.axvline(0, color="black", linewidth=0.9)
    axis.set_title("8. Detrended-height distributions")
    axis.set_xlabel("Residual height (µm)")
    axis.set_ylabel("Density")
    axis.legend(frameon=False)
    summary_text = (
        f"Vertices/mm²: {features['roi_vertices_per_nominal_mm2']:.2f}\n"
        f"Median edge: {features['median_unique_edge_length_mm']:.3f} mm\n"
        f"Native Sq: {1000 * features['native_Sq_mm']:.2f} µm\n"
        f"Grid Sq: {1000 * features['grid_Sq_mm']:.2f} µm\n"
        f"Grid coverage: {100 * features['grid_coverage']:.1f}%"
    )
    axis.text(
        0.98,
        0.98,
        summary_text,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#CCCCCC"},
    )

    # I: make the per-scan plane logic and key quality values explicit.
    axis = axes[8]
    axis.axis("off")
    plane_text = (
        "PLANE USED FOR THIS SCAN\n\n"
        f"z = {plane_coefficients[0]:+.6f}x "
        f"{plane_coefficients[1]:+.6f}y "
        f"{plane_coefficients[2]:+.6f}\n\n"
        f"Plane tilt: {features['plane_tilt_deg']:.3f}°\n"
        f"Plane RMSE: {1000 * features['plane_rmse_mm']:.2f} µm\n"
        f"Projected coverage: {100 * features['projected_area_coverage']:.1f}%\n\n"
        "A new robust plane is fitted independently to every ROI.\n"
        "The method is identical across scans; the plane coefficients are not.\n\n"
        "Residual height = perpendicular distance from each ROI vertex\n"
        "to this scan-specific fitted plane."
    )
    axis.text(
        0.03,
        0.97,
        plane_text,
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.35,
        bbox={"facecolor": "#F4F4F4", "edgecolor": "#CCCCCC", "boxstyle": "round,pad=0.8"},
    )

    for axis in [axes[0], axes[1], axes[3], axes[4], axes[5], axes[6], axes[7]]:
        axis.grid(alpha=0.15)
    figure.suptitle(
        f"Geometry-processing example — {scanner} | {sample_id}\n"
        "Full extracted ROI analysed; colour/height comparison is visual only",
        fontsize=16,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    figure.savefig(str(output), dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_group3_stage_comparison(
    stage_records: dict[int, tuple[dict[str, object], dict[str, float], str]],
    scanner: str,
    sample_number: int,
    output: Path,
) -> None:
    """Place true colour and detrended height side by side for G3 T0/T1/T2."""
    available = [timepoint for timepoint in EXAMPLE_TIMEPOINTS if timepoint in stage_records]
    if not available:
        return
    output.parent.mkdir(parents=True, exist_ok=True)

    height_values: list[np.ndarray] = []
    for timepoint in available:
        diagnostic = stage_records[timepoint][0]
        native = np.asarray(diagnostic["roi_heights"], dtype=float)
        valid = np.asarray(diagnostic["grid_valid"], dtype=bool)
        grid = np.asarray(diagnostic["grid_height"], dtype=float)
        height_values.append(native[np.isfinite(native)])
        height_values.append(grid[valid & np.isfinite(grid)])
    combined = np.concatenate([values for values in height_values if len(values)])
    colour_limit_um = max(float(np.percentile(np.abs(combined), 98)) * 1000.0, 1e-3)

    figure, axes = plt.subplots(
        len(available),
        3,
        figsize=(15.5, 4.6 * len(available)),
        squeeze=False,
    )
    last_map = None
    for row_index, timepoint in enumerate(available):
        diagnostic, _features, sample_id = stage_records[timepoint]
        stage = biological_stage(EXAMPLE_GROUP, timepoint)
        roi_points = np.asarray(diagnostic["roi_points"], dtype=float)
        roi_faces = np.asarray(diagnostic["roi_faces"], dtype=int)
        roi_rgb = diagnostic["roi_rgb"]
        colour_source = str(diagnostic["colour_source"])
        roi_heights = np.asarray(diagnostic["roi_heights"], dtype=float)
        triangulation = Triangulation(
            roi_points[:, 0], roi_points[:, 1], triangles=roi_faces
        )

        plot_true_colour_roi(
            axes[row_index, 0],
            roi_points,
            roi_faces,
            roi_rgb if isinstance(roi_rgb, np.ndarray) else None,
            f"T{timepoint} — {stage}\nActual ROI colour",
            colour_source,
        )
        axes[row_index, 0].text(
            0.98,
            0.02,
            sample_id,
            transform=axes[row_index, 0].transAxes,
            ha="right",
            fontsize=7.5,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )

        last_map = axes[row_index, 1].tripcolor(
            triangulation,
            roi_heights * 1000.0,
            shading="gouraud",
            cmap="coolwarm",
            vmin=-colour_limit_um,
            vmax=colour_limit_um,
        )
        axes[row_index, 1].set_title(
            f"T{timepoint} — {stage}\nNative residual height"
        )

        grid_x = np.asarray(diagnostic["grid_x"], dtype=float)
        grid_y = np.asarray(diagnostic["grid_y"], dtype=float)
        grid_height = np.asarray(diagnostic["grid_height"], dtype=float)
        grid_valid = np.asarray(diagnostic["grid_valid"], dtype=bool)
        masked_grid = np.ma.masked_where(~grid_valid, grid_height * 1000.0)
        last_map = axes[row_index, 2].pcolormesh(
            grid_x,
            grid_y,
            masked_grid,
            shading="auto",
            cmap="coolwarm",
            vmin=-colour_limit_um,
            vmax=colour_limit_um,
        )
        axes[row_index, 2].set_title(
            f"T{timepoint} — {stage}\nStandardized-grid height"
        )

        for axis in axes[row_index, 1:]:
            axis.set_xlabel("Canonical x (mm)")
            axis.set_ylabel("Canonical y (mm)")
            axis.set_aspect("equal")
            axis.grid(alpha=0.15)

    missing = sorted(set(EXAMPLE_TIMEPOINTS) - set(available))
    missing_note = f" Missing: {', '.join(f'T{x}' for x in missing)}." if missing else ""
    figure.suptitle(
        f"Group 3 matched-stage colour and surface example — {scanner} | "
        f"BCG3-{sample_number}\n"
        "Shared height scale within scanner; visual comparison only; surfaces are "
        f"not pointwise registered.{missing_note}",
        fontsize=15,
    )
    figure.subplots_adjust(top=0.91, right=0.86, hspace=0.34, wspace=0.28)
    if last_map is not None:
        colourbar_axis = figure.add_axes([0.89, 0.16, 0.018, 0.66])
        figure.colorbar(
            last_map,
            cax=colourbar_axis,
            label="Residual height from each scan's fitted plane (µm)",
        )
    try:
        figure.savefig(str(output), dpi=220, bbox_inches="tight")
    except OSError as first_error:
        # Some Windows/Pillow combinations reject an otherwise valid long or
        # currently locked output pathname. Retry with a short fresh filename
        # directly in figures/ instead of terminating the whole analysis.
        scanner_code = SCANNER_FILE_CODES.get(scanner, safe_filename(scanner)[:8])
        fallback_base = output.parent.parent / f"G3_{scanner_code}_{sample_number}"
        fallback = fallback_base.with_suffix(".png")
        counter = 1
        while fallback.exists():
            fallback = output.parent.parent / f"{fallback_base.name}_{counter}.png"
            counter += 1
        print(
            f"WARNING: could not save stage example as {output.name}: "
            f"{first_error}. Retrying as {fallback.name}."
        )
        figure.savefig(str(fallback), dpi=220, bbox_inches="tight")
    finally:
        plt.close(figure)


def feature_dictionary() -> pd.DataFrame:
    """Definitions for every geometric feature family written by the script."""
    rows: list[dict[str, str]] = []

    def add(
        category: str,
        feature: str,
        calculation: str,
        tells_you: str,
        limitation: str,
        role: str,
    ) -> None:
        rows.append(
            {
                "category": category,
                "feature": feature,
                "calculation": calculation,
                "what_it_tells_you": tells_you,
                "important_limitation": limitation,
                "recommended_role": role,
            }
        )

    add("Size", "roi_total_vertices", "All points stored in the extracted ROI VTP.", "Overall ROI mesh size.", "Includes the clipped peripheral region.", "Descriptive")
    add("Size", "roi_total_cells", "All cells stored in the extracted ROI VTP before triangulation.", "Overall ROI mesh complexity.", "Cell types may differ before triangulation.", "Descriptive")
    add("ROI", "analysis_roi_radius_mm", "Saved radius used when preprocessing extracted this scanner's ROI.", "Nominal ROI radius used for area normalization, grid extent and radial zones.", "It was manually defined on the scanner reference and is not a measured outcome.", "Design variable")
    add("ROI", "roi_nominal_area_mm2", "π × analysis_roi_radius_mm².", "Nominal area of the already-extracted ROI.", "Does not account for gaps between retained boundary triangles.", "Denominator")
    add("ROI", "roi_vertices_referenced_by_faces", "Unique vertices referenced by any triangle already present in roi.vtp.", "Number of sampled locations in the complete extracted ROI mesh.", "Affected by coverage and scanner tessellation.", "Primary sampling feature")
    add("ROI", "roi_faces", "Every triangle already present in the extracted ROI VTP.", "Number of reconstructed surface elements analysed.", "Strongly related to vertex count.", "Descriptive")
    add("ROI", "roi_unique_edges", "Unique vertex pairs belonging to all ROI triangles.", "Connectivity size of the extracted triangular mesh.", "Strongly related to faces and vertices.", "Descriptive")
    add("Mesh QC", "roi_duplicate_vertices_1e_6mm", "ROI coordinates rounded to 10⁻⁶ mm and counted more than once.", "Potential duplicate points in the export.", "Only detects effectively identical coordinates.", "QC")
    add("Sampling", "roi_vertices_per_nominal_mm2", "ROI referenced vertices divided by πr².", "Scanner sampling density normalized by its saved ROI area.", "Reduced coverage can also lower this value; inspect coverage simultaneously.", "Primary")
    add("Sampling", "roi_faces_per_nominal_mm2", "All ROI triangles divided by πr².", "Triangle density.", "Nearly redundant with vertex density.", "Secondary")
    add("Sampling", "roi_vertices_per_covered_mm2", "ROI vertices divided by summed projected triangle area.", "Sampling density after accounting for covered surface.", "Can increase if coverage is fragmented.", "Sensitivity analysis")
    add("Coverage", "projected_triangle_area_mm2", "Sum of all ROI triangle areas after projection to the canonical xy plane.", "Amount of planar ROI represented by triangles.", "Overlapping/folded surfaces could inflate it.", "QC")
    add("Coverage", "projected_area_coverage", "Projected ROI triangle area divided by πr².", "Completeness of the extracted ROI surface.", "Small boundary gaps occur because the extracted edge follows complete triangles.", "Primary QC")
    add("Coverage", "convex_hull_projected_area_mm2", "Area of the convex hull around projected ROI points.", "Outer spatial extent of available ROI points.", "Does not detect holes inside the hull.", "QC")
    add("Coverage", "convex_hull_area_coverage", "ROI convex-hull area divided by πr².", "Whether points reach most of the saved ROI radius.", "Can remain high despite internal missing regions.", "QC")
    add("Area", "surface_area_mm2", "Sum of all 3D triangle areas in the extracted ROI.", "Reconstructed 3D surface area.", "Depends on reconstructed roughness and tessellation.", "Secondary")
    add("Area", "surface_to_projected_area_ratio", "3D triangle area divided by projected triangle area.", "Surface complexity; 1 is nearly planar and larger values indicate more slope/texture.", "Sensitive to noise and scanner smoothing; not plaque volume.", "Secondary")

    for prefix, description in [
        ("nearest_neighbor", "distance from each ROI point to its closest other point"),
        ("unique_edge_length", "length of each unique triangular-mesh edge"),
        ("triangle_area", "3D area of each retained triangle"),
    ]:
        for statistic, meaning in [
            ("p05", "lower tail"),
            ("median", "typical value"),
            ("p95", "upper tail"),
        ]:
            unit = "mm2" if prefix == "triangle_area" else "mm"
            feature = f"{statistic}_{prefix}_{unit}"
            add("Sampling", feature, f"{statistic.upper()} of the {description}.", f"The {meaning} of the scanner's spatial sampling scale.", "Describes reconstruction density, not geometric accuracy.", "Primary" if statistic == "median" else "Distribution/QC")

    add("Triangle quality", "degenerate_triangle_percent", "Percentage of extracted ROI triangles with near-zero 3D area.", "Numerical/topological mesh problems.", "Threshold is computational rather than a biological criterion.", "QC")
    add("Triangle quality", "median_triangle_edge_ratio", "Median longest-edge/shortest-edge ratio per triangle.", "Typical triangle elongation; 1 is equilateral.", "Affected by scanner meshing strategy.", "Primary triangle-quality feature")
    add("Triangle quality", "p95_triangle_edge_ratio", "95th percentile of longest/shortest edge ratio.", "Upper tail of elongated triangles.", "Can be driven by local boundary/coverage behaviour.", "QC")
    add("Triangle quality", "median_triangle_min_angle_deg", "Median of each triangle's smallest interior angle.", "Typical triangle shape; larger values are generally less elongated.", "Redundant with edge ratio.", "Secondary")
    add("Triangle quality", "p05_triangle_min_angle_deg", "5th percentile of triangle minimum angles.", "Worst-tail triangle quality.", "Small values may occur around gaps.", "QC")
    add("Connectivity", "median_vertex_valence", "Median number of unique mesh edges incident to a referenced vertex.", "Typical local mesh connectivity.", "Mainly describes triangulation convention.", "Descriptive")
    add("Connectivity", "p95_vertex_valence", "95th percentile of incident-edge counts.", "Upper tail of local mesh connectivity.", "Not a biological feature.", "QC")

    add("Plane", "plane_a / plane_b / plane_c", "Coefficients of robust z = ax + by + c plane fitted independently to every complete extracted ROI.", "The scan-specific detrending plane removed before roughness calculation.", "The fitting method is shared, but the plane itself is different for every scan.", "Diagnostic")
    add("Plane", "plane_tilt_deg", "atan(sqrt(a²+b²)) converted to degrees.", "Residual ROI orientation after preprocessing alignment.", "Removed before topography analysis; use mainly for QC.", "QC")
    add("Topography", "plane_rmse_mm", "RMS perpendicular distance of ROI points from their scan-specific robust plane.", "Overall reconstructed departure from planarity.", "Strongly correlated with Sq and scanner resolution.", "Secondary")
    add("Topography", "plane_residual_mad_mm", "Median absolute deviation of point-to-plane residuals.", "Robust spread of reconstructed heights.", "Not area weighted.", "Secondary")
    add("Topography", "native_Sa_mm", "Area-weighted mean absolute detrended height.", "Average native-mesh roughness amplitude.", "Depends on scanner resolution and smoothing.", "Secondary")
    add("Topography", "native_Sq_mm", "Square root of the area-weighted mean squared detrended height.", "RMS native-mesh roughness amplitude.", "Depends on scanner resolution and smoothing.", "Primary native feature")
    for percentile in ["p05", "p50", "p95"]:
        add("Topography", f"native_height_{percentile}_mm", f"Area-weighted {percentile.upper()} of detrended native height.", "Location of the native height distribution.", "Scanner-dependent and not a physical thickness reference.", "Distribution")
    add("Topography", "native_height_p95_minus_p05_mm", "Area-weighted P95 minus P05 detrended height.", "Robust vertical range of reconstructed surface variation.", "Does not localize or measure removed plaque.", "Secondary")
    add("Topography", "native_height_skewness", "Skewness of detrended vertex heights.", "Asymmetry of peaks versus depressions.", "Not area weighted and sensitive to sampling.", "Exploratory")
    add("Topography", "native_height_excess_kurtosis", "Excess kurtosis of detrended vertex heights.", "Whether the height distribution has heavy tails/outliers.", "Sensitive to isolated reconstruction artefacts.", "Exploratory")
    for statistic, meaning in [("median", "typical"), ("p95", "upper-tail"), ("rms", "RMS")]:
        add("Normals", f"area_weighted_normal_angle_{statistic}_deg", f"Area-weighted {statistic} angle between triangle normals and the fitted-plane normal.", f"{meaning} local surface-orientation variation.", "Higher density/noise can increase normal variation.", "Primary" if statistic == "median" else "Secondary")
    add("Grid", "grid_spacing_mm", "Fixed xy distance between standardized-grid pixels.", "Physical scale of grid comparison.", "Analysis setting, not an observed feature.", "Design variable")
    add("Grid", "grid_n_valid_pixels / grid_disk_pixels", "Valid interpolated pixels and all pixels geometrically inside the saved ROI-radius disk.", "Amount of standardized-grid data available.", "Counts depend on selected grid spacing.", "QC")
    add("Grid", "grid_coverage", "Valid grid pixels divided by all pixels inside the saved ROI-radius disk.", "Completeness after interpolation to a common pixel spacing.", "Interpolation does not create information outside the measured convex hull.", "Primary QC")
    add("Grid", "grid_Sa_mm", "Mean absolute detrended height on equal-area grid pixels.", "Sampling-density-standardized average roughness.", "Still reflects scanner reconstruction and interpolation.", "Secondary")
    add("Grid", "grid_Sq_mm", "RMS detrended height on equal-area grid pixels.", "Main cross-scanner reconstructed-roughness measure.", "Not validated physical microroughness or plaque thickness.", "Primary topography feature")
    for percentile in ["p05", "p50", "p95"]:
        add("Grid", f"grid_height_{percentile}_mm", f"{percentile.upper()} of detrended valid grid heights.", "Location of the standardized height distribution.", "No pointwise stage correspondence is implied.", "Distribution")
    add("Grid", "grid_height_p95_minus_p05_mm", "Grid P95 minus P05 height.", "Robust standardized vertical range.", "Does not measure volume or localized loss.", "Secondary")
    add("Grid", "grid_rms_slope", "RMS magnitude of finite-difference x/y height gradients on valid interior pixels.", "Overall local steepness of reconstructed topography.", "Depends on grid spacing and reconstruction smoothing.", "Secondary")
    for scale_mm in ROUGHNESS_SCALES_MM:
        suffix = str(scale_mm).replace(".", "p")
        add("Scale-specific", f"grid_highpass_Sq_below_{suffix}mm", f"RMS difference between grid height and Gaussian-smoothed height at {scale_mm:.2f} mm scale.", f"Fine surface variation below approximately {scale_mm:.2f} mm.", "Strongly scanner- and interpolation-dependent.", "Exploratory")
        add("Scale-specific", f"grid_abs_curvature_median_at_{suffix}mm", f"Median absolute small-slope mean-curvature approximation after {scale_mm:.2f} mm smoothing.", "Typical bending at the specified physical scale.", "Approximation only; not reference-grade curvature.", "Exploratory")
        add("Scale-specific", f"grid_abs_curvature_p95_at_{suffix}mm", f"95th percentile absolute curvature approximation after {scale_mm:.2f} mm smoothing.", "Upper tail of local bending at that scale.", "Sensitive to holes, boundaries and reconstruction artefacts.", "Exploratory/QC")

    add("Radial", "ring", "Central (0–0.33r), middle (0.33–0.66r), or outer-ROI (0.66–1.00r).", "Rotation-independent surface location.", "Does not provide pointwise correspondence.", "Spatial summary")
    add("Radial", "ring mean_height_mm", "Mean detrended grid height in each radial zone.", "Whether a zone lies relatively above or below the global fitted plane.", "Relative within-scan value, not physical thickness.", "Exploratory")
    add("Radial", "ring Sa_mm / Sq_mm", "Mean absolute and RMS detrended grid height within each radial zone.", "Radial distribution of reconstructed roughness.", "Affected by valid-pixel coverage in each ring.", "Secondary")
    add("Radial", "ring height_p05_mm / height_p95_mm", "5th and 95th percentiles within each radial zone.", "Radial height-distribution limits.", "Not spatially matched between scans.", "Exploratory")

    return pd.DataFrame(rows)


def write_feature_guide(table_root: Path, output_root: Path) -> None:
    dictionary = feature_dictionary()
    dictionary.to_csv(table_root / "geometry_feature_dictionary.csv", index=False)

    lines = [
        "# BioCal3D geometry feature guide",
        "",
        "All surface features are calculated independently from every triangle already present in each extracted ROI. ",
        "They describe reconstructed geometry, not validated physical trueness, plaque ",
        "thickness, or localized material removal.",
        "",
        "| Category | Feature | Estimation | What it tells you | Limitation | Role |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in dictionary.iterrows():
        values = [str(row[column]).replace("|", "/") for column in dictionary.columns]
        lines.append("| " + " | ".join(values) + " |")
    (output_root / "GEOMETRY_FEATURE_GUIDE.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ============================================================================
# STATISTICAL SUMMARIES
# ============================================================================

def scanner_descriptive_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ANALYSIS_METRICS:
        if metric not in data:
            continue
        for scanner, values in data.groupby("scanner", observed=True)[metric]:
            values = pd.to_numeric(values, errors="coerce").dropna()
            if len(values) == 0:
                continue
            rows.append(
                {
                    "metric": metric,
                    "scanner": scanner,
                    "n": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "median": values.median(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                    "minimum": values.min(),
                    "maximum": values.max(),
                }
            )
    return pd.DataFrame(rows)


def scanner_matched_tests(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairwise_rows = []
    friedman_rows = []

    for metric in ANALYSIS_METRICS:
        if metric not in data:
            continue
        pivot = data.pivot_table(
            index="sample_id",
            columns="scanner",
            values=metric,
            aggfunc="first",
            observed=True,
        )

        available_scanners = [scanner for scanner in SCANNER_ORDER if scanner in pivot]
        if len(available_scanners) >= 3:
            complete = pivot[available_scanners].dropna()
            if len(complete) >= 3:
                statistic, p_value = friedmanchisquare(
                    *[complete[scanner].to_numpy() for scanner in available_scanners]
                )
                friedman_rows.append(
                    {
                        "metric": metric,
                        "scanners": " | ".join(available_scanners),
                        "n_complete_sample_ids": len(complete),
                        "friedman_chi2": statistic,
                        "p_value": p_value,
                    }
                )

        metric_start = len(pairwise_rows)
        for scanner_a, scanner_b in combinations(available_scanners, 2):
            paired = pivot[[scanner_a, scanner_b]].dropna()
            if len(paired) < 3:
                continue
            differences = paired[scanner_b].to_numpy() - paired[scanner_a].to_numpy()
            if np.allclose(differences, 0.0, rtol=0.0, atol=1e-15):
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value = wilcoxon(
                    differences, zero_method="wilcox", alternative="two-sided"
                )
            pairwise_rows.append(
                {
                    "metric": metric,
                    "scanner_A": scanner_a,
                    "scanner_B": scanner_b,
                    "n_pairs": len(paired),
                    "median_A": paired[scanner_a].median(),
                    "median_B": paired[scanner_b].median(),
                    "median_difference_B_minus_A": float(np.median(differences)),
                    "wilcoxon_W": statistic,
                    "p_value": p_value,
                    "rank_biserial_B_minus_A": rank_biserial_from_differences(
                        differences
                    ),
                }
            )

        metric_indices = list(range(metric_start, len(pairwise_rows)))
        if metric_indices:
            raw_p = [pairwise_rows[index]["p_value"] for index in metric_indices]
            adjusted = holm_adjust(raw_p)
            for index, adjusted_p in zip(metric_indices, adjusted):
                pairwise_rows[index]["p_holm_within_metric"] = adjusted_p

    return pd.DataFrame(pairwise_rows), pd.DataFrame(friedman_rows)


def create_stage_changes(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stage_pairs = [
        ("Baseline", "Partial"),
        ("Baseline", "Clean"),
        ("Partial", "Clean"),
    ]
    for (scanner, specimen), subset in data.groupby(
        ["scanner", "physical_sample_id"], observed=True
    ):
        if not specimen:
            continue
        group_value = subset["group"].dropna()
        group = int(group_value.iloc[0]) if len(group_value) else np.nan
        stage_rows = subset.drop_duplicates("stage").set_index("stage")
        for stage_a, stage_b in stage_pairs:
            if stage_a not in stage_rows.index or stage_b not in stage_rows.index:
                continue
            for metric in ANALYSIS_METRICS:
                if metric not in stage_rows:
                    continue
                value_a = finite_float(stage_rows.loc[stage_a, metric])
                value_b = finite_float(stage_rows.loc[stage_b, metric])
                if not np.isfinite(value_a) or not np.isfinite(value_b):
                    continue
                rows.append(
                    {
                        "scanner": scanner,
                        "physical_sample_id": specimen,
                        "group": group,
                        "stage_A": stage_a,
                        "stage_B": stage_b,
                        "stage_pair": f"{stage_a} -> {stage_b}",
                        "metric": metric,
                        "value_A": value_a,
                        "value_B": value_b,
                        "difference_B_minus_A": value_b - value_a,
                    }
                )
    return pd.DataFrame(rows)


def stage_tests(stage_changes: pd.DataFrame) -> pd.DataFrame:
    if stage_changes.empty:
        return pd.DataFrame()
    rows = []
    for (scanner, stage_pair, metric), subset in stage_changes.groupby(
        ["scanner", "stage_pair", "metric"], observed=True
    ):
        differences = subset["difference_B_minus_A"].dropna().to_numpy()
        if len(differences) < 3:
            continue
        if np.allclose(differences, 0.0, rtol=0.0, atol=1e-15):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(
                differences, zero_method="wilcox", alternative="two-sided"
            )
        rows.append(
            {
                "scanner": scanner,
                "stage_pair": stage_pair,
                "metric": metric,
                "n_pairs": len(differences),
                "median_difference": float(np.median(differences)),
                "q25_difference": float(np.percentile(differences, 25)),
                "q75_difference": float(np.percentile(differences, 75)),
                "wilcoxon_W": statistic,
                "p_value": p_value,
                "rank_biserial": rank_biserial_from_differences(differences),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["p_holm_within_metric"] = np.nan
    for metric, indices in result.groupby("metric").groups.items():
        index_list = list(indices)
        result.loc[index_list, "p_holm_within_metric"] = holm_adjust(
            result.loc[index_list, "p_value"].to_numpy()
        )
    return result


# ============================================================================
# FIGURES AND PCA
# ============================================================================

def save_scanner_overview(data: pd.DataFrame, output: Path) -> None:
    metrics = [metric for metric in PLOT_METRICS if metric in data]
    n_columns = 3
    n_rows = math.ceil(len(metrics) / n_columns)
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(15, 4.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, metric in zip(axes, metrics):
        groups = [
            pd.to_numeric(
                data.loc[data["scanner"] == scanner, metric], errors="coerce"
            ).dropna()
            for scanner in SCANNER_ORDER
        ]
        axis.boxplot(groups, tick_labels=SCANNER_ORDER, showfliers=False)
        axis.set_title(FEATURE_LABELS.get(metric, metric))
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)

    for axis in axes[len(metrics) :]:
        axis.axis("off")
    figure.suptitle("Registration-independent ROI geometry by scanner", fontsize=16)
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_qc_figure(data: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("projected_area_coverage", "Core projected-area coverage"),
        ("grid_coverage", "Standardized-grid coverage"),
        ("plane_tilt_deg", "Residual plane tilt (°)"),
    ]
    for axis, (metric, title) in zip(axes, metrics):
        groups = [
            pd.to_numeric(
                data.loc[data["scanner"] == scanner, metric], errors="coerce"
            ).dropna()
            for scanner in SCANNER_ORDER
        ]
        axis.boxplot(groups, tick_labels=SCANNER_ORDER, showfliers=True)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_correlation_figure(data: pd.DataFrame, output: Path) -> None:
    metrics = [metric for metric in ANALYSIS_METRICS if metric in data]
    correlation = data[metrics].corr(method="spearman")
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [FEATURE_LABELS.get(metric, metric) for metric in metrics]
    axis.set_xticks(range(len(metrics)), labels=labels, rotation=90, fontsize=8)
    axis.set_yticks(range(len(metrics)), labels=labels, fontsize=8)
    axis.set_title("Spearman correlations between geometric features")
    figure.colorbar(image, ax=axis, label="Spearman rho", shrink=0.8)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_pca(
    data: pd.DataFrame,
    table_root: Path,
    figure_root: Path,
) -> None:
    metrics = [metric for metric in ANALYSIS_METRICS if metric in data]
    complete = data.dropna(subset=metrics).copy()
    if len(complete) < max(5, len(SCANNER_ORDER)):
        return

    scaled = StandardScaler().fit_transform(complete[metrics])
    pca = PCA(n_components=min(len(metrics), len(complete)))
    scores = pca.fit_transform(scaled)

    score_table = complete[
        ["scanner", "sample_id", "physical_sample_id", "group", "stage"]
    ].reset_index(drop=True)
    score_table["PC1"] = scores[:, 0]
    score_table["PC2"] = scores[:, 1] if scores.shape[1] > 1 else 0.0
    score_table.to_csv(table_root / "geometry_pca_scores.csv", index=False)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=metrics,
        columns=[f"PC{i + 1}" for i in range(pca.components_.shape[0])],
    )
    loadings.insert(0, "metric", loadings.index)
    loadings.reset_index(drop=True).to_csv(
        table_root / "geometry_pca_loadings.csv", index=False
    )

    variance = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    variance.to_csv(table_root / "geometry_pca_variance.csv", index=False)

    figure, axis = plt.subplots(figsize=(8, 6.5))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for scanner, color in zip(SCANNER_ORDER, colors):
        mask = score_table["scanner"].astype(str) == scanner
        axis.scatter(
            score_table.loc[mask, "PC1"],
            score_table.loc[mask, "PC2"],
            label=scanner,
            color=color,
            alpha=0.75,
            s=34,
        )
    axis.axhline(0, color="gray", linewidth=0.8)
    axis.axvline(0, color="gray", linewidth=0.8)
    axis.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)")
    pc2_variance = pca.explained_variance_ratio_[1] if len(pca.explained_variance_ratio_) > 1 else 0
    axis.set_ylabel(f"PC2 ({100 * pc2_variance:.1f}%)")
    axis.set_title("PCA of registration-independent geometric features")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(figure_root / "geometry_pca_by_scanner.png", dpi=220)
    plt.close(figure)


def save_stage_change_figure(stage_changes: pd.DataFrame, output: Path) -> None:
    if stage_changes.empty:
        return
    selected = [
        "roi_vertices_per_nominal_mm2",
        "median_unique_edge_length_mm",
        "native_Sq_mm",
        "grid_Sq_mm",
    ]
    subset = stage_changes[
        (stage_changes["stage_pair"] == "Baseline -> Clean")
        & stage_changes["metric"].isin(selected)
    ]
    if subset.empty:
        return

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, metric in zip(axes.ravel(), selected):
        metric_data = subset[subset["metric"] == metric]
        groups = [
            metric_data.loc[
                metric_data["scanner"].astype(str) == scanner,
                "difference_B_minus_A",
            ].dropna()
            for scanner in SCANNER_ORDER
        ]
        axis.boxplot(groups, tick_labels=SCANNER_ORDER, showfliers=True)
        axis.axhline(0, color="black", linewidth=0.9)
        axis.set_title(FEATURE_LABELS.get(metric, metric))
        axis.set_ylabel("Clean minus baseline")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Global stage-associated geometry changes", fontsize=15)
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main() -> None:
    args = parse_args()
    data_root = args.data_root
    processed_root = data_root / "_processed_ROI"
    manifest_path = args.manifest or processed_root / "manifest.csv"
    output_root = args.output or data_root / "_geometry_analysis"
    table_root = output_root / "tables"
    figure_root = output_root / "figures"
    example_root = figure_root / "processing_examples"
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    example_root.mkdir(parents=True, exist_ok=True)
    write_feature_guide(table_root, output_root)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    required = ["scanner", "group", "timepoint", "sample_number", "sample_id"]
    missing = [column for column in required if column not in manifest]
    if missing:
        raise ValueError("Manifest is missing columns: " + ", ".join(missing))

    print("=" * 80)
    print("REGISTRATION-INDEPENDENT ROI GEOMETRY ANALYSIS")
    print("=" * 80)
    print(f"Manifest: {manifest_path}")
    print(f"Rows: {len(manifest)}")

    example_sample_number = choose_group3_example_sample(manifest)
    if example_sample_number is None:
        print("Group 3 example: unavailable (no usable G3 T0/T1/T2 rows)")
    else:
        print(
            "Group 3 example: "
            f"BCG3-{example_sample_number}, matched T0/T1/T2 where available"
        )

    source_ply_index = build_source_ply_index(data_root)
    print(f"Rediscovered raw source PLYs: {len(source_ply_index)}")

    # Resolve the already-extracted ROI and its saved radius. No second radial
    # clipping is performed by this analysis.
    prepared_rows = []
    radii = []
    for _, row in manifest.iterrows():
        roi_path = resolve_output_file(
            row.get("roi_vtp"), processed_root, row, "roi.vtp"
        )
        geometry_path = resolve_output_file(
            row.get("geometry_npz"), processed_root, row, "geometry.npz"
        )
        radius = load_roi_radius(geometry_path)
        prepared_rows.append((row, roi_path, geometry_path, radius))
        if roi_path.is_file() and np.isfinite(radius) and radius > 0:
            radii.append(radius)

    if not radii:
        raise RuntimeError(
            "No valid ROI radii were found. Check the geometry.npz paths."
        )

    print("ROI use: every triangle already present in each roi.vtp")
    print(
        f"Saved ROI-radius range: {np.min(radii):.4f}–{np.max(radii):.4f} mm"
    )
    print(f"Grid spacing: {args.grid_spacing_mm:.3f} mm")

    feature_rows = []
    qc_rows = []
    radial_rows = []
    group3_stage_records: dict[
        str, dict[int, tuple[dict[str, object], dict[str, float], str]]
    ] = {}

    for index, (row, roi_path, geometry_path, saved_radius) in enumerate(
        prepared_rows, start=1
    ):
        scanner = str(row.get("scanner", ""))
        sample_id_value = str(row.get("sample_id", ""))
        time_number = parse_timepoint_number(row.get("timepoint"), sample_id_value)
        specimen_id = physical_sample_id(row.get("group"), row.get("sample_number"))
        stage = biological_stage(row.get("group"), time_number)
        source_ply = resolve_recorded_file(row.get("source_ply"))
        if source_ply is None:
            source_ply = path_stored_in_geometry(geometry_path, "source_ply")
        if source_ply is None:
            source_ply = source_ply_index.get(
                (scanner.casefold(), sample_id_value.upper())
            )
        texture_path = resolve_texture_file(row.get("source_texture"))
        if texture_path is None:
            texture_path = path_stored_in_geometry(geometry_path, "source_texture")
        if texture_path is None:
            texture_path = find_texture_for_source(source_ply, sample_id_value)
        sample_number_value = finite_float(row.get("sample_number"))
        group_value = finite_float(row.get("group"))
        is_group3_example = (
            example_sample_number is not None
            and np.isfinite(group_value)
            and int(group_value) == EXAMPLE_GROUP
            and np.isfinite(sample_number_value)
            and int(sample_number_value) == example_sample_number
            and np.isfinite(time_number)
            and int(time_number) in EXAMPLE_TIMEPOINTS
            and scanner in SCANNER_ORDER
        )

        metadata = {
            "scanner": scanner,
            "group": finite_float(row.get("group")),
            "timepoint": row.get("timepoint"),
            "timepoint_number": time_number,
            "sample_number": finite_float(row.get("sample_number")),
            "sample_id": sample_id_value,
            "physical_sample_id": specimen_id,
            "stage": stage,
            "manifest_status": row.get("status", ""),
            "roi_vtp": str(roi_path),
            "geometry_npz": str(geometry_path),
            "source_ply": str(source_ply) if source_ply is not None else "",
            "source_texture": str(texture_path) if texture_path is not None else "",
            "saved_roi_radius_mm": saved_radius,
        }
        qc = {
            **metadata,
            "registration_method": row.get("registration_method", ""),
            "global_fitness": finite_float(row.get("global_fitness")),
            "global_rmse": finite_float(row.get("global_rmse")),
            "full_fitness": finite_float(row.get("full_fitness")),
            "full_rmse": finite_float(row.get("full_rmse")),
            "holder_fitness": finite_float(row.get("holder_fitness")),
            "holder_rmse": finite_float(row.get("holder_rmse")),
            "preprocessing_roi_points": finite_float(row.get("roi_points")),
            "preprocessing_roi_area_coverage": finite_float(
                row.get("roi_area_coverage")
            ),
            "preprocessing_roi_edge_reach": finite_float(row.get("roi_edge_reach")),
            "preprocessing_roi_z_span": finite_float(row.get("roi_z_span")),
            "analysis_ok": False,
            "analysis_qc_reasons": "",
            "analysis_error": "",
        }

        print(f"[{index:03d}/{len(prepared_rows):03d}] {scanner} | {sample_id_value}")

        reasons = []
        if str(row.get("status", "")).upper() == "FAILED":
            reasons.append("preprocessing_failed")
        if not roi_path.is_file():
            reasons.append("roi_file_missing")
        if not geometry_path.is_file():
            reasons.append("geometry_file_missing")
        if not np.isfinite(saved_radius) or saved_radius <= 0:
            reasons.append("saved_roi_radius_missing_or_invalid")

        if reasons:
            qc["analysis_qc_reasons"] = ";".join(reasons)
            qc_rows.append(qc)
            continue

        try:
            want_diagnostic = args.save_all_diagnostics or is_group3_example
            features, rings, diagnostic = analyse_one_roi(
                roi_path,
                saved_radius,
                args.grid_spacing_mm,
                texture_path=texture_path,
                source_ply=source_ply,
                geometry_path=geometry_path,
                return_diagnostic=want_diagnostic,
            )
            qc["analysis_roi_points"] = features[
                "roi_vertices_referenced_by_faces"
            ]
            qc["analysis_roi_faces"] = features["roi_faces"]
            qc["analysis_projected_area_coverage"] = features[
                "projected_area_coverage"
            ]
            qc["analysis_convex_hull_coverage"] = features[
                "convex_hull_area_coverage"
            ]
            qc["analysis_grid_coverage"] = features["grid_coverage"]
            qc["analysis_plane_tilt_deg"] = features["plane_tilt_deg"]

            if features["roi_vertices_referenced_by_faces"] < MIN_ROI_POINTS:
                reasons.append("too_few_roi_points")
            if features["roi_faces"] < MIN_ROI_TRIANGLES:
                reasons.append("too_few_roi_triangles")
            if features["projected_area_coverage"] < MIN_PROJECTED_AREA_COVERAGE:
                reasons.append("low_projected_area_coverage")
            if features["grid_coverage"] < MIN_GRID_COVERAGE:
                reasons.append("low_grid_coverage")
            if str(row.get("status", "")).upper() == "CHECK":
                reasons.append("preprocessing_status_check")

            qc["analysis_ok"] = len(reasons) == 0
            qc["analysis_qc_reasons"] = ";".join(reasons)
            feature_rows.append({**metadata, **features, "analysis_ok": qc["analysis_ok"]})
            for ring in rings:
                radial_rows.append({**metadata, **ring, "analysis_ok": qc["analysis_ok"]})

            if qc["analysis_ok"] and diagnostic is not None:
                example_name = (
                    f"{safe_filename(scanner)}_"
                    f"{safe_filename(sample_id_value)}_geometry_processing.png"
                )
                if is_group3_example:
                    diagnostic_directory = example_root / "G3_matched_stages"
                else:
                    diagnostic_directory = example_root / "all_rois"
                save_processing_example(
                    diagnostic=diagnostic,
                    features=features,
                    scanner=scanner,
                    sample_id=sample_id_value,
                    saved_roi_radius_mm=saved_radius,
                    output=diagnostic_directory / example_name,
                )
                if is_group3_example:
                    group3_stage_records.setdefault(scanner, {})[
                        int(time_number)
                    ] = (diagnostic, features, sample_id_value)
        except Exception as exc:
            qc["analysis_error"] = f"{type(exc).__name__}: {exc}"
            qc["analysis_qc_reasons"] = "analysis_exception"
            print(f"  FAILED: {exc}")

        qc_rows.append(qc)

    if example_sample_number is not None:
        stage_example_root = figure_root / "G3_examples"
        for scanner in SCANNER_ORDER:
            records = group3_stage_records.get(scanner, {})
            if not records:
                print(f"WARNING: no accepted G3 example rows for {scanner}")
                continue
            missing_stages = sorted(set(EXAMPLE_TIMEPOINTS) - set(records))
            if missing_stages:
                print(
                    f"WARNING: {scanner} G3 example is missing "
                    + ", ".join(f"T{value}" for value in missing_stages)
                )
            scanner_code = SCANNER_FILE_CODES.get(
                scanner, safe_filename(scanner)[:8]
            )
            try:
                save_group3_stage_comparison(
                    records,
                    scanner,
                    example_sample_number,
                    stage_example_root
                    / f"{scanner_code}_G3-{example_sample_number}_stages.png",
                )
            except OSError as exc:
                print(
                    f"WARNING: the {scanner} G3 stage figure could not be saved: "
                    f"{exc}. The remaining analysis will continue."
                )

    features_df = pd.DataFrame(feature_rows)
    qc_df = pd.DataFrame(qc_rows)
    radial_df = pd.DataFrame(radial_rows)

    features_df.to_csv(table_root / "roi_geometry_features.csv", index=False)
    qc_df.to_csv(table_root / "roi_geometry_qc.csv", index=False)
    radial_df.to_csv(table_root / "roi_radial_geometry_features.csv", index=False)

    if features_df.empty:
        raise RuntimeError(
            "No ROI could be analysed. Inspect tables/roi_geometry_qc.csv."
        )

    # Only fully accepted rows enter comparisons; rejected rows remain available
    # for transparent sensitivity checks in roi_geometry_features.csv.
    accepted = features_df[features_df["analysis_ok"]].copy()
    accepted["scanner"] = pd.Categorical(
        accepted["scanner"], categories=SCANNER_ORDER, ordered=True
    )

    scanner_summary = scanner_descriptive_summary(accepted)
    pairwise, friedman = scanner_matched_tests(accepted)
    stage_change_df = create_stage_changes(accepted)
    stage_test_df = stage_tests(stage_change_df)

    scanner_summary.to_csv(table_root / "scanner_geometry_summary.csv", index=False)
    pairwise.to_csv(
        table_root / "scanner_pairwise_geometry_tests.csv", index=False
    )
    friedman.to_csv(table_root / "scanner_friedman_geometry_tests.csv", index=False)
    stage_change_df.to_csv(
        table_root / "individual_stage_geometry_changes.csv", index=False
    )
    stage_test_df.to_csv(table_root / "stage_geometry_tests.csv", index=False)

    if not accepted.empty:
        save_scanner_overview(
            accepted, figure_root / "scanner_geometry_overview.png"
        )
        save_qc_figure(features_df, figure_root / "geometry_qc_overview.png")
        save_correlation_figure(
            accepted, figure_root / "geometry_feature_correlation.png"
        )
        run_pca(accepted, table_root, figure_root)
        save_stage_change_figure(
            stage_change_df, figure_root / "baseline_to_clean_geometry_changes.png"
        )

    notes = f"""BIOCAL3D REGISTRATION-INDEPENDENT GEOMETRY ANALYSIS

ROI selection: every triangle already present in each extracted roi.vtp
Saved ROI-radius range: {np.min(radii):.6f}–{np.max(radii):.6f} mm
Standardized-grid spacing: {args.grid_spacing_mm:.6f} mm

Interpretation rules:
- The manually imposed ROI radius, diameter, perimeter and circularity are not outcomes.
- No second inner radius or radial clipping is applied in this script.
- Boundary topology is not interpreted because preprocessing created the ROI edge.
- A separate robust plane is fitted to every ROI. The method is the same for all
  scans, but the plane coefficients are scan-specific.
- Native-mesh metrics describe scanner reconstruction and tessellation.
- Grid metrics compare surfaces after standardizing spatial sampling.
- Stage results are global stage-associated differences, not pointwise plaque loss.
- Vertices and pixels are not independent observations; the scan/specimen is the
  statistical unit.
- No result from this script should be described as trueness or accuracy without an
  independent geometric reference.

Rows in manifest: {len(manifest)}
Successfully computed feature rows: {len(features_df)}
Rows passing all analysis QC: {len(accepted)}
Rows failing or flagged by analysis QC: {len(qc_df) - len(accepted)}
Processing examples saved: {len(list(example_root.rglob('*.png')))}
Selected Group 3 example specimen: {example_sample_number}
Matched G3 stage comparisons saved: {len(list((figure_root / 'G3_examples').glob('*.png'))) if (figure_root / 'G3_examples').is_dir() else 0}
"""
    (output_root / "README_results.txt").write_text(notes, encoding="utf-8")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Feature rows computed: {len(features_df)}")
    print(f"Rows passing QC:       {len(accepted)}")
    print(f"Output: {output_root}")
    print("\nStart with:")
    print(f"  {table_root / 'roi_geometry_qc.csv'}")
    print(f"  {table_root / 'scanner_geometry_summary.csv'}")
    print(f"  {figure_root / 'scanner_geometry_overview.png'}")
    print(f"  {example_root}")
    print(f"  {output_root / 'GEOMETRY_FEATURE_GUIDE.md'}")


if __name__ == "__main__":
    main()
