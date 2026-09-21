from pathlib import Path
from collections import defaultdict
import csv
import math
import re

import numpy as np
import pyvista as pv
import open3d as o3d

from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull, QhullError


# ============================================================
# DATA ROOT
# ============================================================

DATA_ROOT = Path(
    r"C:\Users\johan\repos\biocal3d\data"
    r"\02-09-2026-Data collected"
)


# ============================================================
# SCANNER-SPECIFIC REFERENCES
# ============================================================
#
# Each scanner is registered ONLY to a reference from the
# same scanner.
#
# Change these if another scan is a better reference.
# ============================================================

REFERENCE_SAMPLE_BY_SCANNER = {
    "LABscanner": "BCG1T0-1",
    "iTERO": "BCG1T0-1",
    "TRIOS3": "BCG1T0-1",
    "TRIOS5": "BCG1T0-1",
}


# ============================================================
# REDEFINE REFERENCES
# ============================================================
#
# FIRST RUN / if references should be redefined:
#
# REDEFINE_REFERENCE_SCANNERS = {
#     "LABscanner",
#     "iTERO",
#     "TRIOS3",
#     "TRIOS5",
# }
#
# Once correct references are already saved:
#
# REDEFINE_REFERENCE_SCANNERS = set()
#
# ============================================================

REDEFINE_REFERENCE_SCANNERS = set()


# ============================================================
# REFERENCE ROI
# ============================================================

MIN_REFERENCE_EDGE_POINTS = 6

# Slight shrink relative to manually fitted physical boundary.
ROI_RADIUS_FACTOR = 0.985


# ============================================================
# SPECIMEN SURFACE / ROI
# ============================================================

SPECIMEN_HEIGHT_TOLERANCE = 1.5


# ============================================================
# ROBUST SPECIMEN Z DETECTION
# ============================================================

Z_MODE_SEARCH_RANGE = 8.0

Z_MODE_BIN_WIDTH = 0.20

Z_MODE_LOCAL_HALF_WIDTH = 0.50

Z_MODE_PRIOR_SIGMA = 3.0


# ============================================================
# ROI QUALITY
# ============================================================
#
# Do NOT use 500 points as an absolute criterion.
#
# TRIOS has substantially fewer points per specimen.
# ============================================================

MIN_ROI_POINTS_ABSOLUTE = 100

# Convex hull area / expected circular area.
MIN_ROI_AREA_COVERAGE = 0.65

# 98th percentile radius / expected ROI radius.
MIN_ROI_EDGE_REACH = 0.82


# ============================================================
# REGISTRATION
# ============================================================

VOXEL_SIZE = 0.7

FULL_ICP_DISTANCE = 1.2

HOLDER_ICP_DISTANCE_1 = 0.8
HOLDER_ICP_DISTANCE_2 = 0.35

MIN_FGR_FITNESS = 0.05


# ============================================================
# HOLDER REGION
# ============================================================

HOLDER_HALF_WIDTH_FACTOR = 1.60

HOLDER_EXCLUSION_FACTOR = 1.02

HOLDER_Z_RANGE = 8.0

TARGET_HOLDER_SEARCH_FACTOR = 1.90


# ============================================================
# REGISTRATION QC
# ============================================================

MIN_FULL_FITNESS = 0.15

MIN_HOLDER_FITNESS = 0.25

MAX_HOLDER_RMSE = 0.70


# ============================================================
# TRIOS FALLBACK
# ============================================================
#
# TRIOS can be harder to register globally because the dental
# model around the holder changes considerably.
#
# If normal registration is weak, embedded RGB is used ONLY
# as a rough localization cue.
#
# Final ROI is still determined geometrically from the
# reference.
# ============================================================

TRIOS_SCANNERS = {
    "TRIOS3",
    "TRIOS5",
}

ENABLE_TRIOS_RGB_FALLBACK = True


# Primary registration considered strong if:
TRIOS_PRIMARY_MIN_HOLDER_FITNESS = 0.35
TRIOS_PRIMARY_MAX_HOLDER_RMSE = 0.65


# RGB candidate thresholds
RGB_MIN_RED = 90
RGB_MIN_RED_MINUS_GREEN = 15
RGB_MIN_RED_MINUS_BLUE = 5


# Clustering
RGB_CLUSTER_VOXEL = 0.40

RGB_CLUSTER_EPS_FACTOR = 0.25

RGB_CLUSTER_MIN_POINTS = 15

RGB_MAX_CLUSTERS = 4


# Wide -> medium -> fine holder ICP for fallback
TRIOS_HOLDER_ICP_WIDE = 2.5
TRIOS_HOLDER_ICP_MEDIUM = 1.0
TRIOS_HOLDER_ICP_FINE = 0.35


# Circular specimen cannot resolve rotation around its own
# normal, so try several rotations and let holder geometry
# select the best one.
TRIOS_INPLANE_ANGLES_DEG = [
    0,
    15,
    30,
    45,
    60,
    75,
]


# ============================================================
# OPTIONAL PROCESSING FILTERS
# ============================================================
#
# Examples:
#
# SCANNERS_TO_PROCESS = {"TRIOS3", "TRIOS5"}
# GROUPS_TO_PROCESS = {1}
# MAX_FILES = 20
#
# None = everything.
# ============================================================

SCANNERS_TO_PROCESS = None

GROUPS_TO_PROCESS = None

MAX_FILES = None


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_ROOT = DATA_ROOT / "_processed_ROI"

REFERENCE_ROOT = OUTPUT_ROOT / "_references"

CONTACT_SHEET_ROOT = OUTPUT_ROOT / "_contact_sheets"

MANIFEST_FILE = OUTPUT_ROOT / "manifest.csv"


for folder in [
    OUTPUT_ROOT,
    REFERENCE_ROOT,
    CONTACT_SHEET_ROOT,
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# QC PREVIEW
# ============================================================

PREVIEW_WIDTH = 900
PREVIEW_HEIGHT = 450

CONTEXT_SIZE_FACTOR = 1.40
CONTEXT_HEIGHT_TOLERANCE = 4.0


# ============================================================
# CONTACT SHEETS
# ============================================================

CONTACT_SHEET_COLUMNS = 4
CONTACT_SHEET_ROWS = 4

CONTACT_SHEET_TILE_WIDTH = 920
CONTACT_SHEET_TILE_HEIGHT = 505


# ============================================================
# METADATA REGEX
# ============================================================

SAMPLE_PATTERN = re.compile(
    r"BCG(?P<group>\d+)T(?P<timepoint>\d+)-(?P<sample>\d+)",
    re.IGNORECASE,
)

GROUP_PATTERN = re.compile(
    r"^G(?P<group>\d+)",
    re.IGNORECASE,
)


np.random.seed(42)


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_name(text):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(text),
    )


def get_template_file(scanner):

    folder = (
        REFERENCE_ROOT
        / safe_name(scanner)
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    return folder / "reference_roi.npz"


# ============================================================
# EMBEDDED RGB
# ============================================================

def prepare_embedded_rgb(mesh):
    """
    Detect point-level RGB/RGBA.

    Preferred visualization source whenever available.

    Known:
        TRIOS3 -> RGB
        TRIOS5 -> RGB
        iTERO  -> RGB after PyVista loading
    """

    preferred_names = [
        "RGB",
        "rgb",
        "RGBA",
        "rgba",
        "Colors",
        "colors",
        "Color",
        "color",
    ]

    for name in preferred_names:

        if name not in mesh.point_data:
            continue

        array = np.asarray(
            mesh.point_data[name]
        )

        if not (
            array.ndim == 2
            and array.shape[0] == mesh.n_points
            and array.shape[1] in (3, 4)
        ):
            continue

        if np.issubdtype(
            array.dtype,
            np.floating,
        ):

            converted = array.copy()

            if np.nanmax(converted) <= 1.0:
                converted *= 255.0

            converted = np.clip(
                converted,
                0,
                255,
            ).astype(np.uint8)

            mesh.point_data["RGB"] = converted

            return mesh, "RGB"

        return mesh, name


    # Separate R/G/B arrays
    triplets = [
        ("red", "green", "blue"),
        ("Red", "Green", "Blue"),
        ("RED", "GREEN", "BLUE"),
    ]

    keys = set(mesh.point_data.keys())

    for r_name, g_name, b_name in triplets:

        if not all(
            name in keys
            for name in (
                r_name,
                g_name,
                b_name,
            )
        ):
            continue

        rgb = np.column_stack(
            [
                mesh.point_data[r_name],
                mesh.point_data[g_name],
                mesh.point_data[b_name],
            ]
        )

        if np.issubdtype(
            rgb.dtype,
            np.floating,
        ):

            if np.nanmax(rgb) <= 1.0:
                rgb *= 255.0

        rgb = np.clip(
            rgb,
            0,
            255,
        ).astype(np.uint8)

        mesh.point_data["RGB"] = rgb

        return mesh, "RGB"

    return mesh, None


def get_rgb_array_name(mesh):

    preferred_names = [
        "RGB",
        "rgb",
        "RGBA",
        "rgba",
        "Colors",
        "colors",
        "Color",
        "color",
    ]

    for name in preferred_names:

        if name not in mesh.point_data:
            continue

        array = np.asarray(
            mesh.point_data[name]
        )

        if (
            array.ndim == 2
            and array.shape[0] == mesh.n_points
            and array.shape[1] in (3, 4)
        ):
            return name

    return None


# ============================================================
# UV / EXTERNAL TEXTURE
# ============================================================

def activate_tcoords(mesh):

    possible_names = [
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
        "texcoord",
        "texcoords",
    ]

    # Deliberately no arbitrary Nx2 fallback.
    for name in possible_names:

        if name not in mesh.point_data:
            continue

        array = np.asarray(
            mesh.point_data[name]
        )

        if not (
            array.ndim == 2
            and array.shape[0] == mesh.n_points
            and array.shape[1] == 2
        ):
            continue

        vtk_array = (
            mesh.GetPointData()
            .GetArray(name)
        )

        if vtk_array is not None:

            mesh.GetPointData().SetTCoords(
                vtk_array
            )

            return mesh, name

    return mesh, None


def has_active_tcoords(mesh):

    return (
        mesh.GetPointData()
        .GetTCoords()
        is not None
    )


# ============================================================
# LOAD MESH
# ============================================================

def load_mesh(path):

    mesh = pv.read(path)

    mesh = mesh.extract_surface()
    mesh = mesh.triangulate()
    mesh = mesh.clean()

    # RGB first
    mesh, rgb_name = prepare_embedded_rgb(
        mesh
    )

    # Keep UV too
    mesh, uv_name = activate_tcoords(
        mesh
    )

    return (
        mesh,
        rgb_name,
        uv_name,
    )


# ============================================================
# FIND EXTERNAL TEXTURE
# ============================================================

def find_texture(
    ply_path,
    sample_id=None,
):

    folder = ply_path.parent

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    images = sorted(
        [
            path
            for path in folder.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in extensions
            )
        ]
    )

    if not images:
        return None

    ply_stem = ply_path.stem.lower()

    # Exact stem
    for image in images:

        if image.stem.lower() == ply_stem:
            return image


    # stem_texture
    expected_names = {
        ply_stem + "_texture",
        ply_stem + "-texture",
        ply_stem + "texture",
    }

    for image in images:

        if image.stem.lower() in expected_names:
            return image


    # Sample ID
    if sample_id is not None:

        sample_lower = sample_id.lower()

        matches = [
            image
            for image in images
            if sample_lower
            in image.stem.lower()
        ]

        if len(matches) == 1:
            return matches[0]

        texture_matches = [
            image
            for image in matches
            if "texture"
            in image.stem.lower()
        ]

        if len(texture_matches) == 1:
            return texture_matches[0]


    # Dedicated sample folder
    if len(images) == 1:
        return images[0]

    return None


def load_texture(path):

    if path is None:
        return None

    try:
        return pv.read_texture(path)

    except Exception as exc:

        print(
            f"WARNING: texture failed: "
            f"{path}\n{exc}"
        )

        return None


# ============================================================
# VISUALIZATION
# ============================================================

def add_scan_mesh(
    plotter,
    mesh,
    texture=None,
):
    """
    Priority:

    1. embedded RGB
    2. external JPG + UV
    3. gray

    This avoids scrambled iTERO G1 texture mapping.
    """

    rgb_name = get_rgb_array_name(
        mesh
    )

    if rgb_name is not None:

        plotter.add_mesh(
            mesh,
            scalars=rgb_name,
            rgb=True,
            preference="point",
            lighting=False,
        )

        return


    if (
        texture is not None
        and has_active_tcoords(mesh)
    ):

        plotter.add_mesh(
            mesh,
            texture=texture,
            lighting=False,
        )

        return


    plotter.add_mesh(
        mesh,
        color="lightgray",
        smooth_shading=True,
    )


# ============================================================
# DATASET METADATA
# ============================================================

def extract_scanner(path):

    relative = path.relative_to(
        DATA_ROOT
    )

    if len(relative.parts) < 2:
        return "UNKNOWN"

    return relative.parts[0]


def extract_sample_match(path):

    match = SAMPLE_PATTERN.search(
        path.stem
    )

    if match is not None:
        return match

    for parent in path.parents:

        if parent == DATA_ROOT:
            break

        match = SAMPLE_PATTERN.search(
            parent.name
        )

        if match is not None:
            return match

    return None


def extract_group_folder(path):

    relative = path.relative_to(
        DATA_ROOT
    )

    for part in relative.parts:

        match = GROUP_PATTERN.match(
            part
        )

        if match is not None:

            return (
                int(
                    match.group("group")
                ),
                part,
            )

    return None, None


# ============================================================
# DISCOVER DATASET
# ============================================================

def discover_dataset():

    excluded_top_level = {
        "_processed_ROI",
        "_loading_QC",
        "_loading_QC_v2",
    }

    ply_files = []

    for path in DATA_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() != ".ply":
            continue

        relative = path.relative_to(
            DATA_ROOT
        )

        if (
            len(relative.parts) > 0
            and relative.parts[0]
            in excluded_top_level
        ):
            continue

        ply_files.append(path)


    records = []

    for path in ply_files:

        scanner = extract_scanner(
            path
        )

        (
            folder_group,
            group_folder,
        ) = extract_group_folder(
            path
        )

        match = extract_sample_match(
            path
        )

        if match is not None:

            group = int(
                match.group("group")
            )

            time_number = int(
                match.group("timepoint")
            )

            sample_number = int(
                match.group("sample")
            )

            timepoint = (
                f"T{time_number}"
            )

            sample_id = (
                f"BCG{group}"
                f"T{time_number}"
                f"-{sample_number}"
            )

        else:

            group = folder_group
            time_number = None
            sample_number = None
            timepoint = None
            sample_id = path.stem


        texture_path = find_texture(
            path,
            sample_id,
        )


        group_output = (
            f"G{group}"
            if group is not None
            else "G_UNKNOWN"
        )


        output_dir = (
            OUTPUT_ROOT
            / scanner
            / group_output
            / sample_id
        )


        records.append(
            {
                "scanner": scanner,
                "group": group,
                "group_folder": group_folder,
                "timepoint": timepoint,
                "timepoint_number": time_number,
                "sample_number": sample_number,
                "sample_id": sample_id,
                "ply_path": path,
                "texture_path": texture_path,
                "output_dir": output_dir,
            }
        )


    records.sort(
        key=lambda r: (
            r["scanner"].lower(),

            (
                r["group"]
                if r["group"]
                is not None
                else 999
            ),

            (
                r["timepoint_number"]
                if r["timepoint_number"]
                is not None
                else 999
            ),

            (
                r["sample_number"]
                if r["sample_number"]
                is not None
                else 999
            ),
        )
    )

    return records


# ============================================================
# GEOMETRY
# ============================================================

def transform_points(
    points,
    matrix,
):

    homogeneous = np.column_stack(
        [
            points,
            np.ones(len(points)),
        ]
    )

    return (
        matrix
        @ homogeneous.T
    ).T[:, :3]


def points_to_o3d(points):

    cloud = o3d.geometry.PointCloud()

    cloud.points = (
        o3d.utility.Vector3dVector(
            np.asarray(points)
        )
    )

    return cloud


def fit_plane(points):

    center = np.mean(
        points,
        axis=0,
    )

    centered = (
        points
        - center
    )

    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    normal = vh[-1]

    normal /= np.linalg.norm(
        normal
    )

    return (
        center,
        normal,
    )


def create_plane_basis(normal):

    normal = (
        normal
        / np.linalg.norm(normal)
    )

    reference = np.array(
        [1.0, 0.0, 0.0]
    )

    if abs(
        np.dot(
            reference,
            normal,
        )
    ) > 0.90:

        reference = np.array(
            [0.0, 1.0, 0.0]
        )


    u_axis = np.cross(
        normal,
        reference,
    )

    u_axis /= np.linalg.norm(
        u_axis
    )


    v_axis = np.cross(
        normal,
        u_axis,
    )

    v_axis /= np.linalg.norm(
        v_axis
    )


    return (
        u_axis,
        v_axis,
    )


def fit_circle_2d(x, y):

    A = np.column_stack(
        [
            x,
            y,
            np.ones_like(x),
        ]
    )

    b = -(
        x**2
        + y**2
    )

    params, _, _, _ = (
        np.linalg.lstsq(
            A,
            b,
            rcond=None,
        )
    )

    d, e, f = params

    cx = -d / 2.0
    cy = -e / 2.0

    radius_squared = (
        cx**2
        + cy**2
        - f
    )

    if radius_squared <= 0:

        raise RuntimeError(
            "Invalid fitted circle."
        )

    return (
        cx,
        cy,
        np.sqrt(radius_squared),
    )


def local_coordinates(
    points,
    center,
    u_axis,
    v_axis,
    normal,
):

    relative = (
        points
        - center
    )

    return (
        relative @ u_axis,
        relative @ v_axis,
        relative @ normal,
    )


# ============================================================
# REFERENCE CIRCLE
# ============================================================

def fit_reference_circle(
    clicked_points,
    mesh,
):

    clicked_points = np.asarray(
        clicked_points
    )

    (
        plane_center,
        normal,
    ) = fit_plane(
        clicked_points
    )

    (
        u_axis,
        v_axis,
    ) = create_plane_basis(
        normal
    )


    normal_mesh = mesh.compute_normals(
        point_normals=True,
        cell_normals=False,
        consistent_normals=True,
        auto_orient_normals=False,
        inplace=False,
    )


    mesh_points = np.asarray(
        normal_mesh.points
    )

    mesh_normals = np.asarray(
        normal_mesh.point_data[
            "Normals"
        ]
    )


    distances = np.linalg.norm(
        mesh_points
        - plane_center,
        axis=1,
    )


    nearest = np.argsort(
        distances
    )[:200]


    mean_normal = np.mean(
        mesh_normals[
            nearest
        ],
        axis=0,
    )


    if (
        np.dot(
            normal,
            mean_normal,
        )
        < 0
    ):

        normal *= -1

        (
            u_axis,
            v_axis,
        ) = create_plane_basis(
            normal
        )


    relative = (
        clicked_points
        - plane_center
    )

    x = relative @ u_axis
    y = relative @ v_axis

    (
        cx,
        cy,
        radius,
    ) = fit_circle_2d(
        x,
        y,
    )


    center = (
        plane_center
        + cx * u_axis
        + cy * v_axis
    )


    return {
        "center": center,
        "normal": normal,
        "u_axis": u_axis,
        "v_axis": v_axis,
        "radius": radius,
    }


# ============================================================
# ROBUST SPECIMEN Z
# ============================================================

def estimate_specimen_z(
    radial,
    z,
    reference,
):
    """
    Find the dominant Z layer near the expected specimen
    center instead of requiring >=50 points within a rigid
    +/-3 mm interval.
    """

    radius = reference["radius"]
    expected_z = reference["z_offset"]


    central_mask = (
        radial
        <
        radius * 0.55
    )


    central_z = (
        z[
            central_mask
        ]
    )


    central_z = (
        central_z[
            np.isfinite(
                central_z
            )
        ]
    )


    if len(central_z) < 20:

        raise RuntimeError(
            "Too few central points to estimate specimen surface."
        )


    search_z = (
        central_z[
            np.abs(
                central_z
                - expected_z
            )
            <= Z_MODE_SEARCH_RANGE
        ]
    )


    if len(search_z) < 20:
        search_z = central_z


    z_min = float(
        np.min(search_z)
    )

    z_max = float(
        np.max(search_z)
    )


    if (
        z_max - z_min
        <
        Z_MODE_BIN_WIDTH
    ):

        return float(
            np.median(search_z)
        )


    bins = np.arange(
        z_min - Z_MODE_BIN_WIDTH,
        z_max + 2 * Z_MODE_BIN_WIDTH,
        Z_MODE_BIN_WIDTH,
    )


    histogram, edges = np.histogram(
        search_z,
        bins=bins,
    )


    centers = (
        0.5
        * (
            edges[:-1]
            + edges[1:]
        )
    )


    # Weak prior toward reference specimen height.
    prior = np.exp(
        -0.5
        * (
            (
                centers
                - expected_z
            )
            / Z_MODE_PRIOR_SIGMA
        )**2
    )


    scores = (
        histogram
        * prior
    )


    best_index = int(
        np.argmax(scores)
    )


    best_center = (
        centers[
            best_index
        ]
    )


    layer = (
        search_z[
            np.abs(
                search_z
                - best_center
            )
            <= Z_MODE_LOCAL_HALF_WIDTH
        ]
    )


    if len(layer) < 10:

        order = np.argsort(
            np.abs(
                search_z
                - best_center
            )
        )

        layer = (
            search_z[
                order[
                    :min(
                        30,
                        len(search_z),
                    )
                ]
            ]
        )


    return float(
        np.median(layer)
    )


# ============================================================
# CANONICAL TRANSFORM
# ============================================================

def make_canonical_transform(
    center,
    u_axis,
    v_axis,
    normal,
    z_offset,
):

    surface_center = (
        center
        + z_offset
        * normal
    )


    rotation = np.vstack(
        [
            u_axis,
            v_axis,
            normal,
        ]
    )


    matrix = np.eye(4)

    matrix[
        :3,
        :3
    ] = rotation

    matrix[
        :3,
        3
    ] = -(
        rotation
        @ surface_center
    )


    return matrix


# ============================================================
# ROI EXTRACTION
# ============================================================

def extract_specimen(
    mesh,
    reference,
):

    points = np.asarray(
        mesh.points
    )


    (
        x,
        y,
        z,
    ) = local_coordinates(
        points,
        reference["center"],
        reference["u_axis"],
        reference["v_axis"],
        reference["normal"],
    )


    radial = np.sqrt(
        x**2
        + y**2
    )


    z_offset = estimate_specimen_z(
        radial,
        z,
        reference,
    )


    roi_radius = (
        reference["radius"]
        * ROI_RADIUS_FACTOR
    )


    roi_mask = (
        (
            radial
            <= roi_radius
        )
        &
        (
            np.abs(
                z - z_offset
            )
            <=
            SPECIMEN_HEIGHT_TOLERANCE
        )
    )


    specimen = (
        mesh.extract_points(
            roi_mask,
            adjacent_cells=False,
            include_cells=True,
        )
        .extract_surface()
        .clean()
    )


    specimen, _ = prepare_embedded_rgb(
        specimen
    )

    specimen, _ = activate_tcoords(
        specimen
    )


    canonical = make_canonical_transform(
        reference["center"],
        reference["u_axis"],
        reference["v_axis"],
        reference["normal"],
        z_offset,
    )


    specimen.transform(
        canonical,
        inplace=True,
    )


    specimen, _ = prepare_embedded_rgb(
        specimen
    )

    specimen, _ = activate_tcoords(
        specimen
    )


    return (
        specimen,
        z_offset,
        roi_radius,
        canonical,
    )


# ============================================================
# ROI QUALITY
# ============================================================

def assess_roi_quality(
    specimen,
    roi_radius,
):

    n_points = specimen.n_points


    result = {
        "ok": False,
        "n_points": n_points,
        "area_coverage": 0.0,
        "edge_reach": 0.0,
        "z_span": np.nan,
    }


    if (
        n_points
        <
        MIN_ROI_POINTS_ABSOLUTE
    ):
        return result


    points = np.asarray(
        specimen.points
    )


    xy = points[:, :2]


    radial = np.sqrt(
        xy[:, 0]**2
        + xy[:, 1]**2
    )


    edge_reach = (
        np.percentile(
            radial,
            98,
        )
        /
        roi_radius
    )


    result[
        "edge_reach"
    ] = float(edge_reach)


    expected_area = (
        np.pi
        * roi_radius**2
    )


    try:

        hull = ConvexHull(
            xy
        )

        # For 2D ConvexHull, volume = area.
        hull_area = hull.volume

        area_coverage = (
            hull_area
            /
            expected_area
        )

    except (
        QhullError,
        ValueError,
    ):

        area_coverage = 0.0


    result[
        "area_coverage"
    ] = float(area_coverage)


    z_values = points[:, 2]

    z_span = (
        np.percentile(
            z_values,
            95,
        )
        -
        np.percentile(
            z_values,
            5,
        )
    )


    result[
        "z_span"
    ] = float(z_span)


    result[
        "ok"
    ] = bool(
        (
            n_points
            >= MIN_ROI_POINTS_ABSOLUTE
        )
        and
        (
            area_coverage
            >= MIN_ROI_AREA_COVERAGE
        )
        and
        (
            edge_reach
            >= MIN_ROI_EDGE_REACH
        )
    )


    return result


# ============================================================
# HOLDER MASK
# ============================================================

def holder_mask(
    points,
    reference,
    half_width_factor,
):

    (
        x,
        y,
        z,
    ) = local_coordinates(
        points,
        reference["center"],
        reference["u_axis"],
        reference["v_axis"],
        reference["normal"],
    )


    radial = np.sqrt(
        x**2
        + y**2
    )


    half_width = (
        reference["radius"]
        * half_width_factor
    )


    return (
        (
            np.abs(x)
            < half_width
        )
        &
        (
            np.abs(y)
            < half_width
        )
        &
        (
            np.abs(z)
            < HOLDER_Z_RANGE
        )
        &
        (
            radial
            >
            reference["radius"]
            * HOLDER_EXCLUSION_FACTOR
        )
    )


# ============================================================
# OPEN3D
# ============================================================

def estimate_normals(
    cloud,
    radius,
):

    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=50,
        )
    )


def prepare_cloud(
    points,
    voxel_size,
):

    cloud = points_to_o3d(
        points
    )


    down = (
        cloud.voxel_down_sample(
            voxel_size
        )
    )


    estimate_normals(
        down,
        voxel_size * 2.5,
    )


    feature = (
        o3d.pipelines.registration
        .compute_fpfh_feature(
            down,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=(
                    voxel_size
                    * 5.0
                ),
                max_nn=100,
            ),
        )
    )


    return (
        down,
        feature,
    )


# ============================================================
# GLOBAL REGISTRATION
# ============================================================

def global_registration(
    source_down,
    target_down,
    source_feature,
    target_feature,
):

    threshold = (
        VOXEL_SIZE
        * 1.5
    )


    # --------------------------------------------------------
    # Fast Global Registration
    # --------------------------------------------------------

    try:

        option = (
            o3d.pipelines.registration
            .FastGlobalRegistrationOption(
                maximum_correspondence_distance=(
                    threshold
                )
            )
        )


        result = (
            o3d.pipelines.registration
            .registration_fgr_based_on_feature_matching(
                source_down,
                target_down,
                source_feature,
                target_feature,
                option,
            )
        )


        if (
            result.fitness
            >=
            MIN_FGR_FITNESS
        ):

            return (
                result,
                "FGR",
            )

    except Exception:
        pass


    # --------------------------------------------------------
    # RANSAC
    # --------------------------------------------------------

    result = (
        o3d.pipelines.registration
        .registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_feature,
            target_feature,
            mutual_filter=True,
            max_correspondence_distance=threshold,
            estimation_method=(
                o3d.pipelines.registration
                .TransformationEstimationPointToPoint(
                    False
                )
            ),
            ransac_n=4,
            checkers=[
                (
                    o3d.pipelines.registration
                    .CorrespondenceCheckerBasedOnEdgeLength(
                        0.9
                    )
                ),
                (
                    o3d.pipelines.registration
                    .CorrespondenceCheckerBasedOnDistance(
                        threshold
                    )
                ),
            ],
            criteria=(
                o3d.pipelines.registration
                .RANSACConvergenceCriteria(
                    30000,
                    0.999,
                )
            ),
        )
    )


    return (
        result,
        "RANSAC",
    )


def full_icp(
    source_down,
    reference_down,
    initial_transform,
):

    return (
        o3d.pipelines.registration
        .registration_icp(
            source_down,
            reference_down,
            FULL_ICP_DISTANCE,
            initial_transform,
            o3d.pipelines.registration
            .TransformationEstimationPointToPlane(),
        )
    )


# ============================================================
# HOLDER ICP
# ============================================================

def run_holder_icp(
    aligned_points,
    reference,
    reference_holder_cloud,
    distances,
):

    mask = holder_mask(
        aligned_points,
        reference,
        TARGET_HOLDER_SEARCH_FACTOR,
    )


    target_points = (
        aligned_points[
            mask
        ]
    )


    if len(target_points) < 100:

        return (
            None,
            len(target_points),
        )


    target = points_to_o3d(
        target_points
    )


    target = (
        target.voxel_down_sample(
            0.25
        )
    )


    estimate_normals(
        target,
        1.0,
    )


    transform = np.eye(4)

    final_result = None


    for distance in distances:

        result = (
            o3d.pipelines.registration
            .registration_icp(
                target,
                reference_holder_cloud,
                distance,
                transform,
                o3d.pipelines.registration
                .TransformationEstimationPointToPlane(),
            )
        )

        transform = (
            result.transformation
        )

        final_result = result


    return (
        final_result,
        len(target_points),
    )


# ============================================================
# PRIMARY REGISTRATION
# ============================================================

def primary_registration(
    mesh,
    reference_system,
):

    reference = (
        reference_system[
            "reference"
        ]
    )


    target_points = np.asarray(
        mesh.points
    )


    (
        target_down,
        target_feature,
    ) = prepare_cloud(
        target_points,
        VOXEL_SIZE,
    )


    (
        global_result,
        global_method,
    ) = global_registration(
        target_down,
        reference_system["down"],
        target_feature,
        reference_system["feature"],
    )


    full_result = full_icp(
        target_down,
        reference_system["down"],
        global_result.transformation,
    )


    full_transform = (
        full_result.transformation
    )


    aligned_points = transform_points(
        target_points,
        full_transform,
    )


    (
        holder_result,
        holder_points,
    ) = run_holder_icp(
        aligned_points,
        reference,
        reference_system["holder_cloud"],
        [
            HOLDER_ICP_DISTANCE_1,
            HOLDER_ICP_DISTANCE_2,
        ],
    )


    if holder_result is None:
        return None


    total_transform = (
        holder_result.transformation
        @ full_transform
    )


    return {
        "transform": total_transform,
        "method": (
            global_method
            + "+HOLDER"
        ),
        "global_fitness": global_result.fitness,
        "global_rmse": global_result.inlier_rmse,
        "full_fitness": full_result.fitness,
        "full_rmse": full_result.inlier_rmse,
        "holder_fitness": holder_result.fitness,
        "holder_rmse": holder_result.inlier_rmse,
        "holder_points": holder_points,
    }


# ============================================================
# ROTATION HELPERS
# ============================================================

def rotation_align_vectors(
    source,
    target,
):

    a = np.asarray(
        source,
        dtype=float,
    )

    b = np.asarray(
        target,
        dtype=float,
    )

    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)


    v = np.cross(a, b)

    c = np.clip(
        np.dot(a, b),
        -1.0,
        1.0,
    )

    s = np.linalg.norm(v)


    if s < 1e-8:

        if c > 0:
            return np.eye(3)

        trial = np.array(
            [1.0, 0.0, 0.0]
        )

        if abs(
            np.dot(
                trial,
                a,
            )
        ) > 0.9:

            trial = np.array(
                [0.0, 1.0, 0.0]
            )

        axis = np.cross(
            a,
            trial,
        )

        axis /= np.linalg.norm(
            axis
        )

        return (
            2
            * np.outer(
                axis,
                axis,
            )
            - np.eye(3)
        )


    K = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


    return (
        np.eye(3)
        + K
        + (
            K @ K
        )
        * (
            (1 - c)
            /
            (s**2)
        )
    )


def rotation_about_axis(
    axis,
    angle_radians,
):

    axis = np.asarray(
        axis,
        dtype=float,
    )

    axis /= np.linalg.norm(
        axis
    )


    x, y, z = axis

    c = np.cos(
        angle_radians
    )

    s = np.sin(
        angle_radians
    )

    C = 1 - c


    return np.array(
        [
            [
                c + x*x*C,
                x*y*C - z*s,
                x*z*C + y*s,
            ],
            [
                y*x*C + z*s,
                c + y*y*C,
                y*z*C - x*s,
            ],
            [
                z*x*C - y*s,
                z*y*C + x*s,
                c + z*z*C,
            ],
        ]
    )


# ============================================================
# TRIOS RGB CANDIDATES
# ============================================================

def find_rgb_specimen_candidates(
    mesh,
    reference_radius,
):

    rgb_name = get_rgb_array_name(
        mesh
    )


    if rgb_name is None:
        return []


    rgb = np.asarray(
        mesh.point_data[
            rgb_name
        ]
    )[:, :3].astype(float)


    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]


    redness = (
        r
        -
        0.5
        * (
            g + b
        )
    )


    mask = (
        (r >= RGB_MIN_RED)
        &
        (
            r - g
            >=
            RGB_MIN_RED_MINUS_GREEN
        )
        &
        (
            r - b
            >=
            RGB_MIN_RED_MINUS_BLUE
        )
    )


    # Adaptive fallback
    if np.sum(mask) < 200:

        threshold = np.percentile(
            redness,
            90,
        )

        mask = (
            redness
            >= threshold
        )


    candidate_points = (
        np.asarray(
            mesh.points
        )[
            mask
        ]
    )


    if len(candidate_points) < 100:
        return []


    cloud = points_to_o3d(
        candidate_points
    )


    cloud = (
        cloud.voxel_down_sample(
            RGB_CLUSTER_VOXEL
        )
    )


    points = np.asarray(
        cloud.points
    )


    if len(points) < 50:
        return []


    eps = max(
        0.8,
        reference_radius
        * RGB_CLUSTER_EPS_FACTOR,
    )


    labels = np.array(
        cloud.cluster_dbscan(
            eps=eps,
            min_points=(
                RGB_CLUSTER_MIN_POINTS
            ),
            print_progress=False,
        )
    )


    valid_labels = [
        label
        for label
        in np.unique(labels)
        if label >= 0
    ]


    expected_diameter = (
        reference_radius
        * 2.0
    )


    candidates = []


    for label in valid_labels:

        cluster = points[
            labels == label
        ]


        if len(cluster) < 20:
            continue


        center = np.mean(
            cluster,
            axis=0,
        )


        centered = (
            cluster - center
        )


        covariance = np.cov(
            centered.T
        )


        (
            eigenvalues,
            eigenvectors,
        ) = np.linalg.eigh(
            covariance
        )


        order = np.argsort(
            eigenvalues
        )


        eigenvalues = (
            eigenvalues[
                order
            ]
        )

        eigenvectors = (
            eigenvectors[
                :,
                order
            ]
        )


        normal = (
            eigenvectors[
                :,
                0
            ]
        )

        axis1 = (
            eigenvectors[
                :,
                2
            ]
        )

        axis2 = (
            eigenvectors[
                :,
                1
            ]
        )


        p1 = centered @ axis1
        p2 = centered @ axis2
        pn = centered @ normal


        extent1 = (
            np.percentile(p1, 95)
            -
            np.percentile(p1, 5)
        )

        extent2 = (
            np.percentile(p2, 95)
            -
            np.percentile(p2, 5)
        )

        thickness = (
            np.percentile(pn, 95)
            -
            np.percentile(pn, 5)
        )


        max_extent = max(
            extent1,
            extent2,
        )

        min_extent = max(
            1e-6,
            min(
                extent1,
                extent2,
            ),
        )


        circularity = (
            min_extent
            /
            max(
                max_extent,
                1e-6,
            )
        )


        diameter_score = np.exp(
            -abs(
                max_extent
                - expected_diameter
            )
            /
            max(
                expected_diameter,
                1e-6,
            )
        )


        flatness = np.exp(
            -thickness
            /
            max(
                expected_diameter
                * 0.30,
                0.5,
            )
        )


        size_score = np.log1p(
            len(cluster)
        )


        score = (
            size_score
            * (
                0.3
                +
                0.7
                * circularity
            )
            * (
                0.3
                +
                0.7
                * diameter_score
            )
            * (
                0.3
                +
                0.7
                * flatness
            )
        )


        candidates.append(
            {
                "center": center,
                "normal": normal,
                "score": score,
            }
        )


    candidates.sort(
        key=lambda item:
        item["score"],
        reverse=True,
    )


    return (
        candidates[
            :RGB_MAX_CLUSTERS
        ]
    )


# ============================================================
# TRIOS FALLBACK
# ============================================================

def trios_rgb_fallback_registration(
    mesh,
    reference_system,
):

    reference = (
        reference_system[
            "reference"
        ]
    )


    candidates = (
        find_rgb_specimen_candidates(
            mesh,
            reference["radius"],
        )
    )


    if not candidates:
        return None


    target_points = np.asarray(
        mesh.points
    )


    reference_surface_center = (
        reference["center"]
        +
        reference["z_offset"]
        * reference["normal"]
    )


    reference_normal = (
        reference["normal"]
    )


    best = None


    for cluster_index, candidate in enumerate(
        candidates
    ):

        target_center = candidate[
            "center"
        ]


        for normal_sign in (
            1.0,
            -1.0,
        ):

            target_normal = (
                candidate["normal"]
                * normal_sign
            )


            base_rotation = (
                rotation_align_vectors(
                    target_normal,
                    reference_normal,
                )
            )


            for angle_deg in (
                TRIOS_INPLANE_ANGLES_DEG
            ):

                inplane = (
                    rotation_about_axis(
                        reference_normal,
                        np.deg2rad(
                            angle_deg
                        ),
                    )
                )


                rotation = (
                    inplane
                    @ base_rotation
                )


                translation = (
                    reference_surface_center
                    -
                    rotation
                    @ target_center
                )


                initial = np.eye(4)

                initial[
                    :3,
                    :3
                ] = rotation

                initial[
                    :3,
                    3
                ] = translation


                aligned = transform_points(
                    target_points,
                    initial,
                )


                (
                    holder_result,
                    holder_points,
                ) = run_holder_icp(
                    aligned,
                    reference,
                    reference_system[
                        "holder_cloud"
                    ],
                    [
                        TRIOS_HOLDER_ICP_WIDE,
                        TRIOS_HOLDER_ICP_MEDIUM,
                        TRIOS_HOLDER_ICP_FINE,
                    ],
                )


                if holder_result is None:
                    continue


                total_transform = (
                    holder_result.transformation
                    @ initial
                )


                fitness = (
                    holder_result.fitness
                )

                rmse = (
                    holder_result.inlier_rmse
                )


                score = (
                    fitness
                    -
                    0.10
                    * rmse
                )


                if (
                    best is None
                    or
                    score
                    >
                    best["score"]
                ):

                    best = {
                        "transform": (
                            total_transform
                        ),

                        "method": (
                            "RGB_COARSE+HOLDER"
                        ),

                        "score": score,

                        "global_fitness": np.nan,
                        "global_rmse": np.nan,

                        "full_fitness": np.nan,
                        "full_rmse": np.nan,

                        "holder_fitness": fitness,
                        "holder_rmse": rmse,

                        "holder_points": holder_points,

                        "rgb_cluster": cluster_index,
                        "rgb_cluster_score": (
                            candidate["score"]
                        ),

                        "angle_deg": angle_deg,
                        "normal_sign": normal_sign,
                    }


    return best


# ============================================================
# REGISTRATION QUALITY
# ============================================================

def registration_is_strong(
    result,
):

    if result is None:
        return False


    return (
        (
            result[
                "holder_fitness"
            ]
            >=
            TRIOS_PRIMARY_MIN_HOLDER_FITNESS
        )
        and
        (
            result[
                "holder_rmse"
            ]
            <=
            TRIOS_PRIMARY_MAX_HOLDER_RMSE
        )
    )


def registration_score(
    result,
):

    if result is None:
        return -np.inf


    return (
        result[
            "holder_fitness"
        ]
        -
        0.10
        * result[
            "holder_rmse"
        ]
    )


# ============================================================
# QC CONTEXT
# ============================================================

def create_context(
    aligned_mesh,
    reference,
    z_offset,
    canonical,
):

    points = np.asarray(
        aligned_mesh.points
    )


    (
        x,
        y,
        z,
    ) = local_coordinates(
        points,
        reference["center"],
        reference["u_axis"],
        reference["v_axis"],
        reference["normal"],
    )


    half_width = (
        reference["radius"]
        * CONTEXT_SIZE_FACTOR
    )


    mask = (
        (
            np.abs(x)
            <= half_width
        )
        &
        (
            np.abs(y)
            <= half_width
        )
        &
        (
            np.abs(
                z
                - z_offset
            )
            <=
            CONTEXT_HEIGHT_TOLERANCE
        )
    )


    context = (
        aligned_mesh.extract_points(
            mask,
            adjacent_cells=False,
            include_cells=True,
        )
        .extract_surface()
        .clean()
    )


    context, _ = (
        prepare_embedded_rgb(
            context
        )
    )

    context, _ = (
        activate_tcoords(
            context
        )
    )


    context.transform(
        canonical,
        inplace=True,
    )


    return context


def add_canonical_circle(
    plotter,
    radius,
):

    theta = np.linspace(
        0,
        2 * np.pi,
        300,
    )


    points = np.column_stack(
        [
            radius
            * np.cos(theta),

            radius
            * np.sin(theta),

            np.full(
                len(theta),
                0.05,
            ),
        ]
    )


    line = pv.lines_from_points(
        points,
        close=True,
    )


    plotter.add_mesh(
        line,
        color="red",
        line_width=5,
    )


# ============================================================
# REFERENCE SELECTION
# ============================================================

def define_reference_roi(
    mesh,
    texture,
    scanner,
    sample_id,
):

    while True:

        selected_points = []


        plotter = pv.Plotter(
            window_size=(
                1300,
                950,
            )
        )

        plotter.set_background(
            "white"
        )


        add_scan_mesh(
            plotter,
            mesh,
            texture,
        )


        def redraw():

            try:
                plotter.remove_actor(
                    "roi_points",
                    reset_camera=False,
                )

            except Exception:
                pass


            if selected_points:

                plotter.add_points(
                    np.asarray(
                        selected_points
                    ),
                    color="red",
                    point_size=20,
                    render_points_as_spheres=True,
                    name="roi_points",
                )


            plotter.render()


        def add_point(point):

            if point is None:
                return

            selected_points.append(
                np.asarray(point)
            )

            print(
                f"Added point "
                f"{len(selected_points)}"
            )

            redraw()


        def undo():

            if selected_points:
                selected_points.pop()

            redraw()


        def clear():

            selected_points.clear()

            redraw()


        plotter.add_text(
            (
                f"{scanner} | {sample_id}\n\n"
                "Define TRUE outer specimen edge\n\n"
                "LEFT CLICK = add point\n"
                "U = undo\n"
                "C = clear\n"
                "Q = finish\n\n"
                "Use approximately 8-12 points."
            ),
            position="upper_left",
            font_size=11,
        )


        plotter.enable_point_picking(
            callback=add_point,
            show_message=False,
            show_point=False,
            left_clicking=True,
            pickable_window=False,
        )


        plotter.add_key_event(
            "u",
            undo,
        )

        plotter.add_key_event(
            "c",
            clear,
        )


        plotter.reset_camera()

        plotter.show()


        if (
            len(selected_points)
            <
            MIN_REFERENCE_EDGE_POINTS
        ):

            print(
                "Not enough points."
            )

            continue


        reference = fit_reference_circle(
            selected_points,
            mesh,
        )


        # Initial reference Z directly from center.
        points = np.asarray(
            mesh.points
        )


        (
            x,
            y,
            z,
        ) = local_coordinates(
            points,
            reference["center"],
            reference["u_axis"],
            reference["v_axis"],
            reference["normal"],
        )


        radial = np.sqrt(
            x**2 + y**2
        )


        central = (
            radial
            <
            reference["radius"]
            * 0.50
        )


        if np.sum(central) < 20:

            print(
                "Could not estimate reference surface."
            )

            continue


        reference[
            "z_offset"
        ] = float(
            np.median(
                z[central]
            )
        )


        (
            specimen,
            z_offset,
            roi_radius,
            canonical,
        ) = extract_specimen(
            mesh,
            reference,
        )


        reference[
            "z_offset"
        ] = z_offset

        reference[
            "roi_radius"
        ] = roi_radius


        context = create_context(
            mesh,
            reference,
            z_offset,
            canonical,
        )


        qc = pv.Plotter(
            shape=(1, 2),
            window_size=(
                1700,
                850,
            ),
        )


        qc.subplot(
            0,
            0,
        )

        add_scan_mesh(
            qc,
            context,
            texture,
        )

        add_canonical_circle(
            qc,
            roi_radius,
        )

        qc.view_xy()

        qc.camera.parallel_projection = True

        qc.reset_camera()


        qc.subplot(
            0,
            1,
        )

        add_scan_mesh(
            qc,
            specimen,
            texture,
        )

        qc.view_xy()

        qc.camera.parallel_projection = True

        qc.reset_camera()


        qc.show()


        answer = input(
            f"\nIs {scanner} reference ROI correct? [y/n]: "
        )


        if (
            answer.strip()
            .lower()
            .startswith("y")
        ):

            return reference


# ============================================================
# SAVE / LOAD REFERENCES
# ============================================================

def save_reference(
    scanner,
    reference,
    record,
):

    file = get_template_file(
        scanner
    )


    np.savez(
        file,
        scanner=scanner,

        center=reference[
            "center"
        ],

        normal=reference[
            "normal"
        ],

        u_axis=reference[
            "u_axis"
        ],

        v_axis=reference[
            "v_axis"
        ],

        radius=reference[
            "radius"
        ],

        roi_radius=reference[
            "roi_radius"
        ],

        z_offset=reference[
            "z_offset"
        ],

        reference_sample_id=(
            record[
                "sample_id"
            ]
        ),

        reference_ply=str(
            record[
                "ply_path"
            ]
        ),
    )


def load_reference(scanner):

    data = np.load(
        get_template_file(
            scanner
        ),
        allow_pickle=True,
    )


    return {
        "center": np.asarray(
            data["center"]
        ),

        "normal": np.asarray(
            data["normal"]
        ),

        "u_axis": np.asarray(
            data["u_axis"]
        ),

        "v_axis": np.asarray(
            data["v_axis"]
        ),

        "radius": float(
            data["radius"]
        ),

        "roi_radius": float(
            data["roi_radius"]
        ),

        "z_offset": float(
            data["z_offset"]
        ),

        "reference_sample_id": str(
            data[
                "reference_sample_id"
            ]
        ),
    }


# ============================================================
# SAVE QC PREVIEW
# ============================================================

def save_qc_preview(
    context,
    specimen,
    texture,
    roi_radius,
    output_file,
):

    plotter = pv.Plotter(
        shape=(1, 2),
        off_screen=True,
        window_size=(
            PREVIEW_WIDTH,
            PREVIEW_HEIGHT,
        ),
    )


    # LEFT
    plotter.subplot(
        0,
        0,
    )

    plotter.set_background(
        "white"
    )


    add_scan_mesh(
        plotter,
        context,
        texture,
    )


    add_canonical_circle(
        plotter,
        roi_radius,
    )


    plotter.add_text(
        "Original + ROI",
        font_size=9,
    )


    plotter.view_xy()

    plotter.camera.parallel_projection = True

    plotter.reset_camera()


    # RIGHT
    plotter.subplot(
        0,
        1,
    )

    plotter.set_background(
        "white"
    )


    add_scan_mesh(
        plotter,
        specimen,
        texture,
    )


    plotter.add_text(
        "Extracted ROI",
        font_size=9,
    )


    plotter.view_xy()

    plotter.camera.parallel_projection = True

    plotter.reset_camera()


    plotter.screenshot(
        str(output_file)
    )

    plotter.close()


# ============================================================
# CONTACT SHEETS
# ============================================================

def make_contact_sheets(
    records,
    output_folder,
    prefix,
):

    if not records:
        return


    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )


    per_page = (
        CONTACT_SHEET_COLUMNS
        * CONTACT_SHEET_ROWS
    )


    pages = math.ceil(
        len(records)
        /
        per_page
    )


    font = (
        ImageFont.load_default()
    )


    for page in range(pages):

        subset = records[
            page * per_page:
            (page + 1) * per_page
        ]


        sheet = Image.new(
            "RGB",
            (
                CONTACT_SHEET_COLUMNS
                * CONTACT_SHEET_TILE_WIDTH,

                CONTACT_SHEET_ROWS
                * CONTACT_SHEET_TILE_HEIGHT,
            ),
            "white",
        )


        draw = ImageDraw.Draw(
            sheet
        )


        for i, record in enumerate(
            subset
        ):

            row = (
                i
                //
                CONTACT_SHEET_COLUMNS
            )

            col = (
                i
                %
                CONTACT_SHEET_COLUMNS
            )


            x0 = (
                col
                * CONTACT_SHEET_TILE_WIDTH
            )

            y0 = (
                row
                * CONTACT_SHEET_TILE_HEIGHT
            )


            preview = record.get(
                "qc_preview"
            )


            if (
                preview
                and
                Path(preview).exists()
            ):

                image = (
                    Image.open(preview)
                    .convert("RGB")
                )


                image.thumbnail(
                    (
                        CONTACT_SHEET_TILE_WIDTH
                        - 20,

                        CONTACT_SHEET_TILE_HEIGHT
                        - 65,
                    )
                )


                sheet.paste(
                    image,
                    (
                        x0
                        +
                        (
                            CONTACT_SHEET_TILE_WIDTH
                            -
                            image.width
                        )
                        // 2,

                        y0 + 5,
                    ),
                )


            label = (
                f"{record['scanner']} | "
                f"{record['sample_id']} | "
                f"{record['status']}\n"
                f"{record.get('registration_method', '')}"
            )


            draw.text(
                (
                    x0 + 8,
                    y0
                    +
                    CONTACT_SHEET_TILE_HEIGHT
                    - 52,
                ),
                label,
                fill="black",
                font=font,
            )


            draw.rectangle(
                [
                    x0,
                    y0,
                    x0
                    + CONTACT_SHEET_TILE_WIDTH
                    - 1,
                    y0
                    + CONTACT_SHEET_TILE_HEIGHT
                    - 1,
                ],
                outline="gray",
                width=1,
            )


        output = (
            output_folder
            /
            f"{prefix}_{page + 1:03d}.jpg"
        )


        sheet.save(
            output,
            quality=92,
        )


# ============================================================
# EVALUATE ONE REGISTRATION
# ============================================================

def evaluate_registration(
    mesh,
    reference,
    registration,
):

    aligned_mesh = mesh.copy(
        deep=True
    )


    aligned_mesh.transform(
        registration[
            "transform"
        ],
        inplace=True,
    )


    aligned_mesh, _ = (
        prepare_embedded_rgb(
            aligned_mesh
        )
    )

    aligned_mesh, _ = (
        activate_tcoords(
            aligned_mesh
        )
    )


    (
        specimen,
        z_offset,
        roi_radius,
        canonical,
    ) = extract_specimen(
        aligned_mesh,
        reference,
    )


    quality = assess_roi_quality(
        specimen,
        roi_radius,
    )


    return {
        "registration": registration,
        "aligned_mesh": aligned_mesh,
        "specimen": specimen,
        "z_offset": z_offset,
        "roi_radius": roi_radius,
        "canonical": canonical,
        "quality": quality,
    }


# ============================================================
# DISCOVER DATASET
# ============================================================

dataset = discover_dataset()


# ============================================================
# DATASET DISCOVERY DIAGNOSTIC
# ============================================================

print("\n" + "=" * 80)
print("DATASET DISCOVERY DIAGNOSTIC")
print("=" * 80)

print(
    f"\nTotal discovered PLY files: "
    f"{len(dataset)}"
)


# ------------------------------------------------------------
# COUNT BY SCANNER
# ------------------------------------------------------------

scanner_counts = defaultdict(int)

for record in dataset:
    scanner_counts[
        record["scanner"]
    ] += 1


print("\nDiscovered per scanner:")

for scanner, count in sorted(
    scanner_counts.items()
):
    print(
        f"  {scanner:<20} {count}"
    )


# ------------------------------------------------------------
# COUNT BY SCANNER + GROUP
# ------------------------------------------------------------

scanner_group_counts = defaultdict(
    int
)

for record in dataset:

    scanner_group_counts[
        (
            record["scanner"],
            record["group"],
        )
    ] += 1


print(
    "\nDiscovered per scanner/group:"
)

for (
    scanner,
    group,
), count in sorted(
    scanner_group_counts.items(),
    key=lambda item: (
        item[0][0].lower(),
        (
            item[0][1]
            if item[0][1]
            is not None
            else 999
        ),
    ),
):

    print(
        f"  {scanner:<20} "
        f"G{group}: "
        f"{count}"
    )


# ------------------------------------------------------------
# SHOW ALL ITERO FILES
# ------------------------------------------------------------

print(
    "\nAll discovered iTERO samples:"
)

itero_records = [
    record
    for record in dataset
    if record[
        "scanner"
    ].lower()
    ==
    "itero"
]


for record in itero_records:

    print(
        f"  "
        f"{record['group']} | "
        f"{record['sample_id']} | "
        f"{record['ply_path']}"
    )


# ============================================================
# BUILD SCANNER-SPECIFIC REFERENCES
# ============================================================

scanner_references = {}


scanners = sorted(
    {
        record[
            "scanner"
        ]
        for record in dataset
    }
)


for scanner in scanners:

    if (
        scanner
        not in
        REFERENCE_SAMPLE_BY_SCANNER
    ):

        print(
            f"No reference configured for "
            f"{scanner}"
        )

        continue


    reference_sample = (
        REFERENCE_SAMPLE_BY_SCANNER[
            scanner
        ]
    )


    matches = [
        record
        for record in dataset
        if (
            record[
                "scanner"
            ] == scanner
            and
            record[
                "sample_id"
            ].upper()
            ==
            reference_sample.upper()
        )
    ]


    if not matches:

        print(
            f"Reference "
            f"{reference_sample} "
            f"not found for {scanner}"
        )

        continue


    reference_record = (
        matches[0]
    )


    print(
        "\n" + "=" * 80
    )

    print(
        f"Preparing {scanner} reference: "
        f"{reference_sample}"
    )

    print(
        "=" * 80
    )


    (
        reference_mesh,
        _,
        _,
    ) = load_mesh(
        reference_record[
            "ply_path"
        ]
    )


    reference_texture = (
        load_texture(
            reference_record[
                "texture_path"
            ]
        )
    )


    template_file = (
        get_template_file(
            scanner
        )
    )


    if (
        scanner
        in
        REDEFINE_REFERENCE_SCANNERS
        or
        not template_file.exists()
    ):

        reference = (
            define_reference_roi(
                reference_mesh,
                reference_texture,
                scanner,
                reference_sample,
            )
        )


        save_reference(
            scanner,
            reference,
            reference_record,
        )


    else:

        reference = load_reference(
            scanner
        )


    reference_points = np.asarray(
        reference_mesh.points
    )


    (
        reference_down,
        reference_feature,
    ) = prepare_cloud(
        reference_points,
        VOXEL_SIZE,
    )


    holder_selection = holder_mask(
        reference_points,
        reference,
        HOLDER_HALF_WIDTH_FACTOR,
    )


    holder_points = (
        reference_points[
            holder_selection
        ]
    )


    holder_cloud = points_to_o3d(
        holder_points
    )


    holder_cloud = (
        holder_cloud.voxel_down_sample(
            0.25
        )
    )


    estimate_normals(
        holder_cloud,
        1.0,
    )


    scanner_references[
        scanner
    ] = {
        "record": reference_record,
        "reference": reference,
        "down": reference_down,
        "feature": reference_feature,
        "holder_cloud": holder_cloud,
        "holder_points": holder_points,
    }


# ============================================================
# FILTER DATASET
# ============================================================

processing_dataset = []


for record in dataset:

    if (
        record["scanner"]
        not in
        scanner_references
    ):
        continue


    if (
        SCANNERS_TO_PROCESS
        is not None
        and
        record["scanner"]
        not in
        SCANNERS_TO_PROCESS
    ):
        continue


    if (
        GROUPS_TO_PROCESS
        is not None
        and
        record["group"]
        not in
        GROUPS_TO_PROCESS
    ):
        continue


    processing_dataset.append(
        record
    )


if MAX_FILES is not None:

    processing_dataset = (
        processing_dataset[
            :MAX_FILES
        ]
    )


# ============================================================
# PROCESS
# ============================================================

manifest_rows = []


for index, record in enumerate(
    processing_dataset,
    start=1,
):

    scanner = record[
        "scanner"
    ]

    sample_id = record[
        "sample_id"
    ]

    system = scanner_references[
        scanner
    ]

    reference = system[
        "reference"
    ]

    reference_record = system[
        "record"
    ]


    print(
        "\n" + "=" * 80
    )

    print(
        f"[{index}/"
        f"{len(processing_dataset)}] "
        f"{scanner} | {sample_id}"
    )

    print(
        "=" * 80
    )


    output_dir = record[
        "output_dir"
    ]

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    roi_file = (
        output_dir
        / "roi.vtp"
    )

    geometry_file = (
        output_dir
        / "geometry.npz"
    )

    qc_file = (
        output_dir
        / "qc.png"
    )


    try:

        (
            mesh,
            _,
            _,
        ) = load_mesh(
            record[
                "ply_path"
            ]
        )


        texture = load_texture(
            record[
                "texture_path"
            ]
        )


        is_reference = (
            record[
                "ply_path"
            ].resolve()
            ==
            reference_record[
                "ply_path"
            ].resolve()
        )


        # ====================================================
        # REGISTRATION CANDIDATES
        # ====================================================

        registration_candidates = []


        if is_reference:

            registration_candidates.append(
                {
                    "transform": np.eye(4),
                    "method": "REFERENCE",

                    "global_fitness": 1.0,
                    "global_rmse": 0.0,

                    "full_fitness": 1.0,
                    "full_rmse": 0.0,

                    "holder_fitness": 1.0,
                    "holder_rmse": 0.0,

                    "holder_points": len(
                        system[
                            "holder_points"
                        ]
                    ),
                }
            )


        else:

            # ------------------------------------------------
            # PRIMARY
            # ------------------------------------------------

            try:

                primary = primary_registration(
                    mesh,
                    system,
                )

                if primary is not None:

                    registration_candidates.append(
                        primary
                    )

                    print(
                        f"Primary holder fitness: "
                        f"{primary['holder_fitness']:.3f}"
                    )

                    print(
                        f"Primary holder RMSE: "
                        f"{primary['holder_rmse']:.3f}"
                    )

            except Exception as exc:

                print(
                    f"Primary registration failed: "
                    f"{exc}"
                )


            # ------------------------------------------------
            # PRECOMPUTE FALLBACK IF PRIMARY IS WEAK
            # ------------------------------------------------

            need_fallback = (
                scanner
                in
                TRIOS_SCANNERS
                and
                ENABLE_TRIOS_RGB_FALLBACK
                and
                (
                    len(
                        registration_candidates
                    ) == 0
                    or
                    not registration_is_strong(
                        registration_candidates[
                            0
                        ]
                    )
                )
            )


            if need_fallback:

                print(
                    "Trying TRIOS RGB-assisted fallback..."
                )


                fallback = (
                    trios_rgb_fallback_registration(
                        mesh,
                        system,
                    )
                )


                if fallback is not None:

                    registration_candidates.append(
                        fallback
                    )


        if not registration_candidates:

            raise RuntimeError(
                "No registration candidate found."
            )


        # ====================================================
        # EVALUATE CANDIDATES
        # ====================================================

        evaluations = []

        evaluation_errors = []


        for registration in (
            registration_candidates
        ):

            try:

                evaluation = (
                    evaluate_registration(
                        mesh,
                        reference,
                        registration,
                    )
                )


                quality = (
                    evaluation[
                        "quality"
                    ]
                )


                print(
                    f"\nCandidate: "
                    f"{registration['method']}"
                )

                print(
                    f"  ROI points: "
                    f"{quality['n_points']}"
                )

                print(
                    f"  Area coverage: "
                    f"{quality['area_coverage']:.3f}"
                )

                print(
                    f"  Edge reach: "
                    f"{quality['edge_reach']:.3f}"
                )


                evaluations.append(
                    evaluation
                )


            except Exception as exc:

                evaluation_errors.append(
                    (
                        registration[
                            "method"
                        ],
                        str(exc),
                    )
                )


                print(
                    f"{registration['method']} "
                    f"ROI evaluation failed: "
                    f"{exc}"
                )


        # ====================================================
        # IF TRIOS PRIMARY PRODUCED BAD ROI, FORCE FALLBACK
        # ====================================================

        valid_evaluations = [
            result
            for result
            in evaluations
            if result[
                "quality"
            ][
                "ok"
            ]
        ]


        fallback_already_tested = any(
            result[
                "registration"
            ][
                "method"
            ]
            ==
            "RGB_COARSE+HOLDER"
            for result
            in evaluations
        )


        if (
            not valid_evaluations
            and
            scanner
            in
            TRIOS_SCANNERS
            and
            ENABLE_TRIOS_RGB_FALLBACK
            and
            not fallback_already_tested
        ):

            print(
                "\nPrimary ROI invalid. "
                "Forcing TRIOS RGB-assisted fallback..."
            )


            fallback = (
                trios_rgb_fallback_registration(
                    mesh,
                    system,
                )
            )


            if fallback is not None:

                try:

                    fallback_evaluation = (
                        evaluate_registration(
                            mesh,
                            reference,
                            fallback,
                        )
                    )


                    evaluations.append(
                        fallback_evaluation
                    )


                    if (
                        fallback_evaluation[
                            "quality"
                        ][
                            "ok"
                        ]
                    ):

                        valid_evaluations.append(
                            fallback_evaluation
                        )


                except Exception as exc:

                    evaluation_errors.append(
                        (
                            "RGB_COARSE+HOLDER",
                            str(exc),
                        )
                    )


        # ====================================================
        # SELECT BEST VALID ROI
        # ====================================================

        if not valid_evaluations:

            details = "; ".join(
                (
                    f"{method}: {error}"
                    for method, error
                    in evaluation_errors
                )
            )


            if evaluations:

                details += "; ROI QC: " + "; ".join(
                    (
                        f"{result['registration']['method']} "
                        f"points="
                        f"{result['quality']['n_points']} "
                        f"coverage="
                        f"{result['quality']['area_coverage']:.3f} "
                        f"edge="
                        f"{result['quality']['edge_reach']:.3f}"
                    )
                    for result in evaluations
                )


            raise RuntimeError(
                "No candidate produced a valid ROI. "
                + details
            )


        # Prefer best holder registration among valid ROIs.
        chosen = max(
            valid_evaluations,
            key=lambda result:
            registration_score(
                result[
                    "registration"
                ]
            ),
        )


        registration = (
            chosen[
                "registration"
            ]
        )

        aligned_mesh = (
            chosen[
                "aligned_mesh"
            ]
        )

        specimen = (
            chosen[
                "specimen"
            ]
        )

        z_offset = (
            chosen[
                "z_offset"
            ]
        )

        roi_radius = (
            chosen[
                "roi_radius"
            ]
        )

        canonical = (
            chosen[
                "canonical"
            ]
        )

        roi_quality = (
            chosen[
                "quality"
            ]
        )


        # ====================================================
        # STATUS
        # ====================================================

        status = "OK"


        if (
            registration[
                "holder_fitness"
            ]
            <
            MIN_HOLDER_FITNESS
        ):

            status = "CHECK"


        if (
            registration[
                "holder_rmse"
            ]
            >
            MAX_HOLDER_RMSE
        ):

            status = "CHECK"


        # ====================================================
        # SAVE ROI
        # ====================================================

        specimen.save(
            roi_file
        )


        original_to_reference = (
            registration[
                "transform"
            ]
        )


        original_to_canonical = (
            canonical
            @
            original_to_reference
        )


        np.savez(
            geometry_file,

            scanner=scanner,

            group=(
                record["group"]
                if record["group"]
                is not None
                else -1
            ),

            timepoint=(
                record["timepoint"]
                if record["timepoint"]
                is not None
                else ""
            ),

            sample_id=sample_id,

            source_ply=str(
                record[
                    "ply_path"
                ]
            ),

            source_texture=(
                str(
                    record[
                        "texture_path"
                    ]
                )
                if record[
                    "texture_path"
                ]
                is not None
                else ""
            ),

            reference_scanner=scanner,

            reference_sample_id=(
                reference_record[
                    "sample_id"
                ]
            ),

            reference_radius=(
                reference[
                    "radius"
                ]
            ),

            roi_radius=roi_radius,

            z_offset=z_offset,

            original_to_reference=(
                original_to_reference
            ),

            reference_to_canonical=(
                canonical
            ),

            original_to_canonical=(
                original_to_canonical
            ),

            registration_method=(
                registration[
                    "method"
                ]
            ),

            global_fitness=(
                registration[
                    "global_fitness"
                ]
            ),

            global_rmse=(
                registration[
                    "global_rmse"
                ]
            ),

            full_fitness=(
                registration[
                    "full_fitness"
                ]
            ),

            full_rmse=(
                registration[
                    "full_rmse"
                ]
            ),

            holder_fitness=(
                registration[
                    "holder_fitness"
                ]
            ),

            holder_rmse=(
                registration[
                    "holder_rmse"
                ]
            ),

            roi_points=(
                roi_quality[
                    "n_points"
                ]
            ),

            roi_area_coverage=(
                roi_quality[
                    "area_coverage"
                ]
            ),

            roi_edge_reach=(
                roi_quality[
                    "edge_reach"
                ]
            ),

            roi_z_span=(
                roi_quality[
                    "z_span"
                ]
            ),
        )


        # ====================================================
        # QC IMAGE
        # ====================================================

        context = create_context(
            aligned_mesh,
            reference,
            z_offset,
            canonical,
        )


        save_qc_preview(
            context,
            specimen,
            texture,
            roi_radius,
            qc_file,
        )


        # ====================================================
        # CONSOLE
        # ====================================================

        print(
            f"\nSELECTED: "
            f"{registration['method']}"
        )

        print(
            f"Holder fitness: "
            f"{registration['holder_fitness']:.3f}"
        )

        print(
            f"Holder RMSE: "
            f"{registration['holder_rmse']:.3f}"
        )

        print(
            f"ROI points: "
            f"{roi_quality['n_points']}"
        )

        print(
            f"Area coverage: "
            f"{roi_quality['area_coverage']:.3f}"
        )

        print(
            f"Edge reach: "
            f"{roi_quality['edge_reach']:.3f}"
        )

        print(
            f"Status: {status}"
        )


        # ====================================================
        # MANIFEST ROW
        # ====================================================

        manifest_rows.append(
            {
                "scanner": scanner,

                "group": (
                    record[
                        "group"
                    ]
                ),

                "group_folder": (
                    record[
                        "group_folder"
                    ]
                ),

                "timepoint": (
                    record[
                        "timepoint"
                    ]
                ),

                "sample_number": (
                    record[
                        "sample_number"
                    ]
                ),

                "sample_id": sample_id,

                "status": status,

                "registration_method": (
                    registration[
                        "method"
                    ]
                ),

                "global_fitness": (
                    registration[
                        "global_fitness"
                    ]
                ),

                "global_rmse": (
                    registration[
                        "global_rmse"
                    ]
                ),

                "full_fitness": (
                    registration[
                        "full_fitness"
                    ]
                ),

                "full_rmse": (
                    registration[
                        "full_rmse"
                    ]
                ),

                "holder_fitness": (
                    registration[
                        "holder_fitness"
                    ]
                ),

                "holder_rmse": (
                    registration[
                        "holder_rmse"
                    ]
                ),

                "holder_points": (
                    registration[
                        "holder_points"
                    ]
                ),

                "roi_points": (
                    roi_quality[
                        "n_points"
                    ]
                ),

                "roi_area_coverage": (
                    roi_quality[
                        "area_coverage"
                    ]
                ),

                "roi_edge_reach": (
                    roi_quality[
                        "edge_reach"
                    ]
                ),

                "roi_z_span": (
                    roi_quality[
                        "z_span"
                    ]
                ),

                "roi_vtp": str(
                    roi_file
                ),

                "geometry_npz": str(
                    geometry_file
                ),

                "qc_preview": str(
                    qc_file
                ),

                "source_ply": str(
                    record[
                        "ply_path"
                    ]
                ),

                "source_texture": (
                    str(
                        record[
                            "texture_path"
                        ]
                    )
                    if record[
                        "texture_path"
                    ]
                    is not None
                    else ""
                ),
            }
        )


    # ========================================================
    # FAILED
    # ========================================================

    except Exception as exc:

        print(
            f"\nFAILED: {exc}"
        )


        manifest_rows.append(
            {
                "scanner": scanner,

                "group": (
                    record[
                        "group"
                    ]
                ),

                "group_folder": (
                    record[
                        "group_folder"
                    ]
                ),

                "timepoint": (
                    record[
                        "timepoint"
                    ]
                ),

                "sample_number": (
                    record[
                        "sample_number"
                    ]
                ),

                "sample_id": sample_id,

                "status": "FAILED",

                "registration_method": "",

                "global_fitness": "",

                "global_rmse": "",

                "full_fitness": "",

                "full_rmse": "",

                "holder_fitness": "",

                "holder_rmse": "",

                "holder_points": "",

                "roi_points": "",

                "roi_area_coverage": "",

                "roi_edge_reach": "",

                "roi_z_span": "",

                "roi_vtp": "",

                "geometry_npz": "",

                "qc_preview": "",

                "source_ply": str(
                    record[
                        "ply_path"
                    ]
                ),

                "source_texture": (
                    str(
                        record[
                            "texture_path"
                        ]
                    )
                    if record[
                        "texture_path"
                    ]
                    is not None
                    else ""
                ),

                "error": str(exc),
            }
        )


# ============================================================
# SAVE MANIFEST
# ============================================================

if manifest_rows:

    fields = sorted(
        {
            key
            for row in manifest_rows
            for key in row.keys()
        }
    )


    with open(
        MANIFEST_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            manifest_rows
        )


# ============================================================
# CONTACT SHEETS
# ============================================================

make_contact_sheets(
    manifest_rows,
    CONTACT_SHEET_ROOT / "ALL",
    "ROI_overview",
)


by_scanner_group = defaultdict(
    list
)


for row in manifest_rows:

    group = row.get(
        "group"
    )

    group_name = (
        f"G{group}"
        if group is not None
        else "G_UNKNOWN"
    )


    by_scanner_group[
        (
            row["scanner"],
            group_name,
        )
    ].append(
        row
    )


for (
    scanner,
    group_name,
), rows in (
    by_scanner_group.items()
):

    make_contact_sheets(
        rows,
        (
            CONTACT_SHEET_ROOT
            / scanner
            / group_name
        ),
        (
            f"{safe_name(scanner)}_"
            f"{group_name}_ROI"
        ),
    )


# ============================================================
# SUMMARY
# ============================================================

n_ok = sum(
    row["status"] == "OK"
    for row in manifest_rows
)

n_check = sum(
    row["status"] == "CHECK"
    for row in manifest_rows
)

n_failed = sum(
    row["status"] == "FAILED"
    for row in manifest_rows
)


method_counts = defaultdict(
    int
)


for row in manifest_rows:

    method_counts[
        row.get(
            "registration_method",
            "",
        )
    ] += 1


print(
    "\n" + "=" * 80
)

print(
    "FINISHED"
)

print(
    "=" * 80
)


print(
    f"\nProcessed: "
    f"{len(manifest_rows)}"
)

print(
    f"OK:        "
    f"{n_ok}"
)

print(
    f"CHECK:     "
    f"{n_check}"
)

print(
    f"FAILED:    "
    f"{n_failed}"
)


print(
    "\nRegistration methods:"
)


for method, count in sorted(
    method_counts.items()
):

    print(
        f"  {method:<25} "
        f"{count}"
    )


print(
    f"\nManifest:\n"
    f"{MANIFEST_FILE}"
)

print(
    f"\nQC sheets:\n"
    f"{CONTACT_SHEET_ROOT}"
)