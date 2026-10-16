"""Registration/reference functions adapted from your preprocessing.py; no import-time batch execution."""
from pathlib import Path
from collections import defaultdict
import csv
import math
import json
from dataclasses import asdict
from roi_geometry import Extraction, select_surface, DEFAULT_HEIGHT_HALF_WIDTH
import re
import numpy as np
import pyvista as pv
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull, QhullError
REFERENCE_SAMPLE_BY_SCANNER = {'LABscanner': 'BCG1T0-1', 'iTERO': 'BCG1T0-1', 'TRIOS3': 'BCG1T0-1', 'TRIOS5': 'BCG1T0-1', 'PRIMESCAN': 'BCG1T0-1'}
REDEFINE_REFERENCE_SCANNERS = set()
MIN_REFERENCE_EDGE_POINTS = 6
ROI_RADIUS_FACTOR = 0.985
SPECIMEN_HEIGHT_TOLERANCE = DEFAULT_HEIGHT_HALF_WIDTH
Z_MODE_SEARCH_RANGE = 8.0
Z_MODE_BIN_WIDTH = 0.2
Z_MODE_LOCAL_HALF_WIDTH = 0.5
Z_MODE_PRIOR_SIGMA = 3.0
MIN_ROI_POINTS_ABSOLUTE = 100
MIN_ROI_AREA_COVERAGE = 0.65
MIN_ROI_EDGE_REACH = 0.82
VOXEL_SIZE = 0.7
FULL_ICP_DISTANCE = 1.2
HOLDER_ICP_DISTANCE_1 = 0.8
HOLDER_ICP_DISTANCE_2 = 0.35
MIN_FGR_FITNESS = 0.05
HOLDER_HALF_WIDTH_FACTOR = 1.6
HOLDER_EXCLUSION_FACTOR = 1.02
HOLDER_Z_RANGE = 8.0
TARGET_HOLDER_SEARCH_FACTOR = 1.9
MIN_FULL_FITNESS = 0.15
MIN_HOLDER_FITNESS = 0.25
MAX_HOLDER_RMSE = 0.7
TRIOS_SCANNERS = {'TRIOS3', 'TRIOS5'}
ENABLE_TRIOS_RGB_FALLBACK = True
TRIOS_PRIMARY_MIN_HOLDER_FITNESS = 0.35
TRIOS_PRIMARY_MAX_HOLDER_RMSE = 0.65
RGB_MIN_RED = 90
RGB_MIN_RED_MINUS_GREEN = 15
RGB_MIN_RED_MINUS_BLUE = 5
RGB_CLUSTER_VOXEL = 0.4
RGB_CLUSTER_EPS_FACTOR = 0.25
RGB_CLUSTER_MIN_POINTS = 15
RGB_MAX_CLUSTERS = 4
TRIOS_HOLDER_ICP_WIDE = 2.5
TRIOS_HOLDER_ICP_MEDIUM = 1.0
TRIOS_HOLDER_ICP_FINE = 0.35
TRIOS_INPLANE_ANGLES_DEG = [0, 15, 30, 45, 60, 75]
SCANNERS_TO_PROCESS = None
GROUPS_TO_PROCESS = None
MAX_FILES = None
PREVIEW_WIDTH = 900
PREVIEW_HEIGHT = 450
CONTEXT_SIZE_FACTOR = 1.4
CONTEXT_HEIGHT_TOLERANCE = 4.0
CONTACT_SHEET_COLUMNS = 4
CONTACT_SHEET_ROWS = 4
CONTACT_SHEET_TILE_WIDTH = 920
CONTACT_SHEET_TILE_HEIGHT = 505
SAMPLE_PATTERN = re.compile('BCG(?P<group>\\d+)T(?P<timepoint>\\d+)-(?P<sample>\\d+)', re.IGNORECASE)
GROUP_PATTERN = re.compile('^G(?P<group>\\d+)', re.IGNORECASE)

def safe_name(text):
    return re.sub('[^A-Za-z0-9_.-]+', '_', str(text))

def get_template_file(scanner):
    folder = REFERENCE_ROOT / safe_name(scanner)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / 'reference_roi.npz'

def prepare_embedded_rgb(mesh):
    """
    Detect point-level RGB/RGBA.

    Preferred visualization source whenever available.

    Known:
        TRIOS3 -> RGB
        TRIOS5 -> RGB
        iTERO  -> RGB after PyVista loading
    """
    preferred_names = ['RGB', 'rgb', 'RGBA', 'rgba', 'Colors', 'colors', 'Color', 'color']
    for name in preferred_names:
        if name not in mesh.point_data:
            continue
        array = np.asarray(mesh.point_data[name])
        if not (array.ndim == 2 and array.shape[0] == mesh.n_points and (array.shape[1] in (3, 4))):
            continue
        if np.issubdtype(array.dtype, np.floating):
            converted = array.copy()
            if np.nanmax(converted) <= 1.0:
                converted *= 255.0
            converted = np.clip(converted, 0, 255).astype(np.uint8)
            mesh.point_data['RGB'] = converted
            return (mesh, 'RGB')
        return (mesh, name)
    triplets = [('red', 'green', 'blue'), ('Red', 'Green', 'Blue'), ('RED', 'GREEN', 'BLUE')]
    keys = set(mesh.point_data.keys())
    for r_name, g_name, b_name in triplets:
        if not all((name in keys for name in (r_name, g_name, b_name))):
            continue
        rgb = np.column_stack([mesh.point_data[r_name], mesh.point_data[g_name], mesh.point_data[b_name]])
        if np.issubdtype(rgb.dtype, np.floating):
            if np.nanmax(rgb) <= 1.0:
                rgb *= 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        mesh.point_data['RGB'] = rgb
        return (mesh, 'RGB')
    return (mesh, None)

def get_rgb_array_name(mesh):
    preferred_names = ['RGB', 'rgb', 'RGBA', 'rgba', 'Colors', 'colors', 'Color', 'color']
    for name in preferred_names:
        if name not in mesh.point_data:
            continue
        array = np.asarray(mesh.point_data[name])
        if array.ndim == 2 and array.shape[0] == mesh.n_points and (array.shape[1] in (3, 4)):
            return name
    return None

def activate_tcoords(mesh):
    possible_names = ['TCoords', 'tcoords', 'Texture Coordinates', 'TextureCoordinates', 'texture_coordinates', 'UV', 'UVs', 'uv', 'uvs', 'TexCoord', 'TexCoords', 'texcoord', 'texcoords']
    for name in possible_names:
        if name not in mesh.point_data:
            continue
        array = np.asarray(mesh.point_data[name])
        if not (array.ndim == 2 and array.shape[0] == mesh.n_points and (array.shape[1] == 2)):
            continue
        vtk_array = mesh.GetPointData().GetArray(name)
        if vtk_array is not None:
            mesh.GetPointData().SetTCoords(vtk_array)
            return (mesh, name)
    return (mesh, None)

def has_active_tcoords(mesh):
    return mesh.GetPointData().GetTCoords() is not None

def load_mesh(path):
    mesh = pv.read(path)
    mesh.point_data['RawVertexId'] = np.arange(mesh.n_points, dtype=np.int64)
    mesh = mesh.extract_surface()
    mesh = mesh.triangulate()
    mesh.cell_data['RawTriangleId'] = np.arange(mesh.n_cells, dtype=np.int64)
    mesh = mesh.clean(point_merging=False)
    mesh, rgb_name = prepare_embedded_rgb(mesh)
    mesh, uv_name = activate_tcoords(mesh)
    return (mesh, rgb_name, uv_name)

def find_texture(ply_path, sample_id=None):
    folder = ply_path.parent
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    images = sorted([path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in extensions])
    if not images:
        return None
    ply_stem = ply_path.stem.lower()
    for image in images:
        if image.stem.lower() == ply_stem:
            return image
    expected_names = {ply_stem + '_texture', ply_stem + '-texture', ply_stem + 'texture'}
    for image in images:
        if image.stem.lower() in expected_names:
            return image
    if sample_id is not None:
        sample_lower = sample_id.lower()
        matches = [image for image in images if sample_lower in image.stem.lower()]
        if len(matches) == 1:
            return matches[0]
        texture_matches = [image for image in matches if 'texture' in image.stem.lower()]
        if len(texture_matches) == 1:
            return texture_matches[0]
    if len(images) == 1:
        return images[0]
    return None

def load_texture(path):
    if path is None:
        return None
    try:
        return pv.read_texture(path)
    except Exception as exc:
        print(f'WARNING: texture failed: {path}\n{exc}')
        return None

def add_scan_mesh(plotter, mesh, texture=None):
    """
    Priority:

    1. embedded RGB
    2. external JPG + UV
    3. gray

    This avoids scrambled iTERO G1 texture mapping.
    """
    rgb_name = get_rgb_array_name(mesh)
    if rgb_name is not None:
        plotter.add_mesh(mesh, scalars=rgb_name, rgb=True, preference='point', lighting=False)
        return
    if texture is not None and has_active_tcoords(mesh):
        plotter.add_mesh(mesh, texture=texture, lighting=False)
        return
    plotter.add_mesh(mesh, color='lightgray', smooth_shading=True)

def extract_scanner(path):
    relative = path.relative_to(DATA_ROOT)
    if len(relative.parts) < 2:
        return 'UNKNOWN'
    return relative.parts[0]

def extract_sample_match(path):
    match = SAMPLE_PATTERN.search(path.stem)
    if match is not None:
        return match
    for parent in path.parents:
        if parent == DATA_ROOT:
            break
        match = SAMPLE_PATTERN.search(parent.name)
        if match is not None:
            return match
    return None

def extract_group_folder(path):
    relative = path.relative_to(DATA_ROOT)
    for part in relative.parts:
        match = GROUP_PATTERN.match(part)
        if match is not None:
            return (int(match.group('group')), part)
    return (None, None)

def discover_dataset():
    excluded_top_level = {'_processed_ROI', '_loading_QC', '_loading_QC_v2'}
    ply_files = []
    for path in DATA_ROOT.rglob('*'):
        if not path.is_file():
            continue
        if path.suffix.lower() != '.ply':
            continue
        relative = path.relative_to(DATA_ROOT)
        if len(relative.parts) > 0 and relative.parts[0] in excluded_top_level:
            continue
        ply_files.append(path)
    records = []
    for path in ply_files:
        scanner = extract_scanner(path)
        folder_group, group_folder = extract_group_folder(path)
        match = extract_sample_match(path)
        if match is not None:
            group = int(match.group('group'))
            time_number = int(match.group('timepoint'))
            sample_number = int(match.group('sample'))
            timepoint = f'T{time_number}'
            sample_id = f'BCG{group}T{time_number}-{sample_number}'
        else:
            group = folder_group
            time_number = None
            sample_number = None
            timepoint = None
            sample_id = path.stem
        texture_path = find_texture(path, sample_id)
        group_output = f'G{group}' if group is not None else 'G_UNKNOWN'
        output_dir = OUTPUT_ROOT / scanner / group_output / sample_id
        records.append({'scanner': scanner, 'group': group, 'group_folder': group_folder, 'timepoint': timepoint, 'timepoint_number': time_number, 'sample_number': sample_number, 'sample_id': sample_id, 'ply_path': path, 'texture_path': texture_path, 'output_dir': output_dir})
    records.sort(key=lambda r: (r['scanner'].lower(), r['group'] if r['group'] is not None else 999, r['timepoint_number'] if r['timepoint_number'] is not None else 999, r['sample_number'] if r['sample_number'] is not None else 999))
    return records

def transform_points(points, matrix):
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (matrix @ homogeneous.T).T[:, :3]

def points_to_o3d(points):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points))
    return cloud

def fit_plane(points):
    center = np.mean(points, axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    return (center, normal)

def create_plane_basis(normal):
    normal = normal / np.linalg.norm(normal)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(reference, normal)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u_axis = np.cross(normal, reference)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)
    return (u_axis, v_axis)

def fit_circle_2d(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    d, e, f = params
    cx = -d / 2.0
    cy = -e / 2.0
    radius_squared = cx ** 2 + cy ** 2 - f
    if radius_squared <= 0:
        raise RuntimeError('Invalid fitted circle.')
    return (cx, cy, np.sqrt(radius_squared))

def local_coordinates(points, center, u_axis, v_axis, normal):
    relative = points - center
    return (relative @ u_axis, relative @ v_axis, relative @ normal)

def fit_reference_circle(clicked_points, mesh):
    clicked_points = np.asarray(clicked_points)
    plane_center, normal = fit_plane(clicked_points)
    u_axis, v_axis = create_plane_basis(normal)
    normal_mesh = mesh.compute_normals(point_normals=True, cell_normals=False, consistent_normals=True, auto_orient_normals=False, inplace=False)
    mesh_points = np.asarray(normal_mesh.points)
    mesh_normals = np.asarray(normal_mesh.point_data['Normals'])
    distances = np.linalg.norm(mesh_points - plane_center, axis=1)
    nearest = np.argsort(distances)[:200]
    mean_normal = np.mean(mesh_normals[nearest], axis=0)
    if np.dot(normal, mean_normal) < 0:
        normal *= -1
        u_axis, v_axis = create_plane_basis(normal)
    relative = clicked_points - plane_center
    x = relative @ u_axis
    y = relative @ v_axis
    cx, cy, radius = fit_circle_2d(x, y)
    center = plane_center + cx * u_axis + cy * v_axis
    return {'center': center, 'normal': normal, 'u_axis': u_axis, 'v_axis': v_axis, 'radius': radius}

def estimate_specimen_z(radial, z, reference):
    """
    Find the dominant Z layer near the expected specimen
    center instead of requiring >=50 points within a rigid
    +/-3 mm interval.
    """
    radius = reference['radius']
    expected_z = reference['z_offset']
    central_mask = radial < radius * 0.55
    central_z = z[central_mask]
    central_z = central_z[np.isfinite(central_z)]
    if len(central_z) < 20:
        raise RuntimeError('Too few central points to estimate specimen surface.')
    search_z = central_z[np.abs(central_z - expected_z) <= Z_MODE_SEARCH_RANGE]
    if len(search_z) < 20:
        search_z = central_z
    z_min = float(np.min(search_z))
    z_max = float(np.max(search_z))
    if z_max - z_min < Z_MODE_BIN_WIDTH:
        return float(np.median(search_z))
    bins = np.arange(z_min - Z_MODE_BIN_WIDTH, z_max + 2 * Z_MODE_BIN_WIDTH, Z_MODE_BIN_WIDTH)
    histogram, edges = np.histogram(search_z, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    prior = np.exp(-0.5 * ((centers - expected_z) / Z_MODE_PRIOR_SIGMA) ** 2)
    scores = histogram * prior
    best_index = int(np.argmax(scores))
    best_center = centers[best_index]
    layer = search_z[np.abs(search_z - best_center) <= Z_MODE_LOCAL_HALF_WIDTH]
    if len(layer) < 10:
        order = np.argsort(np.abs(search_z - best_center))
        layer = search_z[order[:min(30, len(search_z))]]
    return float(np.median(layer))

def make_canonical_transform(center, u_axis, v_axis, normal, z_offset):
    surface_center = center + z_offset * normal
    rotation = np.vstack([u_axis, v_axis, normal])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = -(rotation @ surface_center)
    return matrix

def extract_specimen(mesh, reference):
    points = np.asarray(mesh.points)
    x, y, z = local_coordinates(points, reference['center'], reference['u_axis'], reference['v_axis'], reference['normal'])
    radial = np.sqrt(x ** 2 + y ** 2)
    z_offset = estimate_specimen_z(radial, z, reference)
    roi_radius = reference['radius'] * ROI_RADIUS_FACTOR
    canonical = make_canonical_transform(reference['center'], reference['u_axis'], reference['v_axis'], reference['normal'], z_offset)
    canonical_mesh = mesh.copy(deep=True)
    canonical_mesh.transform(canonical, inplace=True)
    raw_faces = np.asarray(canonical_mesh.faces).reshape(-1, 4)
    if not np.all(raw_faces[:, 0] == 3):
        raise ValueError('ROI extraction requires triangulated source geometry.')
    settings = Extraction(radius=roi_radius, below=SPECIMEN_HEIGHT_TOLERANCE, above=SPECIMEN_HEIGHT_TOLERANCE)
    keep, metrics = select_surface(np.asarray(canonical_mesh.points), raw_faces[:, 1:], settings)
    if not keep.any():
        raise ValueError('No central top surface passed extraction; inspect the reference and plane.')
    specimen = canonical_mesh.extract_cells(np.flatnonzero(keep)).extract_surface().clean(point_merging=False)
    specimen, _ = prepare_embedded_rgb(specimen)
    specimen, _ = activate_tcoords(specimen)
    specimen.field_data['ROIExtractionJSON'] = [json.dumps({'settings': asdict(settings), **metrics})]
    return (specimen, z_offset, roi_radius, canonical)

def assess_roi_quality(specimen, roi_radius):
    n_points = specimen.n_points
    result = {'ok': False, 'n_points': n_points, 'area_coverage': 0.0, 'edge_reach': 0.0, 'z_span': np.nan}
    if n_points < MIN_ROI_POINTS_ABSOLUTE:
        return result
    points = np.asarray(specimen.points)
    xy = points[:, :2]
    radial = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    edge_reach = np.percentile(radial, 98) / roi_radius
    result['edge_reach'] = float(edge_reach)
    expected_area = np.pi * roi_radius ** 2
    try:
        hull = ConvexHull(xy)
        hull_area = hull.volume
        area_coverage = hull_area / expected_area
    except (QhullError, ValueError):
        area_coverage = 0.0
    result['area_coverage'] = float(area_coverage)
    z_values = points[:, 2]
    z_span = np.percentile(z_values, 95) - np.percentile(z_values, 5)
    result['z_span'] = float(z_span)
    result['ok'] = bool(n_points >= MIN_ROI_POINTS_ABSOLUTE and area_coverage >= MIN_ROI_AREA_COVERAGE and (edge_reach >= MIN_ROI_EDGE_REACH))
    return result

def holder_mask(points, reference, half_width_factor):
    x, y, z = local_coordinates(points, reference['center'], reference['u_axis'], reference['v_axis'], reference['normal'])
    radial = np.sqrt(x ** 2 + y ** 2)
    half_width = reference['radius'] * half_width_factor
    return (np.abs(x) < half_width) & (np.abs(y) < half_width) & (np.abs(z) < HOLDER_Z_RANGE) & (radial > reference['radius'] * HOLDER_EXCLUSION_FACTOR)

def estimate_normals(cloud, radius):
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))

def prepare_cloud(points, voxel_size):
    cloud = points_to_o3d(points)
    down = cloud.voxel_down_sample(voxel_size)
    estimate_normals(down, voxel_size * 2.5)
    feature = o3d.pipelines.registration.compute_fpfh_feature(down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100))
    return (down, feature)

def global_registration(source_down, target_down, source_feature, target_feature):
    threshold = VOXEL_SIZE * 1.5
    try:
        option = o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=threshold)
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(source_down, target_down, source_feature, target_feature, option)
        if result.fitness >= MIN_FGR_FITNESS:
            return (result, 'FGR')
    except Exception:
        pass
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(source_down, target_down, source_feature, target_feature, mutual_filter=True, max_correspondence_distance=threshold, estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n=4, checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold)], criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(30000, 0.999))
    return (result, 'RANSAC')

def full_icp(source_down, reference_down, initial_transform):
    return o3d.pipelines.registration.registration_icp(source_down, reference_down, FULL_ICP_DISTANCE, initial_transform, o3d.pipelines.registration.TransformationEstimationPointToPlane())

def run_holder_icp(aligned_points, reference, reference_holder_cloud, distances):
    mask = holder_mask(aligned_points, reference, TARGET_HOLDER_SEARCH_FACTOR)
    target_points = aligned_points[mask]
    if len(target_points) < 100:
        return (None, len(target_points))
    target = points_to_o3d(target_points)
    target = target.voxel_down_sample(0.25)
    estimate_normals(target, 1.0)
    transform = np.eye(4)
    final_result = None
    for distance in distances:
        result = o3d.pipelines.registration.registration_icp(target, reference_holder_cloud, distance, transform, o3d.pipelines.registration.TransformationEstimationPointToPlane())
        transform = result.transformation
        final_result = result
    return (final_result, len(target_points))

def primary_registration(mesh, reference_system):
    reference = reference_system['reference']
    target_points = np.asarray(mesh.points)
    target_down, target_feature = prepare_cloud(target_points, VOXEL_SIZE)
    global_result, global_method = global_registration(target_down, reference_system['down'], target_feature, reference_system['feature'])
    full_result = full_icp(target_down, reference_system['down'], global_result.transformation)
    full_transform = full_result.transformation
    aligned_points = transform_points(target_points, full_transform)
    holder_result, holder_points = run_holder_icp(aligned_points, reference, reference_system['holder_cloud'], [HOLDER_ICP_DISTANCE_1, HOLDER_ICP_DISTANCE_2])
    if holder_result is None:
        return None
    total_transform = holder_result.transformation @ full_transform
    return {'transform': total_transform, 'method': global_method + '+HOLDER', 'global_fitness': global_result.fitness, 'global_rmse': global_result.inlier_rmse, 'full_fitness': full_result.fitness, 'full_rmse': full_result.inlier_rmse, 'holder_fitness': holder_result.fitness, 'holder_rmse': holder_result.inlier_rmse, 'holder_points': holder_points}

def rotation_align_vectors(source, target):
    a = np.asarray(source, dtype=float)
    b = np.asarray(target, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.clip(np.dot(a, b), -1.0, 1.0)
    s = np.linalg.norm(v)
    if s < 1e-08:
        if c > 0:
            return np.eye(3)
        trial = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(trial, a)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, trial)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)

def rotation_about_axis(axis, angle_radians):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle_radians)
    s = np.sin(angle_radians)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s], [y * x * C + z * s, c + y * y * C, y * z * C - x * s], [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])

def find_rgb_specimen_candidates(mesh, reference_radius):
    rgb_name = get_rgb_array_name(mesh)
    if rgb_name is None:
        return []
    rgb = np.asarray(mesh.point_data[rgb_name])[:, :3].astype(float)
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    redness = r - 0.5 * (g + b)
    mask = (r >= RGB_MIN_RED) & (r - g >= RGB_MIN_RED_MINUS_GREEN) & (r - b >= RGB_MIN_RED_MINUS_BLUE)
    if np.sum(mask) < 200:
        threshold = np.percentile(redness, 90)
        mask = redness >= threshold
    candidate_points = np.asarray(mesh.points)[mask]
    if len(candidate_points) < 100:
        return []
    cloud = points_to_o3d(candidate_points)
    cloud = cloud.voxel_down_sample(RGB_CLUSTER_VOXEL)
    points = np.asarray(cloud.points)
    if len(points) < 50:
        return []
    eps = max(0.8, reference_radius * RGB_CLUSTER_EPS_FACTOR)
    labels = np.array(cloud.cluster_dbscan(eps=eps, min_points=RGB_CLUSTER_MIN_POINTS, print_progress=False))
    valid_labels = [label for label in np.unique(labels) if label >= 0]
    expected_diameter = reference_radius * 2.0
    candidates = []
    for label in valid_labels:
        cluster = points[labels == label]
        if len(cluster) < 20:
            continue
        center = np.mean(cluster, axis=0)
        centered = cluster - center
        covariance = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        normal = eigenvectors[:, 0]
        axis1 = eigenvectors[:, 2]
        axis2 = eigenvectors[:, 1]
        p1 = centered @ axis1
        p2 = centered @ axis2
        pn = centered @ normal
        extent1 = np.percentile(p1, 95) - np.percentile(p1, 5)
        extent2 = np.percentile(p2, 95) - np.percentile(p2, 5)
        thickness = np.percentile(pn, 95) - np.percentile(pn, 5)
        max_extent = max(extent1, extent2)
        min_extent = max(1e-06, min(extent1, extent2))
        circularity = min_extent / max(max_extent, 1e-06)
        diameter_score = np.exp(-abs(max_extent - expected_diameter) / max(expected_diameter, 1e-06))
        flatness = np.exp(-thickness / max(expected_diameter * 0.3, 0.5))
        size_score = np.log1p(len(cluster))
        score = size_score * (0.3 + 0.7 * circularity) * (0.3 + 0.7 * diameter_score) * (0.3 + 0.7 * flatness)
        candidates.append({'center': center, 'normal': normal, 'score': score})
    candidates.sort(key=lambda item: item['score'], reverse=True)
    return candidates[:RGB_MAX_CLUSTERS]

def trios_rgb_fallback_registration(mesh, reference_system):
    reference = reference_system['reference']
    candidates = find_rgb_specimen_candidates(mesh, reference['radius'])
    if not candidates:
        return None
    target_points = np.asarray(mesh.points)
    reference_surface_center = reference['center'] + reference['z_offset'] * reference['normal']
    reference_normal = reference['normal']
    best = None
    for cluster_index, candidate in enumerate(candidates):
        target_center = candidate['center']
        for normal_sign in (1.0, -1.0):
            target_normal = candidate['normal'] * normal_sign
            base_rotation = rotation_align_vectors(target_normal, reference_normal)
            for angle_deg in TRIOS_INPLANE_ANGLES_DEG:
                inplane = rotation_about_axis(reference_normal, np.deg2rad(angle_deg))
                rotation = inplane @ base_rotation
                translation = reference_surface_center - rotation @ target_center
                initial = np.eye(4)
                initial[:3, :3] = rotation
                initial[:3, 3] = translation
                aligned = transform_points(target_points, initial)
                holder_result, holder_points = run_holder_icp(aligned, reference, reference_system['holder_cloud'], [TRIOS_HOLDER_ICP_WIDE, TRIOS_HOLDER_ICP_MEDIUM, TRIOS_HOLDER_ICP_FINE])
                if holder_result is None:
                    continue
                total_transform = holder_result.transformation @ initial
                fitness = holder_result.fitness
                rmse = holder_result.inlier_rmse
                score = fitness - 0.1 * rmse
                if best is None or score > best['score']:
                    best = {'transform': total_transform, 'method': 'RGB_COARSE+HOLDER', 'score': score, 'global_fitness': np.nan, 'global_rmse': np.nan, 'full_fitness': np.nan, 'full_rmse': np.nan, 'holder_fitness': fitness, 'holder_rmse': rmse, 'holder_points': holder_points, 'rgb_cluster': cluster_index, 'rgb_cluster_score': candidate['score'], 'angle_deg': angle_deg, 'normal_sign': normal_sign}
    return best

def registration_is_strong(result):
    if result is None:
        return False
    return result['holder_fitness'] >= TRIOS_PRIMARY_MIN_HOLDER_FITNESS and result['holder_rmse'] <= TRIOS_PRIMARY_MAX_HOLDER_RMSE

def registration_score(result):
    if result is None:
        return -np.inf
    return result['holder_fitness'] - 0.1 * result['holder_rmse']

def create_context(aligned_mesh, reference, z_offset, canonical):
    points = np.asarray(aligned_mesh.points)
    x, y, z = local_coordinates(points, reference['center'], reference['u_axis'], reference['v_axis'], reference['normal'])
    half_width = reference['radius'] * CONTEXT_SIZE_FACTOR
    mask = (np.abs(x) <= half_width) & (np.abs(y) <= half_width) & (np.abs(z - z_offset) <= CONTEXT_HEIGHT_TOLERANCE)
    context = aligned_mesh.extract_points(mask, adjacent_cells=False, include_cells=True).extract_surface().clean(point_merging=False)
    context, _ = prepare_embedded_rgb(context)
    context, _ = activate_tcoords(context)
    context.transform(canonical, inplace=True)
    return context

def add_canonical_circle(plotter, radius):
    theta = np.linspace(0, 2 * np.pi, 300)
    points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full(len(theta), 0.05)])
    line = pv.lines_from_points(points, close=True)
    plotter.add_mesh(line, color='red', line_width=5)

def define_reference_roi(mesh, texture, scanner, sample_id):
    while True:
        selected_points = []
        plotter = pv.Plotter(window_size=(1300, 950))
        plotter.set_background('white')
        add_scan_mesh(plotter, mesh, texture)

        def redraw():
            try:
                plotter.remove_actor('roi_points', reset_camera=False)
            except Exception:
                pass
            if selected_points:
                plotter.add_points(np.asarray(selected_points), color='red', point_size=20, render_points_as_spheres=True, name='roi_points')
            plotter.render()

        def add_point(point):
            if point is None:
                return
            selected_points.append(np.asarray(point))
            print(f'Added point {len(selected_points)}')
            redraw()

        def undo():
            if selected_points:
                selected_points.pop()
            redraw()

        def clear():
            selected_points.clear()
            redraw()
        plotter.add_text(f'{scanner} | {sample_id}\n\nDefine TRUE outer specimen edge\n\nLEFT CLICK = add point\nU = undo\nC = clear\nQ = finish\n\nUse approximately 8-12 points.', position='upper_left', font_size=11)
        plotter.enable_point_picking(callback=add_point, show_message=False, show_point=False, left_clicking=True, pickable_window=False)
        plotter.add_key_event('u', undo)
        plotter.add_key_event('c', clear)
        plotter.reset_camera()
        plotter.show()
        if len(selected_points) < MIN_REFERENCE_EDGE_POINTS:
            print('Not enough points.')
            continue
        reference = fit_reference_circle(selected_points, mesh)
        points = np.asarray(mesh.points)
        x, y, z = local_coordinates(points, reference['center'], reference['u_axis'], reference['v_axis'], reference['normal'])
        radial = np.sqrt(x ** 2 + y ** 2)
        central = radial < reference['radius'] * 0.5
        if np.sum(central) < 20:
            print('Could not estimate reference surface.')
            continue
        reference['z_offset'] = float(np.median(z[central]))
        specimen, z_offset, roi_radius, canonical = extract_specimen(mesh, reference)
        reference['z_offset'] = z_offset
        reference['roi_radius'] = roi_radius
        context = create_context(mesh, reference, z_offset, canonical)
        qc = pv.Plotter(shape=(1, 2), window_size=(1700, 850))
        qc.subplot(0, 0)
        add_scan_mesh(qc, context, texture)
        add_canonical_circle(qc, roi_radius)
        qc.view_xy()
        qc.camera.parallel_projection = True
        qc.reset_camera()
        qc.subplot(0, 1)
        add_scan_mesh(qc, specimen, texture)
        qc.view_xy()
        qc.camera.parallel_projection = True
        qc.reset_camera()
        qc.show()
        answer = input(f'\nIs {scanner} reference ROI correct? [y/n]: ')
        if answer.strip().lower().startswith('y'):
            return reference

def save_reference(scanner, reference, record):
    file = get_template_file(scanner)
    np.savez(file, scanner=scanner, center=reference['center'], normal=reference['normal'], u_axis=reference['u_axis'], v_axis=reference['v_axis'], radius=reference['radius'], roi_radius=reference['roi_radius'], z_offset=reference['z_offset'], reference_sample_id=record['sample_id'], reference_ply=str(record['ply_path']))

def load_reference(scanner):
    data = np.load(get_template_file(scanner), allow_pickle=False)
    return {'center': np.asarray(data['center']), 'normal': np.asarray(data['normal']), 'u_axis': np.asarray(data['u_axis']), 'v_axis': np.asarray(data['v_axis']), 'radius': float(data['radius']), 'roi_radius': float(data['roi_radius']), 'z_offset': float(data['z_offset']), 'reference_sample_id': str(data['reference_sample_id'])}

def save_qc_preview(context, specimen, texture, roi_radius, output_file):
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(PREVIEW_WIDTH, PREVIEW_HEIGHT))
    plotter.subplot(0, 0)
    plotter.set_background('white')
    add_scan_mesh(plotter, context, texture)
    add_canonical_circle(plotter, roi_radius)
    plotter.add_text('Original + ROI', font_size=9)
    plotter.view_xy()
    plotter.camera.parallel_projection = True
    plotter.reset_camera()
    plotter.subplot(0, 1)
    plotter.set_background('white')
    add_scan_mesh(plotter, specimen, texture)
    plotter.add_text('Extracted ROI', font_size=9)
    plotter.view_xy()
    plotter.camera.parallel_projection = True
    plotter.reset_camera()
    plotter.screenshot(str(output_file))
    plotter.close()

def make_contact_sheets(records, output_folder, prefix):
    if not records:
        return
    output_folder.mkdir(parents=True, exist_ok=True)
    per_page = CONTACT_SHEET_COLUMNS * CONTACT_SHEET_ROWS
    pages = math.ceil(len(records) / per_page)
    font = ImageFont.load_default()
    for page in range(pages):
        subset = records[page * per_page:(page + 1) * per_page]
        sheet = Image.new('RGB', (CONTACT_SHEET_COLUMNS * CONTACT_SHEET_TILE_WIDTH, CONTACT_SHEET_ROWS * CONTACT_SHEET_TILE_HEIGHT), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, record in enumerate(subset):
            row = i // CONTACT_SHEET_COLUMNS
            col = i % CONTACT_SHEET_COLUMNS
            x0 = col * CONTACT_SHEET_TILE_WIDTH
            y0 = row * CONTACT_SHEET_TILE_HEIGHT
            preview = record.get('qc_preview')
            if preview and Path(preview).exists():
                image = Image.open(preview).convert('RGB')
                image.thumbnail((CONTACT_SHEET_TILE_WIDTH - 20, CONTACT_SHEET_TILE_HEIGHT - 65))
                sheet.paste(image, (x0 + (CONTACT_SHEET_TILE_WIDTH - image.width) // 2, y0 + 5))
            label = f"{record['scanner']} | {record['sample_id']} | {record['status']}\n{record.get('registration_method', '')}"
            draw.text((x0 + 8, y0 + CONTACT_SHEET_TILE_HEIGHT - 52), label, fill='black', font=font)
            draw.rectangle([x0, y0, x0 + CONTACT_SHEET_TILE_WIDTH - 1, y0 + CONTACT_SHEET_TILE_HEIGHT - 1], outline='gray', width=1)
        output = output_folder / f'{prefix}_{page + 1:03d}.jpg'
        sheet.save(output, quality=92)

def evaluate_registration(mesh, reference, registration):
    aligned_mesh = mesh.copy(deep=True)
    aligned_mesh.transform(registration['transform'], inplace=True)
    aligned_mesh, _ = prepare_embedded_rgb(aligned_mesh)
    aligned_mesh, _ = activate_tcoords(aligned_mesh)
    specimen, z_offset, roi_radius, canonical = extract_specimen(aligned_mesh, reference)
    quality = assess_roi_quality(specimen, roi_radius)
    return {'registration': registration, 'aligned_mesh': aligned_mesh, 'specimen': specimen, 'z_offset': z_offset, 'roi_radius': roi_radius, 'canonical': canonical, 'quality': quality}
