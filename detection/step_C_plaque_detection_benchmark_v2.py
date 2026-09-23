"""
Step C - Weakly supervised plaque-detection benchmark
======================================================

Purpose
-------
Compare six plaque-detection approaches using the SAME specimen-level
cross-validation splits for every method, separately for each scanner.

Weak supervision
----------------
- Baseline (t0): plaque-positive endpoint
- Clean (t1 for G1-G2; t2 for G3-G5): plaque-negative endpoint
- Partial (t1 for G3-G5): unlabeled mixed image; used for inference and,
  for the unsupervised GMM only, may be included in fitting without labels.

Critical anti-leakage rule
--------------------------
All stages from one physical specimen stay in the same fold.  A specimen such
as G3 sample 4 is either training or test; its other stages never cross folds.
Models are fitted per scanner by default.

Methods
-------
M1  a* threshold
M2  Mahalanobis distance from clean appearance
M3  Plaque-vs-clean Lab reference score
M4  Logistic regression on [L*, a*, b*]
M5  HistGradientBoosting nonlinear classifier on [L*, a*, b*]
M6  Gaussian mixture model on Lab; components are interpreted using only
    training Baseline/Clean endpoint enrichment.

Input
-----
The script discovers sample folders whose names look like:
    BCG1T0-1, BCG3T1-4, ...
under scanner folders beneath DATA_ROOT.

It tries, in this order:
1) geometry.npz inside the sample folder
2) roi.vtp inside the sample folder

The loader supports either direct Lab values or RGB vertex colours.  If your
preprocessing uses different array/key names, edit LAB_KEY_CANDIDATES and
RGB_KEY_CANDIDATES near the top.

Output
------
OUTPUT_ROOT/
  tables/
    scan_level_predictions.csv
    endpoint_metrics_by_scanner_method.csv
    stage_ordering_by_scanner_method.csv
    scanner_pairwise_agreement.csv
    scanner_icc_by_method.csv
    method_comparison_summary.csv
  predictions/<scanner>/<sample>.npz
  qc/<scanner>/<sample>.png
  fold_models/<scanner>/... metadata CSV files

Important interpretation
------------------------
Without manually annotated plaque masks, these are PLAQUE-LIKE scores and
coverage estimates, not validated true plaque segmentation.  When manual masks
are added later, Dice/IoU/area error can be appended to the same benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
import math
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr

from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from skimage.color import rgb2lab
except Exception as exc:
    raise ImportError(
        "scikit-image is required. Install with: pip install scikit-image"
    ) from exc

try:
    import pyvista as pv
except Exception:
    pv = None


# =============================================================================
# USER SETTINGS
# =============================================================================

DATA_ROOT = Path(
    r"C:\Users\johan\repos\biocal3d\data\02-09-2026-Data collected"
)

# If your repository is still named biocad3d on this computer, change only the
# path above. Everything else is discovered automatically.

OUTPUT_ROOT = DATA_ROOT / "_plaque_detection_benchmark"
TABLE_ROOT = OUTPUT_ROOT / "tables"
PRED_ROOT = OUTPUT_ROOT / "predictions"
QC_ROOT = OUTPUT_ROOT / "qc"
MODEL_ROOT = OUTPUT_ROOT / "fold_models"

RANDOM_STATE = 42
N_SPLITS = 5

# Equal point sampling prevents dense scans from dominating training.
MAX_TRAIN_POINTS_PER_SCAN = 5000
MAX_EVAL_POINTS_PER_SCAN = 10000

# GMM complexity. 3 is a useful first test: clean / plaque / intermediate or
# illumination-related appearance. You can later benchmark 2,3,4 separately.
GMM_COMPONENTS = 3

# Optional outputs. Prediction NPZs can be large but are useful for later review.
SAVE_PREDICTION_NPZ = True
SAVE_QC_FIGURES = True
QC_MAX_POINTS = 18000
QC_ONLY_PARTIAL = False

# If mesh coordinates are in millimetres (your previous meshes appear to be),
# surface area is reported in mm^2. Set False if units are unknown.
MESH_UNITS_ARE_MM = True

# Candidate keys/array names used by the loader.
LAB_KEY_CANDIDATES = [
    "lab", "Lab", "LAB", "cielab", "CIELAB",
]
L_KEY_CANDIDATES = ["L", "L_star", "Lstar", "lab_L"]
A_KEY_CANDIDATES = ["a", "a_star", "astar", "lab_a"]
B_KEY_CANDIDATES = ["b", "b_star", "bstar", "lab_b"]
RGB_KEY_CANDIDATES = [
    "rgb", "RGB", "colors", "Colors", "colour", "color",
    "vertex_colors", "VertexColors", "texture_rgb",
]
VERTEX_KEY_CANDIDATES = ["vertices", "verts", "points", "xyz"]
FACE_KEY_CANDIDATES = ["faces", "triangles", "cells"]
MASK_KEY_CANDIDATES = ["roi_mask", "valid_mask", "mask"]

METHODS = [
    "M1_a_threshold",
    "M2_clean_mahalanobis",
    "M3_lab_reference_score",
    "M4_logistic_lab",
    "M5_hist_gradient_boosting",
    "M6_gmm_lab",
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ScanData:
    scanner: str
    sample_folder: Path
    sample_name: str
    group: int
    time_idx: int
    specimen_num: int
    physical_id: str
    stage: str
    lab: np.ndarray          # N x 3
    rgb: Optional[np.ndarray]  # N x 3, float 0..1 if available
    vertices: np.ndarray     # N x 3
    faces: Optional[np.ndarray]  # M x 3 triangle indices


# =============================================================================
# HELPERS: PARSING AND DISCOVERY
# =============================================================================

# Accept the naming variants used across the BioCal3D folders, e.g.
# BCG1T0-1, 1T0-1, G1T0-1, BCG1T0_1.
SAMPLE_RE = re.compile(
    r"(?:BCG)?G?(?P<group>\d+)\s*T(?P<time>\d+)\s*[-_]\s*(?P<sample>\d+)",
    re.IGNORECASE,
)


def stage_from_group_time(group: int, time_idx: int) -> str:
    """Map the experimental naming scheme to biological stage."""
    if group in (1, 2):
        if time_idx == 0:
            return "baseline"
        if time_idx == 1:
            return "clean"
    elif group in (3, 4, 5):
        if time_idx == 0:
            return "baseline"
        if time_idx == 1:
            return "partial"
        if time_idx == 2:
            return "clean"
    return f"unknown_t{time_idx}"


def parse_sample_name(name: str) -> Optional[Tuple[int, int, int, str, str]]:
    m = SAMPLE_RE.search(name)
    if not m:
        return None
    group = int(m.group("group"))
    time_idx = int(m.group("time"))
    sample = int(m.group("sample"))
    physical_id = f"G{group}_S{sample}"
    stage = stage_from_group_time(group, time_idx)
    return group, time_idx, sample, physical_id, stage


def _infer_scanner_from_path(path: Path, root: Path) -> str:
    """Infer scanner from any ancestor in the path, not only root/<scanner>."""
    known = {
        "trios3": "TRIOS3",
        "trios 3": "TRIOS3",
        "trios5": "TRIOS5",
        "trios 5": "TRIOS5",
        "itero": "iTERO",
        "labscanner": "LABscanner",
        "lab scanner": "LABscanner",
    }

    ancestors = [path] + list(path.parents)
    for anc in ancestors:
        try:
            anc.relative_to(root)
        except ValueError:
            continue
        name = anc.name.lower().replace("_", " ").replace("-", " ").strip()
        compact = name.replace(" ", "")
        for key, canonical in known.items():
            if key.replace(" ", "") in compact:
                return canonical

    # Fallback: first directory underneath DATA_ROOT.
    try:
        rel = path.relative_to(root)
        if len(rel.parts) > 1:
            return rel.parts[0]
    except ValueError:
        pass
    return "UNKNOWN_SCANNER"


def _nearest_sample_ancestor(path: Path, root: Path) -> Optional[Path]:
    """Find the nearest ancestor whose name contains a BioCal3D sample ID."""
    current = path if path.is_dir() else path.parent
    while True:
        if SAMPLE_RE.search(current.name):
            return current
        if current == root or root not in current.parents:
            break
        current = current.parent
    return None


def discover_sample_folders(root: Path) -> List[Tuple[str, Path]]:
    """
    Return unique (scanner, sample_folder) pairs.

    Discovery is intentionally file-first. The previous version assumed that
    geometry.npz/roi.vtp sat directly inside a folder named BCGxTy-z. In the
    real preprocessing tree those files may be nested one or more levels below
    the sample folder, so we locate the processed files first and then walk
    upward to recover the sample and scanner.
    """
    if not root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")

    assets = []
    assets.extend(root.rglob("geometry.npz"))
    assets.extend(root.rglob("roi.vtp"))

    # Do not accidentally rediscover files produced by this benchmark itself.
    assets = [a for a in assets if OUTPUT_ROOT not in a.parents]

    found = {}
    unmatched_examples = []

    for asset in assets:
        sample_folder = _nearest_sample_ancestor(asset, root)
        if sample_folder is None:
            if len(unmatched_examples) < 10:
                unmatched_examples.append(asset)
            continue

        parsed = parse_sample_name(sample_folder.name)
        if parsed is None:
            continue
        group, time_idx, specimen_num, physical_id, stage = parsed
        scanner = _infer_scanner_from_path(asset.parent, root)

        # De-duplicate geometry.npz + roi.vtp belonging to the same scan.
        key = (scanner.lower(), group, time_idx, specimen_num)
        found[key] = (scanner, sample_folder)

    print(f"Processed colour/geometry files found recursively: {len(assets)}")
    if assets and not found:
        print("No sample IDs could be recovered from their parent paths.")
        if unmatched_examples:
            print("Example processed-file paths:")
            for p in unmatched_examples:
                print(f"  - {p}")

    return sorted(found.values(), key=lambda x: (x[0].lower(), x[1].name.lower()))


# =============================================================================
# HELPERS: COLOUR / MESH LOADING
# =============================================================================


def _first_existing(mapping, candidates: Iterable[str]):
    for key in candidates:
        if key in mapping:
            return key
    return None


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] < 3:
        raise ValueError(f"RGB must be N x 3, got {rgb.shape}")
    rgb = rgb[:, :3]
    finite = np.isfinite(rgb).all(axis=1)
    if finite.any():
        max_val = np.nanpercentile(rgb[finite], 99.9)
        if max_val > 1.5:
            rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = normalize_rgb(rgb)
    return rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)


def _coerce_faces_array(faces: np.ndarray) -> Optional[np.ndarray]:
    """Accept Mx3 triangles or PyVista flat [3,i,j,k,3,...] format."""
    if faces is None:
        return None
    arr = np.asarray(faces)
    if arr.size == 0:
        return None
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return arr[:, :3].astype(np.int64)
    arr = arr.ravel().astype(np.int64)
    tris = []
    i = 0
    while i < len(arr):
        n = int(arr[i])
        ids = arr[i + 1:i + 1 + n]
        if n == 3 and len(ids) == 3:
            tris.append(ids)
        i += n + 1
    return np.asarray(tris, dtype=np.int64) if tris else None


def load_from_npz(path: Path) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]]:
    """Return lab, rgb, vertices, faces; None if colour data absent."""
    z = np.load(path, allow_pickle=False)
    keys = set(z.files)

    vkey = _first_existing(keys, VERTEX_KEY_CANDIDATES)
    if vkey is None:
        return None
    vertices = np.asarray(z[vkey], dtype=float)

    fkey = _first_existing(keys, FACE_KEY_CANDIDATES)
    faces = _coerce_faces_array(z[fkey]) if fkey else None

    mask_key = _first_existing(keys, MASK_KEY_CANDIDATES)
    mask = np.asarray(z[mask_key], dtype=bool) if mask_key else None

    lab = None
    rgb = None

    lab_key = _first_existing(keys, LAB_KEY_CANDIDATES)
    if lab_key:
        arr = np.asarray(z[lab_key], dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            lab = arr[:, :3]

    if lab is None:
        lk = _first_existing(keys, L_KEY_CANDIDATES)
        ak = _first_existing(keys, A_KEY_CANDIDATES)
        bk = _first_existing(keys, B_KEY_CANDIDATES)
        if lk and ak and bk:
            lab = np.column_stack([z[lk], z[ak], z[bk]]).astype(float)

    rgb_key = _first_existing(keys, RGB_KEY_CANDIDATES)
    if rgb_key:
        rgb = normalize_rgb(np.asarray(z[rgb_key]))
        if lab is None:
            lab = rgb_to_lab(rgb)

    if lab is None:
        return None

    n = len(lab)
    if len(vertices) != n:
        # If geometry contains full mesh and colour only ROI, do not guess.
        return None

    if mask is not None and len(mask) == n:
        vertices = vertices[mask]
        lab = lab[mask]
        if rgb is not None:
            rgb = rgb[mask]
        if faces is not None:
            # Face remapping after arbitrary mask is non-trivial. For reliable
            # area estimates, use roi.vtp or save already-cropped faces.
            faces = None

    return lab, rgb, vertices, faces


def _find_vtp_rgb(mesh: pv.PolyData) -> Optional[np.ndarray]:
    names = list(mesh.point_data.keys())
    for candidate in RGB_KEY_CANDIDATES:
        if candidate in names:
            arr = np.asarray(mesh.point_data[candidate])
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return normalize_rgb(arr[:, :3])

    # Fallback: first N x 3 point array that looks like colour values.
    for name in names:
        arr = np.asarray(mesh.point_data[name])
        if arr.ndim == 2 and arr.shape[1] in (3, 4):
            finite = np.isfinite(arr[:, :3])
            if finite.any():
                lo = np.nanmin(arr[:, :3])
                hi = np.nanmax(arr[:, :3])
                if lo >= 0 and hi <= 255.5:
                    return normalize_rgb(arr[:, :3])
    return None


def _find_vtp_lab(mesh: pv.PolyData) -> Optional[np.ndarray]:
    names = list(mesh.point_data.keys())
    for candidate in LAB_KEY_CANDIDATES:
        if candidate in names:
            arr = np.asarray(mesh.point_data[candidate], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return arr[:, :3]

    lk = next((k for k in L_KEY_CANDIDATES if k in names), None)
    ak = next((k for k in A_KEY_CANDIDATES if k in names), None)
    bk = next((k for k in B_KEY_CANDIDATES if k in names), None)
    if lk and ak and bk:
        return np.column_stack([
            mesh.point_data[lk], mesh.point_data[ak], mesh.point_data[bk]
        ]).astype(float)
    return None


def load_from_vtp(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    if pv is None:
        raise ImportError(
            "PyVista is required only when loading roi.vtp. Install with: "
            "pip install pyvista, or save Lab/RGB in geometry.npz."
        )
    mesh = pv.read(path)
    vertices = np.asarray(mesh.points, dtype=float)
    faces = _coerce_faces_array(np.asarray(mesh.faces))

    lab = _find_vtp_lab(mesh)
    rgb = _find_vtp_rgb(mesh)
    if lab is None and rgb is not None:
        lab = rgb_to_lab(rgb)

    if lab is None:
        raise ValueError(
            f"No Lab or RGB point colour array found in {path}. "
            f"Available point arrays: {list(mesh.point_data.keys())}. "
            "Save vertex RGB/Lab in roi.vtp or geometry.npz, or add the array "
            "name to the candidate lists at the top of the script."
        )
    return lab, rgb, vertices, faces


def load_scan(scanner: str, folder: Path) -> ScanData:
    parsed = parse_sample_name(folder.name)
    if parsed is None:
        raise ValueError(f"Could not parse sample folder: {folder.name}")
    group, time_idx, specimen_num, physical_id, stage = parsed

    loaded = None

    # Processed files may sit directly in the sample folder or in a nested
    # preprocessing-output subfolder. Prefer geometry.npz when it contains
    # usable colours; otherwise fall back to roi.vtp.
    npz_candidates = list(folder.rglob("geometry.npz"))
    vtp_candidates = list(folder.rglob("roi.vtp"))

    npz_errors = []
    for npz_path in npz_candidates:
        try:
            loaded = load_from_npz(npz_path)
            if loaded is not None:
                break
        except Exception as exc:
            npz_errors.append(f"{npz_path}: {exc}")

    if loaded is None:
        vtp_errors = []
        for vtp_path in vtp_candidates:
            try:
                loaded = load_from_vtp(vtp_path)
                if loaded is not None:
                    break
            except Exception as exc:
                vtp_errors.append(f"{vtp_path}: {exc}")

        if loaded is None:
            details = ""
            all_errors = npz_errors + vtp_errors
            if all_errors:
                details = "\nTried:\n  " + "\n  ".join(all_errors[:6])
            raise FileNotFoundError(
                f"No usable coloured geometry.npz or roi.vtp beneath {folder}." + details
            )

    lab, rgb, vertices, faces = loaded

    valid = (
        np.isfinite(lab).all(axis=1)
        & np.isfinite(vertices).all(axis=1)
        & (lab[:, 0] >= 0)
        & (lab[:, 0] <= 110)
    )
    if rgb is not None:
        valid &= np.isfinite(rgb).all(axis=1)

    lab = lab[valid]
    vertices = vertices[valid]
    if rgb is not None:
        rgb = rgb[valid]

    # Filtering vertices invalidates face indexing, so keep faces only if every
    # vertex survived. In normal cropped ROI files this should usually be true.
    if faces is not None and not np.all(valid):
        faces = None

    if len(lab) < 50:
        raise ValueError(f"Too few valid coloured ROI points ({len(lab)}) in {folder}")

    return ScanData(
        scanner=scanner,
        sample_folder=folder,
        sample_name=folder.name,
        group=group,
        time_idx=time_idx,
        specimen_num=specimen_num,
        physical_id=physical_id,
        stage=stage,
        lab=lab,
        rgb=rgb,
        vertices=vertices,
        faces=faces,
    )


# =============================================================================
# TRAINING DATA SAMPLING
# =============================================================================


def rng_for(*parts) -> np.random.Generator:
    seed = RANDOM_STATE
    for part in parts:
        seed = (seed * 1315423911 + hash(str(part))) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def sample_rows(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) <= n:
        return x
    idx = rng.choice(len(x), size=n, replace=False)
    return x[idx]


def build_endpoint_training(scans: List[ScanData], scanner: str, fold: int):
    xs = []
    ys = []
    scan_ids = []
    for scan in scans:
        if scan.stage not in {"baseline", "clean"}:
            continue
        r = rng_for(scanner, fold, scan.sample_name, "endpoint")
        pts = sample_rows(scan.lab, MAX_TRAIN_POINTS_PER_SCAN, r)
        label = 1 if scan.stage == "baseline" else 0
        xs.append(pts)
        ys.append(np.full(len(pts), label, dtype=np.int8))
        scan_ids.extend([scan.sample_name] * len(pts))

    if not xs:
        raise RuntimeError("No Baseline/Clean training data found.")
    X = np.vstack(xs)
    y = np.concatenate(ys)
    return X, y, np.asarray(scan_ids, dtype=object)


def build_allstage_gmm_training(scans: List[ScanData], scanner: str, fold: int) -> np.ndarray:
    xs = []
    for scan in scans:
        if scan.stage.startswith("unknown"):
            continue
        r = rng_for(scanner, fold, scan.sample_name, "gmm")
        xs.append(sample_rows(scan.lab, MAX_TRAIN_POINTS_PER_SCAN, r))
    if not xs:
        raise RuntimeError("No training data for GMM.")
    return np.vstack(xs)


# =============================================================================
# THRESHOLD SELECTION
# =============================================================================


def optimal_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    """Youden-J threshold from training endpoint data only."""
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    good = np.isfinite(score)
    y_true = y_true[good]
    score = score[good]
    if len(np.unique(y_true)) < 2:
        return float(np.nanmedian(score))
    fpr, tpr, thresholds = roc_curve(y_true, score)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    thr = thresholds[idx]
    if not np.isfinite(thr):
        finite_thr = thresholds[np.isfinite(thresholds)]
        thr = np.nanmedian(finite_thr) if len(finite_thr) else np.nanmedian(score)
    return float(thr)


# =============================================================================
# SIX PLAQUE DETECTORS
# =============================================================================


class Detector:
    name: str
    threshold: float

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = self.score(X)
        return s, (s >= self.threshold).astype(np.uint8)


class AThresholdDetector(Detector):
    name = "M1_a_threshold"

    def __init__(self):
        self.orientation = 1.0
        self.threshold = 0.0

    def fit(self, X, y):
        base_mean = np.mean(X[y == 1, 1])
        clean_mean = np.mean(X[y == 0, 1])
        self.orientation = 1.0 if base_mean >= clean_mean else -1.0
        s = self.orientation * X[:, 1]
        self.threshold = optimal_threshold(y, s)
        return self

    def score(self, X):
        return self.orientation * X[:, 1]


class CleanMahalanobisDetector(Detector):
    name = "M2_clean_mahalanobis"

    def __init__(self):
        self.mean_ = None
        self.precision_ = None
        self.threshold = 0.0

    def fit(self, X, y):
        Xc = X[y == 0]
        cov = LedoitWolf().fit(Xc)
        self.mean_ = cov.location_.copy()
        self.precision_ = cov.precision_.copy()
        s = self.score(X)
        self.threshold = optimal_threshold(y, s)
        return self

    def score(self, X):
        d = X - self.mean_
        d2 = np.einsum("ni,ij,nj->n", d, self.precision_, d)
        return np.sqrt(np.clip(d2, 0, None))


class LabReferenceDetector(Detector):
    name = "M3_lab_reference_score"

    def __init__(self):
        self.mu_plaque = None
        self.mu_clean = None
        self.threshold = 0.5

    def fit(self, X, y):
        self.mu_plaque = X[y == 1].mean(axis=0)
        self.mu_clean = X[y == 0].mean(axis=0)
        s = self.score(X)
        self.threshold = optimal_threshold(y, s)
        return self

    def score(self, X):
        d_p = np.linalg.norm(X - self.mu_plaque, axis=1)
        d_c = np.linalg.norm(X - self.mu_clean, axis=1)
        return d_c / (d_c + d_p + 1e-12)


class LogisticLabDetector(Detector):
    name = "M4_logistic_lab"

    def __init__(self):
        self.model = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ])
        self.threshold = 0.5

    def fit(self, X, y):
        self.model.fit(X, y)
        s = self.score(X)
        self.threshold = optimal_threshold(y, s)
        return self

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]


class HistGBDetector(Detector):
    name = "M5_hist_gradient_boosting"

    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.06,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )
        self.threshold = 0.5

    def fit(self, X, y):
        w = compute_sample_weight(class_weight="balanced", y=y)
        self.model.fit(X, y, sample_weight=w)
        s = self.score(X)
        self.threshold = optimal_threshold(y, s)
        return self

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]


class GMMDetector(Detector):
    name = "M6_gmm_lab"

    def __init__(self, n_components=3):
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=1e-5,
            n_init=3,
            random_state=RANDOM_STATE,
        )
        self.component_plaque_prob = None
        self.threshold = 0.5

    def fit(self, X_allstage, X_endpoint, y_endpoint):
        self.gmm.fit(X_allstage)

        responsibilities = self.gmm.predict_proba(X_endpoint)
        # Soft counts per component, with a small Beta(1,1) prior.
        plaque = responsibilities[y_endpoint == 1].sum(axis=0)
        clean = responsibilities[y_endpoint == 0].sum(axis=0)
        self.component_plaque_prob = (plaque + 1.0) / (plaque + clean + 2.0)

        s = self.score(X_endpoint)
        self.threshold = optimal_threshold(y_endpoint, s)
        return self

    def score(self, X):
        r = self.gmm.predict_proba(X)
        return r @ self.component_plaque_prob


def fit_detectors(train_scans: List[ScanData], scanner: str, fold: int) -> Dict[str, Detector]:
    X, y, _ = build_endpoint_training(train_scans, scanner, fold)

    detectors: Dict[str, Detector] = {}
    detectors["M1_a_threshold"] = AThresholdDetector().fit(X, y)
    detectors["M2_clean_mahalanobis"] = CleanMahalanobisDetector().fit(X, y)
    detectors["M3_lab_reference_score"] = LabReferenceDetector().fit(X, y)
    detectors["M4_logistic_lab"] = LogisticLabDetector().fit(X, y)
    detectors["M5_hist_gradient_boosting"] = HistGBDetector().fit(X, y)

    X_all = build_allstage_gmm_training(train_scans, scanner, fold)
    detectors["M6_gmm_lab"] = GMMDetector(GMM_COMPONENTS).fit(X_all, X, y)
    return detectors


# =============================================================================
# AREA / COVERAGE
# =============================================================================


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def plaque_area_from_vertex_mask(scan: ScanData, mask: np.ndarray) -> Tuple[float, float, str]:
    """Return plaque area, total area, and area_method."""
    if scan.faces is not None and len(scan.faces) > 0:
        try:
            areas = triangle_areas(scan.vertices, scan.faces)
            face_plaque = mask[scan.faces].mean(axis=1) >= 0.5
            return float(areas[face_plaque].sum()), float(areas.sum()), "triangle_area"
        except Exception:
            pass

    # Fallback: point fraction. Area itself becomes NaN, but coverage is still usable.
    return float("nan"), float("nan"), "vertex_fraction"


def summarize_prediction(scan: ScanData, method: str, fold: int,
                         score: np.ndarray, mask: np.ndarray, threshold: float) -> dict:
    plaque_area, total_area, area_method = plaque_area_from_vertex_mask(scan, mask)

    if area_method == "triangle_area" and total_area > 0:
        coverage = 100.0 * plaque_area / total_area
    else:
        coverage = 100.0 * float(np.mean(mask))

    return {
        "scanner": scan.scanner,
        "sample_name": scan.sample_name,
        "physical_id": scan.physical_id,
        "group": scan.group,
        "time_idx": scan.time_idx,
        "stage": scan.stage,
        "fold": fold,
        "method": method,
        "n_vertices": len(scan.lab),
        "threshold": threshold,
        "mean_score": float(np.mean(score)),
        "median_score": float(np.median(score)),
        "plaque_like_coverage_pct": coverage,
        "plaque_like_area": plaque_area,
        "total_roi_area": total_area,
        "area_unit": "mm^2" if MESH_UNITS_ARE_MM else "mesh_unit^2",
        "area_method": area_method,
    }


# =============================================================================
# QC FIGURES / PREDICTION STORAGE
# =============================================================================


def project_vertices_2d(vertices: np.ndarray) -> np.ndarray:
    if len(vertices) < 3:
        return vertices[:, :2]
    return PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(vertices)


def save_prediction_npz(scan: ScanData, pred_by_method: dict):
    out_dir = PRED_ROOT / scan.scanner
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "vertices": scan.vertices.astype(np.float32),
        "lab": scan.lab.astype(np.float32),
    }
    if scan.rgb is not None:
        payload["rgb"] = scan.rgb.astype(np.float32)
    for method, pred in pred_by_method.items():
        payload[f"{method}__score"] = pred["score"].astype(np.float32)
        payload[f"{method}__mask"] = pred["mask"].astype(np.uint8)
        payload[f"{method}__threshold"] = np.array([pred["threshold"]], dtype=np.float32)
    np.savez_compressed(out_dir / f"{scan.sample_name}.npz", **payload)


def save_qc_figure(scan: ScanData, pred_by_method: dict):
    if QC_ONLY_PARTIAL and scan.stage != "partial":
        return

    out_dir = QC_ROOT / scan.scanner
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(scan.vertices)
    r = rng_for(scan.scanner, scan.sample_name, "qc")
    if n > QC_MAX_POINTS:
        idx = r.choice(n, size=QC_MAX_POINTS, replace=False)
    else:
        idx = np.arange(n)

    xy = project_vertices_2d(scan.vertices[idx])

    ncols = 3
    panels = 1 + len(METHODS)
    nrows = math.ceil(panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.6 * nrows))
    axes = np.asarray(axes).ravel()

    ax = axes[0]
    if scan.rgb is not None:
        ax.scatter(xy[:, 0], xy[:, 1], c=scan.rgb[idx], s=3, linewidths=0)
        ax.set_title("Original vertex colour")
    else:
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=scan.lab[idx, 1], s=3, linewidths=0)
        fig.colorbar(sc, ax=ax, fraction=0.046)
        ax.set_title("a* (RGB unavailable)")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    for j, method in enumerate(METHODS, start=1):
        ax = axes[j]
        s = pred_by_method[method]["score"][idx]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=s, s=3, linewidths=0)
        cov = pred_by_method[method]["coverage_pct"]
        ax.set_title(f"{method}\ncoverage={cov:.1f}%")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        fig.colorbar(sc, ax=ax, fraction=0.046)

    for ax in axes[panels:]:
        ax.axis("off")

    fig.suptitle(
        f"{scan.scanner} | {scan.sample_name} | {scan.stage} | {scan.physical_id}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / f"{scan.sample_name}.png", dpi=180)
    plt.close(fig)


# =============================================================================
# EVALUATION
# =============================================================================


def endpoint_metrics(scan_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scanner, method), df in scan_df.groupby(["scanner", "method"]):
        d = df[df["stage"].isin(["baseline", "clean"])].copy()
        if len(d) < 4 or d["stage"].nunique() < 2:
            continue
        y = (d["stage"] == "baseline").astype(int).to_numpy()
        # Coverage is directly comparable across all methods even though raw score
        # scales differ (e.g., a* versus probability).
        s = d["plaque_like_coverage_pct"].to_numpy(float)
        try:
            auc = roc_auc_score(y, s)
        except Exception:
            auc = np.nan
        pred = (s >= 50.0).astype(int)
        bal = balanced_accuracy_score(y, pred)
        rows.append({
            "scanner": scanner,
            "method": method,
            "n_endpoint_scans": len(d),
            "endpoint_scan_auc_using_coverage": auc,
            "endpoint_balanced_accuracy_at_50pct_coverage": bal,
            "mean_baseline_coverage_pct": d.loc[d.stage == "baseline", "plaque_like_coverage_pct"].mean(),
            "mean_clean_coverage_pct": d.loc[d.stage == "clean", "plaque_like_coverage_pct"].mean(),
        })
    return pd.DataFrame(rows)


def stage_ordering_metrics(scan_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scanner, method), df in scan_df.groupby(["scanner", "method"]):
        wide = df.pivot_table(
            index="physical_id",
            columns="stage",
            values="plaque_like_coverage_pct",
            aggfunc="mean",
        )
        n_bc = 0
        ok_bc = 0
        n_bpc = 0
        ok_bpc = 0
        diffs = []
        for _, row in wide.iterrows():
            if "baseline" in row.index and "clean" in row.index and pd.notna(row.get("baseline")) and pd.notna(row.get("clean")):
                n_bc += 1
                ok_bc += int(row["baseline"] > row["clean"])
                diffs.append(row["baseline"] - row["clean"])
            if all(c in row.index for c in ["baseline", "partial", "clean"]):
                if pd.notna(row["baseline"]) and pd.notna(row["partial"]) and pd.notna(row["clean"]):
                    n_bpc += 1
                    ok_bpc += int(row["baseline"] > row["partial"] > row["clean"])
        rows.append({
            "scanner": scanner,
            "method": method,
            "n_baseline_clean_pairs": n_bc,
            "prop_baseline_gt_clean": ok_bc / n_bc if n_bc else np.nan,
            "mean_baseline_minus_clean_coverage_pct": np.mean(diffs) if diffs else np.nan,
            "n_baseline_partial_clean_triplets": n_bpc,
            "prop_baseline_gt_partial_gt_clean": ok_bpc / n_bpc if n_bpc else np.nan,
        })
    return pd.DataFrame(rows)


def pairwise_scanner_agreement(scan_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, dfm in scan_df.groupby("method"):
        wide = dfm.pivot_table(
            index=["physical_id", "stage"],
            columns="scanner",
            values="plaque_like_coverage_pct",
            aggfunc="mean",
        )
        scanners = list(wide.columns)
        for i in range(len(scanners)):
            for j in range(i + 1, len(scanners)):
                s1, s2 = scanners[i], scanners[j]
                pair = wide[[s1, s2]].dropna()
                if len(pair) < 3:
                    continue
                x = pair[s1].to_numpy(float)
                y = pair[s2].to_numpy(float)
                try:
                    rp = pearsonr(x, y).statistic
                except Exception:
                    rp = np.nan
                try:
                    rs = spearmanr(x, y).statistic
                except Exception:
                    rs = np.nan
                rows.append({
                    "method": method,
                    "scanner_1": s1,
                    "scanner_2": s2,
                    "n_matched_sample_stages": len(pair),
                    "pearson_r": rp,
                    "spearman_r": rs,
                    "mean_abs_difference_pct_points": np.mean(np.abs(x - y)),
                    "mean_signed_difference_scanner1_minus_scanner2": np.mean(x - y),
                })
    return pd.DataFrame(rows)


def icc2_1_from_matrix(X: np.ndarray) -> float:
    """Two-way random effects, absolute agreement, single measure ICC(2,1).

    X shape = targets x raters/scanners. Requires complete cases.
    Formula follows the standard Shrout-Fleiss / McGraw-Wong ANOVA form.
    """
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if n < 2 or k < 2:
        return np.nan

    grand = X.mean()
    mean_rows = X.mean(axis=1)
    mean_cols = X.mean(axis=0)

    ss_rows = k * np.sum((mean_rows - grand) ** 2)
    ss_cols = n * np.sum((mean_cols - grand) ** 2)
    ss_total = np.sum((X - grand) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denom == 0:
        return np.nan
    return float((ms_rows - ms_error) / denom)


def scanner_icc(scan_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, dfm in scan_df.groupby("method"):
        wide = dfm.pivot_table(
            index=["physical_id", "stage"],
            columns="scanner",
            values="plaque_like_coverage_pct",
            aggfunc="mean",
        )
        complete = wide.dropna(axis=0, how="any")
        icc = icc2_1_from_matrix(complete.to_numpy(float)) if complete.shape[1] >= 2 else np.nan
        rows.append({
            "method": method,
            "n_complete_targets": len(complete),
            "n_scanners": complete.shape[1],
            "scanner_icc2_1_absolute_agreement": icc,
        })
    return pd.DataFrame(rows)


def make_method_comparison(endpoint_df, ordering_df, pair_df, icc_df) -> pd.DataFrame:
    """Create one method-level benchmark table without declaring a winner."""
    methods = sorted(set(METHODS))
    rows = []
    for method in methods:
        e = endpoint_df[endpoint_df.method == method]
        o = ordering_df[ordering_df.method == method]
        p = pair_df[pair_df.method == method]
        ic = icc_df[icc_df.method == method]

        rows.append({
            "method": method,
            "mean_endpoint_auc_across_scanners": e["endpoint_scan_auc_using_coverage"].mean() if len(e) else np.nan,
            "mean_endpoint_balanced_accuracy_across_scanners": e["endpoint_balanced_accuracy_at_50pct_coverage"].mean() if len(e) else np.nan,
            "mean_prop_baseline_gt_clean_across_scanners": o["prop_baseline_gt_clean"].mean() if len(o) else np.nan,
            "mean_prop_baseline_gt_partial_gt_clean_across_scanners": o["prop_baseline_gt_partial_gt_clean"].mean() if len(o) else np.nan,
            "mean_baseline_minus_clean_coverage_pct": o["mean_baseline_minus_clean_coverage_pct"].mean() if len(o) else np.nan,
            "mean_pairwise_scanner_abs_difference_pct_points": p["mean_abs_difference_pct_points"].mean() if len(p) else np.nan,
            "scanner_icc2_1": ic["scanner_icc2_1_absolute_agreement"].iloc[0] if len(ic) else np.nan,
        })
    return pd.DataFrame(rows)


# =============================================================================
# MODEL METADATA
# =============================================================================


def detector_metadata(detectors: Dict[str, Detector], scanner: str, fold: int) -> pd.DataFrame:
    rows = []
    for name, d in detectors.items():
        row = {
            "scanner": scanner,
            "fold": fold,
            "method": name,
            "threshold": float(d.threshold),
        }
        if isinstance(d, AThresholdDetector):
            row["a_orientation"] = d.orientation
        elif isinstance(d, CleanMahalanobisDetector):
            row["clean_L"] = d.mean_[0]
            row["clean_a"] = d.mean_[1]
            row["clean_b"] = d.mean_[2]
        elif isinstance(d, LabReferenceDetector):
            row["plaque_L"] = d.mu_plaque[0]
            row["plaque_a"] = d.mu_plaque[1]
            row["plaque_b"] = d.mu_plaque[2]
            row["clean_L"] = d.mu_clean[0]
            row["clean_a"] = d.mu_clean[1]
            row["clean_b"] = d.mu_clean[2]
        elif isinstance(d, LogisticLabDetector):
            clf = d.model.named_steps["clf"]
            row["logistic_coef_scaled_L"] = clf.coef_[0, 0]
            row["logistic_coef_scaled_a"] = clf.coef_[0, 1]
            row["logistic_coef_scaled_b"] = clf.coef_[0, 2]
        elif isinstance(d, GMMDetector):
            row["gmm_component_plaque_prob"] = json.dumps(
                [float(v) for v in d.component_plaque_prob]
            )
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# MAIN BENCHMARK
# =============================================================================


def main():
    for folder in [OUTPUT_ROOT, TABLE_ROOT, PRED_ROOT, QC_ROOT, MODEL_ROOT]:
        folder.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP C - WEAKLY SUPERVISED PLAQUE-DETECTION BENCHMARK")
    print("=" * 80)
    print(f"Data root:   {DATA_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")

    discovered = discover_sample_folders(DATA_ROOT)
    print(f"\nDiscovered candidate sample folders: {len(discovered)}")

    scans: List[ScanData] = []
    failures = []
    for i, (scanner, folder) in enumerate(discovered, 1):
        try:
            scan = load_scan(scanner, folder)
            if scan.stage.startswith("unknown"):
                warnings.warn(f"Skipping unknown stage: {scan.sample_name}")
                continue
            scans.append(scan)
            print(
                f"[{i:>3}/{len(discovered)}] OK  {scanner:12s} "
                f"{scan.sample_name:12s} stage={scan.stage:8s} N={len(scan.lab):,}"
            )
        except Exception as exc:
            failures.append({
                "scanner": scanner,
                "folder": str(folder),
                "error": str(exc),
            })
            print(f"[{i:>3}/{len(discovered)}] FAIL {scanner} {folder.name}: {exc}")

    if failures:
        pd.DataFrame(failures).to_csv(TABLE_ROOT / "loading_failures.csv", index=False)

    if not scans:
        raise RuntimeError(
            "No usable scans loaded. The script now searches recursively for "
            "geometry.npz/roi.vtp. Check the diagnostic lines above: if processed "
            "files found = 0, point DATA_ROOT to the preprocessing-output tree; "
            "if files are found but loading fails, use loading_failures.csv to "
            "identify the colour-array mismatch."
        )

    inventory = pd.DataFrame([{
        "scanner": s.scanner,
        "sample_name": s.sample_name,
        "physical_id": s.physical_id,
        "group": s.group,
        "time_idx": s.time_idx,
        "stage": s.stage,
        "n_vertices": len(s.lab),
        "has_rgb": s.rgb is not None,
        "has_faces": s.faces is not None,
        "folder": str(s.sample_folder),
    } for s in scans])
    inventory.to_csv(TABLE_ROOT / "input_inventory.csv", index=False)

    print("\nLoaded scans by scanner/stage:")
    print(inventory.groupby(["scanner", "stage"]).size().unstack(fill_value=0))

    all_scan_rows = []
    all_model_meta = []
    cached_predictions: Dict[Tuple[str, str], dict] = {}

    scanners = sorted({s.scanner for s in scans})

    # -------------------------------------------------------------------------
    # GLOBAL specimen-level folds
    # -------------------------------------------------------------------------
    # The same physical specimen receives the same test fold for every scanner.
    # This makes cross-scanner method comparisons directly comparable and also
    # protects against leakage across stages of the same specimen.
    all_physical_ids = sorted({s.physical_id for s in scans})
    n_global_splits = min(N_SPLITS, len(all_physical_ids))
    if n_global_splits < 2:
        raise RuntimeError("Fewer than two physical specimens were found.")

    kfold = KFold(
        n_splits=n_global_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    fold_of = {}
    ids_array = np.asarray(all_physical_ids, dtype=object)
    for fold, (_, test_idx) in enumerate(kfold.split(ids_array), start=1):
        for idx in test_idx:
            fold_of[str(ids_array[idx])] = fold

    pd.DataFrame({
        "physical_id": all_physical_ids,
        "fold": [fold_of[x] for x in all_physical_ids],
    }).to_csv(TABLE_ROOT / "global_specimen_fold_assignments.csv", index=False)

    for scanner in scanners:
        scanner_scans = [s for s in scans if s.scanner == scanner]
        unique_groups = sorted({s.physical_id for s in scanner_scans})
        if len(unique_groups) < 2:
            warnings.warn(f"Skipping {scanner}: fewer than 2 physical specimens.")
            continue

        print("\n" + "-" * 80)
        print(
            f"SCANNER: {scanner} | scans={len(scanner_scans)} | "
            f"specimens={len(unique_groups)} | global folds={n_global_splits}"
        )
        print("-" * 80)

        for fold in range(1, n_global_splits + 1):
            test_scans = [s for s in scanner_scans if fold_of[s.physical_id] == fold]
            train_scans = [s for s in scanner_scans if fold_of[s.physical_id] != fold]

            if not test_scans:
                print(f"Fold {fold}: no test scans for this scanner; skipping.")
                continue

            train_ids = sorted({s.physical_id for s in train_scans})
            test_ids = sorted({s.physical_id for s in test_scans})
            overlap = set(train_ids) & set(test_ids)
            if overlap:
                raise RuntimeError(f"Group leakage detected: {overlap}")

            print(f"Fold {fold}: train specimens={len(train_ids)}, test specimens={len(test_ids)}")

            detectors = fit_detectors(train_scans, scanner, fold)
            meta = detector_metadata(detectors, scanner, fold)
            all_model_meta.append(meta)

            for scan in test_scans:
                pred_by_method = {}
                for method, detector in detectors.items():
                    score, mask = detector.predict(scan.lab)
                    summary = summarize_prediction(
                        scan=scan,
                        method=method,
                        fold=fold,
                        score=score,
                        mask=mask,
                        threshold=detector.threshold,
                    )
                    all_scan_rows.append(summary)
                    pred_by_method[method] = {
                        "score": score,
                        "mask": mask,
                        "threshold": detector.threshold,
                        "coverage_pct": summary["plaque_like_coverage_pct"],
                    }

                cached_predictions[(scan.scanner, scan.sample_name)] = pred_by_method
                if SAVE_PREDICTION_NPZ:
                    save_prediction_npz(scan, pred_by_method)
                if SAVE_QC_FIGURES:
                    save_qc_figure(scan, pred_by_method)

    if not all_scan_rows:
        raise RuntimeError("Benchmark produced no predictions.")

    scan_df = pd.DataFrame(all_scan_rows)
    scan_df.to_csv(TABLE_ROOT / "scan_level_predictions.csv", index=False)

    if all_model_meta:
        pd.concat(all_model_meta, ignore_index=True).to_csv(
            TABLE_ROOT / "fold_model_metadata.csv", index=False
        )

    endpoint_df = endpoint_metrics(scan_df)
    ordering_df = stage_ordering_metrics(scan_df)
    pair_df = pairwise_scanner_agreement(scan_df)
    icc_df = scanner_icc(scan_df)
    comparison_df = make_method_comparison(endpoint_df, ordering_df, pair_df, icc_df)

    endpoint_df.to_csv(TABLE_ROOT / "endpoint_metrics_by_scanner_method.csv", index=False)
    ordering_df.to_csv(TABLE_ROOT / "stage_ordering_by_scanner_method.csv", index=False)
    pair_df.to_csv(TABLE_ROOT / "scanner_pairwise_agreement.csv", index=False)
    icc_df.to_csv(TABLE_ROOT / "scanner_icc_by_method.csv", index=False)
    comparison_df.to_csv(TABLE_ROOT / "method_comparison_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Predictions: {TABLE_ROOT / 'scan_level_predictions.csv'}")
    print(f"Comparison:  {TABLE_ROOT / 'method_comparison_summary.csv'}")
    print("\nMethod comparison (descriptive; no automatic winner is chosen):")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(comparison_df.round(3).to_string(index=False))

    print("\nInterpretation reminder:")
    print(
        "These are weakly supervised PLAQUE-LIKE maps. Baseline/Clean endpoint "
        "labels are not pixel-level ground truth. Use Partial scans to inspect "
        "localization, and add manual masks later for Dice/IoU/true area error."
    )


if __name__ == "__main__":
    main()
