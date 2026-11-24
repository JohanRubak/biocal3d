"""Shared spatial projections for the learned-attention and PCA figures."""
from functools import partial
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


def patch_overlay(scores, coverage, width, height):
    support = coverage > 0
    def resize(array):
        image = Image.fromarray(array.astype(np.float32))
        return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR))
    total = resize(np.where(support, scores, 0))
    weight = resize(support)
    return np.divide(total, weight, out=np.full_like(total, np.nan), where=weight > .05)

def point_overlay(xyz, scores, center, field, mask):
    # Interpolate for the flat preview only, in the same physical projection as
    # the original render. The cached point coordinates define the locations.
    height, width = mask.shape
    rows, cols = np.where(mask)
    xy = np.column_stack(((cols + .5) * field / width + center[0] - field / 2,
                          center[1] + field / 2 - (rows + .5) * field / height))
    distance, ids = cKDTree(xyz[:, :2]).query(xy, k=min(8, len(xyz)))
    if distance.ndim == 1:
        distance, ids = distance[:, None], ids[:, None]
    squared_distance = distance ** 2
    # Shift by the nearest distance so each ROI pixel has a nonzero weight,
    # even where the sampled points are widely spaced.
    weight = np.exp(-(squared_distance - squared_distance[:, :1]) / (2 * .28 ** 2))
    overlay = np.full(mask.shape, np.nan)
    overlay[rows, cols] = (weight * scores[ids]).sum(axis=1) / weight.sum(axis=1)
    return overlay

def save_png(path, encode):
    # Close the temporary file before publishing it, so Windows can rename it.
    with BytesIO() as buffer:
        encode(buffer, format="png")
        temporary = None
        try:
            with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                output.write(buffer.getbuffer())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

def save_figure_png(fig, path, dpi=160):
    try:
        save_png(path, partial(fig.savefig, dpi=dpi, bbox_inches="tight", facecolor="white"))
    finally:
        plt.close(fig)

def points_to_cells(tokens, xyz, shape, center, field):
    """Average Concerto embeddings inside each cell of the DINOv3 patch grid."""
    height, width = shape
    cols = np.floor(((xyz[:, 0] - center[0]) / field + .5) * width).astype(np.intp)
    rows = np.floor(((center[1] - xyz[:, 1]) / field + .5) * height).astype(np.intp)
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    cell = rows[inside] * width + cols[inside]
    counts = np.bincount(cell, minlength=height * width)
    totals = np.zeros((height * width, tokens.shape[1]), dtype=np.float32)
    np.add.at(totals, cell, tokens[inside])
    np.divide(totals, counts[:, None], out=totals, where=counts[:, None] > 0)
    return totals, counts > 0

def match_scan(local, center, field, scan):
    shape = local["dinov3_rgb"][3]["shape"]
    cell_features, supported = {}, []
    for mode, (tokens, eligible, _, meta) in local.items():
        if "shape" in meta:
            if meta["shape"] != shape:
                raise ValueError(f"DINOv3 inputs have different patch grids for {scan}.")
            values = np.zeros((eligible.size, tokens.shape[1]), dtype=np.float32)
            values[eligible] = tokens
            support = eligible
        else:
            values, support = points_to_cells(tokens, meta["xyz_mm"], shape, center, field)
        cell_features[mode] = values
        supported.append(support)
    common = np.logical_and.reduce(supported)
    if not common.any():
        raise ValueError(f"No spatial cells contain features for all configured inputs in {scan}.")
    return {mode: (values[common], common, None, {"shape": shape})
            for mode, values in cell_features.items()}
