"""Paper-inspired attention pseudo-labels with a dense CRF on the common grid."""
import numpy as np
from PIL import Image
from scipy.special import expit

from images_2d import to_grayscale
from spatial import patch_overlay

GAMMA = .75
THRESHOLD = .10
CRF_SEED_THRESHOLD = .50
ITERATIONS = 5


def box(values, shape):
    return np.asarray(Image.fromarray(values.astype(np.float32)).resize(shape[::-1], Image.Resampling.BOX))


def normalized_kernel(distance, sigma):
    weights = np.exp(-.5 * distance / sigma**2)
    np.fill_diagonal(weights, 0)
    degrees = np.sqrt(np.maximum(weights.sum(axis=1), 1e-8))
    return weights / degrees[:, None] / degrees[None, :]


def dense_refinement(blend, luminance, coverage):
    """Binary mean-field inference with spatial and optional intensity affinities."""
    valid = coverage > 0
    points = np.argwhere(valid).astype(np.float32)
    delta = points[:, None, :] - points[None, :, :]
    distance = np.sum(delta * delta, axis=-1)
    affinity = .55 * normalized_kernel(distance, 1.5)
    if luminance is not None:
        values = luminance[valid]
        intensity_delta = (values[:, None] - values[None, :]) / .12
        bilateral_distance = distance / 3**2 + intensity_delta**2
        affinity += .75 * normalized_kernel(bilateral_distance, 1)
    # The thresholded map initializes the labels, with the pre-threshold
    # blend providing unary evidence about the strength of each label.
    unary = np.clip(blend[valid], .08, .92)
    logit = np.log(unary / (1 - unary))
    probability = np.where(blend[valid] >= CRF_SEED_THRESHOLD, .7, .3).astype(np.float32)
    for _ in range(ITERATIONS):
        probability = expit(logit + affinity @ (2 * probability - 1))
    result = np.zeros_like(blend)
    result[valid] = probability
    return result


def plaque_scores(attention, rgb, roi, shape, predicted_stage, geometry_only):
    """Return the refined scores, with NaN wherever plaque is ineligible."""
    support = roi & np.isfinite(attention)
    if predicted_stage == "clean" or not support.any():
        return np.full(roi.shape, np.nan, dtype=np.float32)
    values = attention[support]
    low, high = np.quantile(values, [.01, .99])
    scaled = np.clip((attention - low) / max(high - low, 1e-8), 0, 1)
    luminance = None if geometry_only else to_grayscale(rgb)[..., 0].astype(np.float32) / 255
    blend = scaled if luminance is None else GAMMA * scaled + (1 - GAMMA) * (1 - luminance)
    coverage = box(support, shape)
    valid = coverage > 0

    def coarse(values):
        total = box(np.where(support, values, 0), shape)
        return np.divide(total, coverage, out=np.zeros_like(total), where=valid)

    score = dense_refinement(coarse(blend), None if luminance is None else coarse(luminance), coverage)
    refined = patch_overlay(score, coverage, roi.shape[1], roi.shape[0])
    return np.where(support, refined, np.nan)


def propose_mask(attention, rgb, roi, shape, predicted_stage, geometry_only):
    return plaque_scores(attention, rgb, roi, shape, predicted_stage, geometry_only) >= THRESHOLD
