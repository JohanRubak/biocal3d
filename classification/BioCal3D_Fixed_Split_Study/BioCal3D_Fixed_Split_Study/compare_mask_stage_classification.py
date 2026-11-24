"""Apply the fitted booster and summarise candidate masks."""

import numpy as np
from scipy import ndimage as ndi

import segment_color_boosting as boosting


def boosted_mask(image, roi, model):
    # Match the standalone boosting predictor's gate, cutoff, and cleanup.
    luminance_only = model.get("luminance_only", False)
    features, eligible, scale = boosting.pixel_features(image, roi, luminance_only)
    scores = np.zeros(len(features))
    for step, tree in model["trees"]:
        scores += step * tree.predict(features)
    candidate = np.zeros_like(roi)
    candidate[eligible] = (scores > model["threshold"]) & boosting.color_gate(features, luminance_only)
    components, _ = ndi.label(candidate)
    sizes = np.bincount(components.ravel())
    return candidate & (sizes[components] >= max(1, round(10 * scale ** 2)))


def describe_mask(mask, roi):
    components, count = ndi.label(mask)
    sizes = np.bincount(components.ravel())[1:]
    area = roi.sum()
    return (float(mask.sum() / area), float(sizes.max(initial=0) / area), float(np.log1p(count)))

