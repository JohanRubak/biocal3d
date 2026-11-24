"""Attention-mass footprints and class-specific evidence for the fitted pilot."""

import base64
from io import BytesIO

import matplotlib
from matplotlib import colors, colormaps
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

from explain_stage_classifier import image_values
from paper_attention import normalize
from study_data import STAGES

CONTRASTS = ("partial", "baseline")
OUTER = (255, 202, 64)
INNER = (27, 228, 245)
EVIDENCE_CMAP = colormaps["RdBu_r"]


def attention_groups(weights):
    """Mark the highest-weight locations that reach 50% and 80% of total mass."""
    sorted_weights = np.sort(weights.astype(np.float64))[::-1]
    cumulative = np.cumsum(sorted_weights)
    cutoffs = sorted_weights[np.searchsorted(cumulative, np.array([.5, .8]) * cumulative[-1])]
    # Keep equally weighted locations together at the boundary.
    inner, outer = (weights >= cutoff for cutoff in cutoffs)
    groups = inner.astype(np.uint8) + outer.astype(np.uint8)
    mass = np.array([weights[inner].sum(dtype=np.float64), weights[outer].sum(dtype=np.float64)])
    return groups, mass / cumulative[-1]


def project_group(cache, members, roi, center, field):
    meta = cache["meta"]
    if "shape" in meta:
        patch = Image.fromarray(members.reshape(meta["shape"]).astype(np.uint8))
        raster = np.asarray(patch.resize((roi.shape[1], roi.shape[0]), Image.Resampling.NEAREST))
        return roi & (raster > 0)
    return roi & (image_values(cache, members.astype(np.float32), roi, center, field) >= .5)


def attention_views(cache, attention, model, support, center, field):
    eligible = cache["eligible"]
    groups, mass = attention_groups(attention)
    native_groups = np.zeros(len(eligible), dtype=np.uint8)
    native_groups[eligible] = groups
    views = {
        "contour_50": project_group(cache, native_groups == 2, support, center, field),
        "contour_80": project_group(cache, native_groups >= 1, support, center, field),
        "attention_mass": mass,
        "native_attention": attention,
        "native_attention_group": groups,
        "native_location_index": np.flatnonzero(eligible),
    }
    if "xyz_mm" in cache["meta"]:
        views["native_xyz_mm"] = cache["meta"]["xyz_mm"][eligible].astype(np.float32)

    x = normalize(cache["x"][eligible])
    clean = STAGES.index("clean")
    for stage in CONTRASTS:
        target = STAGES.index(stage)
        contrast = model["classifier"][:, target] - model["classifier"][:, clean]
        evidence = attention.astype(np.float64) * (x.astype(np.float64) @ contrast.astype(np.float64))
        bias = float(model["bias"][target]) - float(model["bias"][clean])
        native = np.zeros(len(eligible), dtype=np.float64)
        native[eligible] = evidence
        projected = image_values(cache, native, support, center, field)
        views[f"evidence_{stage}"] = np.where(support, projected, np.nan).astype(np.float32)
        views[f"native_evidence_{stage}"] = evidence
        views[f"bias_{stage}"] = np.array(bias)
    return views


def contour_overlay(views):
    inner, outer = (views[name] for name in ("contour_50", "contour_80"))
    image = np.zeros((*inner.shape, 4), dtype=np.uint8)
    width = max(1, round(inner.shape[0] / 225))
    for region, rgb, thickness in ((outer, OUTER, 2 * width), (inner, INNER, width)):
        edge = region & ~binary_erosion(region, iterations=thickness)
        image[edge] = (*rgb, 255)
    return image


def evidence_overlay(values):
    finite = np.isfinite(values)
    limit = max(1e-8, float(np.quantile(np.abs(values[finite]), .99))) if finite.any() else 1e-8
    unit = np.clip(np.where(finite, values, 0) / limit, -1, 1)
    rgba = EVIDENCE_CMAP((unit + 1) / 2, bytes=True)
    rgba[..., 3] = np.rint(230 * np.abs(unit)).astype(np.uint8)
    return rgba, limit


def png_data(image):
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def inspection_payload(views, contours, evidence):
    native = None
    if "native_xyz_mm" in views:
        xyz = views["native_xyz_mm"]
        center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
        span = max(1e-6, float(np.linalg.norm(xyz - center, axis=1).max()))
        display = np.ascontiguousarray((xyz - center) / span, dtype="<f4")
        native = {
            "xyz": base64.b64encode(display.tobytes()).decode("ascii"),
            "groups": base64.b64encode(views["native_attention_group"].tobytes()).decode("ascii"),
        }
    return {
        "contours": png_data(contours), "mass": views["attention_mass"].tolist(), "native": native,
        "evidence": {stage: {"image": png_data(rgba), "limit": limit}
                     for stage, (rgba, limit) in evidence.items()},
    }


def export_inspection(path, scan, stage, raw, data, labels, paper_threshold, mask_colour):
    """Export a static partial-versus-clean triptych and the interactive overlays."""
    from matplotlib import pyplot as plt

    fig, grid = plt.subplots(len(data), 3, figsize=(11, 12.8), layout="constrained")
    payload = {}
    for axes, (mode, item) in zip(grid, data.items()):
        views = item["views"]
        contours = contour_overlay(views)
        evidence = {stage: evidence_overlay(views[f"evidence_{stage}"]) for stage in CONTRASTS}
        payload[mode] = inspection_payload(views, contours, evidence)
        for axis in axes:
            axis.imshow(raw)
            axis.set_axis_off()

        rgba, limit = evidence["partial"]
        mask = item["paper"] > paper_threshold
        axes[0].imshow(contours)
        axes[1].imshow(rgba)
        axes[2].imshow(np.ma.masked_where(~mask, mask), cmap=mask_colour, vmin=0, vmax=1, alpha=.72)
        axes[0].set_title(f"{labels[mode]}\n50% and 80% attention mass", fontsize=10)
        axes[1].set_title("Partial vs fully cleaned", fontsize=10)
        gate = "closed" if item["prediction"] == "clean" else "open"
        axes[2].set_title(f"Candidate mask · threshold {paper_threshold:g}\n"
                          f"{item['prediction'].title()} prediction · gate {gate}", fontsize=10)
        bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=colors.Normalize(-limit, limit),
                                                       cmap=EVIDENCE_CMAP),
                           ax=axes[1], orientation="horizontal", fraction=.055, pad=.015, ticks=(-limit, 0, limit))
        bar.ax.set_xticklabels([f"{-limit:.2g}", "0", f"{limit:.2g}"], fontsize=7)
        bar.ax.set_title("Local logit contribution", fontsize=7)
    fig.suptitle(f"{scan} · recorded stage {stage}\nCyan shows 50% mass. Gold shows 80% mass. "
                 "Red favours partial and blue favours fully cleaned.", fontsize=11)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return payload
