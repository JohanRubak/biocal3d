"""Create a validation montage of highest-confidence correct classifications.

Uses existing outputs from biocal3d_stage_gradcam_updated_v5.py:
  stage_classification/stage_predictions.csv
  stage_classification/gradcam/<split>/<scanner>/<sample>/gradcam_<class>.npz

Default selection: top 2 correctly classified validation scans per stage.
This avoids a montage dominated by the very easy baseline class.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from biocal3d_prepare_dino import DATA_ROOT

CLASSES = ["baseline", "partial", "clean"]


def make_cam_overlay(rgb, cam, valid, max_alpha=0.65):
    """Overlay CAM with opacity proportional to CAM intensity."""
    from matplotlib import colormaps

    heat = colormaps["inferno"](cam)[..., :3]
    overlay = rgb.copy()
    alpha = (max_alpha * np.clip(cam, 0, 1))[..., None]
    overlay[valid] = (
        (1.0 - alpha[valid]) * rgb[valid]
        + alpha[valid] * heat[valid]
    )
    return np.clip(overlay, 0, 1)


def partial_is_relevant(row):
    try:
        return int(row.get("group", 0)) >= 3
    except Exception:
        name = str(row.get("sample_name", ""))
        return name.startswith(("BCG3", "BCG4", "BCG5"))


def load_overlay(row, root, class_name, max_alpha):
    map_dir = Path(row["map_dir"])
    with np.load(map_dir / "maps.npz") as z:
        rgb = z["rgb"]
        valid = z["valid"]

    cam_path = (
        root / "gradcam" / row["split"] / row["scanner"] / row["sample_name"]
        / f"gradcam_{class_name}.npz"
    )
    with np.load(cam_path) as z:
        cam = z["normalized_cam"]

    return rgb, make_cam_overlay(rgb, cam, valid, max_alpha=max_alpha)


def select_examples(predictions, n_per_stage):
    val = predictions[(predictions["split"] == "val") & (predictions["correct"] == True)].copy()
    if val.empty:
        raise RuntimeError("No correctly classified validation scans found.")

    selected = []
    for stage in CLASSES:
        sub = val[val["stage"] == stage].copy()
        sub = sub.sort_values(
            ["true_class_probability", "probability_margin"],
            ascending=[False, False],
        )
        selected.append(sub.head(n_per_stage))

    out = pd.concat(selected, ignore_index=True)
    out["stage_order"] = pd.Categorical(out["stage"], CLASSES, ordered=True)
    out = out.sort_values(
        ["stage_order", "true_class_probability"], ascending=[True, False]
    ).drop(columns="stage_order").reset_index(drop=True)
    return out


def plot_montage(selected, root, output_png, max_alpha):
    nrows = len(selected)
    ncols = 4
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(14.5, max(3.15 * nrows, 4.0)),
        squeeze=False,
    )

    titles = ["RGB", "Baseline Grad-CAM", "Partial Grad-CAM", "Clean Grad-CAM"]
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=11, pad=8)

    for i, row in selected.iterrows():
        r = row.to_dict()
        rgb, base = load_overlay(r, root, "baseline", max_alpha)
        _, clean = load_overlay(r, root, "clean", max_alpha)

        axes[i, 0].imshow(rgb)
        axes[i, 1].imshow(base)

        if partial_is_relevant(r):
            _, partial = load_overlay(r, root, "partial", max_alpha)
            axes[i, 2].imshow(partial)
        else:
            axes[i, 2].text(
                0.5, 0.5, "Not applicable\n(G1-G2)",
                ha="center", va="center",
                transform=axes[i, 2].transAxes,
                fontsize=10,
            )

        axes[i, 3].imshow(clean)

        label = (
            f"{row['scanner']} {row['sample_name']}\n"
            f"true={row['stage']} → pred={row['predicted_stage']}\n"
            f"P(true)={row['true_class_probability']:.3f} | "
            f"margin={row['probability_margin']:.3f}"
        )

        for j in range(ncols):
            axes[i, j].axis("off")

        axes[i, 0].text(
            -0.055, 0.5, label,
            transform=axes[i, 0].transAxes,
            ha="right", va="center", fontsize=8.5,
        )

    fig.suptitle(
        "Highest-confidence correct validation classifications",
        fontsize=14, y=0.998,
    )
    fig.tight_layout(rect=[0.08, 0.01, 1, 0.98])
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classification-root",
        type=Path,
        default=DATA_ROOT / "_dino_preparation_80_10_10" / "stage_classification",
    )
    parser.add_argument("--n-per-stage", type=int, default=2)
    parser.add_argument("--cam-alpha", type=float, default=0.65)
    args = parser.parse_args()

    root = args.classification_root
    pred_path = root / "stage_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing {pred_path}")

    predictions = pd.read_csv(pred_path)
    required = {
        "split", "correct", "stage", "scanner", "sample_name", "map_dir",
        "predicted_stage", "true_class_probability", "probability_margin",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"stage_predictions.csv is missing columns: {sorted(missing)}")

    # CSV may load booleans as strings in some environments.
    if predictions["correct"].dtype == object:
        predictions["correct"] = predictions["correct"].astype(str).str.lower().eq("true")

    selected = select_examples(predictions, args.n_per_stage)

    fig_dir = root / "analysis_figures"
    table_dir = root / "analysis_tables"
    fig_dir.mkdir(exist_ok=True)
    table_dir.mkdir(exist_ok=True)

    csv_out = table_dir / "validation_highest_confidence_examples.csv"
    png_out = fig_dir / "validation_highest_confidence_examples.png"

    selected.to_csv(csv_out, index=False)
    plot_montage(selected, root, png_out, args.cam_alpha)

    print("Selected examples:")
    print(selected[[
        "stage", "scanner", "sample_name", "true_class_probability", "probability_margin"
    ]].to_string(index=False))
    print(f"\nSaved: {png_out}")
    print(f"Saved: {csv_out}")


if __name__ == "__main__":
    main()
