"""Find candidate dark-red regions in BioCal3D RGB renders."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont
from scipy import ndimage as ndi
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

# Equivalent ROI diameter in the original iTERO screenshot experiment.
REFERENCE_DIAMETER = np.sqrt(4 * 21551 / np.pi)
SCANNERS = ("iTERO", "LABscanner", "TRIOS3", "TRIOS5")
STAGES = ("baseline", "partial", "clean")


def segment(image, roi, membership_threshold):
    """Fit a fresh two-component mixture to each scan's interior."""
    if roi.shape != (image.height, image.width) or not roi.any():
        raise ValueError("ROI mask must be nonempty and match the RGB image size.")

    scale = np.sqrt(4 * roi.sum() / np.pi) / REFERENCE_DIAMETER
    distance = ndi.distance_transform_edt(np.pad(roi, 1))[1:-1, 1:-1]
    interior = distance >= max(1.0, 7 * scale)
    if interior.sum() < 32:
        raise ValueError("ROI has too few pixels after excluding the rim.")

    srgb = ImageCms.createProfile("sRGB")
    lab_profile = ImageCms.createProfile("LAB")
    lab = np.asarray(ImageCms.profileToProfile(
        image, srgb, lab_profile, outputMode="LAB"
    ), dtype=np.float64)[..., :2].copy()
    lab[..., 0] *= 100 / 255
    support = roi.astype(float)

    def local_average(sigma):
        weights = ndi.gaussian_filter(support, sigma)
        values = ndi.gaussian_filter(lab * support[..., None], (sigma, sigma, 0))
        return values / np.maximum(weights[..., None], 1e-8)

    residual = local_average(1.5 * scale) - local_average(18 * scale)
    residual[..., 0] *= -1  # Positive values mean darker than the local background.
    features = residual[interior]
    # The screenshot experiment fitted every interior pixel. For larger renders,
    # use a deterministic sample for the fit and score all interior pixels.
    fitting = features
    if len(features) > 30000:
        fitting = features[np.random.default_rng(3).choice(len(features), 30000, replace=False)]
    candidate = np.zeros_like(roi)
    if np.ptp(fitting, axis=0).max() < 1e-8:
        return candidate, "uniform_color"

    with threadpool_limits(limits=1):
        mixture = GaussianMixture(n_components=2, random_state=3, n_init=5).fit(fitting)
        cluster = int(np.argmax(mixture.means_[:, 0]))
        means = mixture.means_[cluster]
        other = mixture.means_[1 - cluster]
        if not np.all(means > np.maximum(other, 0)):
            return candidate, "no_darker_redder_component"
        candidate[interior] = mixture.predict_proba(features)[:, cluster] > membership_threshold

    labels, _ = ndi.label(candidate)
    sizes = np.bincount(labels.ravel())
    candidate &= sizes[labels] >= max(1, round(10 * scale ** 2))
    return candidate, "candidates" if candidate.any() else "no_regions_at_threshold"


def select_scans(args):
    if args.image:
        if args.mask is None:
            raise ValueError("--image requires a separate --mask PNG with a nonzero ROI.")
        if args.scan or args.scanner or args.stage or args.limit or args.partition != "development":
            raise ValueError("Run selection options require --run.")
        return [(args.image.stem, "", "", "", args.image, args.mask)]

    if args.mask:
        raise ValueError("--mask is used with --image. A run uses its cached ROI masks.")
    manifest = json.loads((args.run / "cohort.json").read_text(encoding="utf-8-sig"))
    rows = [row for row in manifest["rows"]
            if (args.partition == "all" or row["partition"] == args.partition)
            and (args.stage is None or row["stage"] == args.stage)
            and (args.scanner is None or row["scanner"].casefold() == args.scanner.casefold())]
    if args.scan:
        unknown = set(args.scan) - {row["scan_id"] for row in rows}
        if unknown:
            raise ValueError(f"Scan IDs outside the selected cohort: {', '.join(sorted(unknown))}.")
        rows = [row for row in rows if row["scan_id"] in args.scan]
    if not rows:
        raise ValueError("No scans match the selection.")
    scans = []
    for row in sorted(rows, key=lambda row: row["scan_id"])[:args.limit]:
        folder = (args.run / "renders" / row["partition"] / "images"
                  / row["scanner"] / row["sample_name"])
        scans.append((row["scan_id"], row["scanner"], row["stage"], row["partition"],
                      folder / "rgb.png", folder / "valid_mask.png"))
    return scans


def save_comparison(original, overlay, mask, path):
    panel_width = max(original.width, 120)
    gutter, heading = 12, 42
    canvas = Image.new("RGB", (3 * panel_width + 4 * gutter, original.height + heading + gutter),
                       "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    for index, (label, panel) in enumerate(zip(("Original", "Overlay", "Mask"),
                                              (original, overlay, mask))):
        left = gutter + index * (panel_width + gutter)
        draw.text((left + panel_width / 2, heading / 2), label, fill="#18212b",
                  font=font, anchor="mm")
        canvas.paste(panel, (left + (panel_width - panel.width) // 2, heading))
    canvas.save(path)


def process(scan, output, threshold):
    scan_id, scanner, stage, partition, image_path, mask_path = scan
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    with Image.open(mask_path) as source:
        roi = np.asarray(source.convert("L")) > 0
    candidate, status = segment(image, roi, threshold)

    folder = output / re.sub(r"[^A-Za-z0-9_.-]+", "_", scan_id).strip(".")
    folder.mkdir(parents=True, exist_ok=True)
    mask_image = Image.fromarray(candidate.astype(np.uint8) * 255)
    mask_image.save(folder / "candidate_mask.png")
    overlay = np.array(image)
    overlay[candidate] = np.rint(
        .4 * overlay[candidate] + .6 * np.array([255, 226, 94])
    ).astype(np.uint8)
    overlay_image = Image.fromarray(overlay)
    overlay_image.save(folder / "overlay.png")
    save_comparison(image, overlay_image, mask_image, folder / "comparison.png")

    area = int(candidate.sum())
    roi_area = int(roi.sum())
    print(f"{scan_id}: {area / roi_area:.1%} of projected ROI ({status})", flush=True)
    return {
        "scan_id": scan_id, "scanner": scanner, "stage": stage, "partition": partition,
        "status": status, "candidate_pixels": area, "roi_pixels": roi_area,
        "candidate_fraction": area / roi_area, "mixture_membership_threshold": threshold,
        "output_folder": str(folder),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="Stage ablation run with cached DINOv3 renders.")
    source.add_argument("--image", type=Path, help="One RGB image. Provide its ROI with --mask.")
    parser.add_argument("--mask", type=Path, help="Nonzero pixels mark the ROI in --image.")
    parser.add_argument("--out", type=Path, default=Path("gmm_preview"))
    parser.add_argument("--partition", choices=("development", "test", "all"), default="development")
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--scanner", choices=SCANNERS)
    parser.add_argument("--scan", nargs="+", help="One or more exact scan IDs from the manifest.")
    parser.add_argument("--limit", type=int, help="Process only this many scans after filtering.")
    parser.add_argument("--membership", type=float, default=.95, help="Mixture membership cutoff.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    if not .5 <= args.membership < 1:
        parser.error("--membership must be at least 0.5 and less than 1.")

    try:
        scans = select_scans(args)
        missing = next((path for scan in scans for path in scan[-2:] if not path.is_file()), None)
        if missing is not None:
            hint = " Generate the RGB renders for this partition with DINOv3 extraction." if args.run else ""
            raise ValueError(f"Missing input: {missing}.{hint}")
        records = [process(scan, args.out, args.membership) for scan in scans]
        summary = args.out / "summary.csv"
        with summary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"Saved {summary}")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
