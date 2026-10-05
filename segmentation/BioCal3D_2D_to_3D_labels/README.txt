BioCal3D 2D-to-3D annotation projection

Install: python -m pip install numpy pillow matplotlib plotly

Run on your original sample folder (with maps.npz, metadata.json, rgb.json):
python biocal3d_project_labels_3d.py --sample-dir "C:/path/to/sample" --open

Only after reviewing all pixels in the ROI:
python biocal3d_project_labels_3d.py --sample-dir "C:/path/to/sample" --reviewed-background --open

Open demo/labels_3d.html in your browser to inspect the supplied sample.
Each panel can be rotated separately. Red=plaque, blue=no-plaque, gray=unknown.
The 3D scatter shows exact pixel-to-surface locations. Face labels are coarser
majority votes among known observations, not exact continuous boundaries.
Original surface colour is approximated from the RGB raster, not recovered
from original scanner vertex colours. Original geometric indices are preserved.

The supplied sample is iTERO BCG3T1-2, split TEST. Keep it reserved for testing.
The annotation has checked=false. Demo labels.npz verified=false, so it will
not be silently used for fitting. Fully review then use --reviewed-background,
or mark checked in X-AnyLabeling for verified positive-only annotations.
To use labels.npz in the ablation loader, place it under:
LABELS_ROOT/split/scanner/sample_name/labels.npz

The sample demonstrated 5 plaque polygons and 34,299 valid plaque pixels.
Maximum coordinate reprojection error: 0.0000104 pixels. This verifies mapping
consistency, not that the manually drawn plaque boundaries are accurate.
