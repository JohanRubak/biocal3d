# Continue the BioCal3D research

After completing [setup](setup.md) and [dataset validation](datasets.md), choose the workflow that matches your research question. Keep the released data as the reference input and place new work in a separate output directory or working copy.

## Available workflows

| Goal | Entry point | Guide |
| --- | --- | --- |
| Inspect scanner colour/geometry and full-model correspondence | Studio Inspect | [Studio](../studio/README.md) |
| Review ROI boundaries or create independent plaque annotations | Studio Review / Label | [Studio](../studio/README.md) |
| Generate the first unedited crops from native scans | `preprocessing/preprocessing.py` | [Preprocessing](../preprocessing/README.md) |
| Characterize colour/geometry and compare harmonization | `analysis/colour-harmonization/run_sections_3_1_3_2.py` | [Characterization](../analysis/colour-harmonization/README.md) |
| Compare cleaning-stage classification and attention maps | `analysis/classification-study/run_study_on_this_pc.py` | [Classification](../analysis/classification-study/README.md) |
| Transfer existing harmonized full-model colour onto approved ROI geometry | `analysis/classification-study/export_harmonized_rois.py` | [ROI colour export](../analysis/classification-study/README.md#export-a-dataset-of-harmonized-approved-rois) |

## Establish a baseline before extending the study

1. Audit a fresh copy of the pinned data release and inspect `BCG3T1-1` across all scanners in Studio.
2. Run the classification preparation check and small smoke experiment before scheduling a full GPU experiment.
3. Reproduce the colour/geometry summaries using the supplied approved ROI variants.
4. Save the commands, specimen split, package versions and output paths with the experiment. Follow [the reproduction guide](reproduce.md).

The current implementation assigns 30 specimens to training, 6 to validation and 12 to test, with all scanner/timepoint views of each specimen in one partition and BCG3-1 fixed in test. Compare conditions using the same saved split. A fresh run is required when the input release or its location changes because caches and run records include dataset paths.

## Research extensions

For plaque segmentation, define an annotation protocol with the clinical collaborators and obtain independent spatial validation. Geometry approval is separate from plaque-label approval, and cleaning-stage accuracy does not validate attention masks as segmentation ground truth.

For new scanner acquisitions or colour calibration, preserve native geometry, textures, acquisition settings and reference information. The existing harmonization was fitted on the study data; evaluate any new mapping on acquisitions excluded from its fitting before claiming generalization.

For new preprocessing or geometry review, export derivatives separately and preserve native source hashes, vertex/triangle correspondence and transforms. Rerun downstream rasterization, audit and analysis checks when geometry or coordinate frames change.

The [release record](handover-status.md) lists the supplied crop-reference fits and final classification run, the pending harmonization/manuscript materials, and records still needed for exact replay. Prioritize those materials when exact reproduction of earlier preparation or numerical results is required.
