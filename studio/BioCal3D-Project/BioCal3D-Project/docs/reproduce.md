# Reproduce the current analysis

Use cleaned data snapshot `92fdb7e462b426ccd3ed738e5ca5c1063ea5c093` for the reference analyses. Download it following the [dataset guide](datasets.md). Set BIOCAL3D_DATA_ROOT to the local parent of the four release folders in the terminal used below. Follow [setup](setup.md) first.

## Audit the cohort

From the repository root, in the biocal3d environment:

```bat
python tools\audit_datasets.py --output "%USERPROFILE%\Downloads\biocal3d_cohort_audit.csv"
```

There are 48 physical specimens, eight in each group, with 512 acquisitions across four scanners. Groups 1–2 have baseline and clean stages. Groups 3–6 have baseline, partial, and clean stages.

## Geometry and colour characterization

```bat
python analysis\colour-harmonization\run_sections_3_1_3_2.py
```

The default inputs are approved_rois_reoriented and approved_rois_reoriented_harmonized. Full-model examples use raw_native and full_harmonized. Outputs go under analysis/colour-harmonization/results unless --output selects another folder. See that folder's README for the figure and table mapping.

## Classification

From analysis/classification-study, validate the full cohort and fixed split:

```bat
python run_study_on_this_pc.py --prepare-only --harmonized-roi
```

Check feature extraction, training, and report generation on the three-specimen smoke cohort:

```bat
python run_study_on_this_pc.py --smoke --harmonized-roi
```

After that check passes, run the full six-condition experiment:

```bat
python run_study_on_this_pc.py --harmonized-roi --epochs 1000 --patience 30
```

The protocol assigns 30 specimens to training, 6 to validation, and 12 to test. BCG3T1-1 is fixed in test across the four scanners. Both encoder backbones are frozen. DINOv3 compares grayscale, raw RGB, and harmonized RGB. Concerto compares geometry, geometry plus raw RGB, and geometry plus harmonized RGB.

Open runs/full/results_harmonized/index.html for the report. An explicit --run path selects another output directory. The same command resumes compatible caches and heads. Use a fresh run when the input release or its location changes. Keep each physical specimen in its saved partition when comparing configurations.

To regenerate figures from an existing compatible run:

```bat
python run_study_on_this_pc.py --figures-only --harmonized-roi
```

## Initial crop extraction and later geometry changes

The [preprocessing workflow](../preprocessing/README.md) regenerates initial unedited crops from raw_native using scanner-specific reference circles. The supplied [reference fits](../preprocessing/references/README.md) reuse the saved scanner circle/crop parameters. They do not contain the original click sequence.

Use Studio Review and Orient for geometry review and scanner orientation. After initial extraction, each ROI was inspected manually in BioCal3D Studio and corrected when necessary. The supplied approved packages are the saved results of that curation stage; no separate missing builder script is required. Use these packages for the reference analyses, since initial extraction alone does not repeat manual review decisions.

The [harmonized ROI exporter](../analysis/classification-study/README.md#export-a-dataset-of-harmonized-approved-rois) transfers colour from the supplied harmonized full models onto approved geometry. Reproducing the harmonization model fitting additionally requires its original training implementation, configuration, and fitted artifact.

## Record an experiment

Keep the following together with each run in the project archive:

- GitHub code commit and Hugging Face dataset revision.
- Exact command and all path or parameter overrides.
- Saved specimen split, encoder checkouts and checkpoint hashes.
- Environment package versions and GPU/driver information.
- Final run directory, heads, predictions, metrics, and figures.

Export packages from each environment used:

```bat
conda run -n biocal3d python -m pip freeze > biocal3d-packages.txt
conda run -n biocal3d-dinov3 python -m pip freeze > dinov3-packages.txt
conda run -n biocal3d-concerto python -m pip freeze > concerto-packages.txt
```

The existing setup pins core Studio and encoder dependencies, while several research requirements accept a range of versions. Package exports document the environment actually used. Exact historical numerical reproduction still depends on recovering the original run artifacts and environment records.

## Read the supplied final classification results

The [final-run archive](../analysis/classification-study/archives/README.md) contains the saved six-condition report, predictions, split and trained heads. Open its `results_harmonized/index.html` directly to inspect existing outputs. Feature settings/bags are absent, so the archive alone does not support the current `--figures-only` reconstruction path. Obtain the frozen encoders from their [official authors' documentation](frozen-models.md) and recover experiment-specific settings/environment records for exact replay.
