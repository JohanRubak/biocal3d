# Working on BioCal3D

Create a branch for changes and include the commands and inputs used for verification in the pull request. Identify the GitHub commit and dataset revision when comparing scientific results.

## Preserve data correspondence

Review and labeling depend on source hashes, face and vertex IDs, saved transforms, and approved geometry. If a geometry operation changes these, verify the downstream review, label migration, raster maps, and classification inputs. Generate derived datasets into a separate output folder.

Use the fixed specimen-level split for comparisons. Report any changes to the cohort, split, field of view, feature sampling, preprocessing, model inputs, or stopping criterion.

## Local checks

In the research environment, from the repository root:

```bat
python -m compileall -q studio preprocessing analysis tools
python -m pip install pytest
python -m pytest -q studio/tests/test_browser_core.py studio/tests/test_browser_originals.py studio/tests/test_original_models.py studio/tests/test_studio_integration.py studio/tests/test_roi_ui.py
```

On a headless machine set QT_QPA_PLATFORM=offscreen. The integration tests disable graphics drawing. Desktop rendering and hardware picking still require target-machine checks. Run the dataset audit against the intended release when changing data resolution or correspondence logic.

For installer changes, test fresh installation and update on a Windows x64 computer. For model changes, run a smoke study before starting the full experiment and retain the resulting settings and outputs.

## Keep the repository reviewable

The .gitignore excludes source scan arrays/meshes, weights, caches and common credential files. It does not inspect the contents of an allowed file. Review staged changes before a commit and keep access tokens in the credential store. Link archived results and datasets from docs/handover-status.md using project-approved access locations.

Access and sharing are currently restricted to the BioCal3D project team. Contact [Katrin](https://www.au.dk/kkresse@dent.au.dk) for access requests; obtain Katrin's approval before any external sharing or public release.

## Keep user-facing documentation aligned

The main README, setup guide, dataset guide and workflow guides should describe the current inputs and actions for a new researcher. Put research limitations and unavailable historical inputs in `docs/handover-status.md`. Keep one-time repository creation instructions and personal troubleshooting transcripts out of the maintained documentation.

The Hugging Face README is maintained in `docs/dataset-card.md`. After editing it, an authorized maintainer can run `python tools/upload_dataset.py --repo-id JohanRubak/BioCal3D-Data --readme-only`. This updates that one document and checks that every other remote file is unchanged. A documentation-only commit does not change the pinned reference data snapshot. Record any actual new data release separately and update the download examples intentionally.
