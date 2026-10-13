BioCal3D clean labeling environment (Windows)

1. Extract this package into your existing BioCal3D_labeling_workflow_v4_6
   folder, beside biocal3d_prepare_all.py and the biocal3d_workflow folder.
   Replace start_labelling.bat with the one in this package.
2. Open Anaconda Prompt (or Miniforge Prompt) on each PC. Change directory
   to that workflow folder and run setup_labeling_environment.bat.
3. After setup passes, double-click start_labelling.bat. In the annotator,
   use Open dataset folder to select the prepared _development_... dataset.

The environment is created in .labeling-env relative to the setup script.
The launcher uses that same relative location and requires no manual activation.
It does not use activate_biocal3d.bat or your existing biocal3d environment.
Conda installs Python and pip; pip installs the application dependencies.
The launcher clears inherited Qt/Python overrides within its own process.
Qt plugin directories are discovered normally by PySide6, with no manual paths.

The launcher opens the existing annotator directly, instead of routing through
biocal3d_prepare_all.py. It accepts the annotator's existing options, for example:
  start_labelling.bat --root "D:\prepared_dataset" --annotator johan
Other options include --output, --manifest, --lock-annotator and --review-mode.
Preparation and training still use their existing launchers/environments.

Package versions for the core runtime are pinned for consistency between PCs.
Supporting packages use bounded ranges; setup records exact resolved versions
in labeling_installed_versions.txt. Setup verifies package dependencies, helper
imports, actual Qt window initialization, and a VTK render.
The uploaded annotator was inspected; its sibling mesh_io.py and
annotation_features.py were not supplied. If they need another dependency,
the import check will report it before setup declares success.

Keep your uploaded annotator unchanged. No edited annotator is required.
Create the environment separately on each PC; do not copy .labeling-env between
PCs. If moving the workflow to another location, recreate its environment there.
To use identical supporting versions on another PC, copy the generated
labeling_installed_versions.txt over requirements_labeling.txt before setup.
