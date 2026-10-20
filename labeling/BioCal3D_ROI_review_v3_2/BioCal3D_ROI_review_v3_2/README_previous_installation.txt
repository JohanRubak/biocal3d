BioCal3D Annotator - Windows 64-bit

SETUP ON EACH PC
1. Extract the complete ZIP into any writable folder. Folder names with spaces
   are supported. Do not run scripts from inside the ZIP.
2. Open Anaconda Prompt or Miniforge Prompt, change directory to the extracted
   BioCal3D_Annotator folder, and run setup_labeling_environment.bat.
   An internet connection is needed to install dependencies.
3. Setup checks helper imports, Qt window startup, and VTK rendering. After it
   passes, double-click start_labelling.bat.
4. Select your prepared dataset with Open dataset folder in the annotator.
   Select/create your own rater ID before labeling.

The setup and launcher belong at the top level, beside biocal3d_workflow.
The original helper modules remain in biocal3d_workflow/annotator.
Use this top-level launcher for normal labeling. The old inner
annotator/start_annotator.bat now forwards to the same launcher.

PORTABILITY
The dedicated .labeling-env is located relative to these scripts. No username,
Conda installation path, Qt plugin path, or dataset location is hardcoded.
Conda is needed in the setup prompt only; the installed launcher locates its
own Python without manual environment activation. Core and direct dependency
versions are pinned. Setup writes labeling_installed_versions.txt locally.
For exact transitive versions, reuse that report as requirements_labeling.txt
when setting up another PC.
Create .labeling-env on each PC; do not distribute or copy it. If you move or
rename the application folder after setup, recreate its environment in the new
location. Prefer choosing the final folder name/location before setup.
The launchers clear inherited Qt/Python settings within their own process.
All included workflow launchers now select this dedicated environment.

POINTS
Points are always bright yellow, independent of scanner RGB, textures, color
checkbox, sliders, and surface shading. Scanner surface colors and annotation
label colors retain their normal meaning. P toggles points; the toolbar also
controls point size. Points are offset slightly for visibility over the surface.

OPTIONAL COMMAND-LINE LABELING
start_labelling.bat --root "path_to_prepared_dataset" --annotator rater1
Supports --output, --manifest, --lock-annotator, and --review-mode as before.
Dataset paths are supplied by the user, not inferred from a private repository.

OPTIONAL STUDY/PREPROCESSING TOOLS
Existing study, agreement, and preparation sources are retained. Supply --root,
--output, or --data-root as the corresponding command requires. BIOCAL3D_DATA_ROOT
is an optional explicit raw-data override. No project-specific default is used.
Raw preprocessing additionally needs requirements_preprocessing.txt. To install:
.labeling-env\python.exe -m pip install -r requirements_preprocessing.txt
Recreate the dedicated environment if changing its dependency set causes issues.
Generated study queue launchers use relative data paths. When moving only a
queue to another location, regenerate it or set BIOCAL3D_APP_DIR to your current
biocal3d_workflow folder before running the queue launcher.

The former bundled environment has been excluded from this distribution.
This is a Windows batch-script package; macOS/Linux need separate launchers.

SAVING AND CLEARING LABELS
Each Save draft, autosave, and Save & Next writes:
  roi.vtp.labels.npz        - labels, completion status, notes and rater metadata
  roi.vtp.label_colors.ply  - colored viewing copy (pink plaque, green clean,
                             orange uncertain, gray unreviewed)
  roi.vtp.labeled.ply       - original-color training mesh, when Auto-save PLY
                             too is enabled
Equivalent filenames are used for other input mesh names. The colored viewing
copy duplicates vertices per face for crisp class boundaries; use the original
mesh/NPZ for training and geometry analysis. In MeshLab enable vertex colors.
Export mesh also offers PLY label colors; Export all includes viewing copies.
Previously saved samples get their colored copy when saved again, or through
Export all (completed annotations by default).
Clear all labels resets active ROI faces to Unreviewed and marks the sample
Draft. It preserves notes, outside-ROI faces, original geometry, and scanner
appearance. The reset is autosaved; Ctrl+Z restores the labels during the current
session, and Ctrl+Y/Ctrl+Shift+Z reapplies the reset. Undo still leaves the sample
Draft: use Save & Next again when the annotation is reviewed and complete.
Clear all labels is disabled for read-only independent ratings.
