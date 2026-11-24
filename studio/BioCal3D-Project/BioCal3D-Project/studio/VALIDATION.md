# Studio validation and platform coverage

Use this record when deciding which checks to run before deploying a modified Studio build. It describes recorded release checks and their limits; it is not a claim that every target computer has been tested.

## Recorded checks

The historical Studio 1.0.0 regression record reported **102 passed and 2 skipped** with the pinned dependencies. Earlier functional builds reported 98 passed and 2 skipped. These are historical results, not a fresh execution on the current workstation.

The covered behaviours include:

- Dataset discovery, native/harmonized full-model lookup and recorded source checksums.
- Shared camera rotation, zoom and pan, landmark direction, and full-model coordinate transforms.
- Approved-surface edits, undo/redo, source-face correspondence and label migration.
- Saving and workspace switching, annotation locks, independent-rater data and failed-save guards.
- Rigid orientation copies, transformed maps/history, texture correspondence and source preservation.
- Shared styles, accessible status rows and a 1366 × 768 application layout.
- Source compilation, launcher syntax and dependency import checks.

Interface integration tests use synthetic data and can disable actual render-window drawing. The optional external real-data fixture and hardware-visible-surface picking checks were skipped in the recorded run. They do not establish real GPU drawing or mouse-picking behaviour.

The recorded pinned stack includes NumPy 1.26.4, SciPy 1.13.1, Matplotlib 3.9.4, Pillow 11.3.0, PySide6 6.8.3, VTK 9.3.1 and plyfile 1.1.3. Use `requirements.txt` for the current dependency definition and record actual package versions with new validation runs.

## Windows installation

The portable per-user installer has been used successfully on Windows x64. Bootstrap checks covered verified downloads, corrupt-cache replacement, runtime reuse, paths containing spaces and dependency imports. The installer downloads its runtime and packages during setup.

Fresh installation, updates, OpenGL rendering, brushing/picking, orientation and figure export should be checked on the target Windows computer after relevant changes. Restricted institutional installations and unsupported processor/OS combinations need separate assessment.

## Other platforms and packaging

Native macOS and Linux desktop launcher support has not received the same end-to-end platform validation. Headless Qt/VTK tests are useful for state/loader regressions but do not replace desktop graphics checks.

Windows PyInstaller and Inno Setup recipes are included. A built and code-signed offline distribution is not supplied. Validate the resulting package on a clean target machine before distributing it.

## Run checks for a change

Follow [CONTRIBUTING.md](../CONTRIBUTING.md) for test commands and dataset invariants. Use the [dataset guide](../docs/datasets.md) to audit a reference release. Record the code commit, OS, package versions, commands, results and remaining platform limitations with the validation report for the modified build.
