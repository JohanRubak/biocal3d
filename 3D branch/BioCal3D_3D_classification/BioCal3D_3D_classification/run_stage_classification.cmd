@echo off
setlocal
rem Edit these three paths when using a new snapshot or experiment directory.
set "BIOCAL_PYTHON=C:\Users\au662213\AppData\Local\anaconda3\envs\biocal3d\python.exe"
set "BIOCAL_INDEX=C:\Users\au662213\repos\biocal3d\data\02-09-2026-Data collected\_development_20261007_100936_877172\dataset_index.json"
set "BIOCAL_OUTPUT=C:\Users\au662213\repos\biocal3d\runs\3d_stage_v1"

"%BIOCAL_PYTHON%" "%~dp0biocal3d_train_3d.py" run --index "%BIOCAL_INDEX%" --output "%BIOCAL_OUTPUT%" --resume
if errorlevel 1 (
    echo Training stopped. Read the error above; setup instructions are in README.md.
) else (
    echo Open "%BIOCAL_OUTPUT%\comparison.html" for the results.
)
pause
