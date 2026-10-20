@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0"
call "%~dp0activate_labeling_runtime.bat"
set "BIOCAL3D_SNAPSHOT=%~1"
if defined BIOCAL3D_SNAPSHOT goto selected
set /p "BIOCAL3D_SNAPSHOT=Paste prepared snapshot folder: "
:selected
set "BIOCAL3D_SNAPSHOT=%BIOCAL3D_SNAPSHOT:"=%"
set "BIOCAL3D_CORRECTED=%~2"
if defined BIOCAL3D_CORRECTED goto output_ready
set /p "BIOCAL3D_CORRECTED=Paste NEW corrected snapshot folder (must be empty): "
:output_ready
set "BIOCAL3D_CORRECTED=%BIOCAL3D_CORRECTED:"=%"
set "BIOCAL3D_REVIEW_OUTPUT=%~3"
if defined BIOCAL3D_REVIEW_OUTPUT goto custom_reviews
"%~dp0.labeling-env\python.exe" "%~dp0biocal3d_workflow\rebuild_reviewed_snapshot.py" --root "%BIOCAL3D_SNAPSHOT%" --output "%BIOCAL3D_CORRECTED%"
goto rebuild_finished
:custom_reviews
"%~dp0.labeling-env\python.exe" "%~dp0biocal3d_workflow\rebuild_reviewed_snapshot.py" --root "%BIOCAL3D_SNAPSHOT%" --output "%BIOCAL3D_CORRECTED%" --annotations "%BIOCAL3D_REVIEW_OUTPUT%"
:rebuild_finished
if errorlevel 1 goto failed
echo Corrected snapshot created. Point training to this NEW folder and regenerate model feature/surface caches.
pause
popd
exit /b 0
:failed
echo Rebuild failed. Check the message above. Do not train on an incomplete output.
pause
popd
exit /b 1
