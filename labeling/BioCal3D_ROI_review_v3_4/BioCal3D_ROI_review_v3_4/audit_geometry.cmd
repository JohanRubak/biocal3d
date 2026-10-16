@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0"
call "%~dp0activate_labeling_runtime.bat"
set "BIOCAL3D_SNAPSHOT=%~1"
if defined BIOCAL3D_SNAPSHOT goto selected
set /p "BIOCAL3D_SNAPSHOT=Paste prepared snapshot folder: "
:selected
set "BIOCAL3D_SNAPSHOT=%BIOCAL3D_SNAPSHOT:"=%"
"%~dp0.labeling-env\python.exe" "%~dp0biocal3d_workflow\audit_roi_geometry.py" --root "%BIOCAL3D_SNAPSHOT%" --output "%BIOCAL3D_SNAPSHOT%_geometry_audit"
if errorlevel 1 goto failed
echo Open "%BIOCAL3D_SNAPSHOT%_geometry_audit\geometry_audit.html" to inspect the report.
pause
popd
exit /b 0
:failed
echo Audit failed. Review the error above.
pause
popd
exit /b 1
