@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0"
if errorlevel 1 exit /b 1
call "%~dp0activate_labeling_runtime.bat"
if errorlevel 1 goto failed
if not "%~1"=="" goto arguments
set /p "BIOCAL3D_STUDY_SOURCE=Paste the SUCCESSFULLY REBUILT approved snapshot folder: "
set "BIOCAL3D_STUDY_SOURCE=%BIOCAL3D_STUDY_SOURCE:"=%"
"%~dp0.labeling-env\python.exe" "%~dp0biocal3d_workflow\labeling_study.py" folders --root "%BIOCAL3D_STUDY_SOURCE%"
goto finished
:arguments
"%~dp0.labeling-env\python.exe" "%~dp0biocal3d_workflow\labeling_study.py" folders %*
:finished
if errorlevel 1 goto failed
echo Study folders created with approved ROIs and empty independent plaque annotations.
pause
popd
exit /b 0
:failed
echo Study preparation stopped. Check the error above. Existing study folders were not overwritten.
pause
popd
exit /b 1
