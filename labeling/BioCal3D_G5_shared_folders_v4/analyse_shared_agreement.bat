@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0"
if errorlevel 1 exit /b 1
if not defined BIOCAL3D_STUDY_PYTHON set "BIOCAL3D_STUDY_PYTHON=%LOCALAPPDATA%\anaconda3\envs\biocal3d\python.exe"
if not exist "%BIOCAL3D_STUDY_PYTHON%" goto failure
"%BIOCAL3D_STUDY_PYTHON%" "%~dp0analyse_labeling_agreement.py" %*
set "BIOCAL3D_EXIT_CODE=%ERRORLEVEL%"
pause
popd
exit /b %BIOCAL3D_EXIT_CODE%
:failure
echo Set BIOCAL3D_STUDY_PYTHON to the biocal3d python.exe path.
pause
popd
exit /b 1
