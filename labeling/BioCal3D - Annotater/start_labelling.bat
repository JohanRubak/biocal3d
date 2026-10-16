@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "BIOCAL3D_ENV=%~dp0.labeling-env"
set "BIOCAL3D_APP=%~dp0biocal3d_workflow\annotator\biocal3d_annotator.py"
if not exist "%BIOCAL3D_ENV%\python.exe" goto missing_env
if not exist "%BIOCAL3D_APP%" goto missing_app
set "PATH=%BIOCAL3D_ENV%;%BIOCAL3D_ENV%\Scripts;%BIOCAL3D_ENV%\Library\bin;%SystemRoot%\System32;%SystemRoot%"
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONHOME="
set "QT_PLUGIN_PATH="
set "QT_QPA_PLATFORM_PLUGIN_PATH="
set "QT_QPA_PLATFORM="
set "QT_API=pyside6"
echo Using dedicated labeling Python: "%BIOCAL3D_ENV%\python.exe"
"%BIOCAL3D_ENV%\python.exe" "%BIOCAL3D_APP%" %*
set "BIOCAL3D_EXIT_CODE=%ERRORLEVEL%"
if not "%BIOCAL3D_EXIT_CODE%"=="0" goto app_failed
popd
exit /b 0
:missing_env
echo Run setup_labeling_environment.bat once from Anaconda Prompt on this PC.
goto failure
:missing_app
echo The biocal3d_workflow\annotator folder must be beside this launcher.
goto failure
:app_failed
echo Annotator exited with code %BIOCAL3D_EXIT_CODE%. Review the error above.
pause
popd
exit /b %BIOCAL3D_EXIT_CODE%
:failure
pause
popd
exit /b 1
