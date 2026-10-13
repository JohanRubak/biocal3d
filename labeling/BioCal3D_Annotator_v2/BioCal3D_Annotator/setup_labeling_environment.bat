@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "BIOCAL3D_ENV=%~dp0.labeling-env"
where conda >nul 2>nul
if errorlevel 1 goto no_conda
if not exist "%~dp0biocal3d_workflow\annotator\biocal3d_annotator.py" goto missing_app
if exist "%BIOCAL3D_ENV%\python.exe" goto install
call conda create --yes --no-default-packages --prefix "%BIOCAL3D_ENV%" python=3.11 pip
if errorlevel 1 goto failure
:install
rem Restrict this process to the new runtime and Windows system binaries.
set "PATH=%BIOCAL3D_ENV%;%BIOCAL3D_ENV%\Scripts;%BIOCAL3D_ENV%\Library\bin;%SystemRoot%\System32;%SystemRoot%"
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONHOME="
set "QT_PLUGIN_PATH="
set "QT_QPA_PLATFORM_PLUGIN_PATH="
set "QT_QPA_PLATFORM="
set "QT_API=pyside6"
"%BIOCAL3D_ENV%\python.exe" -m pip install -r "%~dp0requirements_labeling.txt"
if errorlevel 1 goto failure
"%BIOCAL3D_ENV%\python.exe" -m pip check
if errorlevel 1 goto failure
"%BIOCAL3D_ENV%\python.exe" "%~dp0check_labeling_environment.py"
if errorlevel 1 goto failure
"%BIOCAL3D_ENV%\python.exe" -m pip freeze > "%~dp0labeling_installed_versions.txt"
echo Setup complete. You can now double-click start_labelling.bat.
pause
popd
exit /b 0
:no_conda
echo Run this setup from Anaconda Prompt or Miniforge Prompt on this PC.
goto failure
:missing_app
echo Extract these files into your existing BioCal3D workflow folder.
echo The biocal3d_workflow\annotator folder must be beside this setup script.
:failure
echo Setup failed. Review the error above.
pause
popd
exit /b 1
