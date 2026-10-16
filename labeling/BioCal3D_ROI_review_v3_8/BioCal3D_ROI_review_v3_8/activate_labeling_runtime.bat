@echo off
rem Called from launchers; variables deliberately remain in the caller scope.
set "BIOCAL3D_ENV=%~dp0.labeling-env"
if not exist "%BIOCAL3D_ENV%\python.exe" (
    echo Run setup_labeling_environment.bat from Anaconda or Miniforge Prompt first.
    exit /b 1
)
set "PATH=%BIOCAL3D_ENV%;%BIOCAL3D_ENV%\Scripts;%BIOCAL3D_ENV%\Library\bin;%SystemRoot%\System32;%SystemRoot%"
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONHOME="
set "QT_PLUGIN_PATH="
set "QT_QPA_PLATFORM_PLUGIN_PATH="
set "QT_QPA_PLATFORM="
set "QT_API=pyside6"
set "BIOCAL3D_APP_DIR=%~dp0biocal3d_workflow"
exit /b 0
