@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0"
if errorlevel 1 exit /b 1

rem Optional override for a nonstandard Conda installation.
rem Example: set "BIOCAL3D_CONDA_BAT=D:\Miniconda3\condabin\conda.bat"
set "BIOCAL3D_CONDA_SCRIPT="
if defined BIOCAL3D_CONDA_BAT if exist "%BIOCAL3D_CONDA_BAT%" set "BIOCAL3D_CONDA_SCRIPT=%BIOCAL3D_CONDA_BAT%"
if not defined BIOCAL3D_CONDA_SCRIPT if defined CONDA_BAT if exist "%CONDA_BAT%" set "BIOCAL3D_CONDA_SCRIPT=%CONDA_BAT%"
if not defined BIOCAL3D_CONDA_SCRIPT for %%D in (
    "%LOCALAPPDATA%\anaconda3"
    "%LOCALAPPDATA%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\miniforge3"
    "%PROGRAMDATA%\anaconda3"
    "%PROGRAMDATA%\miniconda3"
) do if not defined BIOCAL3D_CONDA_SCRIPT if exist "%%~D\condabin\conda.bat" set "BIOCAL3D_CONDA_SCRIPT=%%~D\condabin\conda.bat"
if not defined BIOCAL3D_CONDA_SCRIPT for /f "delims=" %%C in ('where conda.bat 2^>nul') do if not defined BIOCAL3D_CONDA_SCRIPT set "BIOCAL3D_CONDA_SCRIPT=%%C"
if not defined BIOCAL3D_CONDA_SCRIPT (
    echo Could not find Conda. Set BIOCAL3D_CONDA_BAT to your condabin\conda.bat path.
    goto failure
)

call "%BIOCAL3D_CONDA_SCRIPT%" activate biocal3d
if errorlevel 1 (
    echo Could not activate the existing biocal3d environment.
    goto failure
)
python -c "import sys; from pathlib import Path; assert Path(sys.prefix).name.lower() == 'biocal3d', 'Wrong Python environment'; import numpy, vtk, PySide6, plyfile; print('Using Python:', sys.executable)"
if errorlevel 1 (
    echo Check the error above. If annotator dependencies are missing, run:
    echo conda activate biocal3d
    echo python -m pip install -r "%~dp0requirements.txt"
    goto failure
)

python "%~dp0biocal3d_annotator.py" %*
set "BIOCAL3D_EXIT_CODE=%ERRORLEVEL%"
if not "%BIOCAL3D_EXIT_CODE%"=="0" (
    echo Annotator exited with code %BIOCAL3D_EXIT_CODE%.
    pause
)
popd
exit /b %BIOCAL3D_EXIT_CODE%

:failure
pause
popd
exit /b 1
