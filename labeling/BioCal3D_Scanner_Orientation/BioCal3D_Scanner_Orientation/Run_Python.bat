@echo off
setlocal
if defined BIOCAL3D_PYTHON (
  "%BIOCAL3D_PYTHON%" %*
  exit /b
)
if /i "%CONDA_DEFAULT_ENV%"=="biocal3d" (
  python %*
  exit /b
)
where conda >nul 2>nul
if not errorlevel 1 (
  call conda run --no-capture-output -n biocal3d python %*
  exit /b
)
for %%D in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3") do (
  if exist "%%~D\condabin\conda.bat" (
    call "%%~D\condabin\conda.bat" run --no-capture-output -n biocal3d python %*
    exit /b
  )
)
python %*
