@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0"
if errorlevel 1 exit /b 1
call "%~dp0activate_biocal3d.bat"
if errorlevel 1 goto failure
python "%~dp0labeling_study.py" plan %*
set "BIOCAL3D_EXIT_CODE=%ERRORLEVEL%"
pause
popd
exit /b %BIOCAL3D_EXIT_CODE%
:failure
pause
popd
exit /b 1
