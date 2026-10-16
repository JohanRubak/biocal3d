@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0"
if errorlevel 1 exit /b 1
call "%~dp0activate_biocal3d.bat"
if errorlevel 1 goto failure
python "%~dp0biocal3d_prepare_all.py" prepare %*
set "BIOCAL3D_EXIT_CODE=%ERRORLEVEL%"
if not "%BIOCAL3D_EXIT_CODE%"=="0" echo Command exited with code %BIOCAL3D_EXIT_CODE%. Review the error and failures.csv above.
pause
popd
exit /b %BIOCAL3D_EXIT_CODE%
:failure
pause
popd
exit /b 1
