@echo off
<<<<<<< HEAD
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
=======
rem The standard study launcher now makes portable approved-ROI folders.
call "%~dp0..\create_labeling_plan.cmd" %*
exit /b %ERRORLEVEL%
>>>>>>> 3dc377b2c13f84fb1a8141a5ae83a69198e2addb
