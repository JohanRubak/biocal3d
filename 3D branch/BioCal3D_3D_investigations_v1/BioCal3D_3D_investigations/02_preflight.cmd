@echo off
setlocal
pushd "%~dp0"
python biocal3d_train_3d_investigations.py --suite core --preflight %*
set "BIOCAL_EXIT=%ERRORLEVEL%"
echo.
echo Preflight exit code: %BIOCAL_EXIT%
pause
popd
exit /b %BIOCAL_EXIT%
