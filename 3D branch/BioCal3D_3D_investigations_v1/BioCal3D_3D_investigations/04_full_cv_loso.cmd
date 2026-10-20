@echo off
setlocal
pushd "%~dp0"
python biocal3d_train_3d_investigations.py --suite full --resume --live %*
set "BIOCAL_EXIT=%ERRORLEVEL%"
echo.
echo Full suite exit code: %BIOCAL_EXIT%
echo Results: runs\3d_prepared_20261013\comparison.html
pause
popd
exit /b %BIOCAL_EXIT%
