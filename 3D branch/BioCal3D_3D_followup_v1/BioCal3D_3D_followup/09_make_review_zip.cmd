@echo off
setlocal
cd /d "%~dp0"
python export_review.py %*
set "RUN_EXIT=%ERRORLEVEL%"
echo.
echo Exit code: %RUN_EXIT%
exit /b %RUN_EXIT%
