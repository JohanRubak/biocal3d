@echo off
setlocal
cd /d "%~dp0"
python followup.py --action loso %*
set "RUN_EXIT=%ERRORLEVEL%"
echo.
echo Exit code: %RUN_EXIT%
exit /b %RUN_EXIT%
