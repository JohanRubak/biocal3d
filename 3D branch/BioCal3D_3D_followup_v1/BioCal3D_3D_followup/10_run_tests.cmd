@echo off
setlocal
cd /d "%~dp0"
python -m unittest -v test_followup
set "RUN_EXIT=%ERRORLEVEL%"
echo.
echo Exit code: %RUN_EXIT%
exit /b %RUN_EXIT%
