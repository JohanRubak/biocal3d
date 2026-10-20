@echo off
setlocal
cd /d "%~dp0"
call "%~dp0_activate.cmd"
if errorlevel 1 goto failed
set "DATA_ARGS="
set "OUTPUT_ARGS="
if defined BIOCAL_DATA set DATA_ARGS=--data-root "%BIOCAL_DATA%"
if defined BIOCAL_OUTPUT set OUTPUT_ARGS=--output "%BIOCAL_OUTPUT%"
python followup.py %DATA_ARGS% %OUTPUT_ARGS% %*
set "RUN_EXIT=%ERRORLEVEL%"
goto finished
:failed
set "RUN_EXIT=1"
:finished
echo.
echo Exit code: %RUN_EXIT%
if not defined BIOCAL_NO_PAUSE pause
exit /b %RUN_EXIT%
