@echo off
setlocal
cd /d "%~dp0"
call "%~dp0_activate.cmd"
if errorlevel 1 goto failed
python -m unittest -v test_approved test_pathway_routing
set "RUN_EXIT=%ERRORLEVEL%"
goto finished
:failed
set "RUN_EXIT=1"
:finished
echo Exit code: %RUN_EXIT%
if not defined BIOCAL_NO_PAUSE pause
exit /b %RUN_EXIT%
