@echo off
setlocal
cd /d "%~dp0"
call activate_labeling_runtime.bat
if errorlevel 1 exit /b 1
python benchmark_performance.py %*
pause
