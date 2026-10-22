@echo off
setlocal
cd /d "%~dp0"
call "%~dp0Run_Python.bat" orient_scanners.py
if errorlevel 1 echo If a dependency is missing, run Install_Missing_Dependencies.bat or use the biocal3d Anaconda Prompt.
pause
