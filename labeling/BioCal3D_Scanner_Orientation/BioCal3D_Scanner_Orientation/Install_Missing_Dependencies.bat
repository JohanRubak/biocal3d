@echo off
setlocal
cd /d "%~dp0"
call "%~dp0Run_Python.bat" -m pip install -r requirements.txt
pause
