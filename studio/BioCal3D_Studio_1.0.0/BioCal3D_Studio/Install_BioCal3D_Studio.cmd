@echo off
setlocal
if not exist "%~dp0setup\install_windows.ps1" goto extract
if not exist "%~dp0app\studio.py" goto extract
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup\install_windows.ps1" -Source "%~dp0."
if errorlevel 1 (
 echo.
 echo Setup did not finish. See the error above and %%LOCALAPPDATA%%\BioCal3DStudio\logs.
 pause
 exit /b 1
)
exit /b 0
:extract
echo First right-click the downloaded ZIP and select Extract All.
echo Then run Install_BioCal3D_Studio.cmd from the extracted folder.
pause
exit /b 1
