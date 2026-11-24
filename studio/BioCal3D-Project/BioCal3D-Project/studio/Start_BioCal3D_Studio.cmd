@echo off
setlocal
if exist "%LOCALAPPDATA%\BioCal3DStudio\Start_BioCal3D_Studio.cmd" (
 call "%LOCALAPPDATA%\BioCal3DStudio\Start_BioCal3D_Studio.cmd" %*
 exit /b
)
echo Run Install_BioCal3D_Studio.cmd first.
pause
exit /b 1
