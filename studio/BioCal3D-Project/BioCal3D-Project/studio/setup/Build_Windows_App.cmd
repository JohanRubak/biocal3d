@echo off
setlocal
cd /d "%~dp0.."
py -3.11 -m venv .build-env
if errorlevel 1 goto fail
".build-env\Scripts\python.exe" -m pip install -r requirements.txt pyinstaller==6.16.0
if errorlevel 1 goto fail
".build-env\Scripts\python.exe" -m PyInstaller --noconfirm setup\BioCal3D_Studio.spec
if errorlevel 1 goto fail
echo Portable Windows application created in dist\BioCal3D Studio.
echo It contains the runtime and does not need Python on the recipient's PC.
echo To create a single installer, compile setup\BioCal3D_Studio.iss with Inno Setup 6.
pause
exit /b 0
:fail
echo Build failed. Check the message above.
pause
exit /b 1
