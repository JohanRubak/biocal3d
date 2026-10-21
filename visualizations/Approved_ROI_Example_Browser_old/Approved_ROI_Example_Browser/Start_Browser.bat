@echo off
setlocal
cd /d "%~dp0"
rem Prefer the already activated environment, and never nest conda run.
if /I "%CONDA_DEFAULT_ENV%"=="biocal3d" if exist "%CONDA_PREFIX%\python.exe" goto active
where conda >nul 2>nul
if errorlevel 1 goto current_python
call conda activate biocal3d
if errorlevel 1 goto current_python
if exist "%CONDA_PREFIX%\python.exe" goto active
:current_python
python "%~dp0launch_browser.py" %*
set "browser_exit=%ERRORLEVEL%"
goto done
:active
"%CONDA_PREFIX%\python.exe" "%~dp0launch_browser.py" %*
set "browser_exit=%ERRORLEVEL%"
:done
if not "%browser_exit%"=="0" (
  echo.
  echo Startup did not complete. Please share qt_startup.log from this folder.
  pause
)
exit /b %browser_exit%
