@echo off
rem Compatibility entry point: use this package's dedicated environment.
call "%~dp0..\activate_labeling_runtime.bat"
exit /b %ERRORLEVEL%
