@echo off
rem The standard study launcher now makes portable approved-ROI folders.
call "%~dp0..\create_labeling_plan.cmd" %*
exit /b %ERRORLEVEL%
